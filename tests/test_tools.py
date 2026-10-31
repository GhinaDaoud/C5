"""Tests for the three tools and the resource, through a real MCP client (in memory).

OpenAlex is fake here, so the tests need no network and always give the same result.
The decks go to a temporary folder.
"""

import json
from datetime import date

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

import server

THIS_YEAR = date.today().year


def work(n: int, title: str, year: int, citations: int, abstract: str = "") -> dict:
    """One OpenAlex result, in the same shape as the real API."""
    words = abstract.split()
    return {
        "id": f"https://openalex.org/W{n}",
        "doi": f"https://doi.org/10.1234/paper.{n}",
        "title": title,
        "publication_year": year,
        "cited_by_count": citations,
        "authorships": [{"author": {"display_name": "Ada Lovelace"}}, {"author": {"display_name": "Alan Turing"}}],
        "abstract_inverted_index": {w: [i for i, x in enumerate(words) if x == w] for w in set(words)},
        "primary_location": {"landing_page_url": f"https://example.org/{n}", "pdf_url": None,
                             "source": {"display_name": "Journal of Tests"}},
        "open_access": {"oa_url": f"https://example.org/{n}.pdf"},
        "topics": [{"display_name": "Machine Learning"}, {"display_name": "Software Testing"}],
    }


WORKS = [
    work(1, "An old classic", THIS_YEAR - 9, 1000, "This old paper is cited a lot. It started the field."),
    work(2, "A strong new paper", THIS_YEAR - 1, 400, "This new paper is cited fast. It improves on the classic."),
    work(3, "A quiet paper", THIS_YEAR - 3, 10, "This paper is not cited much."),
]


class FakeOpenAlex:
    """Replaces httpx.get. Remembers each request, and answers with `works` (or with an HTTP error)."""

    def __init__(self, works=WORKS, status=200):
        self.works, self.status, self.requests = works, status, []

    def __call__(self, url, params=None, timeout=None):
        self.requests.append(params)
        request = httpx.Request("GET", url)
        if self.status != 200:
            return httpx.Response(self.status, text="error", request=request)
        return httpx.Response(200, json={"meta": {"count": len(self.works)}, "results": self.works}, request=request)


@pytest.fixture
def openalex(monkeypatch):
    fake = FakeOpenAlex()
    monkeypatch.setattr(server.httpx, "get", fake)
    return fake


@pytest.fixture(autouse=True)
def decks_dir(tmp_path, monkeypatch):
    folder = tmp_path / "decks"
    monkeypatch.setattr(server, "DECKS_DIR", folder)
    return folder


async def call(tool: str, **arguments) -> dict:
    async with Client(server.mcp) as client:
        result = await client.call_tool(tool, arguments)
        return result.structured_content


async def read_decks() -> list:
    async with Client(server.mcp) as client:
        contents = await client.read_resource("decks://list")
        return json.loads(contents[0].text)


def deck_paper(title: str, **extra) -> dict:
    return {"title": title, "authors": ["Ada Lovelace"], "year": THIS_YEAR - 2, "citations": 5, **extra}


# ---------- The server ----------

async def test_server_has_three_tools_and_one_resource():
    async with Client(server.mcp) as client:
        tools = {tool.name for tool in await client.list_tools()}
        resources = [str(r.uri) for r in await client.list_resources()]
    assert tools == {"search_papers", "create_research_deck", "research_topic"}
    assert resources == ["decks://list"]


# ---------- search_papers (Data) ----------

async def test_search_returns_papers_with_clean_fields(openalex):
    result = await call("search_papers", topic="testing", limit=3, from_year=2020, sort_by="citations")
    first = result["papers"][0]
    assert result["total_found"] == 3 and len(result["papers"]) == 3
    assert first["id"] == "W1" and first["doi"] == "10.1234/paper.1"
    assert first["abstract"] == "This old paper is cited a lot. It started the field."
    assert first["pdf_url"] == "https://example.org/1.pdf"  # Taken from open_access when the location has none.

    params = openalex.requests[0]
    assert "title_and_abstract.search:testing" in params["filter"]
    assert "from_publication_date:2020-01-01" in params["filter"]
    assert params["sort"] == "cited_by_count:desc"


async def test_search_removes_characters_that_break_the_filter(openalex):
    await call("search_papers", topic="LLM, agents: tool|use")
    assert "title_and_abstract.search:LLM agents tool use," in openalex.requests[0]["filter"]


async def test_search_with_an_empty_topic_fails_before_any_request(openalex):
    with pytest.raises(ToolError, match="topic is empty"):
        await call("search_papers", topic="  ,, ")
    assert openalex.requests == []


async def test_search_with_a_bad_limit_or_year_fails_before_any_request(openalex):
    with pytest.raises(ToolError, match="limit must be between 1 and 25"):
        await call("search_papers", topic="testing", limit=30)
    with pytest.raises(ToolError, match="from_year must be between"):
        await call("search_papers", topic="testing", from_year=THIS_YEAR + 5)
    assert openalex.requests == []


async def test_search_when_openalex_is_unreachable(monkeypatch):
    def no_network(*args, **kwargs):
        raise httpx.ConnectError("no network")
    monkeypatch.setattr(server.httpx, "get", no_network)
    with pytest.raises(ToolError, match="OpenAlex is unreachable"):
        await call("search_papers", topic="testing")


async def test_search_when_the_daily_limit_is_used_up(monkeypatch):
    monkeypatch.setattr(server.httpx, "get", FakeOpenAlex(status=429))
    with pytest.raises(ToolError, match="free daily limit"):
        await call("search_papers", topic="testing")


# ---------- create_research_deck (Action) ----------

async def test_deck_is_saved_with_a_bib_file_and_shown_in_the_resource(decks_dir):
    papers = [deck_paper("First <paper>", takeaway="It works.", doi="10.1/a", venue="Journal of Tests"),
              deck_paper("Second paper")]
    deck = await call("create_research_deck", title="My topic", papers=papers)

    page = (decks_dir / "my-topic.html").read_text(encoding="utf-8")
    assert deck["papers"] == 2 and deck["slides"] == 5
    assert "First &lt;paper&gt;" in page and "<paper>" not in page  # Text from the data is escaped.
    assert "It works." in page
    bib = (decks_dir / "my-topic.bib").read_text(encoding="utf-8")
    assert "@article{lovelace" in bib and "@misc{lovelace" in bib  # No venue: @misc.
    assert [d["file"] for d in await read_decks()] == ["my-topic.html"]


async def test_deck_never_overwrites_an_earlier_deck(decks_dir):
    await call("create_research_deck", title="Same title", papers=[deck_paper("A")])
    second = await call("create_research_deck", title="Same title", papers=[deck_paper("B")])
    assert second["path"].endswith("same-title-2.html")
    assert len(await read_decks()) == 2


@pytest.mark.parametrize("papers, message", [
    ([], "between 1 and 12 papers"),
    ([deck_paper("A"), deck_paper("A")], "in the list twice"),
    ([deck_paper("A", year=3000)], "not possible"),
    ([deck_paper("A", takeaway="x" * 600)], "Keep it under 500"),
])
async def test_bad_deck_input_fails_and_saves_nothing(decks_dir, papers, message):
    with pytest.raises(ToolError, match=message):
        await call("create_research_deck", title="Bad deck", papers=papers)
    assert not decks_dir.exists()


# ---------- research_topic (Orchestration) ----------

async def test_briefing_ranks_by_citations_per_year_and_makes_the_deck(openalex, decks_dir):
    result = await call("research_topic", topic="testing", n=2)

    # The new paper has fewer citations, but more per year (400 / 2 > 1000 / 10), so it comes first.
    assert [p["title"] for p in result["papers"]] == ["A strong new paper", "An old classic"]
    assert result["created"] is True and result["considered"] == 3
    assert result["papers"][0]["takeaway"] == "This new paper is cited fast."  # First sentence of the abstract.
    assert [r["sort"] for r in openalex.requests] == ["cited_by_count:desc", "relevance_score:desc"]
    assert (decks_dir / "research-briefing-testing.html").exists()


async def test_briefing_puts_on_topic_titles_first(openalex):
    # Both searches return the same 3 papers: they are counted once.
    result = await call("research_topic", topic="classic", n=3)
    assert result["considered"] == 3
    assert result["papers"][0]["title"] == "An old classic"  # Title matches, so it beats more citations per year.
    assert result["papers"][0]["title_match"] == 1.0


async def test_briefing_with_no_papers_makes_nothing(monkeypatch, decks_dir):
    monkeypatch.setattr(server.httpx, "get", FakeOpenAlex(works=[]))
    result = await call("research_topic", topic="nothing matches this")
    assert result["created"] is False and result["deck"] is None
    assert "No deck was made" in result["reason"]
    assert not decks_dir.exists()


async def test_briefing_passes_on_errors_from_the_data_tool(monkeypatch, decks_dir):
    monkeypatch.setattr(server.httpx, "get", FakeOpenAlex(status=503))
    with pytest.raises(ToolError, match="HTTP 503"):
        await call("research_topic", topic="testing")
    assert not decks_dir.exists()


# ---------- Small helpers ----------

def test_title_match_knows_plurals_and_acronyms():
    assert server.title_match("LLM agents", "Large Language Model Agent Systems") == 1.0
    assert server.title_match("model context protocol", "Lost in the Middle: How Language Models Use Long Contexts") \
        == 0.67


def test_key_sentence_prefers_what_the_paper_does():
    abstract = "LLMs are popular, e.g. ChatGPT. They often fail at chemistry. We introduce ChemCrow, an agent."
    assert server.key_sentence(abstract) == "We introduce ChemCrow, an agent."
    assert server.key_sentence("No claim here at all. Just background.") == "No claim here at all."


def test_clean_text_removes_latex_and_html():
    assert server.clean_text(r"<jats:p>Uses \textit{agentic skills} and $x$ </jats:p>") == "Uses agentic skills and x"
