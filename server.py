"""Research Briefing: an MCP server with three tools and one resource.
 
Tools:
- search_papers         (Data tool)          finds papers on OpenAlex. It changes nothing.
- create_research_deck  (Action tool)        writes an interactive HTML slide deck (and a .bib file) to data/decks/.
- research_topic        (Orchestration tool) uses the two tools above: it searches, ranks, keeps the best papers,
                                             then makes the deck.

Resource:
- decks://list          the decks made so far, read-only.

Run it alone:   python server.py      (the server talks MCP over stdin/stdout)
"""

import html
import json
import os
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Literal

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

load_dotenv(override=True)

# OpenAlex is free. Without a key, everyone on the same network shares a small daily budget (about 100 searches).
# A free key (https://openalex.org/settings/api) gives you your own budget: set OPENALEX_API_KEY in .env.
OPENALEX_URL = os.getenv("OPENALEX_URL", "https://api.openalex.org/works")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")

# The decks. The tests point this to a temporary folder.
DECKS_DIR = Path(__file__).parent / "data" / "decks"

MAX_SEARCH_RESULTS = 25
MAX_DECK_PAPERS = 12
MAX_BRIEFING_PAPERS = 10
MAX_TAKEAWAY_CHARS = 500
OLDEST_YEAR = 1900

SELECT_FIELDS = ("id,doi,title,publication_year,cited_by_count,authorships,abstract_inverted_index,"
                 "primary_location,open_access,topics")
SORT_ORDERS = {"relevance": "relevance_score:desc", "citations": "cited_by_count:desc"}
SortBy = Literal["relevance", "citations"]


# ---------- Types (FastMCP turns these into the input and output schemas of each tool) ----------

class Paper(BaseModel):
    id: str = Field(default="", description="OpenAlex id, for example 'W2963403868'.")
    title: str = Field(description="Title of the paper.")
    authors: list[str] = Field(default_factory=list, description="Author names, first author first (at most 10).")
    year: int | None = Field(default=None, description="Year of publication.")
    citations: int = Field(default=0, description="How many works cite this paper (OpenAlex count).")
    venue: str | None = Field(default=None, description="Journal, conference, or repository, for example 'arXiv'.")
    doi: str | None = Field(default=None, description="DOI without the https://doi.org/ part.")
    url: str | None = Field(default=None, description="Page of the paper.")
    pdf_url: str | None = Field(default=None, description="Free PDF, if there is one.")
    abstract: str = Field(default="", description="Abstract, if OpenAlex has one.")
    topics: list[str] = Field(default_factory=list, description="Research topics given by OpenAlex.")


class SearchResult(BaseModel):
    topic: str
    sort_by: SortBy
    total_found: int = Field(description="How many works on OpenAlex match the search.")
    papers: list[Paper] = Field(description="In the order of sort_by: best match or most cited first.")


class DeckPaper(Paper):
    takeaway: str = Field(default="", description=f"The key point of the paper in one or two sentences "
                                                  f"(at most {MAX_TAKEAWAY_CHARS} characters).")


class Deck(BaseModel):
    title: str
    papers: int = Field(description="Number of papers in the deck.")
    slides: int = Field(description="Number of slides.")
    path: str = Field(description="Where the HTML deck was saved.")
    file_url: str = Field(description="file:// link to open the deck in a browser.")
    bib_path: str = Field(description="Where the BibTeX file was saved.")


class RankedPaper(BaseModel):
    title: str
    year: int | None
    citations: int
    title_match: float = Field(description="Share of the topic words in the title, from 0 to 1.")
    citations_per_year: float
    takeaway: str


class BriefingResult(BaseModel):
    created: bool = Field(description="True if papers were found and the deck was made.")
    reason: str = Field(description="What was done, or why no deck was made.")
    deck: Deck | None = Field(description="The deck. None if nothing was made.")
    considered: int = Field(description="How many search results were ranked.")
    papers: list[RankedPaper] = Field(description="The papers in the deck, best first.")


# ---------- Plain functions (the logic). The MCP tools below call these. ----------

def fetch_json(url: str, params: dict) -> dict:
    """Send one GET request and return the JSON. Raise a clear ToolError if the service fails."""
    try:
        response = httpx.get(url, params=params, timeout=20)
    except httpx.HTTPError as error:
        raise ToolError(f"OpenAlex is unreachable right now ({type(error).__name__}). Try again later.")
    if response.status_code == 429:
        raise ToolError("OpenAlex refused the request: the free daily limit is used up. "
                        "Try again tomorrow, or add a free OPENALEX_API_KEY to .env.")
    if response.status_code != 200:
        raise ToolError(f"OpenAlex returned an error (HTTP {response.status_code}): {response.text[:200]}")
    return response.json()


def clean_text(text: str) -> str:
    """Remove the markup that some abstracts have: HTML tags (<jats:p>) and LaTeX (\\textit{...}, $x$)."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", text)
    text = text.replace("$", "")
    return " ".join(text.split())


def rebuild_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores the abstract as {word: [positions]}. Put the words back in order."""
    if not inverted_index:
        return ""
    positions = {pos: word for word, places in inverted_index.items() for pos in places}
    return clean_text(" ".join(positions[i] for i in sorted(positions)))


def short_id(url: str | None, prefix: str) -> str | None:
    """'https://doi.org/10.1/x' -> '10.1/x'. 'https://openalex.org/W1' -> 'W1'."""
    return url.removeprefix(prefix) if url else None


def to_paper(work: dict) -> Paper:
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    open_access = work.get("open_access") or {}
    return Paper(
        id=short_id(work.get("id"), "https://openalex.org/") or "",
        title=clean_text(work.get("title") or "Untitled"),
        authors=[a["author"]["display_name"] for a in (work.get("authorships") or [])[:10]
                 if a.get("author", {}).get("display_name")],
        year=work.get("publication_year"),
        citations=work.get("cited_by_count") or 0,
        venue=source.get("display_name"),
        doi=short_id(work.get("doi"), "https://doi.org/"),
        url=location.get("landing_page_url") or work.get("doi"),
        pdf_url=location.get("pdf_url") or open_access.get("oa_url"),
        abstract=rebuild_abstract(work.get("abstract_inverted_index")),
        topics=[t["display_name"] for t in (work.get("topics") or [])[:3]],
    )


def check_from_year(from_year: int | None) -> None:
    this_year = date.today().year
    if from_year is not None and not OLDEST_YEAR <= from_year <= this_year:
        raise ToolError(f"from_year must be between {OLDEST_YEAR} and {this_year}. You gave {from_year}.")


def search(topic: str, limit: int, from_year: int | None = None, sort_by: SortBy = "relevance") -> SearchResult:
    """Check the inputs, then ask OpenAlex for papers whose title or abstract matches the topic."""
    # Commas, colons, and pipes have a meaning in OpenAlex filters, so keep only words.
    topic = " ".join(re.sub(r"[^\w\s'-]", " ", topic).split())
    if not topic:
        raise ToolError("The topic is empty. Give a research topic, for example 'retrieval augmented generation'.")
    if not 1 <= limit <= MAX_SEARCH_RESULTS:
        raise ToolError(f"limit must be between 1 and {MAX_SEARCH_RESULTS}. You gave {limit}.")
    if sort_by not in SORT_ORDERS:
        raise ToolError(f"sort_by must be 'relevance' or 'citations'. You gave '{sort_by}'.")
    check_from_year(from_year)

    filters = [f"title_and_abstract.search:{topic}", "type:article|preprint|review"]
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")
    params = {"filter": ",".join(filters), "sort": SORT_ORDERS[sort_by], "per-page": limit, "select": SELECT_FIELDS}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    data = fetch_json(OPENALEX_URL, params)
    papers = [to_paper(work) for work in data.get("results") or []]
    return SearchResult(topic=topic, sort_by=sort_by, papers=papers,
                        total_found=(data.get("meta") or {}).get("count", len(papers)))


def citations_per_year(paper: Paper) -> float:
    """Citations divided by age, so a strong new paper is not buried under old ones."""
    age = max(0, date.today().year - (paper.year or date.today().year))
    return round(paper.citations / (age + 1), 1)


STOP_WORDS = {"a", "an", "the", "of", "for", "in", "on", "and", "or", "with", "to", "by", "using", "use", "via", "from"}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def title_match(topic: str, title: str) -> float:
    """Share of the topic words that are in the title, from 0 to 1.

    'agents' matches 'agent' (same first 5 letters), and 'llm' matches 'large language models' (initials).
    """
    topic_words = [w for w in words(topic) if w not in STOP_WORDS] or words(topic)
    title_words = words(title)

    def found(word: str) -> bool:
        if any(t.startswith(word[:5]) and word.startswith(t[:5]) for t in title_words):
            return True
        n = len(word)
        return n >= 2 and any("".join(t[0] for t in title_words[i:i + n]) == word
                              for i in range(len(title_words) - n + 1))

    return round(sum(found(w) for w in topic_words) / len(topic_words), 2) if topic_words else 0.0


CONTRIBUTION = re.compile(r"\b(we|this (paper|study|work|article|review|survey))\b.{0,40}?\b(propose|introduce|"
                          r"present|show|develop|demonstrate|find|review|describe|examine|explore|report)",
                          re.IGNORECASE)


def sentences(text: str) -> list[str]:
    """Split on . ! ? followed by a capital letter. Parts under 20 characters stay joined ('e.g. The', 'U.S. Army')."""
    parts, current = [], ""
    for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z])", " ".join(text.split())):
        current = f"{current} {piece}".strip()
        if len(current) >= 20:
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return parts


def key_sentence(abstract: str, limit: int = 300) -> str:
    """The takeaway when no model writes one: the sentence that says what the paper does, else the first one."""
    found = sentences(abstract)
    if not found:
        return ""
    sentence = next((s for s in found if CONTRIBUTION.search(s)), found[0])
    return sentence if len(sentence) <= limit else sentence[:limit].rsplit(" ", 1)[0] + " ..."


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "deck"


def bibtex_entries(papers: list[DeckPaper]) -> str:
    """A BibTeX entry for each paper, made from its data (no extra request)."""
    entries, used_keys = [], Counter()
    for paper in papers:
        last_name = re.sub(r"[^A-Za-z]", "", (paper.authors[0].split()[-1] if paper.authors else "anon"))
        first_word = next((w for w in re.findall(r"[A-Za-z]+", paper.title) if len(w) > 3), "paper")
        key = f"{last_name.lower() or 'anon'}{paper.year or ''}{first_word.lower()}"
        used_keys[key] += 1
        if used_keys[key] > 1:
            key += chr(ord("a") + used_keys[key] - 2)

        kind = "misc" if (paper.venue or "").lower().startswith("arxiv") or not paper.venue else "article"
        fields = {
            "title": "{" + paper.title.replace("{", "").replace("}", "") + "}",
            "author": " and ".join(paper.authors) or "Unknown",
            "year": str(paper.year) if paper.year else None,
            "journal" if kind == "article" else "howpublished": paper.venue,
            "doi": paper.doi,
            "url": paper.url,
        }
        body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields.items() if value)
        entries.append(f"@{kind}{{{key},\n{body}\n}}")
    return "\n\n".join(entries) + "\n"


def check_deck_inputs(title: str, papers: list[DeckPaper]) -> str:
    title = " ".join(title.split())
    if not title:
        raise ToolError("The deck title is empty. Give a title, for example 'Retrieval augmented generation'.")
    if not 1 <= len(papers) <= MAX_DECK_PAPERS:
        raise ToolError(f"A deck needs between 1 and {MAX_DECK_PAPERS} papers. You gave {len(papers)}.")
    seen = set()
    for number, paper in enumerate(papers, start=1):
        if not paper.title.strip():
            raise ToolError(f"Paper {number} has an empty title.")
        if paper.year is not None and not OLDEST_YEAR <= paper.year <= date.today().year + 1:
            raise ToolError(f"Paper {number} ('{paper.title[:60]}') has year {paper.year}, which is not possible.")
        if paper.citations < 0:
            raise ToolError(f"Paper {number} ('{paper.title[:60]}') has a negative citation count.")
        if len(paper.takeaway) > MAX_TAKEAWAY_CHARS:
            raise ToolError(f"The takeaway of paper {number} is {len(paper.takeaway)} characters long. "
                            f"Keep it under {MAX_TAKEAWAY_CHARS}.")
        key = paper.id or paper.title.strip().lower()
        if key in seen:
            raise ToolError(f"Paper {number} ('{paper.title[:60]}') is in the list twice.")
        seen.add(key)
    return title


def create_deck(title: str, papers: list[DeckPaper]) -> Deck:
    """Check the inputs, then write the HTML deck and the .bib file. Nothing is written if a check fails."""
    title = check_deck_inputs(title, papers)
    bibtex = bibtex_entries(papers)
    page = render_deck(title, papers, bibtex)

    DECKS_DIR.mkdir(parents=True, exist_ok=True)
    stem, number = slugify(title), 1
    while (DECKS_DIR / f"{stem}.html").exists():  # Never overwrite an earlier deck.
        number += 1
        stem = f"{slugify(title)}-{number}"
    deck_path, bib_path = DECKS_DIR / f"{stem}.html", DECKS_DIR / f"{stem}.bib"
    deck_path.write_text(page, encoding="utf-8")
    bib_path.write_text(bibtex, encoding="utf-8")

    return Deck(title=title, papers=len(papers), slides=len(papers) + 3, path=str(deck_path.resolve()),
                file_url=deck_path.resolve().as_uri(), bib_path=str(bib_path.resolve()))


def research(topic: str, n: int, from_year: int | None = None) -> BriefingResult:
    """Search two ways, rank, keep the top n, and make the deck. Make nothing if no paper is found."""
    if not 1 <= n <= MAX_BRIEFING_PAPERS:
        raise ToolError(f"n must be between 1 and {MAX_BRIEFING_PAPERS}. You gave {n}.")

    # Step 1: the Data tool logic, two times. The most cited matches can be off topic,
    # and the closest matches can be new papers that nobody cites yet. Together they cover both.
    most_cited = search(topic, MAX_SEARCH_RESULTS, from_year, sort_by="citations")
    closest = search(topic, MAX_SEARCH_RESULTS, from_year, sort_by="relevance")
    topic, since = most_cited.topic, f" since {from_year}" if from_year else ""

    # A preprint and its published version have the same title: keep the one with more citations.
    pool: dict[str, Paper] = {}
    for paper in sorted(most_cited.papers + closest.papers, key=lambda p: p.citations):
        pool[" ".join(words(paper.title))] = paper

    if not pool:
        return BriefingResult(created=False, considered=0, deck=None, papers=[],
                              reason=f"OpenAlex has no papers on '{topic}'{since}. No deck was made. "
                                     f"Try a broader topic or an earlier year.")

    # Step 2: on-topic titles first, then the most cited per year.
    ranked = sorted(pool.values(), key=lambda p: (title_match(topic, p.title), citations_per_year(p)), reverse=True)
    chosen = [DeckPaper(**p.model_dump(), takeaway=key_sentence(p.abstract)) for p in ranked[:n]]
    deck = create_deck(f"Research briefing: {topic}", chosen)  # Step 3: the Action tool logic.

    fewer = f" Only {len(chosen)} were found." if len(chosen) < n else ""
    return BriefingResult(
        created=True, considered=len(pool), deck=deck,
        papers=[RankedPaper(title=p.title, year=p.year, citations=p.citations, title_match=title_match(topic, p.title),
                            citations_per_year=citations_per_year(p), takeaway=p.takeaway) for p in chosen],
        reason=(f"Deck made with the top {len(chosen)} of {len(pool)} papers{since} (the most cited and the closest "
                f"matches), ranked by how well the title matches, then by citations per year.{fewer} "
                f"Saved to {deck.path}."),
    )


def list_decks() -> list[dict]:
    if not DECKS_DIR.exists():
        return []
    decks = sorted(DECKS_DIR.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"file": p.name, "path": str(p.resolve()), "file_url": p.resolve().as_uri(),
             "created": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")} for p in decks]


# ---------- The HTML deck ----------

def esc(text) -> str:
    return html.escape(str(text), quote=True)


def link(url: str | None, label: str) -> str:
    if not url or not url.startswith(("http://", "https://")):
        return ""
    return f'<a class="button" href="{esc(url)}" target="_blank" rel="noopener">{esc(label)}</a>'


def author_line(authors: list[str]) -> str:
    if not authors:
        return "Unknown authors"
    return ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")


def citation_chart(papers: list[DeckPaper]) -> str:
    """Horizontal bars, one per paper: the citation count. Inline SVG, so the deck works offline."""
    most = max((p.citations for p in papers), default=0) or 1
    row, width, label_w = 34, 640, 250
    bars = []
    for i, paper in enumerate(papers):
        y = i * row
        bar = max(2, (width - label_w - 70) * paper.citations / most)
        name = paper.title if len(paper.title) <= 34 else paper.title[:33].rstrip() + "…"
        bars.append(
            f'<text class="chart-label" x="0" y="{y + 21}">{i + 1}. {esc(name)}</text>'
            f'<rect class="chart-bar" x="{label_w}" y="{y + 8}" width="{bar:.0f}" height="18" rx="3">'
            f'<title>{esc(paper.title)}: {paper.citations} citations</title></rect>'
            f'<text class="chart-value" x="{label_w + bar + 8:.0f}" y="{y + 21}">{paper.citations:,}'
            f'{" · " + str(paper.year) if paper.year else ""}</text>')
    return (f'<svg class="chart" viewBox="0 0 {width} {len(papers) * row + 6}" role="img" '
            f'aria-label="Citations per paper">{"".join(bars)}</svg>')


def render_deck(title: str, papers: list[DeckPaper], bibtex: str) -> str:
    years = [p.year for p in papers if p.year]
    year_span = (f"{min(years)}–{max(years)}" if min(years) != max(years) else str(years[0])) if years else "n/a"
    total_citations = sum(p.citations for p in papers)
    free_pdfs = sum(1 for p in papers if p.pdf_url)
    topic_counts = Counter(t for p in papers for t in dict.fromkeys(p.topics)).most_common(6)

    slides = [f"""
<section class="slide title-slide">
  <p class="kicker">Research briefing</p>
  <h1>{esc(title.removeprefix("Research briefing: "))}</h1>
  <p class="lead">{len(papers)} papers · {year_span} · {total_citations:,} citations in total</p>
  <p class="muted">Made on {date.today():%d %B %Y} from OpenAlex data. Use ← → or the buttons below.</p>
</section>""", f"""
<section class="slide">
  <p class="kicker">At a glance</p>
  <h2>The papers in this briefing</h2>
  <div class="stats">
    <div class="stat"><span>{len(papers)}</span>papers</div>
    <div class="stat"><span>{year_span}</span>years</div>
    <div class="stat"><span>{total_citations:,}</span>citations</div>
    <div class="stat"><span>{free_pdfs}/{len(papers)}</span>free PDFs</div>
  </div>
  <div class="two-col">
    <div><h3>Citations</h3>{citation_chart(papers)}</div>
    <div><h3>Shared topics</h3>
      <ul class="topics">{"".join(f'<li><b>{c}/{len(papers)}</b> {esc(t)}</li>' for t, c in topic_counts)
                          or '<li>No topics in the data.</li>'}</ul>
    </div>
  </div>
</section>"""]

    for i, paper in enumerate(papers, start=1):
        meta = " · ".join(esc(part) for part in (author_line(paper.authors), paper.venue, paper.year) if part)
        abstract = (f'<details><summary>Show the abstract</summary><p>{esc(paper.abstract)}</p></details>'
                    if paper.abstract else '<p class="muted">OpenAlex has no abstract for this paper.</p>')
        takeaway = paper.takeaway or key_sentence(paper.abstract) or "No summary available."
        slides.append(f"""
<section class="slide">
  <p class="kicker">Paper {i} of {len(papers)}</p>
  <h2>{esc(paper.title)}</h2>
  <p class="meta">{meta}</p>
  <div class="chips">
    <span class="chip"><b>{paper.citations:,}</b> citations</span>
    <span class="chip"><b>{citations_per_year(paper)}</b> per year</span>
    <span class="chip">{"Free PDF" if paper.pdf_url else "No free PDF"}</span>
    {"".join(f'<span class="chip soft">{esc(t)}</span>' for t in paper.topics)}
  </div>
  <blockquote><span class="label">Key point</span>{esc(takeaway)}</blockquote>
  {abstract}
  <div class="links">{link(paper.url, "Open the paper")}{link(paper.pdf_url, "PDF")}
    {link(f"https://doi.org/{paper.doi}" if paper.doi else None, "DOI")}</div>
</section>""")

    references = "".join(
        f'<li>{esc(author_line(p.authors))} ({p.year or "n.d."}). <i>{esc(p.title)}</i>.'
        f'{" " + esc(p.venue) + "." if p.venue else ""}{" doi:" + esc(p.doi) if p.doi else ""}</li>' for p in papers)
    slides.append(f"""
<section class="slide">
  <p class="kicker">References</p>
  <h2>Cite these papers</h2>
  <ol class="refs">{references}</ol>
  <div class="links">
    <button class="button" id="copy-bib" type="button">Copy BibTeX</button>
    <button class="button" id="download-bib" type="button">Download .bib</button>
    <span id="copy-status" class="muted" aria-live="polite"></span>
  </div>
  <details><summary>Show the BibTeX</summary><pre id="bibtex">{esc(bibtex)}</pre></details>
</section>""")

    return DECK_TEMPLATE.replace("{{TITLE}}", esc(title)).replace("{{SLIDES}}", "".join(slides)).replace(
        "{{BIB_NAME}}", esc(slugify(title) + ".bib"))


DECK_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
:root {
  --bg: #f6f5f1; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6a64; --line: #dedcd3;
  --accent: #2f5bd3; --accent-soft: #e6ecfb; --chip: #efeee8;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #151614; --panel: #1f201d; --text: #ecebe6; --muted: #a3a29b; --line: #34352f;
          --accent: #8aa7ff; --accent-soft: #263055; --chip: #2a2b27; }
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
body { background: var(--bg); color: var(--text);
       font: 17px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; overflow: hidden; }
.slide { position: absolute; inset: 0 0 64px 0; padding: 48px max(16px, calc((100vw - 1040px) / 2));
         overflow-y: auto; opacity: 0; visibility: hidden; transform: translateX(24px);
         transition: opacity .25s ease, transform .25s ease, visibility .25s; }
.slide.active { opacity: 1; visibility: visible; transform: none; }
.title-slide { display: flex; flex-direction: column; justify-content: center; }
h1 { font-size: clamp(34px, 6vw, 60px); line-height: 1.1; margin: 0 0 16px; letter-spacing: -0.02em; }
h2 { font-size: clamp(24px, 3.4vw, 36px); line-height: 1.2; margin: 0 0 10px; letter-spacing: -0.01em; }
h3 { font-size: 15px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 0 0 12px; }
.kicker { color: var(--accent); font-weight: 600; text-transform: uppercase; letter-spacing: .08em;
          font-size: 13px; margin: 0 0 10px; }
.lead { font-size: 22px; margin: 0 0 8px; }
.muted, .meta { color: var(--muted); }
.meta { margin: 0 0 18px; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 22px 0 28px; }
.stat { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;
        color: var(--muted); font-size: 14px; }
.stat span { display: block; font-size: 26px; font-weight: 700; color: var(--text); }
.two-col { display: grid; grid-template-columns: 3fr 2fr; gap: 32px; }
.chart { width: 100%; height: auto; }
.chart-label { fill: var(--text); font-size: 13px; }
.chart-value { fill: var(--muted); font-size: 12px; }
.chart-bar { fill: var(--accent); }
.topics { list-style: none; padding: 0; margin: 0; }
.topics li { padding: 8px 0; border-bottom: 1px solid var(--line); }
.topics b { display: inline-block; min-width: 42px; color: var(--accent); }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 22px; }
.chip { background: var(--chip); border-radius: 999px; padding: 4px 12px; font-size: 14px; }
.chip.soft { background: transparent; border: 1px solid var(--line); color: var(--muted); }
blockquote { margin: 0 0 20px; padding: 18px 22px; background: var(--accent-soft); border-radius: 10px;
             border-left: 4px solid var(--accent); font-size: 19px; }
.label { display: block; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em;
         color: var(--accent); margin-bottom: 6px; }
details { margin: 0 0 20px; }
summary { cursor: pointer; color: var(--accent); font-weight: 600; }
details p { color: var(--muted); max-width: 75ch; }
pre { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px;
      overflow-x: auto; font-size: 13px; }
.links { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 8px 0 18px; }
.button { display: inline-block; background: var(--panel); color: var(--text); border: 1px solid var(--line);
          border-radius: 8px; padding: 8px 14px; font: inherit; font-size: 15px; text-decoration: none;
          cursor: pointer; }
.button:hover { border-color: var(--accent); color: var(--accent); }
.refs li { margin-bottom: 8px; }
nav { position: fixed; left: 0; right: 0; bottom: 0; height: 64px; display: flex; align-items: center;
      justify-content: center; gap: 16px; border-top: 1px solid var(--line); background: var(--bg); }
nav .button { min-width: 44px; }
#counter { min-width: 64px; text-align: center; color: var(--muted); font-variant-numeric: tabular-nums; }
#progress { position: fixed; top: 0; left: 0; height: 3px; background: var(--accent); transition: width .25s; }
@media (max-width: 720px) {
  .slide { padding-top: 28px; }
  .stats { grid-template-columns: repeat(2, 1fr); }
  .two-col { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) { .slide, #progress { transition: none; } }
</style>
</head>
<body>
<div id="progress"></div>
<main>{{SLIDES}}</main>
<nav>
  <button class="button" id="prev" type="button" aria-label="Previous slide">←</button>
  <span id="counter"></span>
  <button class="button" id="next" type="button" aria-label="Next slide">→</button>
</nav>
<script>
(function () {
  var slides = document.querySelectorAll(".slide");
  var current = 0;
  function show(index) {
    current = Math.max(0, Math.min(slides.length - 1, index));
    slides.forEach(function (slide, i) { slide.classList.toggle("active", i === current); });
    document.getElementById("counter").textContent = (current + 1) + " / " + slides.length;
    document.getElementById("progress").style.width = ((current + 1) / slides.length * 100) + "%";
    history.replaceState(null, "", "#" + (current + 1));
  }
  document.getElementById("prev").onclick = function () { show(current - 1); };
  document.getElementById("next").onclick = function () { show(current + 1); };
  document.addEventListener("keydown", function (event) {
    if (event.target.closest && event.target.closest("summary, input, textarea")) return;
    if (["ArrowRight", "PageDown", " "].indexOf(event.key) >= 0) { event.preventDefault(); show(current + 1); }
    if (["ArrowLeft", "PageUp"].indexOf(event.key) >= 0) { event.preventDefault(); show(current - 1); }
    if (event.key === "Home") show(0);
    if (event.key === "End") show(slides.length - 1);
  });
  var touchX = null;
  document.addEventListener("touchstart", function (e) { touchX = e.touches[0].clientX; });
  document.addEventListener("touchend", function (e) {
    if (touchX === null) return;
    var dx = e.changedTouches[0].clientX - touchX;
    if (Math.abs(dx) > 60) show(current + (dx < 0 ? 1 : -1));
    touchX = null;
  });

  var bibtex = document.getElementById("bibtex").textContent;
  var status = document.getElementById("copy-status");
  document.getElementById("copy-bib").onclick = function () {
    function done() { status.textContent = "Copied."; }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(bibtex).then(done, fallback);
    } else { fallback(); }
    function fallback() {
      var area = document.createElement("textarea");
      area.value = bibtex; document.body.appendChild(area); area.select();
      try { document.execCommand("copy"); done(); } catch (e) { status.textContent = "Copy failed. Use 'Show the BibTeX'."; }
      document.body.removeChild(area);
    }
  };
  document.getElementById("download-bib").onclick = function () {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([bibtex], { type: "application/x-bibtex" }));
    a.download = "{{BIB_NAME}}"; a.click();
  };

  var start = parseInt(location.hash.slice(1), 10);
  show(isNaN(start) ? 0 : start - 1);
})();
</script>
</body>
</html>
"""


# ---------- The MCP server ----------

mcp = FastMCP(
    name="Research Briefing",
    instructions=(
        "Tools to find research papers and turn them into an interactive slide deck. "
        "Use search_papers to look at papers only. "
        "Use create_research_deck when the papers are already chosen (for example from search_papers). "
        "Use research_topic when the user wants you to pick the best papers on a topic and make the deck in one go."
    ),
)


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
def search_papers(
    topic: Annotated[str, Field(description="Research topic or keywords, for example 'retrieval augmented generation'.")],
    limit: Annotated[int, Field(description=f"How many papers to return, from 1 to {MAX_SEARCH_RESULTS}.")] = 10,
    from_year: Annotated[int | None, Field(description="Only papers published in this year or later.")] = None,
    sort_by: Annotated[SortBy, Field(description="'relevance' (best match first) or 'citations' "
                                                 "(most cited first).")] = "relevance",
) -> SearchResult:
    """Search OpenAlex for research papers whose title or abstract matches a topic.

    Returns each paper with its title, authors, year, citation count, venue, DOI, links, abstract,
    and topics. This tool changes nothing.
    Fails with a clear error if the topic is empty, `limit` or `from_year` is out of range, or OpenAlex is unreachable.
    """
    return search(topic, limit, from_year, sort_by)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False})
def create_research_deck(
    title: Annotated[str, Field(description="Title of the deck, for example 'LLM agents and tool use'.")],
    papers: Annotated[list[DeckPaper], Field(description=f"The papers to present, in order (1 to {MAX_DECK_PAPERS}). "
                                                         f"Pass them as search_papers returned them, and add a "
                                                         f"short 'takeaway' for each.")],
) -> Deck:
    """Create an interactive HTML slide deck from papers that are already chosen.

    The deck has a title slide, an overview (citation chart and shared topics), one slide per paper
    (key point, abstract, links), and a references slide with BibTeX. A .bib file is saved next to it.
    Saves to data/decks/ and never overwrites an earlier deck. Does not search for papers.
    Fails with a clear error, and saves nothing, if the title is empty, there are 0 or too many papers,
    a paper has no title or an impossible year, or a paper is listed twice.
    """
    return create_deck(title, papers)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
def research_topic(
    topic: Annotated[str, Field(description="Research topic, for example 'LLM agents tool use'.")],
    n: Annotated[int, Field(description=f"How many papers to put in the deck, from 1 to {MAX_BRIEFING_PAPERS}.")] = 5,
    from_year: Annotated[int | None, Field(description="Only papers published in this year or later.")] = None,
) -> BriefingResult:
    """Find the most influential papers on a topic, then make a slide deck about them.

    Steps: (1) gets the 25 most cited and the 25 closest matches on OpenAlex (like search_papers),
    (2) ranks them: papers with more topic words in the title first, then by citations per year,
    so strong new papers are not buried under old ones, (3) keeps the top n, and uses the sentence
    of each abstract that says what the paper does as its key point, (4) makes the deck
    (like create_research_deck).
    If no paper is found, returns created=false with the reason, and makes nothing.
    """
    return research(topic, n, from_year)


@mcp.resource("decks://list", mime_type="application/json", description="All decks made so far, newest first.")
def decks() -> str:
    return json.dumps(list_decks(), indent=2)


if __name__ == "__main__":
    mcp.run(show_banner=False)  # stdio transport by default
