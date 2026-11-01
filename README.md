# Research Briefing: an MCP toolkit with FastMCP and the OpenAI Agents SDK

EECE 503P/798S: Agentic Systems. Assignment C5: Agent Tools.

Research Briefing is a chat assistant for students and researchers.
You give it a research topic. It finds the important papers on OpenAlex.
Then it makes an interactive slide deck about these papers, with a BibTeX file.

The tools are in an **MCP server** (`server.py`, made with FastMCP).
The agent (`agent.py`, made with the OpenAI Agents SDK) is the **MCP Host**.
The agent does not contain tool code. It starts the server, gets the list of tools, and lets the model call them through MCP.

The Part 1 summary is in [SUMMARY.md](SUMMARY.md). The reflection is in [REFLECTION.txt](REFLECTION.txt).

## Demo video

Webex meeting recording: Ghina Daoud (Student)'s meeting 
Recording link: https://aub.webex.com/aub/ldr.php?RCID=f91b58c80dfe4d4f0b39ef22ca93db90
Password: qXZhsr?2

## The three tools

| Category | Tool | What it does | Failure cases (clear error, no silent failure) |
|---|---|---|---|
| **Data** | `search_papers(topic, limit=10, from_year=None, sort_by="relevance")` | Finds papers whose title or abstract matches the topic. It does not change data. | Empty topic, bad `limit`, or bad `from_year` (fails before the request). OpenAlex unreachable. Daily limit used up. |
| **Action** | `create_research_deck(title, papers)` | Writes an HTML slide deck and a `.bib` file to `data/decks/`. It does not overwrite an earlier deck. | Empty title. 0 papers or more than 12. A paper without a title, with a year that is not possible, or two times. The tool saves nothing. |
| **Orchestration** | `research_topic(topic, n=5, from_year=None)` | Searches two times (most cited and closest matches), ranks the papers, keeps the top `n`, and makes the deck. | No paper found: returns `created: false` with the reason, and makes nothing. Errors from the Data tool go to the user. |

**Resource:** `decks://list` gives the list of decks, newest first (read-only JSON).

**Ranking:** papers with more topic words in the title come first ("LLM" also matches "large language models").
Then the tool ranks by citations per year. Thus, a strong new paper is not below an old paper only because of age.

**The deck** is one HTML file that works offline. It has a title slide, an overview (citation chart and shared topics),
one slide for each paper (key point, abstract, links), and a references slide with "Copy BibTeX" and "Download .bib".
To move between the slides, use the arrow keys or the buttons.

## How MCP is used here

```
agent.py  (MCP Host: the model, its instructions, and the chat loop)
   |
   |  MCP client (MCPServerStdio from the Agents SDK)
   |  1. starts server.py as a separate process (stdio transport)
   |  2. "list your tools"  ->  names, descriptions, input schemas
   |  3. "call research_topic with {topic: 'retrieval augmented generation', n: 4}"
   v
server.py  (MCP Server: 3 tools + 1 resource)
   |-- OpenAlex works API (free)
   '-- data/decks/*.html and *.bib
```

Other MCP hosts (the MCP Inspector, Claude Desktop, the `fastmcp` command line) can use `server.py` without a change.

## Project structure

```
C5/
├── server.py            # MCP server: the 3 tools, the resource, and the deck template (FastMCP)
├── agent.py             # MCP Host: OpenAI Agents SDK agent and terminal chat
├── tests/test_tools.py  # 20 pytest tests, through a real MCP client
├── pytest.ini
├── requirements.txt
├── SUMMARY.md           # Part 1: the summary of the MCP article and of the Fetch server
├── REFLECTION.txt       # The reflection (assignment deliverable)
├── results/             # Saved test results, failure cases, and an example briefing (Gaussian splatting)
├── .env                 # API key and settings (not in Git)
└── data/decks/          # The decks and .bib files (made when a deck is made; not in Git)
```

## Setup

You must have Python 3.10 or later. These steps are for Windows PowerShell.

1. Go to the project folder:
   ```powershell
   cd C5
   ```
2. Make the virtual environment:
   ```powershell
   python -m venv .venv
   ```
3. Activate the virtual environment:
   ```powershell
   .venv\Scripts\activate
   ```
   If PowerShell blocks the script, do this command one time, then do step 3 again:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```
4. Install the packages:
   ```powershell
   pip install -r requirements.txt
   ```
5. Make a file with the name `.env` in the project folder. Put these lines in it:
   ```
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4.1-mini
   ```

On macOS or Linux, use `source .venv/bin/activate` in step 3.

### Optional settings in `.env`

| Setting | Use |
|---|---|
| `OPENALEX_API_KEY=...` | Without a key, all users on the same network share a small daily limit (about 100 searches). A free key gives you your own limit. Get it at <https://openalex.org/settings/api>. |
| `OPEN_DECKS=0` | The agent does not open the deck in the browser. |

## How to run

Activate the virtual environment before each command.

**The agent (terminal chat):**

```powershell
python agent.py
```

At the start, the agent shows the tools that it found on the server.
For each question, it shows each `[MCP call]` and `[MCP result]`, then the answer.
When a tool makes a deck, the deck opens in the browser.
To stop, type `quit`.

**Examples of questions:**

- "Find papers on retrieval augmented generation." (uses `search_papers`)
- "Make a deck with papers 1, 2 and 4 from that list." (uses `create_research_deck`)
- "Make me a briefing on LLM agents and tool use, 5 papers, since 2024." (uses `research_topic`)

To ask fixed questions in one conversation and then stop:

```powershell
python agent.py "Find 5 recent papers on retrieval augmented generation" "Make a deck with papers 1, 2 and 4"
```

**The server alone, with the `fastmcp` command line:**

```powershell
fastmcp list server.py                                               # the tools and their schemas
fastmcp call server.py search_papers topic="model context protocol" limit=3
fastmcp call server.py search_papers topic=""                        # a failure case
fastmcp call server.py decks://list                                  # read the resource
```

## Tests

```powershell
pytest -v
```

The tests call the tools through an in-memory MCP client, as the agent does.
OpenAlex is fake in the tests. Thus, the tests do not need the network, and they always give the same result.
The decks go to a temporary folder.

The saved results are in the `results/` folder:

| File | What it contains |
|---|---|
| `pytest_results.txt` | The output of `pytest -v` (20 tests). |
| `failure_cases.txt` | 7 failure cases. Each one calls a tool on `server.py` through MCP with the `fastmcp` command line. |
| `gaussian_splatting/` | A real `research_topic` run on "gaussian splatting": the result data (JSON), the deck, and the `.bib` file. |

| Tool | Tests |
|---|---|
| Server | Has exactly 3 tools and 1 resource. |
| `search_papers` | Gives clean fields and the correct filter. Empty topic, bad `limit`, and bad `from_year` fail before the request. OpenAlex unreachable. Daily limit used up. |
| `create_research_deck` | Saves the HTML and the `.bib` file, and the resource shows them. Escapes the text. Does not overwrite. Bad input fails and saves nothing. |
| `research_topic` | Ranks by title match, then by citations per year. Counts a paper only one time. No papers: makes nothing. Gives Data tool errors to the user. |

## Design decisions

- **One job for each tool.** `search_papers` only reads. `create_research_deck` only writes, and it does not search.
  `research_topic` is the only tool that does more than one job, because that is its role.
- **Thin tools.** The logic is in plain functions. Each MCP tool calls one function.
  The Orchestration tool calls the same functions, so each part of the logic is in one place only.
- **Check the inputs first.** Bad input fails with a clear message before the request or the file write.
- **Tool annotations.** The Data tool has the read-only mark. The other two tools do not.
  Thus, a host can ask the user before it calls them.
