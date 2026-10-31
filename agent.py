"""Research Briefing agent: the MCP Host, built with the OpenAI Agents SDK.

This file has no search code and no tool code. It starts server.py as an MCP server,
asks it which tools it has, and lets the model call them. Every tool call is printed,
so you can see the full trip: agent -> MCP client -> MCP server -> back.
When a tool makes a deck, the agent opens it in the browser.

Run:
    python agent.py                            # chat in the terminal (type 'quit' to stop)
    python agent.py "question 1" "question 2"  # ask these questions in one conversation, then stop
"""

import asyncio
import json
import os
import re
import sys
import webbrowser
from datetime import date
from pathlib import Path

from agents import Agent, Runner, ToolCallItem, ToolCallOutputItem
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv

load_dotenv(override=True)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPEN_DECKS = os.getenv("OPEN_DECKS", "1") != "0"  # Set OPEN_DECKS=0 in .env to not open the browser.
HERE = Path(__file__).parent

INSTRUCTIONS = """You are Research Briefing, an assistant that helps students and researchers start on a topic.
You find research papers and turn them into an interactive slide deck. Today is {today}.

Use your tools:
- To only look at papers, use search_papers. Use sort_by="citations" for the most influential papers,
  and sort_by="relevance" (the default) for the closest matches. Show title, year, and citations.
- When the user chose the papers (for example "make a deck with papers 1, 3 and 4"), or wants a careful deck,
  use create_research_deck. Pass the papers exactly as search_papers returned them, and write a "takeaway"
  for each one: one or two plain sentences, based only on its abstract.
- When the user wants a deck on a topic in one step (for example "make me a briefing on X"), use research_topic.

What you can do: search papers on OpenAlex, and make slide decks with a .bib file.
What you cannot do: read full papers, download PDFs, check if a paper is correct, or write the user's essay.
Never invent papers, authors, numbers, or links. Every paper you mention must come from a tool result.

Rules:
- Always use a tool for papers, even if you think you know the answer. The tool decides, not you.
- Act instead of asking when you have enough to go on. No number given: use 5 papers.
  "Recent" means from_year = {recent_year}. Ask only when the topic is missing.
- Use short search keywords, for example "retrieval augmented generation", not a full question.
- If a tool returns an error, explain it to the user in simple words and suggest a fix. Do not invent data.
- research_topic and create_research_deck save the deck by themselves. When the deck is made, say so,
  give its file path, and say that it opens in the browser. Never ask "should I make it?" after that.
  If research_topic returns created=false, say that no deck was made and why.
- Keep answers short."""


def make_server() -> MCPServerStdio:
    """The MCP client side: it starts server.py as a separate process and talks MCP over stdin/stdout."""
    return MCPServerStdio(
        name="Research Briefing server",
        params={"command": sys.executable, "args": [str(HERE / "server.py")], "cwd": str(HERE)},
        cache_tools_list=True,  # The tool list does not change while the agent runs.
        client_session_timeout_seconds=60,  # OpenAlex can be slow.
    )


def make_agent(mcp_server: MCPServerStdio) -> Agent:
    today = date.today()
    return Agent(
        name="Research Briefing",
        instructions=INSTRUCTIONS.format(today=f"{today:%A}, {today.isoformat()}", recent_year=today.year - 2),
        model=MODEL,
        mcp_servers=[mcp_server],  # The agent gets its tools from the MCP server, not from this file.
    )


def as_text(output) -> str:
    """Keep only the text of a tool result."""
    if isinstance(output, dict) and "text" in output:
        return str(output["text"])
    if isinstance(output, list):
        return " ".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in output)
    return str(output)


def short(text: str, limit: int = 300) -> str:
    """One line, cut if too long."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " ..."


async def ask(agent: Agent, history: list, question: str) -> list:
    """Send one user message, print each tool call and its result, print the answer. Return the new history."""
    result = await Runner.run(agent, history + [{"role": "user", "content": question}])
    decks = []
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            call = item.raw_item
            print(f"  -> [MCP call]   {call.name}({short(json.dumps(json.loads(call.arguments), ensure_ascii=False))})")
        elif isinstance(item, ToolCallOutputItem):
            text = as_text(item.output)
            print(f"  <- [MCP result] {short(text)}")
            decks += re.findall(r'"file_url":\s*"(file:[^"]+\.html)"', text)
    print(f"\nResearch Briefing: {result.final_output}\n")
    if OPEN_DECKS:
        for deck in dict.fromkeys(decks):
            webbrowser.open(deck)
    return result.to_input_list()


async def main(questions: list[str]) -> None:
    async with make_server() as mcp_server:
        tools = await mcp_server.list_tools()
        print(f"Connected to the MCP server. Model: {MODEL}. Tools found:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description.splitlines()[0]}")
        print()

        agent = make_agent(mcp_server)
        history: list = []

        if questions:
            for question in questions:
                print(f"You: {question}")
                history = await ask(agent, history, question)
            return

        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in {"quit", "exit", "q"}:
                break
            if question:
                history = await ask(agent, history, question)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
