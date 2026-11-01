PART 1: MCP

LLMs are smart, but cut off from our files, apps, and tools.  
before MCP, every AI app needed a custom connector for every tool, so the work grew as NxM  
MCP (done by anthropic) turns this into N+M -> we wrap a tool once as an MCP server, and any MCP-compatible app can use it  
a server can offer tools, resources and prompt templates and the messages travel as JSON-RPC  
MCP == agent's 'hands'  
A2A == agent's 'language'

the MCP docs add a third role, the host  
host = app the user talks to, like Claude desktop or the agent itself  
it holds the model and createes one client for each server it connects to. the client speaks the protocol: it asks the server whcih tools it has, then sends the calls  
weak spots: security and auth + more tools do not help if their descriptors are poor

I explored Fetch  
the Fetch server exposes one tool + prompt with the same name  
on the class principles, the interface is well defined: the arguments are typed and bounded, the tool is marked read-only, and long pages come back in chunks with a clear hint  
then a 404 error comes back as error (robots.txt)  
so functionality is less isolated. one tool fetches the page, enforces robots.txt, blocks private IPs, converts HTML to Markdown and paginates, and the raw flag switches it into a different mode  
still the bundling is mostly justified: a separate fetch_html tool would push huge raw HTML into the model's context. My real criticisms are smaller.  
Some messages, like "Page failed to be simplified", come back as plain text in \<error\> tags instead of real tool errors. And the prompt runs the same fetch without the robots.txt check, so the behavior depends on how you call it.
