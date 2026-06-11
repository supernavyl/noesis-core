"""ReAct tool implementations — retrieval, sandbox exec, hypothesis tests.

These are the actions a reasoner can take during an inference loop.
"""

from __future__ import annotations

from typing import Any

from noesis.knowledge.retrieval import HybridRetriever


class RetrievalTool:
    """Wraps HybridRetriever so the ReAct loop can call it.

    Action name: ``search_corpus``
    Action input: {"query": str, "k": int = 6, "source": str|None}
    Observation: pretty-printed top-K snippets.
    """

    name = "search_corpus"

    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or HybridRetriever.default()

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search the NOESIS knowledge corpus (arxiv papers, github repos, blogs, "
                    "books, forums). Returns top-K snippets ranked by hybrid retrieval + source "
                    "priority."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "k": {"type": "integer", "default": 6, "description": "Number of results"},
                        "source": {
                            "type": "string",
                            "description": "Optional source filter (arxiv, github, books, …)",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def __call__(self, _name: str, params: dict[str, Any]) -> str:
        query = params.get("query") or params.get("raw") or ""
        if not query:
            return "ERROR: 'query' required"
        k = int(params.get("k", 6))
        filt = {"source": params["source"]} if params.get("source") else None
        hits = await self.retriever.search(query=query, k_final=k, filter_=filt)
        if not hits:
            return "No matching results."

        rendered: list[str] = []
        for i, h in enumerate(hits, 1):
            title = h.payload.get("title", "")
            src = h.payload.get("source", "?")
            url = h.payload.get("url", "")
            snippet = h.text[:600].replace("\n", " ")
            rendered.append(f"[{i}] {src} | {title}\n    {snippet}\n    URL: {url}")
        return "\n\n".join(rendered)


def default_tools(retriever: HybridRetriever | None = None) -> dict[str, Any]:
    rt = RetrievalTool(retriever=retriever)
    return {rt.name: rt}


def tool_schemas(tools: dict[str, Any]) -> list[dict[str, Any]]:
    return [t.schema for t in tools.values() if hasattr(t, "schema")]
