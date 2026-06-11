"""GitHub watch ingestor — pulls README + recent commits/PRs from priority repos."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from noesis.core.config import ingestion_config, resolve_secret
from noesis.knowledge.ingestion.base import IngestionItem, IngestionSource


class GitHubSource(IngestionSource):
    name = "github"

    def __init__(self) -> None:
        cfg = ingestion_config()["sources"]["github"]
        self.repos: list[str] = cfg["watch_repos"]
        self._token = resolve_secret("GITHUB_TOKEN")

    async def fetch_new(self) -> AsyncIterator[IngestionItem]:
        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        async with httpx.AsyncClient(timeout=20, headers=headers) as client:
            for repo in self.repos:
                try:
                    readme = await client.get(
                        f"https://api.github.com/repos/{repo}/readme",
                        headers={**headers, "Accept": "application/vnd.github.raw"},
                    )
                    readme_text = readme.text if readme.status_code == 200 else ""
                except Exception:
                    readme_text = ""

                commits = await client.get(
                    f"https://api.github.com/repos/{repo}/commits?per_page=20"
                )
                commit_data = commits.json() if commits.status_code == 200 else []
                commit_summaries = "\n".join(
                    f"- {c['commit']['message'].splitlines()[0]} ({c['sha'][:7]})"
                    for c in commit_data
                    if isinstance(c, dict)
                )

                body = f"# {repo}\n\n{readme_text}\n\n## Recent commits\n{commit_summaries}"
                yield IngestionItem(
                    source="github",
                    source_id=repo,
                    url=f"https://github.com/{repo}",
                    title=repo,
                    text=body,
                    tags=["github", repo.split("/", 1)[0]],
                )
