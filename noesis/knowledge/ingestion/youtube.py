"""YouTube ingestor — pulls recent uploads + transcripts from configured channels."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import feedparser
from youtube_transcript_api import YouTubeTranscriptApi

from noesis.core.config import ingestion_config
from noesis.knowledge.ingestion.base import IngestionItem, IngestionSource

CHANNEL_FEED = "https://www.youtube.com/feeds/videos.xml?channel_id={}"


class YouTubeSource(IngestionSource):
    name = "youtube"

    def __init__(self) -> None:
        cfg = ingestion_config()["sources"]["youtube"]
        self.channels: list[str] = cfg["channels"]
        self.extract_transcripts: bool = cfg.get("extract_transcripts", True)

    async def fetch_new(self) -> AsyncIterator[IngestionItem]:
        for ch in self.channels:
            feed = await asyncio.to_thread(feedparser.parse, CHANNEL_FEED.format(ch))
            for entry in feed.entries[:10]:
                video_id = entry.get("yt_videoid")
                if not video_id:
                    continue
                transcript_text = ""
                if self.extract_transcripts:
                    try:
                        segs = await asyncio.to_thread(
                            YouTubeTranscriptApi.get_transcript, video_id
                        )
                        transcript_text = " ".join(s["text"] for s in segs)
                    except Exception:
                        transcript_text = ""
                if not transcript_text:
                    transcript_text = entry.get("summary", "")
                yield IngestionItem(
                    source="youtube",
                    source_id=video_id,
                    url=f"https://youtu.be/{video_id}",
                    title=entry.get("title", "untitled"),
                    text=transcript_text,
                    published_at=entry.get("published"),
                    authors=[entry.get("author", "channel")],
                    tags=["youtube", ch],
                )
