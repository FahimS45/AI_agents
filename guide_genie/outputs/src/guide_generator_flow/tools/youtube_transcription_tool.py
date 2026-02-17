from typing import Type
import json
import urllib.parse

from pydantic import BaseModel, Field, AnyUrl
from crewai.tools import BaseTool
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable
)


class ToolInput(BaseModel):
    """Input schema for YoutubeTranscriptionTool."""
    youtube_url: str = Field(
        description="YouTube video URL to transcribe (e.g., https://www.youtube.com/watch?v=VIDEO_ID)"
    )


class YoutubeTranscriptionTool(BaseTool):
    name: str = "Youtube transcription tool"
    description: str = (
        "Fetches and returns the full transcript/subtitles from a YouTube video. "
        "Prefers manually created subtitles over auto-generated ones. "
        "Useful for analyzing video content, extracting information, or summarizing YouTube videos."
    )
    args_schema: Type[BaseModel] = ToolInput

    def _extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from various URL formats."""
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname.lower() if parsed.hostname else ""

        # Handle youtu.be short links
        if "youtu.be" in host:
            return parsed.path.lstrip("/")
        
        # Handle youtube.com URLs
        if "youtube.com" in host:
            # Standard watch URL
            if parsed.path == "/watch":
                return urllib.parse.parse_qs(parsed.query).get("v", [None])[0]
            # Embed or direct video URLs
            if parsed.path.startswith(("/embed/", "/v/")):
                return parsed.path.split("/")[2]
        
        return None

    def _get_youtube_transcription(self, url: str) -> str:
        """Fetch YouTube video transcription, preferring manual over auto-generated."""
        video_id = self._extract_video_id(url)
        
        if not video_id:
            raise ValueError("Invalid YouTube URL")

        # Get list of available transcripts
        transcript_list = YouTubeTranscriptApi().list(video_id)

        # Prefer manually created subtitles
        try:
            for transcript in transcript_list:
                if not transcript.is_generated:
                    data = transcript.fetch()
                    return " ".join(t.text.replace("\n", " ") for t in data)
        except:
            pass

        # Fallback to auto-generated subtitles
        try:
            for transcript in transcript_list:
                if transcript.is_generated:
                    data = transcript.fetch()
                    return " ".join(t.text.replace("\n", " ") for t in data)
        except:
            pass

        raise NoTranscriptFound(video_id)

    def _run(self, youtube_url: str) -> str:
        """Execute the YouTube transcription tool."""
        try:
            # Handle both JSON input and direct URL string
            if youtube_url.strip().startswith("{"):
                payload = json.loads(youtube_url)
                url = payload.get("url") or payload.get("youtube_url")
            else:
                url = youtube_url.strip()

            if not url:
                raise ValueError("YouTube URL is required")

            return self._get_youtube_transcription(url)

        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as e:
            return f"No subtitles available for this video: {str(e)}"
        except ValueError as e:
            return f"Invalid input: {str(e)}"
        except Exception as e:
            return f"Error fetching transcription: {str(e)}"