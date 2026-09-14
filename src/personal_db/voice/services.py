"""STT / TTS clients for the local audio servers.

The TTS client is ported from the standalone local-pipecat-agent, for the
reason its author documented there: pipecat's own `OpenAITTSService` validates
the voice name against OpenAI's official list, so it refuses a Kokoro voice
like "af_heart" and errors out on every utterance. This client passes the voice
through untouched and streams raw PCM back.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

import httpx
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

from ..config import config

log = logging.getLogger(__name__)

KOKORO_NATIVE_SAMPLE_RATE = 24000


class OpenAICompatTTSService(TTSService):
    """TTS client for any OpenAI-compatible /v1/audio/speech server."""

    def __init__(
        self,
        *,
        base_url: str,
        voice: str,
        model: str = "kokoro",
        api_key: str = "not-needed",
        source_sample_rate: int = KOKORO_NATIVE_SAMPLE_RATE,
        **kwargs,
    ):
        # Every settings field must be initialised or the base class logs an
        # error at pipeline start; language is None because the Kokoro server
        # takes it from its own env, not per request.
        kwargs.setdefault(
            "settings", TTSSettings(model=model, voice=voice, language=None)
        )
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")
        self._voice = voice
        self._model = model
        self._api_key = api_key
        self._source_sample_rate = source_sample_rate
        self._resampler = create_stream_resampler()
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        log.debug("TTS: %r", text[:80])
        try:
            await self.start_tts_usage_metrics(text)
            async with self._client.stream(
                "POST",
                f"{self._base_url}/audio/speech",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "input": text,
                    "voice": self._voice,
                    "response_format": "pcm",
                },
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    yield ErrorFrame(
                        error=f"TTS server error {response.status_code}: {body[:200]!r}"
                    )
                    return
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    if not chunk:
                        continue
                    await self.stop_ttfb_metrics()
                    audio = await self._resampler.resample(
                        chunk, self._source_sample_rate, self.sample_rate
                    )
                    yield TTSAudioRawFrame(
                        audio=audio,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
        except httpx.HTTPError as e:
            yield ErrorFrame(error=f"TTS request failed: {e}")
        finally:
            await self.stop_ttfb_metrics()


def create_stt() -> OpenAISTTService:
    return OpenAISTTService(
        api_key="not-needed",
        base_url=config.stt_base_url,
        settings=OpenAISTTService.Settings(
            model=config.stt_model,
            language=Language.EN,
        ),
    )


def create_tts() -> OpenAICompatTTSService:
    return OpenAICompatTTSService(
        base_url=config.tts_base_url,
        voice=config.tts_voice,
    )
