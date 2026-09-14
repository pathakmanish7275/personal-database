"""Voice pipeline: microphone → STT → the project's agent → TTS → speaker.

Adapted from the standalone local-pipecat-agent, with one structural change:
the LLM stage is `PersonalDBLLMService`, so a spoken turn runs the *same* agent
as a typed one and lands in the same session. Tool calling, loop guards and the
NPU call gate all come from that shared path rather than being reimplemented
for voice.

STT and TTS stay as separate local HTTP servers speaking the OpenAI audio API.
They are model servers, like FLM and Ollama — keeping them out of process means
the web app does not carry whisper/kokoro and either can be swapped by URL.
"""

from __future__ import annotations

import asyncio
import logging

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from .. import sessions as sess_store
from ..chat import VOICE_SYSTEM_PROMPT
from ..config import config
from ..stores import Stores, configure_llama_index
from . import events as vevents
from .llm_bridge import PersonalDBLLMService
from .services import create_stt, create_tts

log = logging.getLogger(__name__)


def _greeting_for(session_id: str) -> str:
    """Greet according to where the conversation already is.

    A call placed on a session that already has turns is a continuation, not a
    fresh start — the agent already carries that history into every turn, so
    opening with "what would you like to know?" misrepresents the state to the
    only participant who cannot see it."""
    try:
        msgs = sess_store.get_messages(session_id)
    except Exception:  # noqa: BLE001 — never let a greeting break the call
        log.debug("could not read session for greeting", exc_info=True)
        return config.voice_greeting
    if not msgs:
        return config.voice_greeting
    return config.voice_resume_greeting


async def run_voice_bot(webrtc_connection, *, stores: Stores, session_id: str) -> None:
    """Drive one voice call for one chat session."""
    # Retrieval runs through LlamaIndex, whose global Settings must be pointed
    # at the local Ollama embedder first. The text path does this at the top of
    # every generation; without it here LlamaIndex falls back to its default
    # OpenAI embeddings, which are not installed — every search_kb call then
    # fails and the agent answers from general knowledge instead of the notes.
    # It degrades silently into a plausible-sounding wrong answer, so it is
    # worth doing explicitly rather than relying on the text path having run.
    configure_llama_index()

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
        ),
    )

    # NB: the VAD goes on the user aggregator, not on TransportParams —
    # TransportParams has no `vad_analyzer` field in pipecat 1.3.0 and silently
    # ignores the kwarg, which leaves speech unsegmented and STT receiving
    # nothing at all (the failure looks like "it can't hear me").
    vad = SileroVADAnalyzer(params=VADParams(stop_secs=0.6))

    llm = PersonalDBLLMService(stores=stores, session_id=session_id)

    # No `tools` on the context: tool calling happens *inside* the agent, one
    # layer down. Pipecat only needs to carry the conversation.
    context = LLMContext(messages=[{"role": "system", "content": VOICE_SYSTEM_PROMPT}])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=vad)
    )

    pipeline = Pipeline(
        [
            transport.input(),
            vevents.make_state_observer(session_id),
            create_stt(),
            user_aggregator,
            llm,
            create_tts(),
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True, enable_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        log.info("voice client connected (session=%s)", session_id)
        vevents.publish(session_id, "connected")
        greeting = await asyncio.to_thread(_greeting_for, session_id)
        if greeting:
            await task.queue_frames([TTSSpeakFrame(greeting)])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        log.info("voice client disconnected (session=%s)", session_id)
        vevents.publish(session_id, "ended")
        vevents.clear(session_id)
        await task.cancel()

    from pipecat.pipeline.runner import PipelineRunner

    await PipelineRunner(handle_sigint=False).run(task)
