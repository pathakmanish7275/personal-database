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
from pipecat.frames.frames import LLMMessagesAppendFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.audio.vad_processor import VADProcessor
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


# Live pipeline tasks by session, so a typed message can be injected into a
# call that is already running. A call is an asyncio task inside this process;
# without a handle on it the only way in is the microphone.
_live_tasks: dict[str, "PipelineTask"] = {}


async def say_text(session_id: str, text: str) -> bool:
    """Feed typed text into a live call as if the caller had spoken it.

    Returns False when no call is running for that session.

    Uses LLMMessagesAppendFrame(run_llm=True) rather than a synthetic
    transcription: the aggregator appends it to the same context the spoken
    turns build, so the reply is spoken, persisted to the chat thread and
    carried in history identically. Typing is the reliable path for anything
    STT mangles — names, spellings, identifiers.
    """
    task = _live_tasks.get(session_id)
    if task is None:
        return False
    await task.queue_frames(
        [
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": text}], run_llm=True
            )
        ]
    )
    return True


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

    # VAD must run as its own processor placed BEFORE the STT service.
    #
    # `OpenAISTTService` is a SegmentedSTTService: it buffers audio and only
    # transcribes when it receives VADUserStoppedSpeakingFrame. Those frames
    # come from VADProcessor. Configuring the analyzer on the user aggregator
    # instead puts the VAD *downstream* of STT, where its frames can never
    # reach it — the call connects, the greeting plays, and then nothing
    # happens no matter how long you talk.
    #
    # `TransportParams` has no `vad_analyzer` field in pipecat 1.3.0, so the
    # transport cannot carry it either; VADProcessor is the supported place.
    # min_volume is lowered from pipecat's default 0.6 because VAD requires
    # BOTH `confidence >= 0.7` AND `volume >= min_volume`, and this machine's
    # mic — through the browser's echo cancellation and AGC — peaks at 0.286
    # on normal speech (measured). At the default the volume gate can never
    # open, so no VADUserStoppedSpeakingFrame is ever emitted and the segmented
    # STT never transcribes: the call connects, greets, and then ignores you
    # forever. Silero's confidence stays the real speech test; this is only a
    # noise floor, and 0.1 sits well above the ~0.04 measured when quiet.
    # stop_secs is the silence needed to end a turn. At 0.6 an ordinary pause —
    # thinking mid-sentence, or spelling a name out letter by letter — ends the
    # turn early, and each fragment is sent to the agent as a separate question.
    # Measured on one call: "look into my database and find anything about
    # MuseTalk" arrived as three turns ("...find anything about new stock." /
    # "Files MUSP" / "C-A-L-T."), so the agent was answering fragments and
    # looked like it could not understand. 1.2s spans a normal pause.
    vad = SileroVADAnalyzer(params=VADParams(stop_secs=1.2, min_volume=0.1))

    llm = PersonalDBLLMService(stores=stores, session_id=session_id)

    # No `tools` on the context: tool calling happens *inside* the agent, one
    # layer down. Pipecat only needs to carry the conversation.
    context = LLMContext(messages=[{"role": "system", "content": VOICE_SYSTEM_PROMPT}])
    # No `vad_analyzer` here on purpose: the aggregator would build a second,
    # independent VADController over the same audio, doubling Silero's cost and
    # letting two controllers disagree about turn boundaries. Its turn
    # strategies consume the VAD frames VADProcessor already emits upstream.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams()
    )

    pipeline = Pipeline(
        [
            transport.input(),
            VADProcessor(vad_analyzer=vad),
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

    _live_tasks[session_id] = task
    try:
        await PipelineRunner(handle_sigint=False).run(task)
    finally:
        # Must not outlive the call: a stale task would accept typed messages
        # into a pipeline that is already torn down.
        if _live_tasks.get(session_id) is task:
            _live_tasks.pop(session_id, None)
