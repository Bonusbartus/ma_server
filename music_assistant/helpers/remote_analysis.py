"""
Shared client/bridge helpers for running an AudioAnalysisProvider on a remote worker.

Wire protocol (one aiohttp WebSocket connection per provider instance, multiplexed by
session_id so several tracks can be analyzed concurrently on the same connection):

- TEXT frames carry a JSON control envelope: ``{"type": ..., "session_id": ..., ...}``.
- BINARY frames carry a PCM chunk: a 1-byte session_id length, the session_id (utf-8),
  then the raw PCM bytes for that chunk.

Control message types (client -> worker): ``start``, ``finalize``, ``cancel``.
Control message types (worker -> client): ``start_ack``, ``chunk_ack``, ``result``, ``error``.

This module intentionally has no dependency on torch/numpy/etc: it is imported by both the
lightweight (thin main server) and heavy (worker) sides of a remote-capable provider.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

import aiohttp
from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.helpers.json import json_dumps, json_loads
from music_assistant.models.audio_analysis import AudioAnalysisData, AudioAnalysisError
from music_assistant.models.audio_analysis_provider import AudioAnalysisProvider

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from aiohttp import web

    from music_assistant.mass import MusicAssistant

MASS_LOGGER_NAME = "music_assistant"
LOGGER = logging.getLogger(f"{MASS_LOGGER_NAME}.remote_analysis")

# Reconnect backoff bounds for the client side.
_RECONNECT_MIN_DELAY = 1.0
_RECONNECT_MAX_DELAY = 30.0
# How long to wait for a control-message reply (start_ack/chunk_ack/result) before treating
# the worker as unresponsive. Generous: a busy worker may queue behind other sessions.
_REPLY_TIMEOUT = 120.0

CONF_REMOTE_WORKER_URL = "remote_worker_url"
CONF_REMOTE_WORKER_TOKEN = "remote_worker_token"

# Path the bridge is registered on via mass.webserver.register_dynamic_route.
REMOTE_ANALYSIS_BRIDGE_PATH = "/api/audio_analysis/remote"


def remote_worker_config_entries(*, required: bool) -> tuple[ConfigEntry, ...]:
    """
    Return the remote-worker config entries shared by every remote-capable provider.

    Text (label/description) is authored in the common strings.json under
    config_entries.remote_worker_url / config_entries.remote_worker_token.

    :param required: Whether a remote worker must be configured (True when this host does
        not meet the hardware requirements to run the provider's analysis locally).
    """
    return (
        ConfigEntry(
            key=CONF_REMOTE_WORKER_URL,
            type=ConfigEntryType.STRING,
            required=required,
        ),
        ConfigEntry(
            key=CONF_REMOTE_WORKER_TOKEN,
            type=ConfigEntryType.SECURE_STRING,
            required=required,
        ),
    )


def _pack_chunk_frame(session_id: str, pcm_chunk: bytes) -> bytes:
    """Pack a session_id + PCM chunk into a single binary websocket frame."""
    session_id_bytes = session_id.encode("utf-8")
    if len(session_id_bytes) > 255:
        msg = f"session_id is too long to frame: {session_id!r}"
        raise ValueError(msg)
    return bytes([len(session_id_bytes)]) + session_id_bytes + pcm_chunk


def _unpack_chunk_frame(frame: bytes) -> tuple[str, bytes]:
    """Unpack a binary websocket frame into (session_id, pcm_chunk)."""
    if not frame:
        msg = "empty binary frame"
        raise ValueError(msg)
    session_id_len = frame[0]
    session_id = frame[1 : 1 + session_id_len].decode("utf-8")
    pcm_chunk = frame[1 + session_id_len :]
    return session_id, pcm_chunk


class RemoteAnalysisClient:
    """
    Client side of the remote-analysis bridge, used by a provider to forward sessions.

    Owns a single persistent WebSocket connection to one worker/domain, multiplexing
    concurrent analysis sessions over it and reconnecting (with backoff) on connection loss.
    """

    def __init__(
        self,
        *,
        mass: MusicAssistant,
        url: str,
        token: str,
        domain: str,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the client.

        :param mass: The MusicAssistant instance (used only for its shared aiohttp session).
        :param url: Worker websocket URL, e.g. ``ws://host:8095/api/audio_analysis/remote``.
        :param token: Bearer token the worker requires to accept the connection.
        :param domain: The AudioAnalysisProvider domain to target on the worker.
        """
        self.mass = mass
        self.url = url
        self.token = token
        self.domain = domain
        self.logger = logger or LOGGER
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._connect_lock = asyncio.Lock()
        self._receiver_task: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._closing = False

    async def start(
        self,
        session_id: str,
        streamdetails: StreamDetails,
        audio_format: AudioFormat,
    ) -> bool:
        """Ask the worker to start an analysis session; return whether it was accepted."""
        try:
            await self._ensure_connected()
            reply = await self._request(
                {
                    "type": "start",
                    "session_id": session_id,
                    "domain": self.domain,
                    "audio_format": audio_format.to_dict(),
                    # StreamDetails already omits path/data/decryption_key/buffer etc. on
                    # serialization, so this never leaks a provider's source URL/credentials.
                    "streamdetails": streamdetails.to_dict(),
                },
                session_id,
            )
        except AudioAnalysisError:
            raise
        except Exception:
            return False
        return bool(reply.get("accepted"))

    async def send_chunk(self, session_id: str, pcm_chunk: bytes) -> None:
        """Forward a PCM chunk to the worker and wait for it to be processed."""
        await self._ensure_connected()
        await self._request_binary(_pack_chunk_frame(session_id, pcm_chunk), session_id)

    async def finalize(self, session_id: str) -> AudioAnalysisData | None:
        """Ask the worker to finalize the session and return its analysis result."""
        await self._ensure_connected()
        reply = await self._request({"type": "finalize", "session_id": session_id}, session_id)
        analysis = reply.get("analysis")
        if not analysis:
            return None
        return AudioAnalysisData.from_dict(analysis)

    async def cancel(self, session_id: str) -> None:
        """Tell the worker to cancel the session (best-effort, does not wait for a reply)."""
        self._pending.pop(session_id, None)
        if self._ws is None or self._ws.closed:
            return
        with contextlib.suppress(Exception):
            await self._ws.send_str(json_dumps({"type": "cancel", "session_id": session_id}))

    async def close(self) -> None:
        """Close the connection and stop reconnecting."""
        self._closing = True
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()
        if self._receiver_task is not None:
            self._receiver_task.cancel()
            self._receiver_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def _ensure_connected(self) -> None:
        if self._ws is not None and not self._ws.closed:
            return
        async with self._connect_lock:
            if self._ws is not None and not self._ws.closed:
                return
            if self._closing:
                msg = "remote analysis client is closed"
                raise AudioAnalysisError(msg)
            await self._connect_with_backoff()

    async def _connect_with_backoff(self) -> None:
        delay = _RECONNECT_MIN_DELAY
        while True:
            try:
                session = self.mass.http_session
                self._ws = await session.ws_connect(
                    self.url,
                    headers={"Authorization": f"Bearer {self.token}"},
                    heartbeat=30,
                )
            except Exception as err:
                self.logger.warning(
                    "Could not connect to remote analysis worker at %s: %s", self.url, err
                )
                if delay >= _RECONNECT_MAX_DELAY:
                    raise AudioAnalysisError(
                        f"remote analysis worker at {self.url} is unreachable"
                    ) from err
                await asyncio.sleep(delay)
                delay = min(delay * 2, _RECONNECT_MAX_DELAY)
                continue
            self._receiver_task = self.mass.create_task(self._receive_loop())
            return

    async def _receive_loop(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._dispatch(json_loads(msg.data))
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        finally:
            self._fail_all_pending("connection to remote analysis worker was lost")

    def _dispatch(self, message: dict[str, Any]) -> None:
        session_id = message.get("session_id")
        future = self._pending.pop(session_id, None) if session_id else None
        if future is None or future.done():
            return
        if message.get("type") == "error":
            future.set_exception(AudioAnalysisError(str(message.get("error", "unknown error"))))
        else:
            future.set_result(message)

    def _fail_all_pending(self, reason: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(AudioAnalysisError(reason))
        self._pending.clear()

    async def _request(self, message: dict[str, Any], session_id: str) -> dict[str, Any]:
        ws = self._ws
        if ws is None:
            msg = "not connected to remote analysis worker"
            raise AudioAnalysisError(msg)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[session_id] = future
        await ws.send_str(json_dumps(message))
        try:
            return await asyncio.wait_for(future, timeout=_REPLY_TIMEOUT)
        except TimeoutError as err:
            self._pending.pop(session_id, None)
            msg = f"remote analysis worker timed out replying for session {session_id}"
            raise AudioAnalysisError(msg) from err

    async def _request_binary(self, frame: bytes, session_id: str) -> dict[str, Any]:
        ws = self._ws
        if ws is None:
            msg = "not connected to remote analysis worker"
            raise AudioAnalysisError(msg)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[session_id] = future
        await ws.send_bytes(frame)
        try:
            return await asyncio.wait_for(future, timeout=_REPLY_TIMEOUT)
        except TimeoutError as err:
            self._pending.pop(session_id, None)
            msg = f"remote analysis worker timed out acking a chunk for session {session_id}"
            raise AudioAnalysisError(msg) from err


def create_remote_analysis_handler(
    mass: MusicAssistant,
    *,
    token: str,
) -> Callable[[web.Request], Coroutine[Any, Any, web.Response | web.StreamResponse]]:
    """
    Build the worker-side aiohttp handler for the remote-analysis bridge.

    Register the returned handler with ``mass.webserver.register_dynamic_route`` on the
    worker. It drives the locally loaded AudioAnalysisProvider's session hooks directly,
    bypassing AudioAnalysisController's own local fan-out bookkeeping (which is only needed
    for local real-time playback, not for a network-driven session).

    :param mass: The MusicAssistant instance running in worker mode.
    :param token: Bearer token required to accept a connection.
    """
    from aiohttp import web  # noqa: PLC0415

    async def handler(request: web.Request) -> web.Response | web.StreamResponse:
        auth_header = request.headers.get("Authorization", "")
        provided = auth_header.removeprefix("Bearer ").strip()
        if not provided or provided != token:
            raise web.HTTPUnauthorized

        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        providers: dict[str, AudioAnalysisProvider] = {}

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await _handle_control_message(mass, ws, providers, json_loads(msg.data))
            elif msg.type == aiohttp.WSMsgType.BINARY:
                await _handle_chunk_frame(ws, providers, msg.data)
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                break

        for session_id, provider in providers.items():
            with contextlib.suppress(Exception):
                await provider.cancel(session_id)
        return ws

    return handler


async def _handle_control_message(
    mass: MusicAssistant,
    ws: web.WebSocketResponse,
    providers: dict[str, AudioAnalysisProvider],
    message: dict[str, Any],
) -> None:
    msg_type = message.get("type")
    session_id = message.get("session_id")
    if not session_id:
        return

    if msg_type == "start":
        provider = mass.get_provider(message["domain"])
        accepted = False
        if isinstance(provider, AudioAnalysisProvider) and provider.available:
            streamdetails = StreamDetails.from_dict(message["streamdetails"])
            audio_format = AudioFormat.from_dict(message["audio_format"])
            providers[session_id] = provider
            try:
                accepted = await provider._start_analysis(session_id, streamdetails, audio_format)
            except Exception as err:
                LOGGER.warning("Remote _start_analysis failed for %s: %s", session_id, err)
                accepted = False
        if not accepted:
            providers.pop(session_id, None)
        await ws.send_str(
            json_dumps({"type": "start_ack", "session_id": session_id, "accepted": accepted})
        )
    elif msg_type == "finalize":
        provider = providers.pop(session_id, None)
        analysis = None
        if provider is not None:
            try:
                result = await provider._finalize(session_id)
                analysis = result.to_dict() if result is not None else None
            except Exception as err:
                LOGGER.warning("Remote _finalize failed for %s: %s", session_id, err)
        await ws.send_str(
            json_dumps({"type": "result", "session_id": session_id, "analysis": analysis})
        )
    elif msg_type == "cancel":
        provider = providers.pop(session_id, None)
        if provider is not None:
            with contextlib.suppress(Exception):
                await provider.cancel(session_id)


async def _handle_chunk_frame(
    ws: web.WebSocketResponse,
    providers: dict[str, AudioAnalysisProvider],
    frame: bytes,
) -> None:
    session_id, pcm_chunk = _unpack_chunk_frame(frame)
    provider = providers.get(session_id)
    if provider is not None:
        try:
            await provider.process_pcm_chunk(session_id, pcm_chunk)
        except Exception as err:
            LOGGER.warning("Remote process_pcm_chunk failed for %s: %s", session_id, err)
    await ws.send_str(json_dumps({"type": "chunk_ack", "session_id": session_id}))
