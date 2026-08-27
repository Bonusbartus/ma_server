"""Smart Fades audio analysis provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from music_assistant.helpers.remote_analysis import RemoteAnalysisClient
from music_assistant.helpers.util import local_ml_analysis_capable
from music_assistant.models.audio_analysis_provider import AudioAnalysisProvider

from ._shared import (
    ANALYSIS_VERSION,
    CONF_REMOTE_WORKER_TOKEN,
    CONF_REMOTE_WORKER_URL,
    MAX_ANALYSIS_DURATION,
    MIN_CPU_CORES,
    MIN_RAM_GB,
    remote_worker_config_entries,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry, ProviderConfig
    from music_assistant_models.enums import ProviderFeature
    from music_assistant_models.media_items import AudioFormat
    from music_assistant_models.provider import ProviderManifest
    from music_assistant_models.streamdetails import StreamDetails

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType
    from music_assistant.models.audio_analysis import AudioAnalysisData

SUPPORTED_FEATURES: set[ProviderFeature] = set()


async def setup(
    mass: MusicAssistant,
    manifest: ProviderManifest,
    config: ProviderConfig,
) -> ProviderInstanceType:
    """Set up the Smart Fades provider."""
    # Never import the heavy torch/beat_this stack on a host that cannot run it; a host
    # that fails this check can still use Smart Fades entirely through a remote worker.
    if await local_ml_analysis_capable(min_memory_gb=MIN_RAM_GB, min_cpu_cores=MIN_CPU_CORES):
        from .provider import SmartFadesProvider  # noqa: PLC0415

        return SmartFadesProvider(mass, manifest, config, SUPPORTED_FEATURES)
    return SmartFadesRemoteOnlyProvider(mass, manifest, config, SUPPORTED_FEATURES)


class SmartFadesRemoteOnlyProvider(AudioAnalysisProvider):
    """
    Smart Fades fallback for hosts that do not meet the local hardware requirements.

    Never imports torch/beat_this; every session is forwarded to a remote worker. A
    worker must be configured (see get_config_entries) since there is no local fallback.
    """

    max_analysis_duration = MAX_ANALYSIS_DURATION
    analysis_version = ANALYSIS_VERSION
    has_unloadable_models = False

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
        supported_features: set[ProviderFeature],
    ) -> None:
        """Initialize the provider."""
        super().__init__(mass, manifest, config, supported_features)
        self._remote_client: RemoteAnalysisClient | None = None
        worker_url = str(config.get_value(CONF_REMOTE_WORKER_URL) or "")
        worker_token = str(config.get_value(CONF_REMOTE_WORKER_TOKEN) or "")
        if worker_url and worker_token:
            self._remote_client = RemoteAnalysisClient(
                mass=mass,
                url=worker_url,
                token=worker_token,
                domain=self.domain,
                logger=self.logger,
            )

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return config entries for this provider."""
        return (*remote_worker_config_entries(required=True),)

    async def process_pcm_chunk(self, session_id: str, pcm_chunk: bytes) -> None:
        """Forward a PCM chunk to the configured remote worker."""
        if self._remote_client is not None:
            await self._remote_client.send_chunk(session_id, pcm_chunk)

    async def cancel(self, session_id: str) -> None:
        """Forward cancellation to the configured remote worker."""
        if self._remote_client is not None:
            await self._remote_client.cancel(session_id)
        await super().cancel(session_id)

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload, closing the remote worker connection."""
        if self._remote_client is not None:
            await self._remote_client.close()
        await super().unload(is_removed)

    async def _start_analysis(
        self,
        session_id: str,
        streamdetails: StreamDetails,
        audio_format: AudioFormat,
    ) -> bool:
        """Forward session start to the configured remote worker."""
        if self._remote_client is None:
            return False
        return await self._remote_client.start(session_id, streamdetails, audio_format)

    async def _finalize(self, session_id: str) -> AudioAnalysisData | None:
        """Forward finalization to the configured remote worker."""
        if self._remote_client is None:
            return None
        return await self._remote_client.finalize(session_id)
