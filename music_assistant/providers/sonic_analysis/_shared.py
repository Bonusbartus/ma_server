"""
Constants and config entries shared between the local (CLAP) and remote-only provider.

Kept dependency-free (no torch/transformers/etc.) so the remote-only fallback class in
__init__.py can use them without ever importing the heavy local implementation.
"""

from __future__ import annotations

from music_assistant.helpers.remote_analysis import (
    CONF_REMOTE_WORKER_TOKEN,
    CONF_REMOTE_WORKER_URL,
    remote_worker_config_entries,
)
from music_assistant.models.audio_analysis_provider import (
    ACCUMULATING_ANALYSIS_MAX_DURATION_SECONDS,
)

__all__ = [
    "ANALYSIS_VERSION",
    "CONF_REMOTE_WORKER_TOKEN",
    "CONF_REMOTE_WORKER_URL",
    "MAX_ANALYSIS_DURATION",
    "MIN_CPU_CORES",
    "MIN_RAM_GB",
    "remote_worker_config_entries",
]

# Sonic Analysis runs on-device CLAP inference; gate it to capable hardware.
# 4GB nominal; the gate's tolerance (meets_memory_target) admits genuine 4GB hosts,
# which report ~3.8GB after the kernel/firmware reservation.
MIN_RAM_GB: float = 4.0
MIN_CPU_CORES: int = 2

MAX_ANALYSIS_DURATION = ACCUMULATING_ANALYSIS_MAX_DURATION_SECONDS
ANALYSIS_VERSION: int = 1
