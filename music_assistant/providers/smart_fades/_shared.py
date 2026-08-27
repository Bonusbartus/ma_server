"""
Constants and config entries shared between the local (torch) and remote-only provider.

Kept dependency-free (no torch/beat_this/etc.) so the remote-only fallback class in
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

# Smart Fades runs on-device ML (torch) inference; gate it to capable hardware.
# 4GB nominal, matching the Balanced buffer threshold (the minimum buffer smart crossfade
# needs). The gate applies meets_memory_target()'s tolerance, so a genuine 4GB host (which
# reports ~3.8GB after the kernel/firmware reservation) still passes.
MIN_RAM_GB = 4.0
MIN_CPU_CORES = 2

MAX_ANALYSIS_DURATION = ACCUMULATING_ANALYSIS_MAX_DURATION_SECONDS
# v3: FireRed AED vocal activity
ANALYSIS_VERSION = 3
