"""Tests for analysis-worker mode: provider loading is restricted to Audio Analysis."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import pytest
from music_assistant_models.enums import ProviderType

from music_assistant.mass import MusicAssistant

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.controllers.config.controller import ConfigController


def _manifest(domain: str, prov_type: ProviderType, *, builtin: bool = False) -> ProviderManifest:
    """Return a minimal stand-in for a ProviderManifest."""
    return cast(
        "ProviderManifest",
        SimpleNamespace(
            domain=domain,
            type=prov_type,
            builtin=builtin,
            allow_disable=True,
            mdns_discovery=None,
        ),
    )


def _config(domain: str, *, enabled: bool = True) -> ProviderConfig:
    """Return a minimal stand-in for a ProviderConfig."""
    return cast(
        "ProviderConfig",
        SimpleNamespace(instance_id=f"{domain}_instance", domain=domain, enabled=enabled),
    )


def _make_mass(*, analysis_worker: bool) -> tuple[MusicAssistant, AsyncMock]:
    """Return a bare MusicAssistant with just enough state to exercise provider loading."""
    mass = MusicAssistant.__new__(MusicAssistant)
    mass.analysis_worker = analysis_worker
    load_provider_mock = AsyncMock()
    mass.load_provider = load_provider_mock  # type: ignore[method-assign]
    # Needed by mass.create_task(), which _load_providers()/_load_builtin_providers() use
    # (via TaskManager/asyncio.TaskGroup) to load each provider concurrently.
    mass.loop = asyncio.get_event_loop()
    mass.loop_thread_id = threading.get_ident()
    mass._tracked_tasks = {}
    mass._tracked_timers = {}
    return mass, load_provider_mock


@pytest.mark.asyncio
async def test_load_builtin_providers_worker_mode_skips_non_analysis() -> None:
    """In analysis-worker mode, only builtin Audio Analysis providers are configured/loaded."""
    mass, load_provider_mock = _make_mass(analysis_worker=True)
    mass._provider_manifests = {
        "loudness_analysis": _manifest(
            "loudness_analysis", ProviderType.AUDIO_ANALYSIS, builtin=True
        ),
        "some_builtin_music": _manifest("some_builtin_music", ProviderType.MUSIC, builtin=True),
        "webserver": _manifest("webserver", ProviderType.CORE, builtin=True),
    }
    created: list[str] = []

    async def _create_builtin_provider_config(domain: str) -> None:
        created.append(domain)

    mass.config = cast(
        "ConfigController",
        SimpleNamespace(
            create_builtin_provider_config=_create_builtin_provider_config,
            get_provider_configs=AsyncMock(
                return_value=[
                    _config("loudness_analysis"),
                    _config("some_builtin_music"),
                ]
            ),
        ),
    )

    await mass._load_builtin_providers()

    assert created == ["loudness_analysis"]
    loaded_instance_ids = {call.args[0] for call in load_provider_mock.call_args_list}
    assert loaded_instance_ids == {"loudness_analysis_instance"}


@pytest.mark.asyncio
async def test_load_providers_worker_mode_skips_mdns_defaults_and_non_analysis() -> None:
    """Worker mode never touches mDNS-driven defaults and only loads Audio Analysis providers."""
    mass, load_provider_mock = _make_mass(analysis_worker=True)
    mass._provider_manifests = {
        "smart_fades": _manifest("smart_fades", ProviderType.AUDIO_ANALYSIS),
        "some_player": _manifest("some_player", ProviderType.PLAYER),
    }
    mass.config = cast(
        "ConfigController",
        SimpleNamespace(
            get_provider_configs=AsyncMock(
                return_value=[_config("smart_fades"), _config("some_player")]
            ),
        ),
    )
    # Deliberately no `mass.discovery` set: worker mode must never touch it (the mDNS-driven
    # default-provider setup block is skipped entirely), so an access would raise AttributeError.

    await mass._load_providers()

    loaded_instance_ids = {call.args[0] for call in load_provider_mock.call_args_list}
    assert loaded_instance_ids == {"smart_fades_instance"}


@pytest.mark.asyncio
async def test_load_providers_normal_mode_still_loads_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside analysis-worker mode, non-Audio-Analysis providers still load as before."""
    mass, load_provider_mock = _make_mass(analysis_worker=False)
    mass._provider_manifests = {
        "smart_fades": _manifest("smart_fades", ProviderType.AUDIO_ANALYSIS),
        "some_player": _manifest("some_player", ProviderType.PLAYER),
    }
    mass.config = cast(
        "ConfigController",
        SimpleNamespace(
            get_provider_configs=AsyncMock(
                return_value=[_config("smart_fades"), _config("some_player")]
            ),
            set_default=lambda *_a, **_kw: None,
            get=lambda *_a, **_kw: set(),
        ),
    )
    monkeypatch.setattr("music_assistant.mass.DEFAULT_PROVIDERS", (), raising=False)

    await mass._load_providers()

    loaded_instance_ids = {call.args[0] for call in load_provider_mock.call_args_list}
    assert loaded_instance_ids == {"smart_fades_instance", "some_player_instance"}
