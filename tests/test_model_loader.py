"""resolve_model_dir must not touch the network when the model is cached."""
import sys
import types

from laya_agent.brain import model_loader


def _fake_hub(monkeypatch, cached: bool):
    calls = []

    class LocalEntryNotFoundError(Exception):
        pass

    def snapshot_download(repo, local_files_only=False, **kw):
        calls.append(local_files_only)
        if local_files_only and not cached:
            raise LocalEntryNotFoundError(repo)
        return f"/cache/{repo}"

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    errors = types.ModuleType("huggingface_hub.errors")
    errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return calls


def test_cached_model_resolves_offline(monkeypatch):
    calls = _fake_hub(monkeypatch, cached=True)
    assert model_loader.resolve_model_dir("org/model") == "/cache/org/model"
    assert calls == [True]  # one local-only lookup, no online call


def test_uncached_model_downloads_once(monkeypatch):
    calls = _fake_hub(monkeypatch, cached=False)
    logs = []
    assert model_loader.resolve_model_dir("org/model", logs.append) == "/cache/org/model"
    assert calls == [True, False] and "downloading" in logs[0]


def test_local_dir_passes_through(tmp_path):
    assert model_loader.resolve_model_dir(str(tmp_path)) == str(tmp_path)
