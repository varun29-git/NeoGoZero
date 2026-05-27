from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class WeightExportResult:
    weights_path: Path
    manifest_path: Path
    bundle_path: Path
    auto_download_started: bool


def export_checkpoint_weights(
    checkpoint_path: Path,
    architecture: str,
    output_dir: Path = Path("trained_model_weights"),
    auto_download: bool = False,
) -> WeightExportResult:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    iteration = int(checkpoint["iteration"])
    output_dir = Path(output_dir)
    architecture_dir = output_dir / architecture
    architecture_dir.mkdir(parents=True, exist_ok=True)

    stem = f"neogozero_{architecture}_iteration_{iteration:04d}"
    weights_path = architecture_dir / f"{stem}_weights.pt"
    manifest_path = architecture_dir / f"{stem}_manifest.json"
    bundle_path = architecture_dir / f"{stem}_download_bundle.zip"

    weights_payload = {
        "architecture": architecture,
        "iteration": iteration,
        "model_state": checkpoint["model_state"],
        "config": checkpoint["config"],
        "source_checkpoint": str(checkpoint_path),
        "promoted": checkpoint.get("promoted"),
        "candidate_win_rate": checkpoint.get("candidate_win_rate"),
    }
    torch.save(weights_payload, weights_path)

    manifest = {
        "architecture": architecture,
        "iteration": iteration,
        "weights_file": weights_path.name,
        "source_checkpoint": str(checkpoint_path),
        "config": checkpoint["config"],
        "promoted": checkpoint.get("promoted"),
        "candidate_win_rate": checkpoint.get("candidate_win_rate"),
    }
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(weights_path, arcname=weights_path.name)
        bundle.write(manifest_path, arcname=manifest_path.name)

    auto_download_started = _try_colab_download(bundle_path) if auto_download else False
    return WeightExportResult(
        weights_path=weights_path,
        manifest_path=manifest_path,
        bundle_path=bundle_path,
        auto_download_started=auto_download_started,
    )


def _try_colab_download(path: Path) -> bool:
    try:
        from google.colab import files  # type: ignore[import-not-found]
    except ImportError:
        return False

    files.download(str(path))
    return True


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
