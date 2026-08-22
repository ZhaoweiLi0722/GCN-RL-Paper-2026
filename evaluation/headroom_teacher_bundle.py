"""Create and verify portable, provenance-locked headroom teacher bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.run_gcn_residual_sweep import (
    load_local_search_demonstrations,
)
from src.rl.config import load_config


BUNDLE_SCHEMA_VERSION = 2
SUPPORTED_BUNDLE_SCHEMA_VERSIONS = frozenset((1, 2))
PROVENANCE_HASH_BASIS = "git_blob_sha256"
CSV_FIELD_SIZE_LIMIT = 64 * 1024 * 1024
ARTIFACT_NAMES = (
    "teacher_cache.npz",
    "state_probe.csv",
    "anchor.csv",
    "teacher.csv",
    "summary.json",
)
SOURCE_FILES = (
    "src/env/patient_capacity_planning.py",
    "src/env/patient_condition.py",
    "evaluation/network_residual_headroom.py",
    "evaluation/run_gcn_residual_sweep.py",
    "evaluation/merge_headroom_state_probe_shards.py",
    "evaluation/merge_headroom_teacher_shards.py",
    "evaluation/run_headroom_teacher_pipeline.py",
    "evaluation/headroom_teacher_bundle.py",
)
TEACHER_SCIENTIFIC_SOURCE_FILES = tuple(
    path
    for path in SOURCE_FILES
    if path
    not in {
        "evaluation/headroom_teacher_bundle.py",
        "evaluation/run_headroom_teacher_pipeline.py",
    }
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(max(csv.field_size_limit(), CSV_FIELD_SIZE_LIMIT))
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def assert_finite_payload(value: Any, label: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_payload(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_payload(item, f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(
        float(value)
    ):
        raise ValueError(f"Non-finite value in {label}: {value}")


def assert_finite_csv(rows: list[dict[str, str]], label: str) -> None:
    for row_index, row in enumerate(rows):
        for key, value in row.items():
            if value in (None, ""):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                raise ValueError(
                    f"Non-finite CSV value in {label} row={row_index} "
                    f"column={key}: {value}"
                )


def validate_teacher_root(
    source_root: Path,
    expected_config: dict[str, Any],
) -> dict[str, Any]:
    for name in ARTIFACT_NAMES:
        path = source_root / name
        if not path.is_file():
            raise ValueError(f"Missing canonical teacher artifact: {path}")

    summary = json.loads((source_root / "summary.json").read_text())
    if summary.get("config") != expected_config:
        raise ValueError("Teacher summary config does not match locked config")
    assert_finite_payload(summary, "summary")

    decision = summary.get("decision", {})
    if decision.get("advance_to_network_residual_training") is not True:
        raise ValueError("Headroom decision did not advance to training")

    state_probe = summary.get("state_probe")
    state_shards = summary.get("state_probe_shards")
    teacher = summary.get("online_teacher")
    teacher_shards = summary.get("teacher_shards")
    if not all(
        isinstance(value, dict)
        for value in (state_probe, state_shards, teacher, teacher_shards)
    ):
        raise ValueError("Bundle requires merged state-probe and teacher shards")

    state_rows = read_csv_rows(source_root / "state_probe.csv")
    anchor_rows = read_csv_rows(source_root / "anchor.csv")
    teacher_rows = read_csv_rows(source_root / "teacher.csv")
    for label, rows in (
        ("state_probe", state_rows),
        ("anchor", anchor_rows),
        ("teacher", teacher_rows),
    ):
        assert_finite_csv(rows, label)

    expected_states = int(state_probe.get("states", -1))
    if len(state_rows) != expected_states:
        raise ValueError(
            f"State-probe row count mismatch: {len(state_rows)} != "
            f"{expected_states}"
        )
    if int(state_shards.get("states", -1)) != expected_states:
        raise ValueError("State-probe shard summary has the wrong state count")
    expected_rollouts = int(expected_config["state_probe_rollouts"])
    observed_rollouts = sorted({int(row["rollout"]) for row in state_rows})
    if observed_rollouts != list(range(expected_rollouts)):
        raise ValueError("State-probe rows do not cover every locked rollout")

    expected_replications = int(expected_config["teacher_replications"])
    if len(anchor_rows) != expected_replications:
        raise ValueError("Anchor row count does not match teacher replications")
    if len(teacher_rows) != expected_replications:
        raise ValueError("Teacher row count does not match teacher replications")
    if int(teacher_shards.get("replications", -1)) != expected_replications:
        raise ValueError("Teacher shard summary has the wrong replication count")
    expected_indices = list(range(expected_replications))
    for label, rows in (("anchor", anchor_rows), ("teacher", teacher_rows)):
        indices = [int(row["replication"]) for row in rows]
        if indices != expected_indices:
            raise ValueError(f"{label} rows are not in canonical CRN order")

    cache = load_local_search_demonstrations(source_root / "teacher_cache.npz")
    for key, value in cache.items():
        if isinstance(value, np.ndarray) and np.issubdtype(
            value.dtype, np.number
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Teacher cache contains non-finite {key}")
    samples = int(np.asarray(cache["states"]).shape[0])
    if samples != int(teacher.get("teacher_demonstration_samples", -1)):
        raise ValueError("Teacher cache sample count does not match summary")
    if samples != int(teacher.get("teacher_total_decisions", -1)):
        raise ValueError("Teacher cache does not cover every teacher decision")
    if expected_config.get("store_option_advantages", False):
        for key in (
            "option_advantages",
            "option_feasible",
            "option_groups",
            "option_epsilons",
            "option_signs",
        ):
            if key not in cache:
                raise ValueError(f"Teacher cache is missing {key}")
    return summary


def git_output(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_blob_sha256(
    repo_root: Path,
    relative_path: str,
    *,
    revision: str = "HEAD",
) -> str:
    result = subprocess.run(
        ("git", "show", f"{revision}:{relative_path}"),
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return sha256_bytes(result.stdout)


def teacher_generation_provenance(
    repo_root: Path,
    config_relative_path: str,
    source_root: Path,
) -> dict[str, Any] | None:
    status_path = source_root / "orchestration" / "teacher.status.json"
    if not status_path.is_file():
        return None
    status = json.loads(status_path.read_text())
    if status.get("state") != "completed":
        raise ValueError("Teacher generation status is not completed")
    generation_commit = str(status.get("git_commit", ""))
    if not generation_commit:
        raise ValueError("Teacher generation status has no git commit")
    git_output(repo_root, "merge-base", "--is-ancestor", generation_commit, "HEAD")
    generation_config_sha256 = git_blob_sha256(
        repo_root,
        config_relative_path,
        revision=generation_commit,
    )
    if status.get("config_sha256") != generation_config_sha256:
        raise ValueError("Teacher generation config hash does not match Git")
    for path in TEACHER_SCIENTIFIC_SOURCE_FILES:
        if git_blob_sha256(
            repo_root,
            path,
            revision=generation_commit,
        ) != git_blob_sha256(repo_root, path):
            raise ValueError(
                f"Scientific source changed after teacher generation: {path}"
            )
    changed_paths = git_output(
        repo_root,
        "diff",
        "--name-only",
        generation_commit,
        "HEAD",
    ).splitlines()
    return {
        "git_commit": generation_commit,
        "config_sha256": generation_config_sha256,
        "completed_at": status.get("completed_at"),
        "teacher_status_sha256": sha256_file(status_path),
        "changed_paths_to_bundle_commit": changed_paths,
    }


def local_provenance(
    repo_root: Path,
    config_path: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    dirty = git_output(repo_root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError("Tracked worktree must be clean before bundling")
    config_relative_path = config_path.relative_to(repo_root).as_posix()
    source_hashes = {
        path: git_blob_sha256(repo_root, path)
        for path in SOURCE_FILES
    }
    provenance = {
        "git_branch": git_output(repo_root, "branch", "--show-current"),
        "git_commit": git_output(repo_root, "rev-parse", "HEAD"),
        "hash_basis": PROVENANCE_HASH_BASIS,
        "config_path": config_relative_path,
        "config_sha256": git_blob_sha256(repo_root, config_relative_path),
        "source_sha256": source_hashes,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }
    if source_root is not None:
        generation = teacher_generation_provenance(
            repo_root,
            config_relative_path,
            source_root,
        )
        if generation is not None:
            provenance["teacher_generation"] = generation
    return provenance


def create_teacher_bundle(
    source_root: Path,
    output_path: Path,
    *,
    expected_config: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    sidecar = Path(f"{output_path}.sha256")
    if output_path.exists() or sidecar.exists():
        raise ValueError("Refusing to overwrite an existing teacher bundle")
    validate_teacher_root(source_root, expected_config)
    artifacts = []
    for name in ARTIFACT_NAMES:
        path = source_root / name
        artifacts.append(
            {
                "archive_path": f"teacher/{name}",
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "routing_name": (
            "patient-indexed, identity-preserving, pre-manufacturing "
            "specimen routing"
        ),
        "artifacts": artifacts,
        "provenance": provenance,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_path,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        write_zip_bytes(archive, "manifest.json", manifest_bytes)
        for artifact in artifacts:
            source = source_root / Path(artifact["archive_path"]).name
            write_zip_bytes(
                archive,
                artifact["archive_path"],
                source.read_bytes(),
            )
    bundle_hash = sha256_file(output_path)
    sidecar.write_text(f"{bundle_hash}  {output_path.name}\n")
    return {**manifest, "bundle_sha256": bundle_hash}


def write_zip_bytes(
    archive: zipfile.ZipFile,
    archive_path: str,
    payload: bytes,
) -> None:
    info = zipfile.ZipInfo(archive_path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def verify_teacher_bundle(bundle_path: Path) -> dict[str, Any]:
    sidecar = Path(f"{bundle_path}.sha256")
    if not bundle_path.is_file() or not sidecar.is_file():
        raise ValueError("Teacher bundle or SHA256 sidecar is missing")
    expected_hash = sidecar.read_text().split()[0].lower()
    actual_hash = sha256_file(bundle_path)
    if expected_hash != actual_hash:
        raise ValueError("Teacher bundle SHA256 sidecar does not match")
    with zipfile.ZipFile(bundle_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Teacher bundle contains duplicate members")
        if "manifest.json" not in names:
            raise ValueError("Teacher bundle has no manifest")
        manifest = json.loads(archive.read("manifest.json"))
        if (
            int(manifest.get("schema_version", -1))
            not in SUPPORTED_BUNDLE_SCHEMA_VERSIONS
        ):
            raise ValueError("Unsupported teacher bundle schema")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("Teacher bundle artifacts are invalid")
        canonical_artifact_names = {
            f"teacher/{name}" for name in ARTIFACT_NAMES
        }
        manifest_artifact_names = []
        expected_names = {"manifest.json"}
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ValueError("Teacher bundle artifact entry is invalid")
            archive_path = str(artifact["archive_path"])
            if archive_path != f"teacher/{Path(archive_path).name}":
                raise ValueError(f"Unsafe teacher bundle path: {archive_path}")
            manifest_artifact_names.append(archive_path)
            payload = archive.read(archive_path)
            if len(payload) != int(artifact["size"]):
                raise ValueError(f"Teacher bundle size mismatch: {archive_path}")
            if sha256_bytes(payload) != artifact["sha256"]:
                raise ValueError(f"Teacher bundle hash mismatch: {archive_path}")
            expected_names.add(archive_path)
        if len(manifest_artifact_names) != len(set(manifest_artifact_names)):
            raise ValueError("Teacher bundle manifest contains duplicate artifacts")
        if set(manifest_artifact_names) != canonical_artifact_names:
            raise ValueError(
                "Teacher bundle does not contain the canonical artifact set"
            )
        if set(names) != expected_names:
            raise ValueError("Teacher bundle contains unexpected members")
    return {**manifest, "bundle_sha256": actual_hash}


def assert_local_provenance(
    manifest: dict[str, Any],
    repo_root: Path,
    config_path: Path,
) -> None:
    provenance = manifest.get("provenance", {})
    if git_output(repo_root, "rev-parse", "HEAD") != provenance.get(
        "git_commit"
    ):
        raise ValueError("Local commit does not match teacher bundle")
    config_relative_path = config_path.relative_to(repo_root).as_posix()
    manifest_config_path = provenance.get("config_path")
    if manifest_config_path not in (None, config_relative_path):
        raise ValueError("Local config path does not match teacher bundle")
    hash_basis = provenance.get("hash_basis", "working_tree_sha256")
    if hash_basis == PROVENANCE_HASH_BASIS:
        config_sha256 = git_blob_sha256(repo_root, config_relative_path)
        source_hashes = provenance.get("source_sha256", {})
        if set(source_hashes) != set(SOURCE_FILES):
            raise ValueError("Teacher bundle source hash set is incomplete")
        actual_source_hashes = {
            path: git_blob_sha256(repo_root, path)
            for path in SOURCE_FILES
        }
    elif hash_basis == "working_tree_sha256":
        config_sha256 = sha256_file(config_path)
        source_hashes = provenance.get("source_sha256", {})
        actual_source_hashes = {
            path: sha256_file(repo_root / path)
            for path in source_hashes
        }
    else:
        raise ValueError(f"Unsupported provenance hash basis: {hash_basis}")
    if config_sha256 != provenance.get("config_sha256"):
        raise ValueError("Local config hash does not match teacher bundle")
    for path, expected_hash in source_hashes.items():
        if actual_source_hashes[path] != expected_hash:
            raise ValueError(f"Local source hash does not match bundle: {path}")


def extract_teacher_bundle(
    bundle_path: Path,
    destination: Path,
    *,
    repo_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    if destination.exists():
        raise ValueError(f"Refusing to overwrite teacher destination: {destination}")
    manifest = verify_teacher_bundle(bundle_path)
    assert_local_provenance(manifest, repo_root, config_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.import-",
            dir=destination.parent,
        )
    )
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            for artifact in manifest["artifacts"]:
                archive_path = str(artifact["archive_path"])
                target = temporary / Path(archive_path).name
                target.write_bytes(archive.read(archive_path))
        (temporary / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        validate_teacher_root(temporary, load_config(config_path))
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--source-root", required=True)
    create.add_argument("--config", required=True)
    create.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--bundle", required=True)
    extract.add_argument("--destination", required=True)
    extract.add_argument("--config", required=True)

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.command == "create":
        config_path = (repo_root / args.config).resolve()
        result = create_teacher_bundle(
            Path(args.source_root),
            Path(args.output),
            expected_config=load_config(config_path),
            provenance=local_provenance(
                repo_root,
                config_path,
                source_root=Path(args.source_root),
            ),
        )
    elif args.command == "verify":
        result = verify_teacher_bundle(Path(args.bundle))
    else:
        config_path = (repo_root / args.config).resolve()
        result = extract_teacher_bundle(
            Path(args.bundle),
            Path(args.destination),
            repo_root=repo_root,
            config_path=config_path,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
