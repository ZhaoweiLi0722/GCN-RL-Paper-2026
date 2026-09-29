"""Non-overwriting, hash-verified snapshots for completed research artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Snapshot source must be a real directory")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink is not an archival payload: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    if not files:
        raise ValueError("Empty snapshot")
    return files


def verify_archive(path: Path, files: dict[str, str]) -> None:
    observed = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.name in observed:
                raise ValueError("Archive contains non-file or duplicate member")
            if member.name not in files:
                raise ValueError(f"Unexpected archive member: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("Unreadable archive member")
            digest = hashlib.sha256()
            with handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            observed[member.name] = digest.hexdigest()
    if observed != files:
        raise ValueError("Archive membership/content differs from inventory")


def create_archive(root: Path, output: Path) -> dict:
    """Verify both archive bytes and unchanged source; never extract a tar."""
    root, output = root.resolve(), output.absolute()
    if output.is_relative_to(root):
        raise ValueError("Archive must be outside its source tree")
    files = inventory(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_name(output.name + ".manifest.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(output)
    with output.open("xb") as handle:
        with tarfile.open(fileobj=handle, mode="w:gz") as archive:
            for relative in files:
                archive.add(root / relative, arcname=relative, recursive=False)
    verify_archive(output, files)
    if inventory(root) != files:
        raise ValueError("Source changed during snapshot; preserve failed archive")
    payload = {
        "schema_version": 1,
        "archive": output.name,
        "archive_sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
        "file_count": len(files),
        "files": files,
        "verified_by_reading_every_archive_member": True,
    }
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return payload


def copy_verified(source: Path, target: Path) -> dict:
    """Copy only complete artifacts; local verification is not cloud receipt."""
    if source.is_symlink() or not source.is_file():
        raise ValueError("Copy source must be a regular file")
    expected = sha256_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or sha256_file(target) != expected:
            raise FileExistsError(f"Destination is different; refusing overwrite: {target}")
    else:
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        if sha256_file(target) != expected or sha256_file(source) != expected:
            raise ValueError("Backup checksum mismatch; partial copy preserved")
    return {"path": str(target), "sha256": expected,
            "local_copy_verified": True, "cloud_sync_verified": False}
