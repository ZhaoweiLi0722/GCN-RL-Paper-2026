"""Bounded publication check of Git blobs not yet reachable from origin."""

import hashlib
import json
from pathlib import Path
import re
import subprocess

BASE = Path(__file__).resolve().parents[2]


def git(*args, **kwargs):
    return subprocess.check_output(["git", *args], cwd=BASE, **kwargs)


rows = git("rev-list", "--objects", "HEAD", "--not", "--remotes=origin", text=True).splitlines()
paths = dict((row.split(" ", 1) + [""])[:2] for row in rows)
metadata = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
               input="\n".join(paths) + "\n", text=True).splitlines()
patterns = {
    "private_key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "github_token": rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{50,}\b",
    "aws_key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "openai_key": rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}\b",
}
findings, blobs = [], []
for row in metadata:
    oid, kind, size = row.split()
    if kind != "blob":
        continue
    size = int(size)
    path = paths[oid]
    if size >= 100 * 1024**2:
        findings.append({"path": path, "reason": "oversized_blob", "oid": oid})
        continue
    data = git("cat-file", "blob", oid)
    for name, pattern in patterns.items():
        if re.search(pattern, data):
            findings.append({"path": path, "reason": name, "oid": oid})
    if Path(path).name in (".env", "id_rsa", "id_ed25519", "credentials"):
        findings.append({"path": path, "reason": "sensitive_filename", "oid": oid})
    blobs.append({"oid": oid, "path": path, "bytes": size,
                  "sha256": hashlib.sha256(data).hexdigest()})
result = dict(scope="new outgoing Git blobs only; not proof of absence of all secrets",
              head=git("rev-parse", "HEAD", text=True).strip(),
              remote="origin", blobs=len(blobs), bytes=sum(b["bytes"] for b in blobs),
              largest_bytes=max((b["bytes"] for b in blobs), default=0),
              findings=findings, passed=not findings)
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(1 if findings else 0)
