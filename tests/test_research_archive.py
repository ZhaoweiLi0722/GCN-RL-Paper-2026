import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest

from src.utils.research_archive import (
    copy_verified, create_archive, inventory, sha256_file, verify_archive,
)


class ResearchArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "payload"
        self.source.mkdir()
        (self.source / "weights.pt").write_bytes(b"synthetic checkpoint bytes")
        (self.source / "config.json").write_text('{"seed": 60}\n')
        self.archive = self.root / "snapshot.tar.gz"

    def test_round_trip_without_extraction_and_manifest(self):
        result = create_archive(self.source, self.archive)
        self.assertEqual(result["files"], inventory(self.source))
        self.assertEqual(result["file_count"], 2)
        verify_archive(self.archive, result["files"])
        self.assertEqual(json.loads(self.archive.with_name(self.archive.name + ".manifest.json").read_text()), result)

    def test_existing_archive_not_overwritten(self):
        create_archive(self.source, self.archive)
        digest = sha256_file(self.archive)
        with self.assertRaises(FileExistsError):
            create_archive(self.source, self.archive)
        self.assertEqual(sha256_file(self.archive), digest)

    def test_source_symlinks_rejected(self):
        (self.source / "link").symlink_to(self.source / "weights.pt")
        with self.assertRaises(ValueError):
            create_archive(self.source, self.archive)

    def test_archive_inside_source_rejected(self):
        with self.assertRaises(ValueError):
            create_archive(self.source, self.source / "recursive.tar.gz")

    def test_empty_source_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            inventory(empty)

    def test_tampered_expected_hash_rejected(self):
        result = create_archive(self.source, self.archive)
        result["files"]["weights.pt"] = "0" * 64
        with self.assertRaises(ValueError):
            verify_archive(self.archive, result["files"])

    def test_duplicate_archive_entries_rejected(self):
        with tarfile.open(self.archive, "w:gz") as handle:
            for _ in range(2):
                entry = tarfile.TarInfo("x")
                entry.size = 1
                handle.addfile(entry, io.BytesIO(b"x"))
        with self.assertRaises(ValueError):
            verify_archive(self.archive, {"x": "irrelevant"})

    def test_copy_verifies_but_does_not_claim_cloud_sync(self):
        create_archive(self.source, self.archive)
        target = self.root / "fake_sync" / self.archive.name
        receipt = copy_verified(self.archive, target)
        self.assertTrue(receipt["local_copy_verified"])
        self.assertFalse(receipt["cloud_sync_verified"])
        self.assertEqual(sha256_file(target), sha256_file(self.archive))
        self.assertEqual(copy_verified(self.archive, target), receipt)

    def test_different_destination_not_overwritten(self):
        target = self.root / "existing"
        target.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            copy_verified(self.source / "weights.pt", target)
        self.assertEqual(target.read_bytes(), b"keep")

    def test_destination_symlink_rejected(self):
        target = self.root / "link"
        target.symlink_to(self.source / "weights.pt")
        with self.assertRaises(FileExistsError):
            copy_verified(self.source / "weights.pt", target)


if __name__ == "__main__":
    unittest.main()
