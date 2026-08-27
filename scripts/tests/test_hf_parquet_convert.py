# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import concurrent.futures
import importlib.util
import types
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "huggingface-convert.py"
SPEC = importlib.util.spec_from_file_location("hf_parquet_convert", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LocalCopyUploaderTest(unittest.TestCase):
    def test_requires_xet_enabled_repository(self):
        class FakeApi:
            def __init__(self, enabled):
                self.enabled = enabled

            def repo_info(self, *args, **kwargs):
                return types.SimpleNamespace(xet_enabled=self.enabled)

        MODULE.require_xet_repository(FakeApi(True), "owner/repo", "main", 1)
        with self.assertRaisesRegex(RuntimeError, "not Xet-enabled"):
            MODULE.require_xet_repository(FakeApi(False), "owner/repo", "main", 1)

    def test_retry_call_retries_transient_failures_only(self):
        calls = []

        def transient_operation():
            calls.append(None)
            if len(calls) < 3:
                raise TimeoutError("temporary")
            return "complete"

        self.assertEqual(MODULE.retry_call(transient_operation, 3, base_delay=0), "complete")
        self.assertEqual(len(calls), 3)

        calls.clear()

        def permanent_operation():
            calls.append(None)
            raise ValueError("permanent")

        with self.assertRaisesRegex(ValueError, "permanent"):
            MODULE.retry_call(permanent_operation, 3, base_delay=0)
        self.assertEqual(len(calls), 1)

    def test_missing_destination_prefix_is_an_empty_listing(self):
        import httpx
        from huggingface_hub.errors import RemoteEntryNotFoundError

        class FakeApi:
            def list_repo_tree(self, *args, **kwargs):
                response = httpx.Response(
                    404, request=httpx.Request("GET", "https://huggingface.co/api/datasets/x/tree"))
                raise RemoteEntryNotFoundError("missing prefix", response=response)

        self.assertEqual(
            MODULE.list_repository_files(FakeApi(), "owner/repo", "main", "vortex"), {})

    def test_huggingface_batches_each_format_by_file_range(self):
        class FakeApi:
            def __init__(self):
                self.preuploads = []
                self.commits = []

            def dataset_info(self, repo, revision=None, timeout=None):
                return types.SimpleNamespace(sha="parent")

            def preupload_lfs_files(self, repo, additions, **kwargs):
                self.preuploads.extend(additions)

            def create_commit(self, repo, operations, **kwargs):
                self.commits.append((list(operations), kwargs))
                number = len(self.commits)
                return types.SimpleNamespace(commit_url=f"https://fixture/commit/{number}",
                                             oid=f"commit-{number}")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            api = FakeApi()
            uploader = MODULE.HuggingFaceBatchUploader(
                api, "owner/repo", "main", 2, {"vortex": 3, "vortex-compact": 3})
            paths = []
            for ordinal in range(1, 4):
                path = root / f"{ordinal}.vortex"
                path.write_bytes(bytes([ordinal]))
                paths.append(path)

            pending = uploader.upload(paths[0], "vortex/1.vortex",
                                      format_name="vortex", ordinal=1)
            committed = uploader.upload(paths[1], "vortex/2.vortex",
                                        format_name="vortex", ordinal=2)
            compact = uploader.upload(paths[2], "vortex-compact/3.vortex",
                                      format_name="vortex-compact", ordinal=3)
            final = uploader.flush()

            self.assertEqual(pending["status"], "preuploaded")
            self.assertEqual(committed["commit_message"], "Upload vortex files 1-2 of 3")
            self.assertEqual(compact["status"], "preuploaded")
            self.assertEqual(final[0]["commit_message"],
                             "Upload vortex-compact files 3-3 of 3")
            self.assertEqual(len(api.preuploads), 3)
            self.assertEqual(len(api.commits), 2)
            self.assertEqual(api.commits[0][1]["parent_commit"], "parent")
            self.assertEqual(api.commits[1][1]["parent_commit"], "commit-1")

    def test_huggingface_batch_byte_limit_flushes_before_file_limit(self):
        class FakeApi:
            def __init__(self):
                self.commits = []

            def dataset_info(self, repo, revision=None, timeout=None):
                return types.SimpleNamespace(sha="parent")

            def preupload_lfs_files(self, repo, additions, **kwargs):
                return None

            def create_commit(self, repo, operations, **kwargs):
                self.commits.append(list(operations))
                return types.SimpleNamespace(commit_url="https://fixture/commit/1",
                                             oid="commit-1")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.vortex"
            second = root / "second.vortex"
            first.write_bytes(b"123")
            second.write_bytes(b"456")
            api = FakeApi()
            uploader = MODULE.HuggingFaceBatchUploader(
                api, "owner/repo", "main", 100, {"vortex": 2}, batch_bytes=5)

            self.assertEqual(uploader.upload(
                first, "vortex/first.vortex", format_name="vortex", ordinal=1
            )["status"], "preuploaded")
            result = uploader.upload(
                second, "vortex/second.vortex", format_name="vortex", ordinal=2)

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["size_bytes"], 6)
            self.assertEqual(len(api.commits), 1)

    def test_download_buffer_allows_one_oversized_shard_only_when_empty(self):
        self.assertTrue(MODULE.fits_download_buffer(0, 20, 10))
        self.assertTrue(MODULE.fits_download_buffer(4, 6, 10))
        self.assertFalse(MODULE.fits_download_buffer(4, 7, 10))

    def test_failed_huggingface_commit_retains_preuploaded_file(self):
        class FakeApi:
            def dataset_info(self, repo, revision=None, timeout=None):
                return types.SimpleNamespace(sha="parent")

            def preupload_lfs_files(self, repo, additions, **kwargs):
                return None

            def create_commit(self, repo, operations, **kwargs):
                raise ValueError("fixture commit failure")

            def list_repo_tree(self, *args, **kwargs):
                return []

        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact.vortex"
            artifact.write_bytes(b"keep until commit")
            uploader = MODULE.HuggingFaceBatchUploader(
                FakeApi(), "owner/repo", "main", 1, {"vortex": 1}, attempts=1)

            with self.assertRaisesRegex(ValueError, "fixture commit failure"):
                uploader.upload(artifact, "vortex/artifact.vortex",
                                format_name="vortex", ordinal=1)
            self.assertEqual(artifact.read_bytes(), b"keep until commit")

    def test_mirrored_destination_path(self):
        self.assertEqual(
            MODULE.destination_path(
                "sample/10BT/000_00000.parquet", "vortex-compact"),
            "vortex-compact/sample/10BT/000_00000.vortex",
        )

    def test_parquet_destination_keeps_parquet_extension(self):
        self.assertEqual(
            MODULE.destination_path("data/train.parquet", "parquet-zstd6"),
            "parquet-zstd6/data/train.parquet",
        )

    def test_full_path_filter(self):
        class FakeApi:
            def list_repo_tree(self, *args, **kwargs):
                return [types.SimpleNamespace(path="sample/10BT/a.parquet", size=10, lfs=None),
                        types.SimpleNamespace(path="sample/100BT/b.parquet", size=20, lfs=None)]

        shards = MODULE.list_shards(FakeApi(), "owner/repo", "main", "sample", "*.parquet",
                                    ["sample/10BT/*"])
        self.assertEqual(shards, [{"path": "sample/10BT/a.parquet", "size": 10,
                                   "sha256": None}])

    def test_copies_to_hub_shaped_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.vortex"
            source.write_bytes(b"vortex fixture")

            uploader = MODULE.LocalCopyUploader(root / "remote", chunk_size=4)
            result = uploader.upload(source, "converted/vortex/source.vortex")

            uploaded = root / "remote/converted/vortex/source.vortex"
            self.assertEqual(uploaded.read_bytes(), source.read_bytes())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["hub_path"], "converted/vortex/source.vortex")
            self.assertEqual(result["url"], uploaded.as_uri())
            self.assertEqual(result["chunks"], 4)
            self.assertEqual(result["bytes_uploaded"], len(b"vortex fixture"))
            self.assertFalse(uploaded.with_suffix(".vortex.part").exists())

    def test_local_sink_can_reconstruct_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "vortex/sample/10BT/a.vortex"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"existing")
            files = MODULE.LocalCopyUploader(root).existing_files("vortex")
            self.assertEqual(files[str(existing.relative_to(root))]["size"], len(b"existing"))

    def test_failed_sink_preserves_local_artifact(self):
        class FailingUploader(MODULE.Uploader):
            def upload(self, local_path, destination_path):
                raise RuntimeError("fixture sink failure")

            def config(self):
                return {"type": "failing-fixture"}

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "artifact.vortex"
            source.write_bytes(b"must survive")
            with self.assertRaisesRegex(RuntimeError, "fixture sink failure"):
                MODULE.upload_then_maybe_delete(FailingUploader(), source, "artifact.vortex", True)
            self.assertEqual(source.read_bytes(), b"must survive")

    def test_parallel_real_encoders_upload_and_delete(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")
        vx = SCRIPT.parents[1] / "target/release/vx"
        if not vx.exists():
            self.skipTest("release vx binary is not built")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.parquet"
            pq.write_table(pa.table({"text": ["alpha", "beta"] * 70_000}), source)
            local = root / "local"
            outputs = {"parquet-zstd6": local / "source.parquet",
                       "vortex": local / "source.vortex",
                       "vortex-compact": local / "source-compact.vortex"}
            sink = MODULE.LocalCopyUploader(root / "sink")

            def encode_upload(fmt):
                destination = outputs[fmt]
                if fmt == "parquet-zstd6":
                    MODULE.parquet_zstd6(source, destination)
                else:
                    strategy = "compact" if fmt == "vortex-compact" else "btrblocks"
                    MODULE.run_vx(vx, source, destination, strategy)
                return MODULE.upload_then_maybe_delete(
                    sink, destination, f"fixture/{fmt}/{destination.name}", True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                results = list(executor.map(encode_upload, outputs))
            source.unlink()

            self.assertTrue(all(result["local_deleted"] for result in results))
            self.assertTrue(all(not output.exists() for output in outputs.values()))
            self.assertFalse(source.exists())
            self.assertEqual(list(root.glob(".*.input.*")), [])
            for fmt, output in outputs.items():
                uploaded = root / "sink" / "fixture" / fmt / output.name
                self.assertGreater(uploaded.stat().st_size, 0)
                if fmt == "parquet-zstd6":
                    metadata = pq.ParquetFile(uploaded).metadata
                    self.assertTrue(all(metadata.row_group(index).num_rows
                                        <= MODULE.PARQUET_BATCH_ROWS
                                        for index in range(metadata.num_row_groups)))


if __name__ == "__main__":
    unittest.main()
