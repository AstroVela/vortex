# SPDX-License-Identifier: MIT
# Copyright 2018-2025 Stichting DuckDB Foundation
# Adapted from AstroVela/duckdb-iceberg test/vortex/test_local_catalog.py.
# See fixtures/LICENSE.duckdb-iceberg.

import copy
from contextlib import contextmanager
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import tempfile
import threading
from urllib.parse import urlparse


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


class AppendCatalog(HTTPServer):
    def __init__(self, metadata_path):
        super().__init__(("127.0.0.1", 0), CatalogHandler)
        self.metadata_path = metadata_path
        self.metadata = json.loads(metadata_path.read_text())
        self.metadata["properties"]["write.format.default"] = "vortex"
        self.metadata["schemas"][0]["fields"][0]["required"] = True
        self.commits = []
        self.errors = []

    def load_table(self):
        return {"metadata-location": str(self.metadata_path), "metadata": self.metadata}

    def commit(self, body):
        metadata = copy.deepcopy(self.metadata)
        for requirement in body["requirements"]:
            kind = requirement["type"]
            if kind == "assert-table-uuid":
                assert requirement["uuid"] == metadata["table-uuid"]
            elif kind == "assert-ref-snapshot-id":
                assert requirement["ref"] == "main"
                assert requirement["snapshot-id"] == metadata.get("current-snapshot-id")
            elif kind == "assert-current-schema-id":
                assert requirement["current-schema-id"] == metadata["current-schema-id"]
            elif kind == "assert-last-assigned-field-id":
                assert requirement["last-assigned-field-id"] == metadata["last-column-id"]
            else:
                raise AssertionError(f"Unexpected commit requirement: {requirement}")
        for update in body["updates"]:
            action = update["action"]
            if action == "add-snapshot":
                snapshot = update["snapshot"]
                metadata["snapshots"].append(snapshot)
                metadata["last-sequence-number"] = snapshot["sequence-number"]
                metadata["last-updated-ms"] = snapshot["timestamp-ms"]
            elif action == "set-snapshot-ref":
                assert update["ref-name"] == "main"
                metadata["current-snapshot-id"] = update["snapshot-id"]
                metadata.setdefault("refs", {})["main"] = {
                    "snapshot-id": update["snapshot-id"],
                    "type": "branch",
                }
            elif action == "set-properties":
                metadata["properties"].update(update["updates"])
            elif action == "add-schema":
                metadata["schemas"].append(update["schema"])
                metadata["last-column-id"] = update.get("last-column-id", metadata["last-column-id"])
            elif action == "set-current-schema":
                metadata["current-schema-id"] = update["schema-id"]
            else:
                raise AssertionError(f"Unexpected commit update: {update}")
        metadata.setdefault("snapshot-log", []).append(
            {
                "timestamp-ms": metadata["last-updated-ms"],
                "snapshot-id": metadata["current-snapshot-id"],
            }
        )
        path = self.metadata_path.parent / f"test-commit-{len(self.commits)}.metadata.json"
        path.write_text(json.dumps(metadata))
        self.metadata_path = path
        self.metadata = metadata
        self.commits.append(body)
        return self.load_table()


class CatalogHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def reply(self, body, status=200):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/v1/config":
            self.reply(
                {
                    "defaults": {},
                    "overrides": {},
                    "endpoints": [
                        "GET /v1/{prefix}/namespaces",
                        "GET /v1/{prefix}/namespaces/{namespace}",
                        "GET /v1/{prefix}/namespaces/{namespace}/tables",
                        "GET /v1/{prefix}/namespaces/{namespace}/tables/{table}",
                        "POST /v1/{prefix}/namespaces/{namespace}/tables/{table}",
                    ],
                }
            )
        elif path == "/v1/namespaces":
            self.reply({"namespaces": [["main"]]})
        elif path == "/v1/namespaces/main":
            self.reply({"namespace": ["main"], "properties": {}})
        elif path == "/v1/namespaces/main/tables":
            self.reply({"identifiers": [{"namespace": ["main"], "name": "items"}]})
        elif path == "/v1/namespaces/main/tables/items":
            self.reply(self.server.load_table())
        else:
            self.server.errors.append(f"Unexpected GET {path}")
            self.reply({"error": {"message": path, "type": "NoSuchTableException", "code": 404}}, 404)

    def do_POST(self):
        try:
            assert urlparse(self.path).path == "/v1/namespaces/main/tables/items"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.reply(self.server.commit(body))
        except Exception as error:
            self.server.errors.append(str(error))
            self.reply({"error": {"message": str(error), "type": "CommitFailedException", "code": 409}}, 409)


@contextmanager
def local_catalog(metadata_path):
    server = AppendCatalog(metadata_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    attach = (
        f"ATTACH '' AS lake (TYPE iceberg, ENDPOINT {quote(endpoint)}, "
        "AUTHORIZATION_TYPE 'none', ACCESS_DELEGATION_MODE 'none', DEFAULT_SCHEMA 'main'); "
    )
    try:
        yield server, attach
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
