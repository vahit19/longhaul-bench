"""Vector retrieval over machine manuals for LongHaul-Bench.

Edge-friendly RAG stack: Qdrant in serverless local mode (in-process, no
server, no Docker) + a small GGUF embedding model (nomic-embed-text-v1.5)
served by llama.cpp with --embeddings. No torch, no cloud.

The index is built once per world; `search` filters by machine so the agent
only sees the manual of the machine it is diagnosing (as on a real HMI).
"""

from __future__ import annotations

import json
import urllib.request

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

COLLECTION = "manual_rows"


def _embed(texts: list, endpoint: str) -> list:
    req = urllib.request.Request(
        endpoint + "/v1/embeddings",
        data=json.dumps({"input": texts, "model": "embed"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read())
    return [d["embedding"] for d in body["data"]]


def _row_text(machine_id: str, row: dict) -> str:
    return (
        f"machine {machine_id} symptom: {row['symptom']} "
        f"possible causes: {', '.join(row['possible_causes'])}"
    )


class VectorIndex:
    """In-process Qdrant index over all manual troubleshooting rows."""

    def __init__(self, world: dict, embed_endpoint: str = "http://127.0.0.1:8081"):
        self.endpoint = embed_endpoint
        self.client = QdrantClient(":memory:")
        rows, texts = [], []
        for m in world["manuals"]:
            for s in m["sections"]:
                for row in s["rows"]:
                    rows.append((m["machine_id"], row))
                    # nomic asymmetric prefixes: documents vs queries
                    texts.append("search_document: " + _row_text(m["machine_id"], row))

        vectors = _embed(texts, self.endpoint)
        self.client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )
        self.client.upsert(
            COLLECTION,
            points=[
                PointStruct(id=i, vector=v, payload={"machine_id": mid, "row": row})
                for i, (v, (mid, row)) in enumerate(zip(vectors, rows))
            ],
        )

    def search(self, machine_id: str, query: str, k: int = 5) -> list:
        vec = _embed(["search_query: " + query], self.endpoint)[0]
        hits = self.client.query_points(
            COLLECTION,
            query=vec,
            limit=k,
            query_filter=Filter(must=[FieldCondition(key="machine_id", match=MatchValue(value=machine_id))]),
        ).points
        return [h.payload["row"] for h in hits]
