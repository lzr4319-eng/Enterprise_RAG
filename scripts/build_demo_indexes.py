"""Build fictional BM25 demo resources without downloading embedding models.

The deterministic hash vectors only satisfy the existing FAISS file format.
Metadata explicitly disables semantic vector retrieval for these demo indexes.
Existing enterprise indexes are never overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path

import faiss
import jieba
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIMENSION = 1024
CATEGORIES = ("employee", "product")


def demo_vector(text):
    vector = np.zeros(DIMENSION, dtype=np.float32)
    for token in jieba.lcut(text):
        if not token.strip():
            continue
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % DIMENSION
        vector[bucket] += 1.0 if digest[4] & 1 else -1.0
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return vector


def load_records(path):
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("text"), str) or not record["text"].strip():
                raise ValueError(f"Missing non-empty text in {path}")
            record["embedding_mode"] = "demo-bm25"
            records.append(record)
    if not records:
        raise ValueError(f"No demo records in {path}")
    return records


def build_demo_indexes(output_dir):
    output_dir = Path(output_dir).resolve()
    targets = [
        output_dir / category / name
        for category in CATEGORIES
        for name in ("faiss_index.index", "faiss_index_metadata.jsonl")
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing knowledge-base files. "
            "Use --output-dir with a new directory. Existing files: "
            + ", ".join(existing)
        )

    # Validate all sources before creating any output.
    sources = {
        category: load_records(PROJECT_ROOT / "examples" / f"{category}.jsonl")
        for category in CATEGORIES
    }
    for category, records in sources.items():
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        vectors = np.stack([demo_vector(record["text"]) for record in records])
        index = faiss.IndexFlatIP(DIMENSION)
        index.add(vectors)
        faiss.write_index(index, str(category_dir / "faiss_index.index"))
        with (category_dir / "faiss_index_metadata.jsonl").open(
            "x", encoding="utf-8"
        ) as destination:
            for record in records:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{category}: {len(records)} fictional demo records -> {category_dir}")
    print("Demo indexes use BM25 only; they are not bge-m3 semantic embeddings.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "indexes",
        help="Output directory; existing index files will not be overwritten.",
    )
    args = parser.parse_args()
    try:
        build_demo_indexes(args.output_dir)
    except (FileExistsError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
