"""
maud_index.py -- embedding retrieval over the MAUD 152 contracts (the RAG baseline).

Same retrieval setup as LegalBench-RAG's best configuration without a reranker
(legalbenchrag/methods/baseline.py): recursive character splitting ("rcts") into
500-character chunks, no overlap, embedded with OpenAI text-embedding-3-large
(served through OpenRouter here), cosine top-k. The query is the MAUD question
with its answer choices. Retrieved chunks are put back in document order and
adjacent chunks are merged, so the answering model reads contiguous passages.

Embeddings are computed once (`build`) and cached on disk, so runs make no
embedding calls; the embedding cost is in maud_workdir/rag_index/<config>/usage.json.

    LLM_PROVIDER=openrouter python -m rag.maud_index build      # paid: ~13M tokens, ~$1.7
    python -m rag.maud_index check                             # free: retrieval recall vs gold spans
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "openai/text-embedding-3-large"
CHUNK_SIZE = 500
CONFIG = f"text-embedding-3-large_rcts{CHUNK_SIZE}"
INDEX_DIR = f"maud_workdir/rag_index/{CONFIG}"
GOLD_PATH = "maud/dicts/maud_152_gold_spans.json"
BATCH = 256          # chunks per embeddings request (~32K tokens)
SEPARATORS = ["\n\n", "\n", "!", "?", ".", ":", ";", ",", " ", ""]   # LegalBench-RAG's rcts


def chunk(text: str) -> list[tuple[int, int]]:
    """Character spans of the rcts chunks; they tile `text` exactly."""
    splitter = RecursiveCharacterTextSplitter(separators=SEPARATORS, chunk_size=CHUNK_SIZE, chunk_overlap=0,
                                              length_function=len, is_separator_regex=False, strip_whitespace=False)
    splits = splitter.split_text(text)
    assert "".join(splits) == text
    spans, pos = [], 0
    for s in splits:
        spans.append((pos, pos + len(s)))
        pos += len(s)
    return spans


def query_text(question_text: str, options: list[str]) -> str:
    return question_text.strip() + "\n" + "\n".join(f"- {o}" for o in options)


# --------------------------------------------------------------------------
# building the index (paid)
# --------------------------------------------------------------------------
_usage_lock = threading.Lock()


def _client():
    from openai import OpenAI
    load_dotenv()
    return OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")


def _embed(client, texts: list[str], usage: dict) -> np.ndarray:
    for attempt in range(6):
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            break
        except Exception as e:  # noqa: BLE001 -- network / 429 / 5xx: back off and retry
            if attempt == 5:
                raise
            print(f"  embeddings retry {attempt + 1}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(2 ** attempt * 5)
    u = resp.usage.model_dump() if resp.usage else {}
    with _usage_lock:
        usage["requests"] += 1
        usage["prompt_tokens"] += u.get("prompt_tokens") or 0
        usage["cost_usd"] += float(u.get("cost") or 0.0)
    return np.array([d.embedding for d in sorted(resp.data, key=lambda d: d.index)], dtype=np.float32)


def _sample():
    from maud.maud_data_sampling import get_maud_data_dir, get_sample_rows_for_maud_152
    with open("maud/dicts/question_full_text_map.json", encoding="utf-8") as f:
        full_text = json.load(f)
    with open("maud/dicts/question_answer_map_unique.json", encoding="utf-8") as f:
        options = json.load(f)
    df = get_sample_rows_for_maud_152(split="train")
    return df, full_text, options, f"{get_maud_data_dir()}/contracts"


def build(workers: int = 4) -> None:
    df, full_text, options, contracts_dir = _sample()
    os.makedirs(f"{INDEX_DIR}/contracts", exist_ok=True)
    usage_path = f"{INDEX_DIR}/usage.json"
    usage = {"model": EMBEDDING_MODEL, "chunk_size": CHUNK_SIZE, "requests": 0, "prompt_tokens": 0, "cost_usd": 0.0}
    if os.path.exists(usage_path):
        with open(usage_path) as f:
            usage.update({k: v for k, v in json.load(f).items() if k in ("requests", "prompt_tokens", "cost_usd")})
    client = _client()

    def one(name: str) -> None:
        out = f"{INDEX_DIR}/contracts/{name}.npz"
        if os.path.exists(out):
            return
        # same decoding as benchmarks/maud_bench.py, so spans index the item's document
        with open(f"{contracts_dir}/{name}.txt", encoding="utf-8-sig") as f:
            text = f.read()
        spans = chunk(text)
        embs = [_embed(client, [text[s:e] for s, e in spans[i:i + BATCH]], usage) for i in range(0, len(spans), BATCH)]
        np.savez_compressed(out, emb=np.concatenate(embs).astype(np.float16), spans=np.array(spans, dtype=np.int32))
        with _usage_lock, open(usage_path, "w") as f:
            json.dump(usage, f, indent=1)
        print(f"  {name}: {len(spans)} chunks  (total {usage['prompt_tokens']:,} tokens, ${usage['cost_usd']:.4f})", flush=True)

    names = sorted(set(df.contract_name))
    print(f"embedding {len(names)} contracts with {EMBEDDING_MODEL} -> {INDEX_DIR}")
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, names))

    qpath = f"{INDEX_DIR}/queries.npz"
    if not os.path.exists(qpath):
        row_ids = [f"{r.contract_name}_{i}" for i, r in enumerate(df.itertuples())]
        texts = [query_text(full_text[r.question], options[r.question]) for r in df.itertuples()]
        np.savez_compressed(qpath, row_ids=np.array(row_ids), emb=_embed(client, texts, usage))
    with open(usage_path, "w") as f:
        json.dump(usage, f, indent=1)
    print(f"done: {usage['prompt_tokens']:,} tokens, ${usage['cost_usd']:.4f} reported by OpenRouter")


# --------------------------------------------------------------------------
# retrieval (no API call)
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _contract(name: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(f"{INDEX_DIR}/contracts/{name}.npz")
    emb = z["emb"].astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True), z["spans"]


@lru_cache(maxsize=1)
def _queries() -> dict[str, np.ndarray]:
    z = np.load(f"{INDEX_DIR}/queries.npz")
    return {rid: e / np.linalg.norm(e) for rid, e in zip(z["row_ids"].tolist(), z["emb"].astype(np.float32))}


def retrieve(contract_name: str, row_id: str, k: int) -> list[tuple[int, int]]:
    """Top-k chunk spans by cosine similarity, in document order, adjacent chunks merged."""
    emb, spans = _contract(contract_name)
    top = np.argsort(-(emb @ _queries()[row_id]))[:k]
    merged: list[tuple[int, int]] = []
    for s, e in sorted(tuple(map(int, spans[i])) for i in top):
        if merged and s == merged[-1][1]:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def excerpts(document: str, spans: list[tuple[int, int]]) -> str:
    return "\n\n[...]\n\n".join(document[s:e].strip() for s, e in spans)


def recall_precision(retrieved: list[tuple[int, int]], gold: list[tuple[int, int]]) -> tuple[float, float]:
    """LegalBench-RAG's character-level recall and precision (legalbenchrag/run_benchmark.py)."""
    common = sum(max(0, min(e1, e2) - max(s1, s2)) for s1, e1 in retrieved for s2, e2 in gold)
    n_gold, n_ret = sum(e - s for s, e in gold), sum(e - s for s, e in retrieved)
    return (common / n_gold if n_gold else float("nan")), (common / n_ret if n_ret else 0.0)


@lru_cache(maxsize=1)
def gold_spans() -> dict[str, list[tuple[int, int]]]:
    with open(GOLD_PATH, encoding="utf-8") as f:
        return {g["row_id"]: [tuple(s) for s in g["spans"]] for g in json.load(f) if g["spans"]}


def check(ks: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)) -> None:
    """Retrieval recall/precision@k against the gold spans (LegalBench-RAG's metric), no API call."""
    df, *_ = _sample()
    gold = gold_spans()
    print(f"{'k':>4} {'recall':>8} {'precision':>10} {'any gold hit':>13}   (n={sum(f'{r.contract_name}_{i}' in gold for i, r in enumerate(df.itertuples()))})")
    for k in ks:
        rs, ps, hit = [], [], []
        for i, r in enumerate(df.itertuples()):
            rid = f"{r.contract_name}_{i}"
            if rid in gold:
                rec, prec = recall_precision(retrieve(r.contract_name, rid, k), gold[rid])
                rs.append(rec), ps.append(prec), hit.append(rec > 0)
        print(f"{k:>4} {np.mean(rs):>8.3f} {np.mean(ps):>10.3f} {np.mean(hit):>13.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    build() if cmd == "build" else check()
