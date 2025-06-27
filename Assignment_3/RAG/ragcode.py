# rag_gemini_enhanced.py
"""RAG pipeline powered 100 % by Google Gemini — jq‑free & source‑free.

Quick CLI
---------
```bash
# crawl a site (optional)
python rag.py crawl  --url https://example.com --max-pages 200 --out corpus.jsonl
# embed & index
python rag.py ingest --in corpus.jsonl        --index faiss_index
# chat
python rag.py ask    --index faiss_index      --k 5
# lightweight TriviaQA evaluation
python rag.py eval   --index faiss_index      --samples 20 --rpm 6
```
Dependencies
```
pip install google-generativeai langchain-community langchain-google-genai faiss-cpu trafilatura playwright beautifulsoup4 datasets evaluate tiktoken
playwright install           # one‑time browser download
export GOOGLE_API_KEY=…      # before any command
```
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import pathlib
import random
import time
from collections import deque
from itertools import islice
from typing import List, Set

import trafilatura
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from google.api_core.exceptions import ResourceExhausted

LOGGER = logging.getLogger("rag")
DEFAULT_UA = "Mozilla/5.0 (compatible; GeminiRAG/1.8)"
MAX_CONCURRENCY = 5
TIMEOUT_MS = 30_000  # ms

# ──────────────────────────── CRAWLER ────────────────────────────
async def crawl_site(start_url: str, max_pages: int, out_path: pathlib.Path) -> None:
    """Breadth‑first crawl; save each page’s trafilatura JSON extract."""
    from playwright.async_api import async_playwright

    queue: deque[str] = deque([start_url])
    seen: Set[str] = set()
    saved = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=DEFAULT_UA)
        sem = asyncio.Semaphore(MAX_CONCURRENCY)
        fh = out_path.open("w", encoding="utf-8")

        async def worker() -> None:
            nonlocal saved
            while queue and saved < max_pages:
                url = queue.popleft()
                if url in seen:
                    continue
                seen.add(url)
                await sem.acquire()
                try:
                    page = await ctx.new_page()
                    try:
                        await page.goto(url, timeout=TIMEOUT_MS)
                        html = await page.content()
                        raw = trafilatura.extract(html, output_format="json", url=url)
                        if not raw:
                            continue
                        data = json.loads(raw)
                        if not data.get("text", "").strip():
                            continue
                        fh.write(json.dumps(data, ensure_ascii=False) + "\n")
                        saved += 1
                        LOGGER.info("Saved %s (%d/%d)", url, saved, max_pages)
                        for link in await page.eval_on_selector_all("a", "els=>els.map(e=>e.href)"):
                            if link.startswith(start_url):
                                queue.append(link)
                    finally:
                        await page.close()
                except Exception as exc:
                    LOGGER.warning("Error %s: %s", url, exc)
                finally:
                    sem.release()

        await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(MAX_CONCURRENCY)])
        fh.close(); await browser.close()
    LOGGER.info("Crawl done — %d pages → %s", saved, out_path)

# ──────────────────────────── INGEST ────────────────────────────

def ingest_corpus(corpus: pathlib.Path, index_dir: pathlib.Path, *, chunk:int=1000, overlap:int=200) -> None:
    embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    docs: List[Document] = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = obj.get("text", "").strip()
        if not text:
            continue
        src = obj.get("url") or obj.get("source") or "unknown"
        docs.append(Document(page_content=text, metadata={"source": src}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk, chunk_overlap=overlap, add_start_index=True)
    chunks = splitter.split_documents(docs)

    unique, seen = [], set()
    for doc in chunks:
        body = doc.page_content.strip()
        if len(body) < 200:
            continue
        h = hashlib.sha256(body.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h); unique.append(doc)

    LOGGER.info("Embedding %d chunks…", len(unique))
    db = FAISS.from_documents(unique, embed)
    db.save_local(str(index_dir))
    LOGGER.info("Index saved → %s (docs=%d)", index_dir, db.index.ntotal)

# ──────────────────────────── RAG CHAIN ────────────────────────────

def build_rag(index_dir: pathlib.Path, *, k:int=5) -> RetrievalQA:
    embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local(str(index_dir), embed, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": k})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.2)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)


def interactive_ask(index_dir: pathlib.Path, *, k:int=5) -> None:
    chain = build_rag(index_dir, k=k)
    while True:
        try:
            q = input("\n❓ Question (blank→exit): ")
        except (EOFError, KeyboardInterrupt):
            print(); return
        if not q:
            return
        res = chain.invoke({"query": q})
        ans = res.get("result") if isinstance(res, dict) else res
        print("\nAnswer:\n", ans, "\n")

# ──────────────────────────── EVALUATION ────────────────────────────

def evaluate_trivia(index_dir: pathlib.Path, *, samples:int=20, seed:int=42, rpm:int=6) -> None:
    """Stream *samples* TriviaQA rows (no 3‑GB download) and compute EM/F1.
    `rpm` keeps requests under the free‑tier 10 req/min limit.
    """
    from datasets import load_dataset; import evaluate as hf_eval

    delay = 60 / max(1, rpm)
    random.seed(seed)
    stream = load_dataset("trivia_qa", "rc", split="validation", streaming=True)
    small_ds = list(islice(stream.shuffle(seed=seed, buffer_size=10_000), samples))

    chain = build_rag(index_dir)
    squad = hf_eval.load("squad")
    preds, refs = [], []
    t0 = time.time()
    for i, ex in enumerate(small_ds, 1):
        if i > 1:
            time.sleep(delay)
        try:
            out = chain.invoke({"query": ex["question"]})
        except ResourceExhausted:
            LOGGER.warning("429 quota hit; sleeping 10 s and retrying…")
            time.sleep(10)
            out = chain.invoke({"query": ex["question"]})
        # normalize answer
        answer_text = out.get("result") if isinstance(out, dict) else str(out)
        preds.append({"id": ex["question_id"], "prediction_text": answer_text})
        refs.append({"id": ex["question_id"], "answers": {"answer_start": [0], "text": [ex["answer"]["value"]]}})

    print(json.dumps(squad.compute(predictions=preds, references=refs), indent=2))
    LOGGER.info("Eval finished in %.1f s", time.time() - t0)
# ──────────────────────────── CLI ENTRY ────────────────────────────
def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY first")

    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description="Gemini-powered RAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # crawl
    crawl_p = sub.add_parser("crawl", help="Crawl a website")
    crawl_p.add_argument("--url", required=True)
    crawl_p.add_argument("--max-pages", type=int, default=200)
    crawl_p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("corpus.jsonl"))

    # ingest
    ingest_p = sub.add_parser("ingest", help="Embed & index corpus")
    ingest_p.add_argument("--in", dest="_in", type=pathlib.Path, required=True)
    ingest_p.add_argument("--index", type=pathlib.Path, default=pathlib.Path("faiss_index"))
    ingest_p.add_argument("--chunk", type=int, default=1000)
    ingest_p.add_argument("--overlap", type=int, default=200)

    # ask
    ask_p = sub.add_parser("ask", help="Interactive Q&A shell")
    ask_p.add_argument("--index", type=pathlib.Path, required=True)
    ask_p.add_argument("--k", type=int, default=5)

    # eval
    eval_p = sub.add_parser("eval", help="Evaluate on TriviaQA")
    eval_p.add_argument("--index", type=pathlib.Path, required=True)
    eval_p.add_argument("--samples", type=int, default=20)
    eval_p.add_argument("--rpm", type=int, default=6)  # requests per minute throttle

    args = parser.parse_args()

    if args.cmd == "crawl":
        asyncio.run(crawl_site(args.url, args.max_pages, args.out))
    elif args.cmd == "ingest":
        ingest_corpus(args._in, args.index, chunk=args.chunk, overlap=args.overlap)
    elif args.cmd == "ask":
        interactive_ask(args.index, k=args.k)
    elif args.cmd == "eval":
        evaluate_trivia(args.index, samples=args.samples, rpm=args.rpm)


if __name__ == "__main__":
    main()


