
# Assignment 3 – Advanced Transformer Mini‑Projects

## Folder map

| Folder | Mini‑project |
| ------ | ------------ |
| **RAG/** | 3.1 Retrieval‑Augmented Generation |
| **Multi Agent system/** | 3.2 LLM Multi‑Agent demo |
| **Fine‑Tune vs Parameter Efficient/** | 3.3 *Fine‑Tuning versus Parameter‑Efficient Tuning* (LoRA) |

---

## 1 · Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r RAG/requirements.txt
playwright install           # one‑time
export GOOGLE_API_KEY=sk‑…   # Gemini key
```

---

## 2 · RAG CLI (Gemini + FAISS)

```bash
cd RAG
# (opt) crawl any site
python ragcode.py crawl  --url https://example.com --max-pages 500 --out corpus.jsonl
# embed & index
python ragcode.py ingest --in corpus.jsonl --index faiss_index
# chat
python ragcode.py ask    --index faiss_index --k 10
# lightweight TriviaQA eval (streaming, throttled)
python ragcode.py eval   --index faiss_index --samples 20 --rpm 6
```

*Default KU corpus → 0 EM / 0.47 F1 on TriviaQA (domain mismatch).  
Swap in a broader crawl or a KU‑specific Q‑A set for meaningful scores.*

---

## 3 · Multi‑Agent LLM Demo

```bash
cd "Multi Agent system"
python multiagent.py
```

Agents: **Planner → Researcher → Summarizer** orchestrated by a Coordinator.

---

## 4 · Fine‑Tuning **vs** Parameter‑Efficient Tuning (LoRA)

Open the notebook:

```bash
jupyter lab "Fine-Tune vs Parameter Efficient/parameter-efficientn-fine-tuning-with-lora.ipynb"
```

*Compares full DistilBERT fine‑tune to LoRA adapters on a sentiment task:*

|   | Trainable params | Δ accuracy | GPU min |
|---|------------------|------------|---------|
| **Full fine‑tune** | 66 M | +4 % | ~7 min |
| **LoRA (rank 8)**  | **1.5 M** | +3 % | **<2 min** |

LoRA reaches near‑par accuracy while updating \< 3 % of parameters and cutting
GPU time by ~70 %.

---

## 5 · Troubleshooting

* **429 quota errors** during eval ⇒ lower `--rpm` (requests/min) or upgrade Gemini quota.  
* _index not found_ ⇒ run `ingest` before `ask` / `eval`.  
* For GPU FAISS install `faiss-gpu` and set `use_gpu=True` in `build_rag()`.

Enjoy!
