# Privacy-Preserving RAG for Clinical Data Systems

**Companion repository for**: "Privacy-Preserving Retrieval-Augmented Generation for Clinical Data Systems: Architecture, Deployment, and Regulatory Compliance for On-Premises Healthcare AI" — DanielMartin Arogyasami

> **IMPORTANT**: This implementation uses exclusively publicly available data (PubMed abstracts, DrugBank open access, BioASQ benchmark). It does not reflect, derive from, or use any proprietary systems, data, or trade secrets of any employer.

## What This Demonstrates

A fully on-premises RAG system for clinical/pharmaceutical environments implementing all six layers:

1. **Document Ingestion** — Parse PDFs, DOCX, PubMed XML; de-identify PHI at ingest
2. **Structure-Aware Chunking** — Preserves clinical document hierarchy
3. **Encrypted Vector Storage** — Local Qdrant with BGE-large-en-v1.5 embeddings
4. **Hybrid Retrieval** — BM25 + dense ANN + cross-encoder reranking with ACL enforcement
5. **Quantized Local LLM Inference** — vLLM serving AWQ 4-bit models; structured prompting
6. **Response Filtering** — PHI scanning, hallucination checking, audit logging

## Quick Start

```bash
# 1. Configure
cp .env.example .env

# 2. Install
pip install -e ".[dev]"
python -m spacy download en_core_web_lg

# 3. Start vector DB
docker compose up -d qdrant

# 4. Download models
python scripts/download_models.py

# 5. Ingest public data
python scripts/ingest_pubmed.py --count 500
python scripts/ingest_drugbank.py

# 6. Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 7. Query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(grep API_SECRET_KEY .env | cut -d= -f2)" \
  -d '{"query": "What are the known drug interactions for metformin?", "user_id": "demo"}'

# 8. Run tests
pytest tests/ -v --tb=short
```

## Public Data Sources

|Source                  |Type                     |Access          |
|------------------------|-------------------------|----------------|
|PubMed (E-utilities API)|36M+ biomedical abstracts|Public domain   |
|DrugBank Open Data      |Drug-target interactions |Creative Commons|
|BioASQ                  |QA benchmark             |Public          |

## Key Design Decisions

- **Prompting-first**: No fine-tuning on clinical data to avoid PHI memorization risk
- **Hybrid retrieval**: BM25 + dense ANN + cross-encoder reranking
- **Defense in depth**: PHI protection at ingestion, retrieval, generation, and output layers
- **Air-gapped capable**: All components run on-premises; zero external API calls at runtime
- **Audit trail**: Every query-response pair logged with retrieved passages, model version, timestamps

## License

MIT — see LICENSE file.
