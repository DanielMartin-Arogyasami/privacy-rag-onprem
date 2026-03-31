Topics: `privacy-preserving-ml`, `vllm`, `milvus`, `on-premises`, `healthcare-ai`


# Privacy-Preserving On-Prem RAG (privacy-rag-onprem)

> **⚠️ INDEPENDENCE & DATA DISCLAIMER:** This reference implementation uses exclusively publicly available literature (PubMed) and synthetic PHI for testing. It does not reflect, derive from, or use any proprietary systems, workflows, data, or trade secrets of any current or past employers. This work is an independent academic and open-source contribution.

## Overview
A fully local, air-gapped Retrieval-Augmented Generation (RAG) system designed for zero-data-leakage environments. Features quantized LLM inference (`vLLM`), local vector storage (`Milvus`), and an active Protected Health Information (PHI) detection firewall (`Presidio`). Companion to: *"Privacy-Preserving Retrieval-Augmented Generation for Clinical Data Systems"*.

## Architecture
* **Inference Engine:** `vLLM` running quantized models (e.g., Llama-3 8B AWQ) entirely on localhost.
* **Vector DB:** `Milvus` Lite for local, access-controlled semantic search.
* **Firewall:** Microsoft `Presidio` integration to detect and redact synthetic PII/PHI in prompts before they hit the LLM, and filtering the generation output.

## Public Dataset Acquisition
1. **PubMed Abstracts:** Download open-access biomedical literature via the E-utilities API:
   ```bash
   python src/data/fetch_pubmed.py --query "drug repurposing" --max-results 1000
Synthetic PHI: We use purely synthetic, fake patient profiles generated programmatically for testing the Presidio firewall.

Quick Start
We use Docker Compose to simulate an air-gapped on-premises deployment.

Bash

# Spin up Milvus, FastAPI, and the vLLM server
docker-compose up -d

# Index the downloaded PubMed abstracts into Milvus
python src/retrieval/milvus_client.py --index data/pubmed_abstracts/

# Test the pipeline with a prompt containing synthetic PHI
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Based on the literature, what are the off-label uses for drug X? Patient John Doe (DOB: 01/01/1980) is asking."}'
# Expected output: PHI redacted prior to RAG processing.
Citation
Code snippet

@article{arogyasami2026privacyrag,
  title={Privacy-Preserving Retrieval-Augmented Generation for Clinical Data Systems},
  author={Arogyasami, DanielMartin},
  year={2026},
  journal={Preprint}
}
