"""
Ingest PubMed abstracts via NCBI E-utilities (public API, public domain data).
FIX #2: Batch ingestion with single flush_bm25() at end.
FIX #15: Uses defusedxml for XML parsing.
"""

from __future__ import annotations

import argparse
import logging
import time

import defusedxml.ElementTree as ET
import httpx
from tqdm import tqdm

from config.settings import get_settings
from src.pipeline import RAGPipeline
from src.models import DocumentType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed(query: str, max_results: int = 100) -> list[str]:
    resp = httpx.get(ESEARCH_URL, params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"})
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    articles = []
    for i in range(0, len(pmids), 100):
        batch = pmids[i:i+100]
        resp = httpx.get(EFETCH_URL, params={"db": "pubmed", "id": ",".join(batch), "rettype": "xml", "retmode": "xml"}, timeout=30.0)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        for article in root.iter("PubmedArticle"):
            title_el = article.find(".//ArticleTitle")
            abstract_el = article.find(".//Abstract/AbstractText")
            pmid_el = article.find(".//PMID")
            title = title_el.text if title_el is not None and title_el.text else ""
            abstract = abstract_el.text if abstract_el is not None and abstract_el.text else ""
            pmid = pmid_el.text if pmid_el is not None and pmid_el.text else ""
            if title and abstract:
                mesh_terms = [m.text for m in article.iter("DescriptorName") if m.text]
                articles.append({"pmid": pmid, "title": title, "abstract": abstract, "mesh_terms": mesh_terms})
        time.sleep(0.34)
    return articles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="drug repurposing clinical trial")
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()

    pmids = search_pubmed(args.query, args.count)
    logger.info("Found %d PMIDs", len(pmids))
    articles = fetch_abstracts(pmids)
    logger.info("Fetched %d abstracts", len(articles))

    pipeline = RAGPipeline(get_settings())
    for art in tqdm(articles, desc="Ingesting"):
        content = f"# {art['title']}\n\n{art['abstract']}".encode("utf-8")
        pipeline.ingest_bytes(content=content, filename=f"pubmed_{art['pmid']}.md", document_type=DocumentType.LITERATURE, source="PubMed", title=art["title"])

    # FIX #2: Single BM25 rebuild after all docs ingested
    pipeline.flush_bm25()
    logger.info("Done: %d documents, %d chunks", len(articles), pipeline.dense_store.count())


if __name__ == "__main__":
    main()
