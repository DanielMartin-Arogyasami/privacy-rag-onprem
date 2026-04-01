"""
Ingest DrugBank open data (Creative Commons).
FIX #2: Batch ingestion with single flush_bm25() at end.
FIX #15: Uses defusedxml for XML parsing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import defusedxml.ElementTree as ET
from tqdm import tqdm

from config.settings import get_settings
from src.pipeline import RAGPipeline
from src.models import DocumentType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NS = {"db": "http://www.drugbank.ca"}


def parse_drugbank_xml(path: Path) -> list[dict]:
    tree = ET.parse(str(path))
    root = tree.getroot()
    drugs = []
    for drug in root.findall("db:drug", NS):
        name_el = drug.find("db:name", NS)
        desc_el = drug.find("db:description", NS)
        dbid_el = drug.find("db:drugbank-id[@primary='true']", NS)
        name = name_el.text if name_el is not None and name_el.text else ""
        if not name:
            continue
        desc = desc_el.text if desc_el is not None and desc_el.text else ""
        db_id = dbid_el.text if dbid_el is not None and dbid_el.text else ""
        targets = [t.find("db:name", NS).text for t in drug.findall(".//db:targets/db:target", NS) if t.find("db:name", NS) is not None and t.find("db:name", NS).text]  # type: ignore
        interactions = []
        for ix in drug.findall(".//db:drug-interactions/db:drug-interaction", NS)[:20]:
            iname = ix.find("db:name", NS)
            idesc = ix.find("db:description", NS)
            if iname is not None and iname.text:
                interactions.append({"drug": iname.text, "description": (idesc.text or "")[:200] if idesc is not None else ""})
        drugs.append({"drugbank_id": db_id, "name": name, "description": desc, "targets": targets, "interactions": interactions})
    return drugs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/drugbank_open.xml")
    args = parser.parse_args()
    path = Path(args.file)
    if not path.exists():
        logger.error("File not found: %s — download from https://go.drugbank.com/releases/latest#open-data", path)
        return

    drugs = parse_drugbank_xml(path)
    logger.info("Parsed %d drugs", len(drugs))
    pipeline = RAGPipeline(get_settings())
    for drug in tqdm(drugs, desc="Ingesting"):
        parts = [f"# {drug['name']} ({drug['drugbank_id']})"]
        if drug["description"]:
            parts.append(f"\n## Description\n{drug['description']}")
        if drug["targets"]:
            parts.append(f"\n## Targets\n" + ", ".join(drug["targets"]))
        if drug["interactions"]:
            parts.append("\n## Drug Interactions")
            for ix in drug["interactions"][:10]:
                parts.append(f"- {ix['drug']}: {ix['description']}")
        pipeline.ingest_bytes(content="\n".join(parts).encode("utf-8"), filename=f"drugbank_{drug['drugbank_id']}.md", document_type=DocumentType.KNOWLEDGE_GRAPH, source="DrugBank", title=drug["name"])

    pipeline.flush_bm25()
    logger.info("Done: %d drugs", len(drugs))


if __name__ == "__main__":
    main()
