"""Download embedding and reranker models to local cache."""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    from sentence_transformers import SentenceTransformer, CrossEncoder

    logger.info("Downloading embedding model: BAAI/bge-large-en-v1.5")
    SentenceTransformer("BAAI/bge-large-en-v1.5")
    logger.info("Embedding model cached.")

    logger.info("Downloading reranker: BAAI/bge-reranker-v2-m3")
    CrossEncoder("BAAI/bge-reranker-v2-m3")
    logger.info("Reranker model cached.")


if __name__ == "__main__":
    main()
