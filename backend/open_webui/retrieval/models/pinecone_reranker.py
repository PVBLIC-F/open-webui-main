import logging
from typing import Optional, List, Tuple
from pinecone import Pinecone

from open_webui.env import SRC_LOG_LEVELS
from open_webui.retrieval.models.base_reranker import BaseReranker

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class PineconeReranker(BaseReranker):
    """
    Pinecone Inference API Reranker
    Docs: https://docs.pinecone.io/guides/search/rerank-results
    """

    def __init__(
        self,
        api_key: str,
        model: str = "pinecone-rerank-v0",
        truncate: str = "END",  # How to handle long inputs
    ):
        self.api_key = api_key
        self.model = model
        self.truncate = truncate
        self.pc = Pinecone(api_key=self.api_key)

        log.info(f"Initialized Pinecone Reranker with model: {self.model}")

    def predict(
        self, sentences: List[Tuple[str, str]], user=None
    ) -> Optional[List[float]]:
        """
        Rerank documents using Pinecone's inference API.

        Args:
            sentences: List of (query, document) tuples
            user: Optional user context (not used by Pinecone)

        Returns:
            List of relevance scores (normalized 0-1, higher = more relevant)
        """
        # Extract query (same for all tuples) and documents
        query = sentences[0][0]
        docs = [{"text": sentence[1]} for sentence in sentences]

        try:
            log.info(f"PineconeReranker: Reranking {len(docs)} documents")
            log.debug(f"PineconeReranker: Query: {query[:100]}...")

            # Call Pinecone inference rerank API
            result = self.pc.inference.rerank(
                model=self.model,
                query=query,
                documents=docs,
                top_n=len(docs),  # Return all documents ranked
                rank_fields=["text"],
                return_documents=False,  # Don't need documents back, just scores
                parameters={"truncate": self.truncate}
                if self.model != "cohere-rerank-3.5"
                else {},
            )

            # Extract scores and maintain original order
            # Pinecone returns results sorted by score, but with original index
            scores = [0.0] * len(docs)
            for item in result.data:
                scores[item.index] = item.score

            log.info(f"PineconeReranker: Successfully reranked with scores: {scores}")
            return scores

        except Exception as e:
            log.exception(f"Error in Pinecone reranking: {e}")
            return None

