from typing import Optional, List, Dict, Any, Union
import logging
from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC

from open_webui.retrieval.vector.main import (
    VectorDBBase,
    VectorItem,
    SearchResult,
    GetResult,
)
from open_webui.config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    PINECONE_DIMENSION,
    PINECONE_METRIC,
    PINECONE_CLOUD,
)
from open_webui.env import SRC_LOG_LEVELS

NO_LIMIT = 10000  # Reasonable limit to avoid overwhelming the system
BATCH_SIZE = 256  # Recommended batch size for Pinecone operations

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class PineconeClient(VectorDBBase):
    def __init__(self):
        self.collection_prefix = "open-webui"

        # Validate required configuration
        self._validate_config()

        # Store configuration values
        self.api_key = PINECONE_API_KEY
        self.environment = PINECONE_ENVIRONMENT
        self.index_name = PINECONE_INDEX_NAME
        self.dimension = PINECONE_DIMENSION
        self.metric = PINECONE_METRIC
        self.cloud = PINECONE_CLOUD

        # Initialize gRPC client
        self.client = PineconeGRPC(api_key=self.api_key, environment=self.environment)

        # Ensure index exists and connect
        self._initialize_index()

    def _validate_config(self) -> None:
        missing_vars = []
        if not PINECONE_API_KEY:
            missing_vars.append("PINECONE_API_KEY")
        if not PINECONE_ENVIRONMENT:
            missing_vars.append("PINECONE_ENVIRONMENT")
        if not PINECONE_INDEX_NAME:
            missing_vars.append("PINECONE_INDEX_NAME")
        if not PINECONE_DIMENSION:
            missing_vars.append("PINECONE_DIMENSION")
        if not PINECONE_CLOUD:
            missing_vars.append("PINECONE_CLOUD")
        if missing_vars:
            raise ValueError(f"Required configuration missing: {', '.join(missing_vars)}")

    def _initialize_index(self) -> None:
        try:
            existing = self.client.list_indexes().names()
            if self.index_name not in existing:
                log.info(f"Creating Pinecone index '{self.index_name}'...")
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud=self.cloud, region=self.environment),
                )
                log.info(f"Successfully created Pinecone index '{self.index_name}'")
            else:
                log.info(f"Using existing Pinecone index '{self.index_name}'")

            # Connect to the index over gRPC
            self.index = self.client.Index(self.index_name)

        except Exception as e:
            log.error(f"Failed to initialize Pinecone index: {e}")
            raise RuntimeError(f"Failed to initialize Pinecone index: {e}")

    def _create_points(
        self, items: List[VectorItem], collection_name_with_prefix: str
    ) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        for item in items:
            metadata = item.get("metadata", {}).copy() if item.get("metadata") else {}
            if "text" in item:
                metadata["text"] = item["text"]
            metadata["collection_name"] = collection_name_with_prefix
            point = {"id": item["id"], "values": item["vector"], "metadata": metadata}
            points.append(point)
        return points

    def _get_collection_name_with_prefix(self, collection_name: str) -> str:
        return f"{self.collection_prefix}_{collection_name}"

    def _normalize_distance(self, score: float) -> float:
        if self.metric.lower() == "cosine":
            return (score + 1.0) / 2.0
        return score

    def _result_to_get_result(self, matches: list) -> GetResult:
        ids, docs, metas = [], [], []
        for m in matches:
            ids.append(m.id)
            docs.append(m.metadata.get("text", ""))
            metas.append(m.metadata)
        return GetResult(ids=[ids], documents=[docs], metadatas=[metas])

    def has_collection(self, collection_name: str) -> bool:
        prefix = self._get_collection_name_with_prefix(collection_name)
        try:
            resp = self.index.query(
                vector=[0.0] * self.dimension, top_k=1, filter={"collection_name": prefix}
            )
            return len(resp.matches) > 0
        except Exception:
            return False

    def delete_collection(self, collection_name: str) -> None:
        prefix = self._get_collection_name_with_prefix(collection_name)
        try:
            self.index.delete(filter={"collection_name": prefix})
            log.info(f"Deleted collection '{prefix}'")
        except Exception as e:
            log.warning(f"Failed to delete collection '{prefix}': {e}")
            raise

    def insert(self, collection_name: str, items: List[VectorItem]) -> None:
        if not items:
            log.warning("No items to insert")
            return
        prefix = self._get_collection_name_with_prefix(collection_name)
        points = self._create_points(items, prefix)
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i : i + BATCH_SIZE]
            try:
                self.index.upsert(vectors=batch)
                log.debug(f"Inserted batch of {len(batch)} into '{prefix}'")
            except Exception as e:
                log.error(f"Error inserting batch into '{prefix}': {e}")
                raise
        log.info(f"Inserted {len(items)} items into '{prefix}'")

    def upsert(self, collection_name: str, items: List[VectorItem]) -> None:
        if not items:
            log.warning("No items to upsert")
            return
        prefix = self._get_collection_name_with_prefix(collection_name)
        points = self._create_points(items, prefix)
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i : i + BATCH_SIZE]
            try:
                self.index.upsert(vectors=batch)
                log.debug(f"Upserted batch of {len(batch)} into '{prefix}'")
            except Exception as e:
                log.error(f"Error upserting batch into '{prefix}': {e}")
                raise
        log.info(f"Upserted {len(items)} items into '{prefix}'")

    def search(
        self,
        collection_name: str,
        vectors: List[List[Union[float, int]]],
        limit: int,
    ) -> Optional[SearchResult]:
        if not vectors:
            log.warning("No vectors to search")
            return None
        prefix = self._get_collection_name_with_prefix(collection_name)
        try:
            resp = self.index.query(
                vector=vectors[0],
                top_k=limit,
                filter={"collection_name": prefix},
                include_metadata=True,
            )
            matches = resp.matches

            ids = [m.id for m in matches]
            docs = [m.metadata.get("text", "") for m in matches]
            metas = [m.metadata for m in matches]
            dists = [self._normalize_distance(m.score) for m in matches]

            return SearchResult(
                ids=[ids],
                documents=[docs],
                metadatas=[metas],
                distances=[dists],
            )
        except Exception as e:
            log.error(f"Search error for '{prefix}': {e}")
            return None

    def fetch(self, ids: List[str]) -> Optional[GetResult]:
        if not ids:
            log.warning("No IDs to fetch")
            return None
        results, metas, vals = [], [], []
        for i in range(0, len(ids), BATCH_SIZE):
            batch_ids = ids[i : i + BATCH_SIZE]
            try:
                resp = self.index.fetch(ids=batch_ids)
                for vid, data in resp.vectors.items():
                    results.append(vid)
                    metas.append(data.metadata)
                    vals.append(data.values)
            except Exception as e:
                log.error(f"Fetch error for batch: {e}")
                raise
        return GetResult(ids=results, documents=["" for _ in results], metadatas=metas)

    # Implement abstract base methods and overloaded query
    def delete(self, ids: List[str]) -> None:
        try:
            self.index.delete(ids=ids)
            log.info(f"Deleted vectors: {ids}")
        except Exception as e:
            log.error(f"Error deleting vectors {ids}: {e}")
            raise

    def get(self, ids: List[str]) -> Optional[GetResult]:
        return self.fetch(ids)

    def query(
        self,
        collection_name: str,
        vector: Optional[List[Union[float, int]]] = None,
        limit: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """
        Query for vectors:
          - If no vector/limit: return all IDs in the collection.
          - Otherwise: return top-K similar vectors.
        """
        prefix = self._get_collection_name_with_prefix(collection_name)

        # Fetch all IDs in the collection when no vector/limit provided
        if vector is None or limit is None:
            resp = self.index.query(
                vector=[0.0] * self.dimension,
                top_k=NO_LIMIT,
                filter={"collection_name": prefix},
                include_metadata=True,
            )
            ids = [m.id for m in resp.matches]
            # Construct a full SearchResult with required fields
            return SearchResult(
                ids=[ids],
                documents=[["" for _ in ids]],
                metadatas=[[{} for _ in ids]],
                distances=[[0.0 for _ in ids]],
            )

        # Similarity search path
        return self.search(collection_name, [vector], limit)

    def reset(self) -> None:
        try:
            self.client.delete_index(self.index_name)
            self.client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.environment),
            )
            self.index = self.client.Index(self.index_name)
            log.info(f"Reset Pinecone index '{self.index_name}' successfully.")
        except Exception as e:
            log.error(f"Failed to reset index '{self.index_name}': {e}")
            raise
