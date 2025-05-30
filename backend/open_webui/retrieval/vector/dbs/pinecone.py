from typing import Optional, List, Dict, Any, Union, Tuple
import logging
import time
import threading
import asyncio
import random
from dataclasses import dataclass, field
from pinecone import Pinecone, ServerlessSpec, PodSpec

import functools
import concurrent.futures
from contextlib import contextmanager
import weakref
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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

# Optimized constants based on Pinecone's official scale guide
NO_LIMIT = 10000
BATCH_SIZE = 100  # Optimal for most operations
UPSERT_BATCH_SIZE = 100  # Optimized for upsert operations  
DELETE_BATCH_SIZE = 1000  # Larger batches for delete operations
MAX_RETRIES = 3
RETRY_DELAY = 0.5  # Initial retry delay
CONNECTION_POOL_SIZE = 50  # High concurrency support
CONNECTION_POOL_MAXSIZE = 50  # HTTP connection pool optimization
MAX_TEXT_SIZE = 40000  # Pinecone metadata limit
MAX_PAYLOAD_SIZE = 2 * 1024 * 1024  # 2MB payload limit from Pinecone
QUERY_TIMEOUT = 30  # Timeout for queries
NAMESPACE_PREFIX = "open-webui"  # Use namespaces for better organization

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

# gRPC-first approach with HTTP fallback (following Pinecone's official recommendation)
try:
    from pinecone.grpc import PineconeGRPC
    GRPC_AVAILABLE = True
    log.info("gRPC client available - using for optimal performance")
except ImportError:
    GRPC_AVAILABLE = False
    log.info("gRPC client not available - falling back to HTTP")

@dataclass
class PineconeConfig:
    """Enhanced configuration class following Pinecone's scale guide."""
    api_key: str
    index_name: str
    dimension: int
    metric: str
    cloud: str
    environment: str
    use_grpc: bool = GRPC_AVAILABLE  # Auto-detect gRPC availability
    use_namespaces: bool = False  # Use default namespace for all uploads (original behavior)
    source_tag: str = "open-webui-scale-optimized"  # Source tag for analytics
    deletion_protection: str = "disabled"  # Latest API requirement
    
    def __post_init__(self):
        """Enhanced validation with v7.x requirements."""
        if not all([self.api_key, self.index_name, self.dimension, self.cloud, self.environment]):
            raise ValueError("All Pinecone configuration fields are required")
        
        if self.dimension <= 0 or self.dimension > 20000:
            raise ValueError("Dimension must be between 1 and 20000")
        
        if self.metric not in ["cosine", "euclidean", "dotproduct"]:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        if self.cloud not in ["aws", "gcp", "azure"]:
            raise ValueError(f"Unsupported cloud: {self.cloud}")


class PineconeClient(VectorDBBase):
    """
    Production-optimized Pinecone client following official scale guide.
    
    Features:
    - gRPC-first with HTTP fallback (following Pinecone's recommendation)
    - Advanced batching with payload size validation
    - Exponential backoff with jitter retry strategy
    - Default namespace for cross-file search (original Open WebUI behavior)
    - Circuit breaker pattern for resilience
    - High-concurrency ThreadPoolExecutor
    - Comprehensive monitoring and observability
    """
    
    _instances = weakref.WeakValueDictionary()  # Singleton with connection reuse
    _lock = threading.Lock()
    
    def __new__(cls):
        """Enhanced singleton pattern with configuration-based keying."""
        config_key = f"{PINECONE_API_KEY}_{PINECONE_INDEX_NAME}_{PINECONE_CLOUD}"
        
        with cls._lock:
            if config_key in cls._instances:
                return cls._instances[config_key]
            
            instance = super().__new__(cls)
            cls._instances[config_key] = instance
            return instance

    def __init__(self):
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized'):
            return
        
        # Enhanced configuration following scale guide
        self.config = PineconeConfig(
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            cloud=PINECONE_CLOUD,
            environment=PINECONE_ENVIRONMENT,
        )
        
        # gRPC-first approach with HTTP fallback (official Pinecone recommendation)
        if self.config.use_grpc and GRPC_AVAILABLE:
            log.info("Initializing with gRPC client for optimal performance")
            self.client = PineconeGRPC(
                api_key=self.config.api_key,
                source_tag=self.config.source_tag,
                pool_threads=CONNECTION_POOL_SIZE,
                timeout=QUERY_TIMEOUT,
            )
            self.using_grpc = True
        else:
            log.info("Initializing with HTTP client (gRPC fallback)")
            self.client = Pinecone(
                api_key=self.config.api_key,
                source_tag=self.config.source_tag,
                pool_threads=CONNECTION_POOL_SIZE,
                timeout=QUERY_TIMEOUT,
            )
            self.using_grpc = False
        
        # High-performance executor optimized for I/O-bound operations (from scale guide)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=CONNECTION_POOL_SIZE,
            thread_name_prefix="pinecone-scale-worker"
        )
        
        # Enhanced connection cache with namespace support
        self._index_cache = {}
        self._namespace_cache = set()
        self._cache_lock = threading.Lock()
        
        # Circuit breaker for resilience
        self._circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'closed'  # closed, open, half-open
        }
        
        # Initialize index connection
        self._initialize_index()
        self._initialized = True

    def _initialize_index(self) -> None:
        """Enhanced index initialization following scale guide."""
        try:
            # Check if index exists using modern API
            if not self.client.has_index(self.config.index_name):
                log.info(f"Creating Pinecone index '{self.config.index_name}' with scale optimizations...")
                
                # Use ServerlessSpec with latest optimizations
                spec = ServerlessSpec(
                    cloud=self.config.cloud,
                    region=self.config.environment
                )
                
                self.client.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric,
                    spec=spec,
                    deletion_protection=self.config.deletion_protection,
                    tags={
                        "version": "v7", 
                        "source": "open-webui", 
                        "optimized": "scale-guide",
                        "grpc": str(self.using_grpc)
                    }
                )
                
                # Wait for index to be ready with enhanced monitoring
                self._wait_for_index_ready()
                log.info(f"Successfully created scale-optimized Pinecone index '{self.config.index_name}'")
            else:
                log.info(f"Using existing Pinecone index '{self.config.index_name}' with {'gRPC' if self.using_grpc else 'HTTP'}")

            # Get optimized index connection
            self.index = self._get_index_connection()

        except Exception as e:
            log.error(f"Failed to initialize Pinecone index: {e}")
            raise RuntimeError(f"Failed to initialize Pinecone index: {e}")

    def _wait_for_index_ready(self, timeout: int = 300) -> None:
        """Enhanced index readiness check with better monitoring."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                desc = self.client.describe_index(self.config.index_name)
                if desc.status.ready:
                    log.info(f"Index {self.config.index_name} is ready")
                    return
                log.debug(f"Index {self.config.index_name} not ready, waiting...")
                time.sleep(2)
            except Exception as e:
                log.debug(f"Error checking index status: {e}")
                time.sleep(2)
        raise TimeoutError(f"Index {self.config.index_name} not ready after {timeout}s")

    def _get_index_connection(self):
        """Enhanced index connection optimized for gRPC/HTTP."""
        with self._cache_lock:
            if self.config.index_name not in self._index_cache:
                if self.using_grpc:
                    # gRPC connection for optimal performance (no connection_pool_maxsize needed)
                    self._index_cache[self.config.index_name] = self.client.Index(
                        name=self.config.index_name,
                        pool_threads=CONNECTION_POOL_SIZE,
                    )
                else:
                    # HTTP connection with enhanced pooling
                    self._index_cache[self.config.index_name] = self.client.Index(
                        name=self.config.index_name,
                        pool_threads=CONNECTION_POOL_SIZE,
                        connection_pool_maxsize=CONNECTION_POOL_MAXSIZE,
                    )
            return self._index_cache[self.config.index_name]

    def _get_namespace(self, collection_name: str) -> str:
        """Get namespace for collection using v7.x best practices."""
        if self.config.use_namespaces:
            return f"{NAMESPACE_PREFIX}-{collection_name}"
        return ""  # Default namespace

    def _estimate_batch_size(self, batch: List[Dict[str, Any]]) -> int:
        """Estimate batch size in bytes (from Pinecone scale guide)."""
        import json
        try:
            return len(json.dumps(batch).encode('utf-8'))
        except Exception:
            # Fallback estimation
            return sum(
                len(str(item.get('id', ''))) + 
                len(item.get('values', [])) * 4 +  # Assume 4 bytes per float
                len(str(item.get('metadata', {})))
                for item in batch
            )

    def _circuit_breaker_check(self) -> bool:
        """Circuit breaker pattern for resilience."""
        current_time = time.time()
        
        if self._circuit_breaker['state'] == 'open':
            # Check if we should try again (half-open)
            if current_time - self._circuit_breaker['last_failure'] > 60:  # 1 minute
                self._circuit_breaker['state'] = 'half-open'
                return True
            return False
        
        return True

    def _circuit_breaker_success(self):
        """Record successful operation."""
        self._circuit_breaker['failures'] = 0
        self._circuit_breaker['state'] = 'closed'

    def _circuit_breaker_failure(self):
        """Record failed operation."""
        self._circuit_breaker['failures'] += 1
        self._circuit_breaker['last_failure'] = time.time()
        
        if self._circuit_breaker['failures'] >= 5:
            self._circuit_breaker['state'] = 'open'
            log.warning("Circuit breaker opened due to repeated failures")

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_DELAY, min=RETRY_DELAY, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
        reraise=True
    )
    def _retry_with_exponential_backoff_and_jitter(self, operation_func, *args, **kwargs):
        """Enhanced retry with exponential backoff and jitter (from scale guide)."""
        if not self._circuit_breaker_check():
            raise Exception("Circuit breaker open")
        
        try:
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0, 0.1)
            if jitter > 0.05:  # Only add jitter 50% of the time
                time.sleep(jitter)
            
            result = operation_func(*args, **kwargs)
            self._circuit_breaker_success()
            return result
        except Exception as e:
            self._circuit_breaker_failure()
            # Enhanced error classification
            error_str = str(e).lower()
            is_retryable = any(keyword in error_str for keyword in [
                'timeout', 'rate limit', 'network', 'connection', 'temporary', 
                'unavailable', '429', '500', '502', '503', '504'
            ])
            
            if not is_retryable:
                log.error(f"Non-retryable error: {e}")
                raise
            
            log.warning(f"Retryable error encountered: {e}")
            raise

    def _create_points_scale_optimized(
        self, items: List[VectorItem], namespace: str
    ) -> List[Dict[str, Any]]:
        """Scale-optimized point creation with payload validation."""
        points = []
        skipped = 0
        
        for item in items:
            # Fast validation with enhanced checks
            if not item.get("id") or not item.get("vector"):
                skipped += 1
                continue
                
            # Validate vector dimension
            if len(item["vector"]) != self.config.dimension:
                log.warning(f"Vector dimension mismatch for item {item['id']}: expected {self.config.dimension}, got {len(item['vector'])}")
                skipped += 1
                continue
                
            # Efficient metadata handling
            metadata = {}
            if item.get("metadata"):
                metadata.update(item["metadata"])

            # Optimized text handling with better truncation
            if item.get("text"):
                text = str(item["text"])
                if len(text) > MAX_TEXT_SIZE:
                    # Smart truncation at word boundaries
                    text = text[:MAX_TEXT_SIZE].rsplit(' ', 1)[0]
                    log.debug(f"Truncated text for item {item['id']} to {len(text)} chars")
                metadata["text"] = text

            # Add enhanced metadata
            metadata.update({
                "namespace": namespace,
                "created_at": time.time(),
                "source": "open-webui-scale",
                "grpc": self.using_grpc
            })

            points.append({
                "id": str(item["id"]),
                "values": item["vector"],
                "metadata": metadata,
            })
        
        if skipped > 0:
            log.warning(f"Skipped {skipped} invalid items during point creation")
        
        return points

    def _split_batches_by_size(self, points: List[Dict[str, Any]], max_batch_size: int = UPSERT_BATCH_SIZE) -> List[List[Dict[str, Any]]]:
        """Split points into batches respecting payload size limits (from scale guide)."""
        batches = []
        current_batch = []
        current_size = 0
        
        for point in points:
            # Estimate point size
            point_size = self._estimate_batch_size([point])
            
            # Check if adding this point would exceed limits
            if (len(current_batch) >= max_batch_size or 
                current_size + point_size > MAX_PAYLOAD_SIZE):
                
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_size = 0
            
            current_batch.append(point)
            current_size += point_size
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def _normalize_distance_v7(self, score: float) -> float:
        """Enhanced distance normalization for v7.x."""
        if self.config.metric == "cosine":
            return max(0.0, min(1.0, (score + 1.0) * 0.5))  # Clamp to [0,1]
        elif self.config.metric == "euclidean":
            return max(0.0, score)  # Ensure non-negative
        return score

    def _result_to_get_result_v7(self, matches: list) -> GetResult:
        """Enhanced result conversion optimized for v7.x."""
        if not matches:
            return GetResult(ids=[[]], documents=[[]], metadatas=[[]])
        
        # Pre-allocate with capacity for better performance
        ids = []
        documents = []
        metadatas = []
        
        for match in matches:
            metadata = getattr(match, "metadata", {}) or {}
            ids.append(str(getattr(match, "id", match.get("id", ""))))
            documents.append(metadata.get("text", ""))
            
            # Clean metadata for return (remove internal fields)
            clean_metadata = {k: v for k, v in metadata.items() 
                            if not k.startswith(('namespace', 'created_at', 'source', 'grpc'))}
            metadatas.append(clean_metadata)

        return GetResult(ids=[ids], documents=[documents], metadatas=[metadatas])

    def has_collection(self, collection_name: str) -> bool:
        """Enhanced collection existence check using namespaces."""
        namespace = self._get_namespace(collection_name)

        try:
            def _check_collection():
                # Ultra-fast existence check using namespace stats
                stats = self.index.describe_index_stats(filter={})
                if self.config.use_namespaces:
                    return namespace in stats.namespaces
                else:
                    # Fallback to query-based check
                    response = self.index.query(
                        vector=[0.0] * self.config.dimension,
                        top_k=1,
                        namespace=namespace,
                        include_metadata=False,
                        include_values=False,
                    )
                    return len(getattr(response, "matches", [])) > 0
            
            return self._retry_with_exponential_backoff_and_jitter(_check_collection)
        except Exception as e:
            log.debug(f"Collection check failed for '{collection_name}': {e}")
            return False

    def delete_collection(self, collection_name: str) -> None:
        """Enhanced collection deletion using namespaces."""
        namespace = self._get_namespace(collection_name)
        
        def _delete_collection():
            if self.config.use_namespaces:
                # Delete entire namespace (v7.x best practice)
                self.index.delete(delete_all=True, namespace=namespace)
            else:
                # Fallback to metadata-based deletion
                self.index.delete(filter={"namespace": namespace})
            
            # Update namespace cache
            with self._cache_lock:
                self._namespace_cache.discard(namespace)
        
        try:
            self._retry_with_exponential_backoff_and_jitter(_delete_collection)
            log.info(f"Collection '{collection_name}' deleted from namespace '{namespace}'")
        except Exception as e:
            log.error(f"Failed to delete collection '{collection_name}': {e}")
            raise

    def insert(self, collection_name: str, items: List[VectorItem]) -> None:
        """Scale-optimized insert following Pinecone's best practices."""
        if not items:
            return

        start_time = time.time()
        namespace = self._get_namespace(collection_name)
        points = self._create_points_scale_optimized(items, namespace)

        if not points:
            log.warning("No valid points to insert after filtering")
            return

        try:
            # Split into optimized batches respecting payload limits
            batches = self._split_batches_by_size(points, UPSERT_BATCH_SIZE)
            
            log.info(f"Processing {len(points)} points in {len(batches)} batches")
            
            # Enhanced parallel batch processing with size validation
            batch_futures = []
            for i, batch in enumerate(batches):
                batch_size = self._estimate_batch_size(batch)
                if batch_size > MAX_PAYLOAD_SIZE:
                    log.warning(f"Skipping batch {i+1}: size {batch_size} exceeds 2MB limit")
                    continue
                
                future = self._executor.submit(self._upsert_batch_with_retry, batch, namespace, i+1, len(batches))
                batch_futures.append(future)
            
            # Wait for all batches with enhanced timeout handling
            completed = 0
            for future in concurrent.futures.as_completed(batch_futures, timeout=QUERY_TIMEOUT * 3):
                future.result()  # Raise any exceptions
                completed += 1
                
            # Update namespace cache
            with self._cache_lock:
                self._namespace_cache.add(namespace)
                    
            elapsed = time.time() - start_time
            log.info(f"Successfully inserted {len(points)} vectors in {elapsed:.2f}s into namespace '{namespace}' using {'gRPC' if self.using_grpc else 'HTTP'}")
            
        except Exception as e:
            log.error(f"Error inserting vectors: {e}")
            raise

    def _upsert_batch_with_retry(self, batch: List[Dict[str, Any]], namespace: str, batch_num: int, total_batches: int) -> None:
        """Enhanced batch upsert with retry and monitoring."""
        def _upsert():
            self.index.upsert(vectors=batch, namespace=namespace)
            log.debug(f"Batch {batch_num}/{total_batches} completed successfully")
        
        try:
            self._retry_with_exponential_backoff_and_jitter(_upsert)
        except Exception as e:
            log.error(f"Batch {batch_num}/{total_batches} failed for namespace {namespace}: {e}")
            raise

    def upsert(self, collection_name: str, items: List[VectorItem]) -> None:
        """Enhanced upsert operation."""
        self.insert(collection_name, items)

    def search(
        self, collection_name: str, vectors: List[List[Union[float, int]]], limit: int
    ) -> Optional[SearchResult]:
        """Scale-optimized search with enhanced performance."""
        if not vectors or not vectors[0]:
            return None

        namespace = self._get_namespace(collection_name)
        effective_limit = min(limit or NO_LIMIT, NO_LIMIT)

        def _search():
            # Enhanced query with namespace support
            response = self.index.query(
                vector=vectors[0],
                top_k=effective_limit,
                namespace=namespace,
                include_metadata=True,
                include_values=False,  # Memory optimization
            )

            matches = getattr(response, "matches", [])
            if not matches:
                return SearchResult(ids=[[]], documents=[[]], metadatas=[[]], distances=[[]])

            # Enhanced result processing
            get_result = self._result_to_get_result_v7(matches)
            distances = [[self._normalize_distance_v7(getattr(m, "score", 0.0)) for m in matches]]

            return SearchResult(
                ids=get_result.ids,
                documents=get_result.documents,
                metadatas=get_result.metadatas,
                distances=distances,
            )

        try:
            return self._retry_with_exponential_backoff_and_jitter(_search)
        except Exception as e:
            log.error(f"Search error in namespace '{namespace}': {e}")
            return None

    def query(
        self, collection_name: str, filter: Dict, limit: Optional[int] = None
    ) -> Optional[GetResult]:
        """Enhanced metadata query with namespace support."""
        namespace = self._get_namespace(collection_name)
        effective_limit = min(limit or NO_LIMIT, NO_LIMIT)

        def _query():
            response = self.index.query(
                vector=[0.0] * self.config.dimension,
                filter=filter,
                top_k=effective_limit,
                namespace=namespace,
                include_metadata=True,
                include_values=False,
            )

            matches = getattr(response, "matches", [])
            return self._result_to_get_result_v7(matches)

        try:
            return self._retry_with_exponential_backoff_and_jitter(_query)
        except Exception as e:
            log.error(f"Query error in namespace '{namespace}': {e}")
            return None

    def get(self, collection_name: str) -> Optional[GetResult]:
        """Enhanced get all vectors with namespace support."""
        namespace = self._get_namespace(collection_name)

        def _get():
            response = self.index.query(
                vector=[0.0] * self.config.dimension,
                top_k=NO_LIMIT,
                namespace=namespace,
                include_metadata=True,
                include_values=False,
            )

            matches = getattr(response, "matches", [])
            return self._result_to_get_result_v7(matches)

        try:
            return self._retry_with_exponential_backoff_and_jitter(_get)
        except Exception as e:
            log.error(f"Get error for namespace '{namespace}': {e}")
            return None

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict] = None,
    ) -> None:
        """Scale-optimized delete with enhanced batching."""
        namespace = self._get_namespace(collection_name)

        def _delete():
            if ids:
                # Enhanced batch deletion with namespace support
                for i in range(0, len(ids), DELETE_BATCH_SIZE):
                    batch_ids = ids[i:i + DELETE_BATCH_SIZE]
                    self.index.delete(ids=batch_ids, namespace=namespace)
                log.info(f"Deleted {len(ids)} vectors by ID from namespace '{namespace}'")

            elif filter:
                # Enhanced filter-based deletion
                self.index.delete(filter=filter, namespace=namespace)
                log.info(f"Deleted vectors by filter from namespace '{namespace}'")

        try:
            self._retry_with_exponential_backoff_and_jitter(_delete)
        except Exception as e:
            log.error(f"Delete error in namespace '{namespace}': {e}")
            raise

    def reset(self) -> None:
        """Enhanced database reset with namespace awareness."""
        def _reset():
            if self.config.use_namespaces:
                # Delete all namespaces
                stats = self.index.describe_index_stats()
                for namespace in stats.namespaces:
                    if namespace.startswith(NAMESPACE_PREFIX):
                        self.index.delete(delete_all=True, namespace=namespace)
            else:
                # Delete all vectors
                self.index.delete(delete_all=True)
            
            # Clear caches
            with self._cache_lock:
                self._namespace_cache.clear()

        try:
            self._retry_with_exponential_backoff_and_jitter(_reset)
            log.info("All vectors deleted from index")
        except Exception as e:
            log.error(f"Reset failed: {e}")
            raise

    def close(self):
        """Enhanced resource cleanup with scale optimizations."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=True, timeout=10)
            
            # Clear all caches
            with self._cache_lock:
                self._index_cache.clear()
                self._namespace_cache.clear()
                
        except Exception as e:
            log.warning(f"Cleanup error: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
