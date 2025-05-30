import logging
import os
from typing import Optional, Union

import requests
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time

from huggingface_hub import snapshot_download
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from open_webui.config import VECTOR_DB
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT

from open_webui.models.users import UserModel
from open_webui.models.files import Files

from open_webui.retrieval.vector.main import GetResult


from open_webui.env import (
    SRC_LOG_LEVELS,
    OFFLINE_MODE,
    ENABLE_FORWARD_USER_INFO_HEADERS,
)
from open_webui.config import (
    RAG_EMBEDDING_QUERY_PREFIX,
    RAG_EMBEDDING_CONTENT_PREFIX,
    RAG_EMBEDDING_PREFIX_FIELD_NAME,
)

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever


class VectorSearchRetriever(BaseRetriever):
    collection_name: Any
    embedding_function: Any
    top_k: int

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        result = VECTOR_DB_CLIENT.search(
            collection_name=self.collection_name,
            vectors=[self.embedding_function(query, RAG_EMBEDDING_QUERY_PREFIX)],
            limit=self.top_k,
        )

        ids = result.ids[0]
        metadatas = result.metadatas[0]
        documents = result.documents[0]

        results = []
        for idx in range(len(ids)):
            results.append(
                Document(
                    metadata=metadatas[idx],
                    page_content=documents[idx],
                )
            )
        return results


def query_doc(
    collection_name: str,
    query_embedding: list[float],
    k: int,
    user: UserModel = None,
    expected_count: int = None,
):
    """
    Query vector database with intelligent indexing wait for web search collections.

    Fixes race condition where queries happened before indexing completed.
    Web search collections get longer timeout and stabilization logic.
    """
    import time

    try:
        log.debug(f"query_doc:doc {collection_name}")

        # Auto-detect web search collections for intelligent waiting
        is_web_search = collection_name.startswith("web-search-")

        if is_web_search or (expected_count is not None and expected_count > 0):
            # Web search gets 15s timeout vs 10s for others
            timeout = 15.0 if is_web_search else 10.0

            if wait_for_indexing(collection_name, expected_count or 1, timeout):
                # Indexing complete, perform search with relevance scores
                result = VECTOR_DB_CLIENT.search(
                    collection_name=collection_name,
                    vectors=[query_embedding],
                    limit=k,
                )
                if result:
                    doc_count = len(result.documents[0]) if result.documents[0] else 0
                    log.info(f"query_doc: Retrieved {doc_count} documents")
                    log.info(f"query_doc:result {result.ids} {result.metadatas}")
                return result
            else:
                # Indexing timeout - return empty result
                log.warning(
                    f"query_doc: Indexing timeout for collection "
                    f"'{collection_name}'"
                )
                return None

        # Legacy retry mechanism for non-web-search collections
        max_retries = 4
        retry_delay = 2.0  # seconds

        for attempt in range(max_retries):
            log.info(
                f"query_doc: Attempt {attempt + 1}/{max_retries} "
                f"for collection '{collection_name}'"
            )

            result = VECTOR_DB_CLIENT.search(
                collection_name=collection_name,
                vectors=[query_embedding],
                limit=k,
            )

            # Check if we got meaningful results
            if result and result.documents and result.documents[0]:
                log.info(
                    f"query_doc: SUCCESS on attempt {attempt + 1} - "
                    f"Found {len(result.documents[0])} documents"
                )
                if result:
                    log.info(f"query_doc:result {result.ids} {result.metadatas}")
                return result

            # If this is not the last attempt and we got empty results, wait and retry
            if attempt < max_retries - 1:
                log.info(
                    f"query_doc: Empty results on attempt {attempt + 1}, "
                    f"retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                log.warning(
                    f"query_doc: All {max_retries} attempts failed "
                    f"for collection '{collection_name}'"
                )

        # Log final result if all retries completed
        if result:
            log.info(f"query_doc:result {result.ids} {result.metadatas}")

        return result
    except Exception as e:
        log.exception(f"Error querying doc {collection_name} with limit {k}: {e}")
        raise e


import threading
import time
from typing import Dict, Tuple

# Global coordination system for stabilization to prevent parallel redundant checks
# This prevents multiple concurrent requests from doing duplicate stabilization work
_stabilization_events: Dict[str, threading.Event] = {}  # collection -> event
_stabilization_results: Dict[str, Tuple[bool, float, int]] = (
    {}
)  # collection -> (success, timestamp, doc_count)
_coordination_lock = threading.Lock()
_RESULT_TTL = 10.0  # Cache successful results for 10 seconds


def wait_for_indexing(
    collection_name: str, expected_count: int, timeout: float = 10.0
) -> bool:
    """
    Optimized wait for vector database indexing with thread coordination.

    Uses threading events to coordinate multiple parallel requests for the same
    collection, ensuring only one thread does the actual stabilization work while
    others wait for the result. Features smart timing optimizations for web search.

    Args:
        collection_name: Collection to check (auto-detects web search by prefix)
        expected_count: Expected document count (1 = "any documents")
        timeout: Max wait time (15s for web search, 10s for others)

    Returns:
        True if indexing complete, False if timeout
    """
    import time

    # Skip wait if no documents expected
    if expected_count == 0:
        log.info(
            f"No documents expected for collection '{collection_name}', "
            "skipping wait"
        )
        return True

    # Quick cache check - return immediately if we have a recent successful result
    current_time = time.time()

    # Determine coordination strategy - wait for other thread or do the work
    should_wait_for_other_thread = False
    event_to_wait_for = None

    with _coordination_lock:
        # Check if we have a cached successful result
        if collection_name in _stabilization_results:
            success, result_time, doc_count = _stabilization_results[collection_name]
            if current_time - result_time < _RESULT_TTL and success and doc_count > 0:
                log.debug(
                    f"Using cached result for '{collection_name}': "
                    f"{doc_count} documents "
                    f"(cached {current_time - result_time:.1f}s ago)"
                )
                return True

        # Check if another thread is already doing stabilization work
        if collection_name in _stabilization_events:
            # Another thread is working - we'll wait for its result
            event_to_wait_for = _stabilization_events[collection_name]
            should_wait_for_other_thread = True
            log.debug(
                f"Waiting for another thread to complete stabilization for "
                f"'{collection_name}'"
            )
        else:
            # We're the first thread - create event and proceed with work
            event = threading.Event()
            _stabilization_events[collection_name] = event

    # Wait for another thread's result (happens outside the lock to avoid blocking)
    if should_wait_for_other_thread and event_to_wait_for:
        # Wait for the other thread to complete (with timeout)
        if event_to_wait_for.wait(timeout=min(timeout, 15.0)):
            # Other thread completed, check if we can use its result
            with _coordination_lock:
                if collection_name in _stabilization_results:
                    success, result_time, doc_count = _stabilization_results[
                        collection_name
                    ]
                    if current_time - result_time < _RESULT_TTL and success:
                        log.debug(
                            f"Got result from other thread: {doc_count} documents"
                        )
                        return True

        # If we get here, the other thread failed or timed out
        log.debug(
            f"Other thread failed/timed out, doing our own check for "
            f"'{collection_name}'"
        )

        # Try to become the worker thread now
        with _coordination_lock:
            if collection_name not in _stabilization_events:
                event = threading.Event()
                _stabilization_events[collection_name] = event
            else:
                # Another thread beat us to it, just return False
                return False

    try:
        # We're the thread doing the actual stabilization work
        start_time = time.time()

        # Optimized timing: fast initial checks, adaptive intervals
        check_interval = 0.3  # Start with 300ms intervals (faster than 500ms)
        max_interval = 1.0  # Max 1 second intervals (reduced from 2s)
        attempt = 1

        # Web search gets special stabilization logic (wait for count to stabilize)
        # Other collections just wait for expected count
        is_web_search = collection_name.startswith("web-search-")
        use_stabilization = is_web_search and expected_count == 1

        if use_stabilization:
            log.info(
                f"Waiting for document indexing to stabilize in '{collection_name}'"
            )
            last_count = 0
            stable_count = 0
            stable_threshold = 2  # Need 2 consecutive checks with same count
            documents_found = False  # Track when we first find documents
        else:
            log.info(
                f"Waiting for {expected_count} documents to be indexed in "
                f"'{collection_name}'"
            )

        # Main stabilization loop with adaptive timing
        while time.time() - start_time < timeout:
            try:
                result = VECTOR_DB_CLIENT.get(collection_name=collection_name)

                if result and result.documents and result.documents[0]:
                    current_count = len(result.documents[0])

                    if use_stabilization:
                        log.info(
                            f"Stabilization check {attempt}: {current_count} "
                            f"documents available"
                        )

                        # Optimization: switch to faster checking once docs appear
                        if current_count > 0 and not documents_found:
                            documents_found = True
                            check_interval = (
                                0.2  # Switch to 200ms intervals once docs appear
                            )
                            max_interval = 0.8  # Cap at 800ms for faster stabilization
                            log.debug(
                                f"Documents found, switching to faster checking "
                                f"(200ms intervals)"
                            )

                        # Check if count has stabilized (same count twice in a row)
                        if current_count == last_count and current_count > 0:
                            stable_count += 1
                            if stable_count >= stable_threshold:
                                elapsed = time.time() - start_time

                                # Success - cache the result for other threads
                                with _coordination_lock:
                                    _stabilization_results[collection_name] = (
                                        True,
                                        time.time(),
                                        current_count,
                                    )

                                log.info(
                                    f"Indexing stabilized! {current_count} "
                                    f"documents available after {elapsed:.1f}s"
                                )
                                return True
                        else:
                            stable_count = 0
                            last_count = current_count
                    else:
                        # Simple count-based check for non-web-search collections
                        log.info(
                            f"Indexing check {attempt}: {current_count}/"
                            f"{expected_count} documents available"
                        )

                        if current_count >= expected_count:
                            elapsed = time.time() - start_time

                            # Success - cache the result
                            with _coordination_lock:
                                _stabilization_results[collection_name] = (
                                    True,
                                    time.time(),
                                    current_count,
                                )

                            log.info(
                                f"Indexing completed! {current_count} documents "
                                f"available after {elapsed:.1f}s"
                            )
                            return True

                else:
                    # No documents found yet
                    if use_stabilization:
                        log.info(
                            f"Stabilization check {attempt}: 0 documents available"
                        )
                    else:
                        log.info(
                            f"Indexing check {attempt}: 0/{expected_count} "
                            f"documents available"
                        )

            except Exception as e:
                log.warning(f"Error checking indexing status: {e}")

            # Smart backoff: faster when stabilizing, slower when waiting
            if use_stabilization and documents_found:
                # Fast stabilization mode - much shorter intervals
                time.sleep(check_interval)
                check_interval = min(
                    check_interval * 1.2, max_interval
                )  # Gentler increase
            else:
                # Normal backoff when waiting for initial documents
                time.sleep(check_interval)
                check_interval = min(
                    check_interval * 1.4, max_interval
                )  # Moderate increase

            attempt += 1

        # Timeout reached - mark as failed and cache the failure
        elapsed = time.time() - start_time
        with _coordination_lock:
            _stabilization_results[collection_name] = (False, time.time(), 0)

        if use_stabilization:
            log.warning(f"Indexing stabilization timeout after {elapsed:.1f}s")
        else:
            log.warning(
                f"Indexing timeout after {elapsed:.1f}s - "
                f"expected {expected_count} documents"
            )
        return False

    finally:
        # Always clean up and notify any waiting threads
        with _coordination_lock:
            if collection_name in _stabilization_events:
                event = _stabilization_events[collection_name]
                del _stabilization_events[collection_name]
                event.set()  # Wake up any waiting threads


def get_doc(collection_name: str, user: UserModel = None, expected_count: int = None):
    """
    Get all documents from collection with intelligent indexing wait.

    Enhanced for web search collections with same race condition fixes as query_doc.
    """
    import time

    try:
        log.debug(f"get_doc:doc {collection_name}")

        # Auto-detect web search collections for intelligent waiting
        is_web_search = collection_name.startswith("web-search-")

        if is_web_search or (expected_count is not None and expected_count > 0):
            # Web search gets 15s timeout vs 10s for others
            timeout = 15.0 if is_web_search else 10.0

            if wait_for_indexing(collection_name, expected_count or 1, timeout):
                # Indexing complete, retrieve documents
                result = VECTOR_DB_CLIENT.get(collection_name=collection_name)
                if result:
                    doc_count = len(result.documents[0]) if result.documents[0] else 0
                    log.info(f"get_doc: Retrieved {doc_count} documents")
                    log.info(f"query_doc:result {result.ids} {result.metadatas}")
                return result
            else:
                # Timeout - return empty result
                log.warning(
                    f"get_doc: Indexing timeout for collection " f"'{collection_name}'"
                )
                return None

        # Fallback: Legacy retry mechanism for non-web-search collections
        max_retries = 4
        retry_delay = 2.0  # seconds

        for attempt in range(max_retries):
            log.info(
                f"get_doc: Attempt {attempt + 1}/{max_retries} "
                f"for collection '{collection_name}'"
            )

            result = VECTOR_DB_CLIENT.get(collection_name=collection_name)

            # Check if we got meaningful results
            if result and result.documents and result.documents[0]:
                log.info(
                    f"get_doc: SUCCESS on attempt {attempt + 1} - "
                    f"Found {len(result.documents[0])} documents"
                )
                if result:
                    log.info(f"query_doc:result {result.ids} {result.metadatas}")
                return result

            # If this is not the last attempt and we got empty results, wait and retry
            if attempt < max_retries - 1:
                log.info(
                    f"get_doc: Empty results on attempt {attempt + 1}, "
                    f"retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                log.warning(
                    f"get_doc: All {max_retries} attempts failed "
                    f"for collection '{collection_name}'"
                )

        # Log final result if all retries completed
        if result:
            log.info(f"query_doc:result {result.ids} {result.metadatas}")

        return result
    except Exception as e:
        log.exception(f"Error getting doc {collection_name}: {e}")
        raise e


def query_doc_with_hybrid_search(
    collection_name: str,
    collection_result: GetResult,
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
    hybrid_bm25_weight: float,
) -> dict:
    try:
        log.debug(f"query_doc_with_hybrid_search:doc {collection_name}")
        bm25_retriever = BM25Retriever.from_texts(
            texts=collection_result.documents[0],
            metadatas=collection_result.metadatas[0],
        )
        bm25_retriever.k = k

        vector_search_retriever = VectorSearchRetriever(
            collection_name=collection_name,
            embedding_function=embedding_function,
            top_k=k,
        )

        if hybrid_bm25_weight <= 0:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_search_retriever], weights=[1.0]
            )
        elif hybrid_bm25_weight >= 1:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever], weights=[1.0]
            )
        else:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_search_retriever],
                weights=[hybrid_bm25_weight, 1.0 - hybrid_bm25_weight],
            )

        compressor = RerankCompressor(
            embedding_function=embedding_function,
            top_n=k_reranker,
            reranking_function=reranking_function,
            r_score=r,
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )

        result = compression_retriever.invoke(query)

        distances = [d.metadata.get("score") for d in result]
        documents = [d.page_content for d in result]
        metadatas = [d.metadata for d in result]

        # retrieve only min(k, k_reranker) items, sort and cut by distance
        # if k < k_reranker
        if k < k_reranker:
            sorted_items = sorted(
                zip(distances, metadatas, documents), key=lambda x: x[0], reverse=True
            )
            sorted_items = sorted_items[:k]
            distances, documents, metadatas = map(list, zip(*sorted_items))

        result = {
            "distances": [distances],
            "documents": [documents],
            "metadatas": [metadatas],
        }

        log.info(
            "query_doc_with_hybrid_search:result "
            + f'{result["metadatas"]} {result["distances"]}'
        )
        return result
    except Exception as e:
        log.exception(f"Error querying doc {collection_name} with hybrid search: {e}")
        raise e


def merge_get_results(get_results: list[dict]) -> dict:
    # Initialize lists to store combined data
    combined_documents = []
    combined_metadatas = []
    combined_ids = []

    for data in get_results:
        combined_documents.extend(data["documents"][0])
        combined_metadatas.extend(data["metadatas"][0])
        combined_ids.extend(data["ids"][0])

    # Create the output dictionary
    result = {
        "documents": [combined_documents],
        "metadatas": [combined_metadatas],
        "ids": [combined_ids],
    }

    return result


def merge_and_sort_query_results(query_results: list[dict], k: int) -> dict:
    # Initialize lists to store combined data
    combined = dict()  # To store documents with unique document hashes

    for data in query_results:
        distances = data["distances"][0]
        documents = data["documents"][0]
        metadatas = data["metadatas"][0]

        for distance, document, metadata in zip(distances, documents, metadatas):
            if isinstance(document, str):
                doc_hash = hashlib.sha256(
                    document.encode()
                ).hexdigest()  # Compute a hash for uniqueness

                if doc_hash not in combined.keys():
                    combined[doc_hash] = (distance, document, metadata)
                    continue  # if doc is new, no further comparison is needed

                # if doc is alredy in, but new distance is better, update
                if distance > combined[doc_hash][0]:
                    combined[doc_hash] = (distance, document, metadata)

    combined = list(combined.values())
    # Sort the list based on distances
    combined.sort(key=lambda x: x[0], reverse=True)

    # Slice to keep only the top k elements
    sorted_distances, sorted_documents, sorted_metadatas = (
        zip(*combined[:k]) if combined else ([], [], [])
    )

    # Create and return the output dictionary
    return {
        "distances": [list(sorted_distances)],
        "documents": [list(sorted_documents)],
        "metadatas": [list(sorted_metadatas)],
    }


def get_all_items_from_collections(collection_names: list[str]) -> dict:
    results = []

    for collection_name in collection_names:
        if collection_name:
            try:
                result = get_doc(collection_name=collection_name)
                if result is not None:
                    results.append(result.model_dump())
            except Exception as e:
                log.exception(f"Error when querying the collection: {e}")
        else:
            pass

    return merge_get_results(results)


def query_collection(
    collection_names: list[str],
    queries: list[str],
    embedding_function,
    k: int,
) -> dict:
    results = []
    error = False

    def process_query_collection(collection_name, query_embedding):
        try:
            if collection_name:
                result = query_doc(
                    collection_name=collection_name,
                    k=k,
                    query_embedding=query_embedding,
                )
                if result is not None:
                    return result.model_dump(), None
            return None, None
        except Exception as e:
            log.exception(f"Error when querying the collection: {e}")
            return None, e

    # Generate all query embeddings (in one call)
    query_embeddings = embedding_function(queries, prefix=RAG_EMBEDDING_QUERY_PREFIX)
    log.debug(
        f"query_collection: processing {len(queries)} queries across "
        f"{len(collection_names)} collections"
    )

    with ThreadPoolExecutor() as executor:
        future_results = []
        for query_embedding in query_embeddings:
            for collection_name in collection_names:
                result = executor.submit(
                    process_query_collection, collection_name, query_embedding
                )
                future_results.append(result)
        task_results = [future.result() for future in future_results]

    for result, err in task_results:
        if err is not None:
            error = True
        elif result is not None:
            results.append(result)

    if error and not results:
        log.warning("All collection queries failed. No results returned.")

    return merge_and_sort_query_results(results, k=k)


def query_collection_with_hybrid_search(
    collection_names: list[str],
    queries: list[str],
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
    hybrid_bm25_weight: float,
) -> dict:
    results = []
    error = False
    # Fetch collection data once per collection sequentially
    # Avoid fetching the same data multiple times later
    collection_results = {}
    for collection_name in collection_names:
        try:
            log.debug(
                f"query_collection_with_hybrid_search:VECTOR_DB_CLIENT.get:"
                f"collection {collection_name}"
            )
            # Use get_doc to benefit from retry mechanism
            collection_results[collection_name] = get_doc(
                collection_name=collection_name
            )
        except Exception as e:
            log.exception(f"Failed to fetch collection {collection_name}: {e}")
            collection_results[collection_name] = None

    log.info(
        f"Starting hybrid search for {len(queries)} queries in "
        f"{len(collection_names)} collections..."
    )

    def process_query(collection_name, query):
        try:
            result = query_doc_with_hybrid_search(
                collection_name=collection_name,
                collection_result=collection_results[collection_name],
                query=query,
                embedding_function=embedding_function,
                k=k,
                reranking_function=reranking_function,
                k_reranker=k_reranker,
                r=r,
                hybrid_bm25_weight=hybrid_bm25_weight,
            )
            return result, None
        except Exception as e:
            log.exception(f"Error when querying the collection with hybrid_search: {e}")
            return None, e

    # Prepare tasks for all collections and queries
    # Avoid running any tasks for collections that failed to fetch data
    # (have assigned None)
    tasks = [
        (cn, q)
        for cn in collection_names
        if collection_results[cn] is not None
        for q in queries
    ]

    with ThreadPoolExecutor() as executor:
        future_results = [executor.submit(process_query, cn, q) for cn, q in tasks]
        task_results = [future.result() for future in future_results]

    for result, err in task_results:
        if err is not None:
            error = True
        elif result is not None:
            results.append(result)

    if error and not results:
        raise Exception(
            "Hybrid search failed for all collections. "
            "Using Non-hybrid search as fallback."
        )

    return merge_and_sort_query_results(results, k=k)


def get_embedding_function(
    embedding_engine,
    embedding_model,
    embedding_function,
    url,
    key,
    embedding_batch_size,
    azure_api_version=None,
):
    if embedding_engine == "":
        return lambda query, prefix=None, user=None: embedding_function.encode(
            query, **({"prompt": prefix} if prefix else {})
        ).tolist()
    elif embedding_engine in ["ollama", "openai", "azure_openai"]:
        func = lambda query, prefix=None, user=None: generate_embeddings(
            engine=embedding_engine,
            model=embedding_model,
            text=query,
            prefix=prefix,
            url=url,
            key=key,
            user=user,
            azure_api_version=azure_api_version,
        )

        def generate_multiple(query, prefix, user, func):
            if isinstance(query, list):
                embeddings = []
                for i in range(0, len(query), embedding_batch_size):
                    embeddings.extend(
                        func(
                            query[i : i + embedding_batch_size],
                            prefix=prefix,
                            user=user,
                        )
                    )
                return embeddings
            else:
                return func(query, prefix, user)

        return lambda query, prefix=None, user=None: generate_multiple(
            query, prefix, user, func
        )
    else:
        raise ValueError(f"Unknown embedding engine: {embedding_engine}")


def get_sources_from_files(
    request,
    files,
    queries,
    embedding_function,
    k,
    reranking_function,
    k_reranker,
    r,
    hybrid_bm25_weight,
    hybrid_search,
    full_context=False,
):
    log.debug(
        f"files: {files} {queries} {embedding_function} "
        f"{reranking_function} {full_context}"
    )

    extracted_collections = []
    relevant_contexts = []

    for file in files:

        context = None
        if file.get("docs"):
            # BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL
            context = {
                "documents": [[doc.get("content") for doc in file.get("docs")]],
                "metadatas": [[doc.get("metadata") for doc in file.get("docs")]],
            }
        elif file.get("context") == "full":
            # Manual Full Mode Toggle
            context = {
                "documents": [[file.get("file").get("data", {}).get("content")]],
                "metadatas": [[{"file_id": file.get("id"), "name": file.get("name")}]],
            }
        elif (
            file.get("type") != "web_search"
            and request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
        ):
            # Exclude web_search files from bypass to ensure vector retrieval
            if file.get("type") == "collection":
                file_ids = file.get("data", {}).get("file_ids", [])

                documents = []
                metadatas = []
                for file_id in file_ids:
                    file_object = Files.get_file_by_id(file_id)

                    if file_object:
                        documents.append(file_object.data.get("content", ""))
                        metadatas.append(
                            {
                                "file_id": file_id,
                                "name": file_object.filename,
                                "source": file_object.filename,
                            }
                        )

                context = {
                    "documents": [documents],
                    "metadatas": [metadatas],
                }

            elif file.get("id"):
                file_object = Files.get_file_by_id(file.get("id"))
                if file_object:
                    context = {
                        "documents": [[file_object.data.get("content", "")]],
                        "metadatas": [
                            [
                                {
                                    "file_id": file.get("id"),
                                    "name": file_object.filename,
                                    "source": file_object.filename,
                                }
                            ]
                        ],
                    }
            elif file.get("file").get("data"):
                context = {
                    "documents": [[file.get("file").get("data", {}).get("content")]],
                    "metadatas": [
                        [file.get("file").get("data", {}).get("metadata", {})]
                    ],
                }
        else:
            # Vector retrieval path - FIXED: Web search files now flow through here
            # Previous bug: Web search files were blocked from vector retrieval
            collection_names = []

            for file in files:
                if file.get("type") == "collection":
                    if file.get("legacy"):
                        collection_names = file.get("collection_names", [])
                    else:
                        collection_names.append(file["id"])
                elif file.get("collection_name"):
                    collection_names.append(file["collection_name"])
                elif file.get("id"):
                    if file.get("legacy"):
                        collection_names.append(f"{file['id']}")
                    else:
                        collection_names.append(f"file-{file['id']}")

            collection_names = set(collection_names).difference(extracted_collections)
            if not collection_names:
                continue

            # Check if any collection is web search - web search always needs query
            has_web_search = any(
                name.startswith("web-search-") for name in collection_names
            )

            # Key fix: Eliminates inconsistency where web search lost scores
            # All collections now return results WITH semantic similarity rankings
            try:
                effective_k = k
                if full_context:
                    # Full context = ALL documents WITH relevance scores (not without!)
                    effective_k = 1000  # High k ensures all documents returned

                if hybrid_search:
                    try:
                        context = query_collection_with_hybrid_search(
                            collection_names=collection_names,
                            queries=queries,
                            embedding_function=embedding_function,
                            k=effective_k,
                            reranking_function=reranking_function,
                            k_reranker=k_reranker,
                            r=r,
                            hybrid_bm25_weight=hybrid_bm25_weight,
                        )
                    except Exception as e:
                        log.debug(
                            "Error when using hybrid search, using"
                            " non hybrid search as fallback."
                        )
                        context = None

                if (not hybrid_search) or (context is None):
                    context = query_collection(
                        collection_names=collection_names,
                        queries=queries,
                        embedding_function=embedding_function,
                        k=effective_k,
                    )
            except Exception as e:
                log.exception(e)

            extracted_collections.extend(collection_names)

        if context:
            if "data" in file:
                del file["data"]

            relevant_contexts.append({**context, "file": file})

    sources = []
    for context in relevant_contexts:
        try:
            if "documents" in context:
                if "metadatas" in context:
                    source = {
                        "source": context["file"],
                        "document": context["documents"][0],
                        "metadata": context["metadatas"][0],
                    }
                    if "distances" in context and context["distances"]:
                        source["distances"] = context["distances"][0]

                    sources.append(source)
        except Exception as e:
            log.exception(e)

    return sources


def get_model_path(model: str, update_model: bool = False):
    # Construct huggingface_hub kwargs with local_files_only to return the snapshot path
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")

    local_files_only = not update_model

    if OFFLINE_MODE:
        local_files_only = True

    snapshot_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }

    log.debug(f"model: {model}")
    log.debug(f"snapshot_kwargs: {snapshot_kwargs}")

    # Inspiration from upstream sentence_transformers
    if (
        os.path.exists(model)
        or ("\\" in model or model.count("/") > 1)
        and local_files_only
    ):
        # If fully qualified path exists, return input, else set repo_id
        return model
    elif "/" not in model:
        # Set valid repo_id for model short-name
        model = "sentence-transformers" + "/" + model

    snapshot_kwargs["repo_id"] = model

    # Attempt to query the huggingface_hub library to determine the local path
    # and/or to update
    try:
        model_repo_path = snapshot_download(**snapshot_kwargs)
        log.debug(f"model_repo_path: {model_repo_path}")
        return model_repo_path
    except Exception as e:
        log.exception(f"Cannot determine model snapshot path: {e}")
        return model


def generate_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str = "https://api.openai.com/v1",
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"generate_openai_batch_embeddings:model {model} batch size: {len(texts)}"
        )
        json_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        r = requests.post(
            f"{url}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                **(
                    {
                        "X-OpenWebUI-User-Name": user.name,
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS and user
                    else {}
                ),
            },
            json=json_data,
        )
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            return [elem["embedding"] for elem in data["data"]]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        log.exception(f"Error generating openai batch embeddings: {e}")
        return None


def generate_azure_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str,
    key: str = "",
    version: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"generate_azure_openai_batch_embeddings:deployment {model} "
            f"batch size: {len(texts)}"
        )
        json_data = {"input": texts}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        url = f"{url}/openai/deployments/{model}/embeddings?api-version={version}"

        for _ in range(5):
            r = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "api-key": key,
                    **(
                        {
                            "X-OpenWebUI-User-Name": user.name,
                            "X-OpenWebUI-User-Id": user.id,
                            "X-OpenWebUI-User-Email": user.email,
                            "X-OpenWebUI-User-Role": user.role,
                        }
                        if ENABLE_FORWARD_USER_INFO_HEADERS and user
                        else {}
                    ),
                },
                json=json_data,
            )
            if r.status_code == 429:
                retry = float(r.headers.get("Retry-After", "1"))
                time.sleep(retry)
                continue
            r.raise_for_status()
            data = r.json()
            if "data" in data:
                return [elem["embedding"] for elem in data["data"]]
            else:
                raise Exception("Something went wrong :/")
        return None
    except Exception as e:
        log.exception(f"Error generating azure openai batch embeddings: {e}")
        return None


def generate_ollama_batch_embeddings(
    model: str,
    texts: list[str],
    url: str,
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"generate_ollama_batch_embeddings:model {model} batch size: {len(texts)}"
        )
        json_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        r = requests.post(
            f"{url}/api/embed",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                **(
                    {
                        "X-OpenWebUI-User-Name": user.name,
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS
                    else {}
                ),
            },
            json=json_data,
        )
        r.raise_for_status()
        data = r.json()

        if "embeddings" in data:
            return data["embeddings"]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        log.exception(f"Error generating ollama batch embeddings: {e}")
        return None


def generate_embeddings(
    engine: str,
    model: str,
    text: Union[str, list[str]],
    prefix: Union[str, None] = None,
    **kwargs,
):
    url = kwargs.get("url", "")
    key = kwargs.get("key", "")
    user = kwargs.get("user")

    if prefix is not None and RAG_EMBEDDING_PREFIX_FIELD_NAME is None:
        if isinstance(text, list):
            text = [f"{prefix}{text_element}" for text_element in text]
        else:
            text = f"{prefix}{text}"

    if engine == "ollama":
        embeddings = generate_ollama_batch_embeddings(
            **{
                "model": model,
                "texts": text if isinstance(text, list) else [text],
                "url": url,
                "key": key,
                "prefix": prefix,
                "user": user,
            }
        )
        return embeddings[0] if isinstance(text, str) else embeddings
    elif engine == "openai":
        embeddings = generate_openai_batch_embeddings(
            model, text if isinstance(text, list) else [text], url, key, prefix, user
        )
        return embeddings[0] if isinstance(text, str) else embeddings
    elif engine == "azure_openai":
        azure_api_version = kwargs.get("azure_api_version", "")
        embeddings = generate_azure_openai_batch_embeddings(
            model,
            text if isinstance(text, list) else [text],
            url,
            key,
            azure_api_version,
            prefix,
            user,
        )
        return embeddings[0] if isinstance(text, str) else embeddings


import operator
from typing import Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document


class RerankCompressor(BaseDocumentCompressor):
    embedding_function: Any
    top_n: int
    reranking_function: Any
    r_score: float

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        reranking = self.reranking_function is not None

        if reranking:
            scores = self.reranking_function.predict(
                [(query, doc.page_content) for doc in documents]
            )
        else:
            from sentence_transformers import util

            query_embedding = self.embedding_function(query, RAG_EMBEDDING_QUERY_PREFIX)
            document_embedding = self.embedding_function(
                [doc.page_content for doc in documents], RAG_EMBEDDING_CONTENT_PREFIX
            )
            scores = util.cos_sim(query_embedding, document_embedding)[0]

        docs_with_scores = list(
            zip(documents, scores.tolist() if not isinstance(scores, list) else scores)
        )
        if self.r_score:
            docs_with_scores = [
                (d, s) for d, s in docs_with_scores if s >= self.r_score
            ]

        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        final_results = []
        for doc, doc_score in result[: self.top_n]:
            metadata = doc.metadata
            metadata["score"] = doc_score
            doc = Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
            final_results.append(doc)
        return final_results
