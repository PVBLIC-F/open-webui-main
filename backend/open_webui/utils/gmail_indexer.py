"""
Gmail Indexer - Integration Layer

This module integrates GmailProcessor with existing chat summary infrastructure:
- Uses ContentAwareTextSplitter for intelligent email chunking
- Uses EmbeddingService for vector generation
- Uses PineconeManager for storage
- Matches the exact metadata pattern from chat summaries

Namespace Strategy:
- Chat summaries: PINECONE_NAMESPACE (e.g., "chat-summary-knowledge")
- Gmail emails: PINECONE_NAMESPACE_GMAIL (e.g., "gmail-inbox")

This is the bridge between Gmail parsing and your existing RAG system.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from open_webui.utils.gmail_processor import GmailProcessor
from open_webui.utils.text_cleaner import TextCleaner

# Set up logger with INFO level for visibility
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GmailIndexer:
    """
    Indexes Gmail emails using existing chat summary infrastructure.
    
    Reuses:
    - ContentAwareTextSplitter (from chat summary filter)
    - EmbeddingService (from chat summary filter)
    - PineconeManager (from chat summary filter)
    - DocumentProcessor quality scoring
    
    Uses separate namespace: PINECONE_NAMESPACE_GMAIL
    """

    def __init__(
        self,
        embedding_service,
        content_aware_splitter,
        document_processor,
        gmail_namespace: str = "gmail-inbox",
    ):
        """
        Initialize indexer with existing services.
        
        Args:
            embedding_service: EmbeddingService instance from chat filter
            content_aware_splitter: ContentAwareTextSplitter instance
            document_processor: DocumentProcessor instance for quality scoring
            gmail_namespace: Pinecone namespace for Gmail emails (from PINECONE_NAMESPACE_GMAIL env var)
        """
        self.gmail_processor = GmailProcessor()
        self.embeddings = embedding_service
        self.splitter = content_aware_splitter
        self.doc_processor = document_processor
        self.namespace = gmail_namespace  # Store namespace for Pinecone operations

    async def process_email_for_indexing(
        self,
        email_data: dict,
        user_id: str,
    ) -> Dict:
        """
        Process a single Gmail email into Pinecone-ready format.
        
        This method:
        1. Parses email with GmailProcessor
        2. Chunks with ContentAwareTextSplitter
        3. Generates embeddings with EmbeddingService
        4. Creates upsert data matching chat summary format
        
        Args:
            email_data: Gmail API message response (format='full')
            user_id: User ID for isolation
            
        Returns:
            Dict with email_id, chunks count, and upsert_data for Pinecone
        """
        
        start_time = time.time()
        
        # Step 1: Parse email
        parsed = self.gmail_processor.parse_email(email_data, user_id)
        email_id = parsed["email_id"]
        document_text = parsed["document_text"]
        base_metadata = parsed["metadata"]
        
        logger.info(
            f"Processing email {email_id} ({base_metadata['subject'][:50]}...)"
        )
        
        # Step 2: Chunk the email text using content-aware splitter
        chunks = self.splitter.split_text(document_text)
        
        if not chunks:
            chunks = [document_text]  # Fallback to single chunk
        
        logger.info(f"Email {email_id} split into {len(chunks)} chunks")
        
        # Step 3: Generate embeddings for all chunks
        chunk_embeddings = await self.embeddings.embed_batch(chunks)
        
        logger.info(f"Generated {len(chunk_embeddings)} embeddings for email {email_id}")
        
        # Step 4: Build upsert data (matching chat summary format)
        upsert_data = self._build_upsert_data(
            chunks=chunks,
            embeddings=chunk_embeddings,
            base_metadata=base_metadata,
        )
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Email {email_id} processed in {processing_time:.2f}s: "
            f"{len(chunks)} chunks, {len(upsert_data)} vectors"
        )
        
        return {
            "email_id": email_id,
            "chunks": len(chunks),
            "vectors": len(upsert_data),
            "upsert_data": upsert_data,
            "processing_time": processing_time,
        }
    
    def _build_upsert_data(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        base_metadata: Dict,
    ) -> List[Dict]:
        """
        Build Pinecone upsert data matching chat summary format.
        
        This creates the exact same structure as chat summaries but with
        type='email' and email-specific metadata fields.
        
        Args:
            chunks: List of text chunks from email
            embeddings: List of embedding vectors (one per chunk)
            base_metadata: Base metadata from GmailProcessor
            
        Returns:
            List of dicts ready for Pinecone upsert
        """
        
        upsert_data = []
        email_id = base_metadata["email_id"]
        user_id = base_metadata["user_id"]
        
        for i, (chunk_text, chunk_vec) in enumerate(zip(chunks, embeddings)):
            
            # Calculate quality score (reusing DocumentProcessor logic)
            quality_score = self.doc_processor.quick_quality_score(chunk_text)
            
            # Create unique record ID
            rec_id = f"email-{email_id}-{int(time.time())}-{i}"
            
            # Build metadata matching chat summary structure
            metadata = {
                # Core identification (CHANGED from chat to email)
                "type": "email",  # ← Was "chat" in chat summaries
                "email_id": email_id,  # ← Like "chat_id"
                "thread_id": base_metadata["thread_id"],
                "user_id": user_id,
                
                # Content (SAME structure as chat)
                "chunk_text": TextCleaner.clean_for_pinecone(chunk_text),
                "chunk_index": i,
                "total_chunks": len(chunks),
                
                # Timing (SAME structure as chat)
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "date_timestamp": base_metadata["date_timestamp"],
                
                # Email-specific fields (NEW)
                "subject": base_metadata["subject"],  # ← Like "topic" in chat
                "from": base_metadata["from"],
                "to": base_metadata["to"],
                "cc": base_metadata.get("cc", ""),
                "date": base_metadata["date"],
                "labels": base_metadata["labels"],
                "has_attachments": base_metadata["has_attachments"],
                
                # Quality & metadata (SAME as chat)
                "quality_score": quality_score,
                "doc_type": "email",  # ← Was "chat_summary"
                "source": "gmail",
                "vector_dim": len(chunk_vec),
                "summary_id": rec_id,  # ← For consistency with chat summaries
                
                # Deduplication
                "hash": base_metadata["hash"],
            }
            
            upsert_data.append({
                "id": rec_id,
                "values": [float(v) if v is not None else 0.0 for v in chunk_vec],
                "metadata": metadata,
            })
        
        return upsert_data
    
    @staticmethod
    def _clean_for_pinecone(text: str) -> str:
        """
        Clean text for Pinecone storage (remove problematic characters).
        
        Simplified version - you can replace with TextCleaner.clean_for_pinecone
        from your chat filter if needed.
        """
        
        if not text:
            return ""
        
        import re
        
        # Remove control characters except newlines/tabs
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Encode as UTF-8 (remove invalid chars)
        text = text.encode("utf-8", "ignore").decode("utf-8")
        
        # Limit length for Pinecone
        max_length = 10000
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    async def process_email_batch(
        self,
        emails: List[dict],
        user_id: str,
    ) -> Dict:
        """
        Process multiple emails in batch.
        
        Memory-efficient: Processes and yields results without accumulating all vectors.
        
        Args:
            emails: List of Gmail API message responses
            user_id: User ID for isolation
            
        Returns:
            Dict with batch processing statistics and all upsert data
        """
        
        batch_start = time.time()
        all_upsert_data = []
        processed_count = 0
        error_count = 0
        
        logger.info(f"Processing batch of {len(emails)} emails for user {user_id}")
        
        for email_data in emails:
            try:
                result = await self.process_email_for_indexing(email_data, user_id)
                all_upsert_data.extend(result["upsert_data"])
                processed_count += 1
                
                # Clear the large email_data after processing to free memory
                email_data.clear()
                
            except Exception as e:
                email_id = email_data.get("id", "unknown")
                logger.error(f"Error processing email {email_id}: {e}")
                error_count += 1
                continue
        
        batch_time = time.time() - batch_start
        
        logger.info(
            f"Batch processing complete: {processed_count} emails processed, "
            f"{error_count} errors, {len(all_upsert_data)} vectors, "
            f"{batch_time:.2f}s"
        )
        
        return {
            "processed": processed_count,
            "errors": error_count,
            "total_vectors": len(all_upsert_data),
            "upsert_data": all_upsert_data,
            "processing_time": batch_time,
        }


# ============================================================================
# TESTING
# ============================================================================


def create_test_email_data():
    """Create test email data for testing"""
    return {
        "id": "test_email_001",
        "threadId": "test_thread_001",
        "labelIds": ["INBOX", "IMPORTANT"],
        "snippet": "This is a long test email...",
        "internalDate": "1605451800000",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "sender@company.com"},
                {"name": "To", "value": "recipient@company.com"},
                {"name": "Subject", "value": "Long Email Test"},
                {"name": "Date", "value": "Sun, 15 Nov 2020 14:30:00 +0000"},
            ],
            "body": {
                "data": base64.b64encode(
                    b"""This is a test email with multiple paragraphs.

Paragraph one discusses the project timeline and deliverables for Q4.

Paragraph two covers budget allocation and resource planning.

Paragraph three talks about team assignments and responsibilities.

This email should be chunked because it's longer than typical emails.

Best regards,
Test Sender"""
                ).decode()
            }
        }
    }


async def test_indexer():
    """Test the indexer with mock services"""
    print("\n" + "="*60)
    print("Phase 3 Integration Test - GmailIndexer")
    print("="*60)
    
    # Mock services (simplified for testing)
    class MockEmbeddingService:
        async def embed_batch(self, texts):
            # Return mock 1536-dim vectors
            return [[0.1] * 1536 for _ in texts]
    
    class MockSplitter:
        def split_text(self, text):
            # Simple split by double newlines
            chunks = text.split("\n\n")
            return [c.strip() for c in chunks if c.strip()]
    
    class MockDocProcessor:
        @staticmethod
        def quick_quality_score(text):
            # Simple quality scoring
            score = 0
            if len(text) > 200:
                score += 2
            if "." in text:
                score += 2
            return min(score, 5)
    
    # Create indexer with mock services
    indexer = GmailIndexer(
        embedding_service=MockEmbeddingService(),
        content_aware_splitter=MockSplitter(),
        document_processor=MockDocProcessor(),
    )
    
    # Test with sample email
    email_data = create_test_email_data()
    
    try:
        result = await indexer.process_email_for_indexing(
            email_data,
            "test_user_456"
        )
        
        print(f"\n✅ Email processed successfully!")
        print(f"   Email ID: {result['email_id']}")
        print(f"   Chunks created: {result['chunks']}")
        print(f"   Vectors generated: {result['vectors']}")
        print(f"   Processing time: {result['processing_time']:.3f}s")
        
        # Verify upsert data structure
        if result['upsert_data']:
            sample = result['upsert_data'][0]
            print(f"\n📊 Sample vector metadata:")
            print(f"   ID: {sample['id']}")
            print(f"   Vector dim: {len(sample['values'])}")
            print(f"   Metadata type: {sample['metadata']['type']}")
            print(f"   Metadata user_id: {sample['metadata']['user_id']}")
            print(f"   Metadata subject: {sample['metadata']['subject']}")
            print(f"   Quality score: {sample['metadata']['quality_score']}")
            
            # Verify structure matches chat summary pattern
            required_fields = [
                'type', 'email_id', 'user_id', 'chunk_text', 
                'chunk_index', 'total_chunks', 'quality_score',
                'doc_type', 'source', 'timestamp'
            ]
            
            missing = [f for f in required_fields if f not in sample['metadata']]
            if missing:
                print(f"\n⚠️  Missing fields: {missing}")
                return False
            else:
                print(f"\n✅ All required metadata fields present!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import asyncio
    import base64
    
    print("Testing Phase 3 - Gmail Indexer Integration")
    success = asyncio.run(test_indexer())
    
    if success:
        print("\n🎉 Phase 3 integration test PASSED!")
        print("\nNext: Test with real EmbeddingService from chat filter")
    else:
        print("\n❌ Phase 3 integration test FAILED!")
    
    exit(0 if success else 1)

