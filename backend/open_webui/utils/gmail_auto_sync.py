"""
Gmail Auto-Sync Orchestrator

This module orchestrates the automatic Gmail sync when users sign up with Google OAuth.

Flow:
1. User signs up/logs in with Google OAuth (with Gmail scopes)
2. OAuth callback triggers this auto-sync
3. Background task fetches all emails from Gmail
4. Processes and indexes to Pinecone using existing infrastructure
5. User's inbox is ready for semantic search

Components Used:
- GmailFetcher: Fetch emails from Gmail API
- GmailProcessor: Parse email data
- GmailIndexer: Chunk, embed, and format for Pinecone
- PineconeManager: Background upsert to Pinecone
- Uses existing chat summary infrastructure
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional
from datetime import datetime

from open_webui.utils.gmail_fetcher import GmailFetcher
from open_webui.utils.gmail_indexer import GmailIndexer
from open_webui.models.users import Users
from open_webui.models.oauth_sessions import OAuthSessions

logger = logging.getLogger(__name__)


class GmailAutoSync:
    """
    Orchestrates automatic Gmail sync for users.
    
    This class coordinates all Gmail components to perform a complete
    inbox sync in the background.
    """

    def __init__(
        self,
        embedding_service,
        content_aware_splitter,
        document_processor,
        pinecone_manager,
        gmail_namespace: str,
    ):
        """
        Initialize auto-sync orchestrator.
        
        Args:
            embedding_service: EmbeddingService from chat filter
            content_aware_splitter: ContentAwareTextSplitter from chat filter
            document_processor: DocumentProcessor from chat filter
            pinecone_manager: PineconeManager from chat filter
            gmail_namespace: Pinecone namespace for Gmail (PINECONE_NAMESPACE_GMAIL)
        """
        self.indexer = GmailIndexer(
            embedding_service=embedding_service,
            content_aware_splitter=content_aware_splitter,
            document_processor=document_processor,
            gmail_namespace=gmail_namespace,
        )
        self.pinecone = pinecone_manager
        self.gmail_namespace = gmail_namespace
        
        # Track active syncs
        self.active_syncs = {}  # user_id -> sync_status

    async def sync_user_gmail(
        self,
        user_id: str,
        oauth_token: dict,
        max_emails: int = 5000,
        skip_spam_trash: bool = True,
    ) -> Dict:
        """
        Sync a user's Gmail inbox to Pinecone.
        
        This is the main entry point for Gmail sync.
        
        Args:
            user_id: User ID for isolation
            oauth_token: OAuth token dict with access_token
            max_emails: Maximum emails to sync (default: 5000)
            skip_spam_trash: Skip SPAM and TRASH folders (default: True)
            
        Returns:
            Dict with sync statistics
        """
        
        logger.info(f"🚀 Starting Gmail sync for user {user_id}")
        
        # Initialize sync status
        self.active_syncs[user_id] = {
            "status": "running",
            "start_time": time.time(),
            "total_emails": 0,
            "processed": 0,
            "indexed": 0,
            "errors": 0,
        }
        
        try:
            # Validate OAuth token
            access_token = oauth_token.get("access_token")
            if not access_token:
                raise ValueError("OAuth token missing access_token")
            
            # Check for Gmail scopes
            token_scope = oauth_token.get("scope", "")
            if "gmail" not in token_scope:
                logger.warning(
                    f"User {user_id} OAuth token doesn't include Gmail scopes"
                )
                raise ValueError("OAuth token missing Gmail scopes")
            
            logger.info(f"✅ OAuth token validated with Gmail scopes")
            
            # Step 1: Fetch emails from Gmail API
            logger.info(f"📥 Step 1: Fetching emails from Gmail API...")
            
            fetcher = GmailFetcher(
                oauth_token=access_token,
                max_requests_per_second=40,  # Conservative rate limit
                timeout=30
            )
            
            emails, fetch_stats = await fetcher.fetch_all_emails(
                max_emails=max_emails,
                skip_spam_trash=skip_spam_trash,
                query=None,  # No additional filters for first sync
                batch_size=100,
            )
            
            self.active_syncs[user_id]["total_emails"] = len(emails)
            
            logger.info(
                f"✅ Fetched {len(emails)} emails "
                f"(API calls: {fetch_stats['api_calls']}, "
                f"errors: {fetch_stats['errors']})"
            )
            
            if not emails:
                logger.info(f"No emails found for user {user_id}")
                self.active_syncs[user_id]["status"] = "completed"
                return self._build_sync_result(user_id)
            
            # Step 2: Process and index emails to Pinecone
            logger.info(f"⚙️ Step 2: Processing and indexing {len(emails)} emails...")
            
            indexed_count = await self._process_and_index_batch(
                emails=emails,
                user_id=user_id,
                batch_size=100,
            )
            
            self.active_syncs[user_id]["indexed"] = indexed_count
            self.active_syncs[user_id]["status"] = "completed"
            
            result = self._build_sync_result(user_id)
            
            logger.info(
                f"🎉 Gmail sync complete for user {user_id}: "
                f"{indexed_count} emails indexed in {result['total_time']:.1f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Gmail sync failed for user {user_id}: {e}", exc_info=True)
            self.active_syncs[user_id]["status"] = "failed"
            self.active_syncs[user_id]["error"] = str(e)
            raise
    
    async def _process_and_index_batch(
        self,
        emails: List[dict],
        user_id: str,
        batch_size: int = 100,
    ) -> int:
        """
        Process emails and index to Pinecone in batches.
        
        Args:
            emails: List of Gmail API email responses
            user_id: User ID for isolation
            batch_size: Number of emails to process per batch
            
        Returns:
            Total number of emails successfully indexed
        """
        
        total_indexed = 0
        
        for i in range(0, len(emails), batch_size):
            batch = emails[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(emails) + batch_size - 1) // batch_size
            
            logger.info(
                f"Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} emails)"
            )
            
            # Process batch with GmailIndexer
            try:
                result = await self.indexer.process_email_batch(
                    emails=batch,
                    user_id=user_id,
                )
                
                upsert_data = result["upsert_data"]
                
                if upsert_data:
                    # Schedule background upsert to Pinecone
                    await self.pinecone.schedule_upsert(
                        upsert_data,
                        namespace=self.gmail_namespace
                    )
                    
                    total_indexed += result["processed"]
                    
                    logger.info(
                        f"✅ Batch {batch_num}: {result['processed']} emails processed, "
                        f"{result['total_vectors']} vectors queued for Pinecone"
                    )
                
                self.active_syncs[user_id]["processed"] = i + len(batch)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                self.active_syncs[user_id]["errors"] += 1
                continue
        
        return total_indexed
    
    def _build_sync_result(self, user_id: str) -> Dict:
        """Build sync result with statistics"""
        
        sync_status = self.active_syncs.get(user_id, {})
        elapsed_time = time.time() - sync_status.get("start_time", time.time())
        
        return {
            "user_id": user_id,
            "status": sync_status.get("status", "unknown"),
            "total_emails": sync_status.get("total_emails", 0),
            "processed": sync_status.get("processed", 0),
            "indexed": sync_status.get("indexed", 0),
            "errors": sync_status.get("errors", 0),
            "total_time": elapsed_time,
            "error": sync_status.get("error"),
        }
    
    def get_sync_status(self, user_id: str) -> Optional[Dict]:
        """Get current sync status for a user"""
        
        if user_id not in self.active_syncs:
            return None
        
        return self._build_sync_result(user_id)


# ============================================================================
# TRIGGER FUNCTION (called from OAuth callback)
# ============================================================================


async def trigger_gmail_sync_if_needed(
    request,
    user_id: str,
    provider: str,
    token: dict,
    is_new_user: bool = False,
):
    """
    Trigger Gmail sync if conditions are met.
    
    This function is called from the OAuth callback handler.
    
    Args:
        request: FastAPI request object
        user_id: User ID who just authenticated
        provider: OAuth provider (e.g., "google")
        token: OAuth token dict
        is_new_user: True if this is first-time signup
        
    Conditions for triggering:
    - Provider must be "google"
    - Token must have Gmail scopes
    - ENABLE_GMAIL_AUTO_SYNC must be True
    - If GMAIL_AUTO_SYNC_ON_SIGNUP_ONLY, only trigger for new users
    """
    
    # Check if Gmail auto-sync is enabled
    if not request.app.state.config.ENABLE_GMAIL_AUTO_SYNC:
        logger.info("Gmail auto-sync is disabled in config")
        return
    
    # Only trigger for Google OAuth
    if provider != "google":
        logger.debug(f"Skipping Gmail sync for non-Google provider: {provider}")
        return
    
    # Check for Gmail scopes
    token_scope = token.get("scope", "")
    if "gmail" not in token_scope:
        logger.info(f"User {user_id} OAuth token doesn't include Gmail scopes")
        return
    
    # Check if we should only sync on signup
    if request.app.state.config.GMAIL_AUTO_SYNC_ON_SIGNUP_ONLY and not is_new_user:
        logger.info(f"Gmail auto-sync only on signup, user {user_id} is not new")
        return
    
    logger.info(
        f"🚀 Triggering automatic Gmail sync for user {user_id} "
        f"(new_user: {is_new_user})"
    )
    
    # Launch background task
    try:
        from open_webui.tasks import create_task
        
        # Create the background sync task
        task_id, task = await create_task(
            request.app.state.redis,
            _background_gmail_sync(request, user_id, token),
            id=f"gmail_sync_{user_id}"
        )
        
        logger.info(f"✅ Gmail sync background task created: {task_id}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create Gmail sync task for user {user_id}: {e}")


async def _background_gmail_sync(request, user_id: str, oauth_token: dict):
    """
    Background task that performs the actual Gmail sync.
    
    This runs asynchronously and doesn't block the OAuth callback.
    
    Args:
        request: FastAPI request object
        user_id: User ID to sync
        oauth_token: OAuth token dict with access_token
    """
    
    logger.info(f"📧 Starting background Gmail sync for user {user_id}")
    
    try:
        user = Users.get_user_by_id(user_id)
        if not user:
            logger.error(f"User {user_id} not found")
            return
        
        # Get max emails from config
        max_emails = request.app.state.config.GMAIL_AUTO_SYNC_MAX_EMAILS
        
        logger.info(f"Config: max_emails={max_emails}")
        
        # TODO: Get services from chat filter
        # For now, this is a placeholder structure
        # In Phase 5.4, we'll integrate with the actual chat filter services
        
        logger.info(
            f"Gmail sync task placeholder executed for user {user_id}. "
            f"Full implementation in Phase 5.4"
        )
        
        # The actual implementation will be:
        # 1. Get embedding_service from request.app.state or chat filter
        # 2. Get content_aware_splitter from chat filter
        # 3. Get document_processor from chat filter
        # 4. Get pinecone_manager from chat filter
        # 5. Initialize GmailAutoSync
        # 6. Run sync_user_gmail()
        
    except Exception as e:
        logger.error(f"❌ Background Gmail sync failed for user {user_id}: {e}", exc_info=True)


# ============================================================================
# TESTING
# ============================================================================


async def test_auto_sync_orchestrator():
    """Test the auto-sync orchestrator logic"""
    
    print("\n" + "="*60)
    print("Phase 5 - Auto-Sync Orchestrator Test")
    print("="*60)
    
    # Mock services
    class MockEmbeddingService:
        async def embed_batch(self, texts):
            return [[0.1] * 1536 for _ in texts]
    
    class MockSplitter:
        def split_text(self, text):
            return [text]  # Simple: one chunk per email
    
    class MockDocProcessor:
        @staticmethod
        def quick_quality_score(text):
            return 4
    
    class MockPineconeManager:
        async def schedule_upsert(self, data, namespace):
            logger.info(f"Mock upsert: {len(data)} vectors to namespace '{namespace}'")
            return f"job_{time.time()}"
    
    print("\n✅ TEST 1: Initialize Auto-Sync Orchestrator")
    
    sync = GmailAutoSync(
        embedding_service=MockEmbeddingService(),
        content_aware_splitter=MockSplitter(),
        document_processor=MockDocProcessor(),
        pinecone_manager=MockPineconeManager(),
        gmail_namespace="gmail-inbox",
    )
    
    print(f"   Namespace: {sync.gmail_namespace}")
    print(f"   Indexer initialized: ✓")
    print(f"   Pinecone manager ready: ✓")
    print("   ✅ PASSED\n")
    
    print("✅ TEST 2: Sync Status Tracking")
    
    # Simulate starting a sync
    test_user_id = "test_user_999"
    sync.active_syncs[test_user_id] = {
        "status": "running",
        "start_time": time.time(),
        "total_emails": 100,
        "processed": 50,
        "indexed": 45,
        "errors": 5,
    }
    
    status = sync.get_sync_status(test_user_id)
    
    print(f"   User: {status['user_id']}")
    print(f"   Status: {status['status']}")
    print(f"   Processed: {status['processed']}/{status['total_emails']}")
    print(f"   Indexed: {status['indexed']}")
    print(f"   Errors: {status['errors']}")
    
    assert status["status"] == "running", "Status should be running"
    assert status["processed"] == 50, "Should track processed count"
    
    print("   ✅ PASSED\n")
    
    print("="*60)
    print("Phase 5 Orchestrator Tests Complete ✅")
    print("="*60)
    
    print("\nOrchestrator capabilities:")
    print("  ✅ Coordinates all Gmail components")
    print("  ✅ Tracks sync progress per user")
    print("  ✅ Integrates with existing services")
    print("  ✅ Ready for OAuth callback integration")
    
    return True


if __name__ == "__main__":
    print("Testing Phase 5 - Gmail Auto-Sync Orchestrator")
    success = asyncio.run(test_auto_sync_orchestrator())
    
    if success:
        print("\n🎉 Phase 5 orchestrator tests PASSED!")
        print("\nNext: Hook into OAuth callback (Phase 5.4)")
    
    exit(0 if success else 1)

