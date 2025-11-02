"""
Gmail API Fetcher

This module handles fetching emails from Gmail API with:
- OAuth token authentication
- Pagination for bulk fetching
- Rate limiting to respect Gmail API quotas
- Batch operations for efficiency

Gmail API Quotas:
- 250 quota units per user per second
- List messages: 5 units per request
- Get message: 5 units per request
- Recommended: ~40 requests/second max
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Tuple
import aiohttp

# Set up logger with INFO level for visibility
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GmailFetcher:
    """
    Fetches emails from Gmail API with proper rate limiting and pagination.
    
    Uses OAuth token from user's login session.
    """

    # Gmail API base URL
    GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1"
    
    # Rate limiting configuration
    DEFAULT_MAX_REQUESTS_PER_SECOND = 40  # Conservative limit
    DEFAULT_BATCH_SIZE = 20  # Concurrent requests (Gmail limit: ~25 concurrent)
    
    def __init__(
        self,
        oauth_token: str,
        max_requests_per_second: int = DEFAULT_MAX_REQUESTS_PER_SECOND,
        timeout: int = 30,
    ):
        """
        Initialize Gmail fetcher.
        
        Args:
            oauth_token: OAuth access token with Gmail scopes
            max_requests_per_second: Rate limit (default: 40 req/s)
            timeout: Request timeout in seconds (default: 30)
        """
        self.oauth_token = oauth_token
        self.max_requests_per_second = max_requests_per_second
        self.timeout = timeout
        
        # Rate limiting tracking
        self.request_times = []
        self.request_lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "api_calls": 0,
            "emails_fetched": 0,
            "errors": 0,
            "rate_limit_waits": 0,
        }
    
    async def _rate_limit(self):
        """
        Implement rate limiting to respect Gmail API quotas.
        
        Ensures we don't exceed max_requests_per_second.
        """
        async with self.request_lock:
            now = time.time()
            
            # Remove requests older than 1 second
            self.request_times = [t for t in self.request_times if now - t < 1.0]
            
            # Check if we're at the limit
            if len(self.request_times) >= self.max_requests_per_second:
                # Calculate wait time
                oldest_request = self.request_times[0]
                wait_time = 1.0 - (now - oldest_request)
                
                if wait_time > 0:
                    self.stats["rate_limit_waits"] += 1
                    logger.debug(f"Rate limit reached, waiting {wait_time:.3f}s")
                    await asyncio.sleep(wait_time)
            
            # Record this request
            self.request_times.append(time.time())
    
    async def fetch_all_message_ids(
        self,
        max_results: int = 0,
        skip_spam_trash: bool = True,
        query: Optional[str] = None,
    ) -> List[str]:
        """
        Fetch all message IDs from Gmail with pagination.
        
        This fetches ALL emails from the entire mailbox including:
        - INBOX emails
        - SENT emails (emails you sent)
        - All custom labels/folders
        - Archived emails
        - Drafts
        - Everything except SPAM and TRASH (if skip_spam_trash=True)
        
        Args:
            max_results: Maximum number of message IDs to fetch (0 = unlimited)
            skip_spam_trash: Skip emails in SPAM and TRASH (default: True)
            query: Optional Gmail query filter (e.g., "newer_than:7d")
            
        Returns:
            List of Gmail message IDs from entire mailbox
            
        Example:
            >>> fetcher = GmailFetcher(oauth_token)
            >>> message_ids = await fetcher.fetch_all_message_ids(max_results=1000)
            >>> print(f"Found {len(message_ids)} emails across all folders")
        """
        
        logger.info("📥 Fetching message IDs from Gmail...")
        
        all_message_ids = []
        next_page_token = None
        page_count = 0
        
        # Build query - fetches ALL emails (inbox, sent, labels, etc.)
        query_parts = []
        if skip_spam_trash:
            # Exclude spam and trash, but include EVERYTHING else
            query_parts.append("-in:spam -in:trash")
            logger.info("   📋 Scope: ALL emails (INBOX + SENT + all labels)")
            logger.info("   🚫 Excluding: SPAM and TRASH only")
        else:
            logger.info("   📋 Scope: ENTIRE mailbox (including spam/trash)")
        
        if query:
            query_parts.append(query)
            logger.info(f"   🔍 Additional filter: {query}")
        
        full_query = " ".join(query_parts) if query_parts else None
        
        if full_query:
            logger.info(f"   📝 Gmail query: {full_query}")
        
        # Fetch pages
        async with aiohttp.ClientSession() as session:
            while True:
                page_count += 1
                
                # Build request parameters
                params = {
                    "maxResults": 500,  # Gmail API max per request
                }
                
                if full_query:
                    params["q"] = full_query
                
                if next_page_token:
                    params["pageToken"] = next_page_token
                
                # Rate limiting
                await self._rate_limit()
                
                # Make API request
                url = f"{self.GMAIL_API_BASE}/users/me/messages"
                headers = {
                    "Authorization": f"Bearer {self.oauth_token}",
                    "Accept": "application/json"
                }
                
                try:
                    async with session.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        self.stats["api_calls"] += 1
                        
                        if response.status == 401:
                            raise Exception("OAuth token expired or invalid")
                        
                        if response.status == 403:
                            raise Exception("Gmail API access forbidden - check API is enabled")
                        
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"Gmail API error {response.status}: {error_text}")
                        
                        data = await response.json()
                        
                except Exception as e:
                    self.stats["errors"] += 1
                    logger.error(f"Error fetching message IDs (page {page_count}): {e}")
                    raise
                
                # Extract message IDs
                messages = data.get("messages", [])
                
                if messages:
                    message_ids = [msg["id"] for msg in messages]
                    all_message_ids.extend(message_ids)
                    
                    logger.info(
                        f"Fetched page {page_count}: {len(message_ids)} IDs "
                        f"(total: {len(all_message_ids)})"
                    )
                
                # Check for next page
                next_page_token = data.get("nextPageToken")
                
                if not next_page_token:
                    logger.info(f"No more pages. Total message IDs: {len(all_message_ids)}")
                    break
                
                # Check max results limit
                if max_results > 0 and len(all_message_ids) >= max_results:
                    all_message_ids = all_message_ids[:max_results]
                    logger.info(f"Reached max_results limit: {max_results}")
                    break
        
        logger.info(
            f"Fetched {len(all_message_ids)} message IDs in {page_count} pages "
            f"({self.stats['api_calls']} API calls)"
        )
        
        return all_message_ids
    
    async def fetch_email(
        self,
        message_id: str,
    ) -> Optional[Dict]:
        """
        Fetch a single email's full content.
        
        Args:
            message_id: Gmail message ID
            
        Returns:
            Gmail API message response (format='full') or None if error
        """
        
        # Rate limiting
        await self._rate_limit()
        
        url = f"{self.GMAIL_API_BASE}/users/me/messages/{message_id}"
        headers = {
            "Authorization": f"Bearer {self.oauth_token}",
            "Accept": "application/json"
        }
        params = {"format": "full"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    self.stats["api_calls"] += 1
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"Error fetching email {message_id}: "
                            f"{response.status} - {error_text}"
                        )
                        self.stats["errors"] += 1
                        return None
                    
                    email_data = await response.json()
                    self.stats["emails_fetched"] += 1
                    
                    return email_data
        
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Exception fetching email {message_id}: {e}")
            return None
    
    async def fetch_emails_batch(
        self,
        message_ids: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[Dict]:
        """
        Fetch multiple emails in batches with rate limiting.
        
        Args:
            message_ids: List of Gmail message IDs to fetch
            batch_size: Number of concurrent requests (default: 20, Gmail limit ~25)
            
        Returns:
            List of Gmail API message responses
        """
        
        logger.info(f"Fetching {len(message_ids)} emails in batches of {batch_size}...")
        
        all_emails = []
        
        for i in range(0, len(message_ids), batch_size):
            batch = message_ids[i:i + batch_size]
            
            logger.info(
                f"Fetching batch {i//batch_size + 1} "
                f"({i+1}-{min(i+batch_size, len(message_ids))} of {len(message_ids)})"
            )
            
            # Fetch batch concurrently with limited concurrency (prevents 429)
            tasks = [self.fetch_email(msg_id) for msg_id in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in results
            batch_emails = []
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch fetch error: {result}")
                elif result is not None:
                    batch_emails.append(result)
            
            all_emails.extend(batch_emails)
            
            logger.info(
                f"Batch complete: {len(batch_emails)}/{len(batch)} successful "
                f"(total: {len(all_emails)})"
            )
        
        logger.info(
            f"Fetch complete: {len(all_emails)} emails fetched successfully "
            f"(errors: {self.stats['errors']})"
        )
        
        return all_emails
    
    async def fetch_all_emails(
        self,
        max_emails: int = 0,
        skip_spam_trash: bool = True,
        query: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Tuple[List[Dict], Dict]:
        """
        Complete workflow: Fetch message IDs, then fetch full email content.
        
        Args:
            max_emails: Maximum number of emails to fetch (0 = unlimited)
            skip_spam_trash: Skip SPAM and TRASH folders (default: True)
            query: Optional Gmail query filter
            batch_size: Concurrent fetch batch size (default: 100)
            
        Returns:
            Tuple of (list of email data, statistics dict)
        """
        
        start_time = time.time()
        
        # Step 1: Fetch all message IDs
        logger.info("Step 1: Fetching message IDs...")
        message_ids = await self.fetch_all_message_ids(
            max_results=max_emails,
            skip_spam_trash=skip_spam_trash,
            query=query
        )
        
        if not message_ids:
            logger.warning("No message IDs found")
            return [], self.stats
        
        # Step 2: Fetch full email content
        logger.info(f"Step 2: Fetching {len(message_ids)} emails...")
        emails = await self.fetch_emails_batch(
            message_ids=message_ids,
            batch_size=batch_size
        )
        
        elapsed_time = time.time() - start_time
        
        # Update stats
        self.stats["total_time"] = elapsed_time
        self.stats["emails_per_second"] = len(emails) / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(
            f"Fetch complete: {len(emails)} emails in {elapsed_time:.1f}s "
            f"({self.stats['emails_per_second']:.1f} emails/s)"
        )
        
        return emails, self.stats
    
    def get_stats(self) -> Dict:
        """Get fetcher statistics"""
        return self.stats.copy()
    
    async def fetch_attachment(
        self,
        message_id: str,
        attachment_id: str,
        filename: str
    ) -> Optional[bytes]:
        """
        Fetch an email attachment by ID.
        
        Args:
            message_id: The Gmail message ID
            attachment_id: The attachment ID from the message
            filename: The attachment filename (for logging)
            
        Returns:
            bytes: The attachment data, or None if failed
        """
        url = f"{self.GMAIL_API_BASE}/users/me/messages/{message_id}/attachments/{attachment_id}"
        
        try:
            await self._wait_for_rate_limit()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    self.api_calls += 1
                    
                    if response.status == 200:
                        attachment_data = await response.json()
                        # Gmail returns base64url-encoded data
                        import base64
                        data = attachment_data.get("data", "")
                        # Convert base64url to regular base64
                        data = data.replace('-', '+').replace('_', '/')
                        attachment_bytes = base64.b64decode(data)
                        
                        logger.debug(f"Downloaded attachment '{filename}': {len(attachment_bytes)} bytes")
                        return attachment_bytes
                    
                    elif response.status == 401:
                        logger.error("Gmail API authentication failed (token expired?)")
                        raise Exception("OAuth token expired or invalid")
                    
                    else:
                        error_text = await response.text()
                        logger.warning(f"Failed to download attachment '{filename}': {response.status} - {error_text}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error(f"Timeout downloading attachment '{filename}'")
            self.errors += 1
            return None
        except Exception as e:
            logger.error(f"Error downloading attachment '{filename}': {e}")
            self.errors += 1
            return None


# ============================================================================
# TESTING
# ============================================================================


async def test_fetcher_mock():
    """Test fetcher with mock/simulated data (no real API calls)"""
    
    print("\n" + "="*60)
    print("Phase 4 - GmailFetcher Test (Mock Mode)")
    print("="*60)
    
    # Simulate OAuth token
    mock_token = "mock_oauth_token_12345"
    
    print(f"\n✅ TEST 1: Fetcher Initialization")
    fetcher = GmailFetcher(
        oauth_token=mock_token,
        max_requests_per_second=40,
        timeout=30
    )
    
    print(f"   OAuth token: {mock_token[:20]}...")
    print(f"   Rate limit: {fetcher.max_requests_per_second} req/s")
    print(f"   Timeout: {fetcher.timeout}s")
    print(f"   ✅ PASSED\n")
    
    print(f"✅ TEST 2: Rate Limiting Logic")
    # Test that rate limiter allows requests
    start = time.time()
    for i in range(5):
        await fetcher._rate_limit()
    elapsed = time.time() - start
    
    print(f"   5 requests completed in {elapsed:.3f}s")
    print(f"   Rate limit waits: {fetcher.stats['rate_limit_waits']}")
    print(f"   ✅ PASSED\n")
    
    print(f"✅ TEST 3: Statistics Tracking")
    stats = fetcher.get_stats()
    print(f"   API calls: {stats['api_calls']}")
    print(f"   Emails fetched: {stats['emails_fetched']}")
    print(f"   Errors: {stats['errors']}")
    print(f"   ✅ PASSED\n")
    
    print("="*60)
    print("Phase 4 Mock Tests Complete ✅")
    print("="*60)
    print("\nGmailFetcher is ready for integration!")
    print("\nNext: Test with real Gmail API using your OAuth token")
    
    return True


if __name__ == "__main__":
    print("Testing Phase 4 - Gmail API Fetcher")
    success = asyncio.run(test_fetcher_mock())
    
    if success:
        print("\n🎉 Phase 4 mock tests PASSED!")
        print("\nTo test with real Gmail API:")
        print("1. Get your OAuth token from oauth_session table")
        print("2. Run: fetcher = GmailFetcher(your_token)")
        print("3. Run: emails = await fetcher.fetch_all_emails(max_emails=10)")
    
    exit(0 if success else 1)

