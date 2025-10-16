"""
Email-Specific Text Cleaning

Advanced email body cleaning to remove:
- Email signatures
- Quoted reply text
- Email headers from forwarded messages
- Excessive whitespace
- Common email artifacts

Produces high-quality, clean text suitable for embeddings and search.
"""

import re
from typing import Tuple
from open_webui.utils.text_cleaner import TextCleaner


class EmailCleaner:
    """Advanced email cleaning specifically for Gmail messages"""
    
    # Common signature patterns
    SIGNATURE_PATTERNS = [
        r'--\s*\n.*',  # Standard signature delimiter
        r'Sent from my (iPhone|iPad|Android|Mobile)',
        r'Get Outlook for (iOS|Android)',
        r'Virus-free\. www\.avg\.com',
        r'This email and any attachments.*confidential',
    ]
    
    # Quoted text patterns
    QUOTED_PATTERNS = [
        r'On .+ wrote:.*',  # "On Mon, Oct 14, 2024, John wrote:"
        r'From:.*\nSent:.*\nTo:.*\nSubject:.*',  # Outlook-style headers
        r'^>+.*$',  # Lines starting with >
        r'_{10,}',  # Long underscores (reply separators)
        r'-{10,}',  # Long dashes
    ]
    
    @staticmethod
    def clean_email_body(body: str) -> Tuple[str, str]:
        """
        Clean email body and extract high-quality content.
        
        Returns:
            Tuple of (cleaned_body, original_message)
            - cleaned_body: Full cleaned text
            - original_message: Just the current message (no quotes/signatures)
        """
        
        if not body:
            return "", ""
        
        # Start with basic text cleaning
        text = TextCleaner.clean_text(body)
        
        # Split into lines for processing
        lines = text.split('\n')
        
        # Step 1: Find where the original message ends (before quoted content)
        original_end = EmailCleaner._find_original_message_end(lines)
        
        # Step 2: Remove signatures
        signature_start = EmailCleaner._find_signature_start(lines[:original_end])
        if signature_start:
            original_end = min(original_end, signature_start)
        
        # Extract original message (no quotes, no signature)
        original_lines = lines[:original_end]
        original_message = '\n'.join(original_lines).strip()
        
        # Step 3: Clean the full text (keep quoted content but clean it)
        cleaned_full = EmailCleaner._deep_clean(text)
        
        # Step 4: Final cleaning pass
        original_message = EmailCleaner._final_polish(original_message)
        cleaned_full = EmailCleaner._final_polish(cleaned_full)
        
        return cleaned_full, original_message
    
    @staticmethod
    def _find_original_message_end(lines: list) -> int:
        """Find where the original message ends (before quoted replies)"""
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Common quote indicators
            if any([
                line_lower.startswith('on ') and ' wrote:' in line_lower,
                line_lower.startswith('from:') and i < len(lines) - 1 and lines[i+1].lower().startswith('sent:'),
                line.startswith('>'),
                line.startswith('>>'),
                re.match(r'^_{5,}$', line),  # Underscore separators
                re.match(r'^-{5,}$', line),  # Dash separators
            ]):
                return i
        
        return len(lines)
    
    @staticmethod
    def _find_signature_start(lines: list) -> int:
        """Find where the email signature starts"""
        
        for i, line in enumerate(lines):
            # Standard signature delimiter
            if line.strip() == '--':
                return i
            
            # Common signature patterns
            if any([
                'sent from my' in line.lower(),
                'get outlook for' in line.lower(),
                re.search(r'^\s*thanks?,?\s*$', line, re.IGNORECASE),
                re.search(r'^\s*best,?\s*$', line, re.IGNORECASE),
                re.search(r'^\s*regards?,?\s*$', line, re.IGNORECASE),
            ]):
                # Check if this is followed by name/contact info (likely signature)
                if i < len(lines) - 1:
                    next_line = lines[i + 1].strip()
                    if next_line and len(next_line) < 50:  # Short line after closing
                        return i
        
        return None
    
    @staticmethod
    def _deep_clean(text: str) -> str:
        """Deep cleaning pass to remove email artifacts"""
        
        # Remove quoted reply blocks
        text = re.sub(r'^>+.*$', '', text, flags=re.MULTILINE)
        
        # Remove forwarded message headers
        text = re.sub(
            r'From:.*?\nSent:.*?\nTo:.*?\nSubject:.*?\n',
            '',
            text,
            flags=re.DOTALL
        )
        
        # Remove "Sent from" footers
        for pattern in EmailCleaner.SIGNATURE_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def _final_polish(text: str) -> str:
        """Final polish pass for high-quality output"""
        
        # Remove orphaned punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Fix spacing after punctuation
        text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)
        
        # Remove URLs (often messy tracking links)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[link]', text)
        
        # Remove email addresses (keep privacy)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[email]', text)
        
        # Clean up result
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def parse_email_address(email_str: str) -> Tuple[str, str]:
        """
        Parse email address string into name and email.
        
        Examples:
            "John Doe <john@example.com>" → ("John Doe", "john@example.com")
            "john@example.com" → ("", "john@example.com")
        
        Returns:
            Tuple of (name, email_address)
        """
        
        if not email_str:
            return "", ""
        
        # Try to extract name and email from "Name <email>" format
        match = re.match(r'^([^<]+?)\s*<([^>]+)>$', email_str.strip())
        if match:
            name = match.group(1).strip()
            email = match.group(2).strip()
            return name, email
        
        # Just an email address
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', email_str)
        if email_match:
            return "", email_match.group(0)
        
        return "", email_str.strip()

