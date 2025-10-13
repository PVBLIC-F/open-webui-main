"""
Unit tests for GmailProcessor

Tests email parsing with various formats:
- Plain text emails
- HTML emails
- Multipart emails
- Nested MIME structures
- Attachments
- Edge cases
"""

import base64
import pytest
from open_webui.utils.gmail_processor import GmailProcessor


class TestGmailProcessor:
    """Test suite for GmailProcessor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = GmailProcessor()
        self.test_user_id = "test_user_123"
    
    def test_parse_simple_plain_text_email(self):
        """Test parsing a simple plain text email"""
        
        email_data = {
            "id": "msg_001",
            "threadId": "thread_001",
            "labelIds": ["INBOX"],
            "snippet": "Simple test email...",
            "internalDate": "1605451800000",
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "recipient@example.com"},
                    {"name": "Subject", "value": "Test Email"},
                    {"name": "Date", "value": "Sun, 15 Nov 2020 14:30:00 +0000"},
                ],
                "body": {
                    "data": base64.urlsafe_b64encode(
                        b"This is a simple test email."
                    ).decode()
                }
            }
        }
        
        result = self.processor.parse_email(email_data, self.test_user_id)
        
        # Verify basic fields
        assert result["email_id"] == "msg_001"
        assert result["thread_id"] == "thread_001"
        assert result["metadata"]["subject"] == "Test Email"
        assert result["metadata"]["from"] == "sender@example.com"
        assert result["metadata"]["to"] == "recipient@example.com"
        assert result["metadata"]["user_id"] == self.test_user_id
        assert result["metadata"]["type"] == "email"
        assert result["metadata"]["doc_type"] == "email"
        assert result["raw_body"] == "This is a simple test email."
        assert "This is a simple test email" in result["document_text"]
    
    def test_parse_html_email(self):
        """Test parsing HTML email with tag stripping"""
        
        html_content = """
        <html>
        <body>
            <h1>Important Update</h1>
            <p>This is an <strong>HTML</strong> email with <a href="http://example.com">links</a>.</p>
            <p>Second paragraph here.</p>
        </body>
        </html>
        """
        
        email_data = {
            "id": "msg_002",
            "threadId": "thread_002",
            "labelIds": ["INBOX"],
            "snippet": "HTML email...",
            "internalDate": "1605451800000",
            "payload": {
                "mimeType": "text/html",
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "recipient@example.com"},
                    {"name": "Subject", "value": "HTML Test"},
                    {"name": "Date", "value": "Sun, 15 Nov 2020 14:30:00 +0000"},
                ],
                "body": {
                    "data": base64.urlsafe_b64encode(html_content.encode()).decode()
                }
            }
        }
        
        result = self.processor.parse_email(email_data, self.test_user_id)
        
        # Verify HTML tags are stripped
        assert "<html>" not in result["raw_body"]
        assert "<strong>" not in result["raw_body"]
        assert "Important Update" in result["raw_body"]
        assert "HTML email" in result["raw_body"]
        assert result["metadata"]["subject"] == "HTML Test"
    
    def test_parse_multipart_email(self):
        """Test parsing multipart/alternative email (plain + HTML)"""
        
        plain_content = "This is the plain text version."
        html_content = "<html><body><p>This is the <b>HTML</b> version.</p></body></html>"
        
        email_data = {
            "id": "msg_003",
            "threadId": "thread_003",
            "labelIds": ["INBOX"],
            "snippet": "Multipart email...",
            "internalDate": "1605451800000",
            "payload": {
                "mimeType": "multipart/alternative",
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "recipient@example.com"},
                    {"name": "Subject", "value": "Multipart Test"},
                    {"name": "Date", "value": "Sun, 15 Nov 2020 14:30:00 +0000"},
                ],
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": {
                            "data": base64.urlsafe_b64encode(plain_content.encode()).decode()
                        }
                    },
                    {
                        "mimeType": "text/html",
                        "body": {
                            "data": base64.urlsafe_b64encode(html_content.encode()).decode()
                        }
                    }
                ]
            }
        }
        
        result = self.processor.parse_email(email_data, self.test_user_id)
        
        # Should prefer plain text version
        assert result["raw_body"] == plain_content
        assert "plain text version" in result["document_text"]
    
    def test_parse_email_with_attachments(self):
        """Test detection of email attachments"""
        
        email_data = {
            "id": "msg_004",
            "threadId": "thread_004",
            "labelIds": ["INBOX"],
            "snippet": "Email with attachment...",
            "internalDate": "1605451800000",
            "payload": {
                "mimeType": "multipart/mixed",
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "recipient@example.com"},
                    {"name": "Subject", "value": "Email with Attachment"},
                    {"name": "Date", "value": "Sun, 15 Nov 2020 14:30:00 +0000"},
                ],
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": {
                            "data": base64.urlsafe_b64encode(
                                b"Please see attached document."
                            ).decode()
                        }
                    },
                    {
                        "mimeType": "application/pdf",
                        "filename": "budget_report.pdf",
                        "body": {"attachmentId": "attach_123"}
                    }
                ]
            }
        }
        
        result = self.processor.parse_email(email_data, self.test_user_id)
        
        # Verify attachment detection
        assert result["metadata"]["has_attachments"] is True
        assert "Please see attached document" in result["raw_body"]
    
    def test_parse_email_without_subject(self):
        """Test handling of emails without subject"""
        
        email_data = {
            "id": "msg_005",
            "threadId": "thread_005",
            "labelIds": ["INBOX"],
            "snippet": "No subject...",
            "internalDate": "1605451800000",
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "recipient@example.com"},
                    {"name": "Date", "value": "Sun, 15 Nov 2020 14:30:00 +0000"},
                ],
                "body": {
                    "data": base64.urlsafe_b64encode(b"Email body text.").decode()
                }
            }
        }
        
        result = self.processor.parse_email(email_data, self.test_user_id)
        
        # Should use default subject
        assert result["metadata"]["subject"] == "(No Subject)"
    
    def test_date_parsing(self):
        """Test various date formats"""
        
        test_cases = [
            ("Sun, 15 Nov 2020 14:30:00 +0000", 2020),
            ("Mon, 1 Jan 2024 00:00:00 GMT", 2024),
            ("15 Nov 2020 14:30:00", 2020),
        ]
        
        for date_str, expected_year in test_cases:
            email_data = {
                "id": f"msg_date_test_{expected_year}",
                "threadId": "thread_date",
                "labelIds": ["INBOX"],
                "payload": {
                    "mimeType": "text/plain",
                    "headers": [
                        {"name": "From", "value": "test@example.com"},
                        {"name": "To", "value": "user@example.com"},
                        {"name": "Subject", "value": "Date Test"},
                        {"name": "Date", "value": date_str},
                    ],
                    "body": {"data": base64.urlsafe_b64encode(b"Test").decode()}
                }
            }
            
            result = self.processor.parse_email(email_data, self.test_user_id)
            
            # Verify year is parsed correctly
            assert str(expected_year) in result["metadata"]["date"]
            assert result["metadata"]["date_timestamp"] > 0
    
    def test_user_id_isolation(self):
        """Test that user_id is properly set for isolation"""
        
        email_data = {
            "id": "msg_isolation",
            "threadId": "thread_isolation",
            "labelIds": ["INBOX"],
            "snippet": "Test",
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "test@example.com"},
                    {"name": "Subject", "value": "Isolation Test"},
                ],
                "body": {"data": base64.urlsafe_b64encode(b"Test").decode()}
            }
        }
        
        # Test with different user IDs
        result1 = self.processor.parse_email(email_data, "user_A")
        result2 = self.processor.parse_email(email_data, "user_B")
        
        # Verify user_id is different
        assert result1["metadata"]["user_id"] == "user_A"
        assert result2["metadata"]["user_id"] == "user_B"
        
        # Verify hash is different (includes user_id)
        assert result1["metadata"]["hash"] != result2["metadata"]["hash"]
    
    def test_metadata_structure(self):
        """Test that metadata matches expected structure for Pinecone"""
        
        email_data = {
            "id": "msg_metadata",
            "threadId": "thread_metadata",
            "labelIds": ["INBOX", "IMPORTANT"],
            "snippet": "Test",
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "test@example.com"},
                    {"name": "To", "value": "user@example.com"},
                    {"name": "Subject", "value": "Metadata Test"},
                ],
                "body": {"data": base64.urlsafe_b64encode(b"Test body").decode()}
            }
        }
        
        result = self.processor.parse_email(email_data, self.test_user_id)
        metadata = result["metadata"]
        
        # Verify required fields for Pinecone
        required_fields = [
            "type", "email_id", "thread_id", "user_id", "subject",
            "from", "to", "date", "date_timestamp", "labels",
            "source", "doc_type", "hash"
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"
        
        # Verify field types
        assert metadata["type"] == "email"
        assert metadata["source"] == "gmail"
        assert metadata["doc_type"] == "email"
        assert isinstance(metadata["date_timestamp"], int)
        assert isinstance(metadata["has_attachments"], bool)
        assert isinstance(metadata["body_length"], int)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

