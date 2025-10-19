"""
Semantic Email Chunker

Intelligent content chunking that preserves semantic coherence and extracts
meaningful sections for better vector representation and search.

Features:
- Section-aware chunking (questions, deadlines, decisions, requests)
- Context preservation
- Semantic coherence maintenance
- Adaptive chunk sizing based on content type
"""

import re
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContentSection:
    """Represents a semantically meaningful section of email content"""
    type: str  # 'question', 'deadline', 'decision', 'request', 'content'
    content: str
    context: str
    importance: float  # 0.0 to 1.0
    start_pos: int
    end_pos: int


@dataclass
class SemanticChunk:
    """Represents a semantically coherent chunk of content"""
    text: str
    chunk_type: str  # 'question', 'deadline', 'decision', 'content', 'summary'
    context: str
    entities: List[str]
    importance: float
    metadata: Dict


class SemanticEmailChunker:
    """
    Intelligent email chunker that creates semantically coherent chunks
    while preserving important information and context.
    """

    def __init__(self, max_chunk_size: int = 800, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Patterns for extracting meaningful sections
        self.section_patterns = {
            'question': [
                r'[.!?]*\?[^.!?]*',  # Questions
                r'(?:can you|could you|would you|will you|should we|do we need to)\s+[^.!?]*',
                r'(?:what|when|where|why|how|who)\s+[^.!?]*',
                r'(?:is|are|was|were|does|did|will|would|should|could|can)\s+[^.!?]*\?'
            ],
            'deadline': [
                r'(?:by|due|deadline|before|until)\s+[^.!?]*',
                r'(?:need|required|must|should)\s+[^.!?]*(?:by|before|until)\s+[^.!?]*',
                r'(?:schedule|meeting|call|review)\s+[^.!?]*(?:for|on|at)\s+[^.!?]*'
            ],
            'decision': [
                r'(?:we decided|agreed|concluded|determined|chose|selected)\s+[^.!?]*',
                r'(?:decision|conclusion|agreement|resolution)\s+[^.!?]*',
                r'(?:will|should|need to|must|going to)\s+[^.!?]*'
            ],
            'request': [
                r'(?:please|kindly|request|ask)\s+[^.!?]*',
                r'(?:could you please|would you mind|can you help)\s+[^.!?]*',
                r'(?:need|require|looking for|seeking)\s+[^.!?]*'
            ],
            'action_item': [
                r'(?:action item|todo|task|follow up|next step)\s*:?\s*[^.!?]*',
                r'(?:assign|delegate|responsible|owner)\s+[^.!?]*',
                r'(?:track|monitor|check|verify|confirm)\s+[^.!?]*'
            ]
        }

    def chunk_email(self, email_body: str, subject: str = "") -> List[SemanticChunk]:
        """
        Create semantically coherent chunks from email content.
        
        Args:
            email_body: The main email content
            subject: Email subject for context
            
        Returns:
            List of SemanticChunk objects
        """
        if not email_body or not email_body.strip():
            return []

        logger.info(f"Chunking email with {len(email_body)} characters")

        # Step 1: Extract meaningful sections
        sections = self._extract_sections(email_body)
        
        # Step 2: Create semantic chunks from sections
        chunks = self._create_semantic_chunks(sections, subject, email_body)
        
        # Step 3: Ensure we have reasonable chunk sizes
        chunks = self._optimize_chunk_sizes(chunks)
        
        # Step 4: Add context and metadata
        chunks = self._enrich_chunks(chunks, subject)
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks

    def _extract_sections(self, text: str) -> List[ContentSection]:
        """Extract semantically meaningful sections from text"""
        sections = []
        
        # Find all section matches with their positions
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                    content = match.group(0).strip()
                    if len(content) < 10:  # Skip very short matches
                        continue
                    
                    # Get surrounding context
                    context = self._get_context(text, match.start(), match.end())
                    
                    # Calculate importance based on section type
                    importance = self._calculate_section_importance(section_type, content)
                    
                    section = ContentSection(
                        type=section_type,
                        content=content,
                        context=context,
                        importance=importance,
                        start_pos=match.start(),
                        end_pos=match.end()
                    )
                    sections.append(section)

        # Sort sections by position to maintain order
        sections.sort(key=lambda s: s.start_pos)
        
        # Remove overlapping sections (keep higher importance)
        sections = self._remove_overlapping_sections(sections)
        
        logger.debug(f"Extracted {len(sections)} sections: {[s.type for s in sections]}")
        return sections

    def _get_context(self, text: str, start_pos: int, end_pos: int, context_size: int = 200) -> str:
        """Get surrounding context for a section"""
        context_start = max(0, start_pos - context_size)
        context_end = min(len(text), end_pos + context_size)
        return text[context_start:context_end].strip()

    def _calculate_section_importance(self, section_type: str, content: str) -> float:
        """Calculate importance score for a section"""
        base_importance = {
            'question': 0.9,
            'deadline': 0.95,
            'decision': 0.85,
            'request': 0.8,
            'action_item': 0.9,
            'content': 0.5
        }.get(section_type, 0.5)
        
        # Boost importance for urgent indicators
        urgent_indicators = ['urgent', 'asap', 'immediately', 'critical', 'important', 'priority']
        if any(indicator in content.lower() for indicator in urgent_indicators):
            base_importance = min(1.0, base_importance + 0.2)
        
        return base_importance

    def _remove_overlapping_sections(self, sections: List[ContentSection]) -> List[ContentSection]:
        """Remove overlapping sections, keeping higher importance ones"""
        if not sections:
            return sections
        
        non_overlapping = [sections[0]]
        
        for current in sections[1:]:
            last = non_overlapping[-1]
            
            # Check for overlap
            if current.start_pos < last.end_pos:
                # Overlap detected - keep the more important one
                if current.importance > last.importance:
                    non_overlapping[-1] = current
                # If equal importance, keep the longer one
                elif current.importance == last.importance:
                    if len(current.content) > len(last.content):
                        non_overlapping[-1] = current
            else:
                # No overlap
                non_overlapping.append(current)
        
        return non_overlapping

    def _create_semantic_chunks(self, sections: List[ContentSection], subject: str, full_text: str) -> List[SemanticChunk]:
        """Create semantic chunks from extracted sections"""
        chunks = []
        
        # Create chunks for each section
        for section in sections:
            chunk_text = self._build_chunk_text(section, subject)
            
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            
            # Extract entities from chunk
            entities = self._extract_entities(chunk_text)
            
            chunk = SemanticChunk(
                text=chunk_text,
                chunk_type=section.type,
                context=section.context,
                entities=entities,
                importance=section.importance,
                metadata={
                    'section_type': section.type,
                    'has_entities': len(entities) > 0,
                    'length': len(chunk_text),
                    'importance': section.importance
                }
            )
            chunks.append(chunk)
        
        # If no sections found, use fallback content chunking
        if not chunks and full_text:
            chunks = self._create_content_chunks(full_text, subject)
        
        return chunks

    def _build_chunk_text(self, section: ContentSection, subject: str) -> str:
        """Build chunk text with proper context"""
        # Add subject for context if not already included
        if subject and subject.lower() not in section.content.lower():
            chunk_text = f"{subject}\n\n{section.content}"
        else:
            chunk_text = section.content
        
        # Add context if it provides additional value
        if section.context and len(section.context) > len(section.content) * 1.5:
            # Only add context if it's substantially longer (more information)
            chunk_text = f"{chunk_text}\n\nContext: {section.context}"
        
        return chunk_text.strip()

    def _extract_entities(self, text: str) -> List[str]:
        """Extract basic entities from text"""
        entities = []
        
        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        entities.extend(emails)
        
        # Extract URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        entities.extend(urls)
        
        # Extract phone numbers
        phones = re.findall(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
        entities.extend(phones)
        
        # Extract money amounts
        money = re.findall(r'\$[\d,]+\.?\d*', text)
        entities.extend(money)
        
        # Extract percentages
        percentages = re.findall(r'\d+%', text)
        entities.extend(percentages)
        
        # Extract dates (basic patterns)
        dates = re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', text, re.IGNORECASE)
        entities.extend(dates)
        
        return list(set(entities))  # Remove duplicates

    def _create_content_chunks(self, text: str, subject: str) -> List[SemanticChunk]:
        """Create content chunks when no meaningful sections are found"""
        # Fallback to paragraph-based chunking
        chunks = []
        
        # Split by paragraphs or double newlines
        paragraphs = re.split(r'\n\n+', text)
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 20:
                continue
            
            # Check if adding this paragraph would exceed max size
            if len(current_chunk + para) > self.max_chunk_size and current_chunk:
                # Create chunk from current content
                chunk_text = f"{subject}\n\n{current_chunk}" if subject else current_chunk
                chunks.append(SemanticChunk(
                    text=chunk_text,
                    chunk_type='content',
                    context='',
                    entities=self._extract_entities(chunk_text),
                    importance=0.5,
                    metadata={'section_type': 'content', 'length': len(chunk_text)}
                ))
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add remaining content
        if current_chunk.strip():
            chunk_text = f"{subject}\n\n{current_chunk}" if subject else current_chunk
            chunks.append(SemanticChunk(
                text=chunk_text,
                chunk_type='content',
                context='',
                entities=self._extract_entities(chunk_text),
                importance=0.5,
                metadata={'section_type': 'content', 'length': len(chunk_text)}
            ))
        
        return chunks

    def _optimize_chunk_sizes(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Optimize chunk sizes to be within acceptable range"""
        optimized = []
        
        for chunk in chunks:
            if len(chunk.text) <= self.max_chunk_size:
                optimized.append(chunk)
            else:
                # Split large chunks while preserving semantic coherence
                split_chunks = self._split_large_chunk(chunk)
                optimized.extend(split_chunks)
        
        return optimized

    def _split_large_chunk(self, chunk: SemanticChunk) -> List[SemanticChunk]:
        """Split chunks that are too large"""
        if len(chunk.text) <= self.max_chunk_size:
            return [chunk]
        
        # Split by sentences or paragraphs
        sentences = re.split(r'[.!?]+\s+', chunk.text)
        
        split_chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) > self.max_chunk_size and current_chunk:
                # Create chunk from current content
                split_chunks.append(SemanticChunk(
                    text=current_chunk.strip(),
                    chunk_type=chunk.chunk_type,
                    context=chunk.context,
                    entities=chunk.entities,
                    importance=chunk.importance,
                    metadata=chunk.metadata.copy()
                ))
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add remaining content
        if current_chunk.strip():
            split_chunks.append(SemanticChunk(
                text=current_chunk.strip(),
                chunk_type=chunk.chunk_type,
                context=chunk.context,
                entities=chunk.entities,
                importance=chunk.importance,
                metadata=chunk.metadata.copy()
            ))
        
        return split_chunks

    def _enrich_chunks(self, chunks: List[SemanticChunk], subject: str) -> List[SemanticChunk]:
        """Add additional metadata and context to chunks"""
        for chunk in chunks:
            # Add semantic tags based on content
            chunk.metadata['semantic_tags'] = self._generate_semantic_tags(chunk.text)
            
            # Add content type indicators
            chunk.metadata['content_indicators'] = self._identify_content_indicators(chunk.text)
            
            # Add subject context
            chunk.metadata['has_subject_context'] = subject.lower() in chunk.text.lower()
        
        return chunks

    def _generate_semantic_tags(self, text: str) -> List[str]:
        """Generate semantic tags based on content"""
        tags = []
        text_lower = text.lower()
        
        # Content type tags
        if '?' in text:
            tags.append('question')
        if any(word in text_lower for word in ['deadline', 'due', 'by']):
            tags.append('deadline')
        if any(word in text_lower for word in ['meeting', 'call', 'schedule']):
            tags.append('meeting')
        if any(word in text_lower for word in ['urgent', 'asap', 'immediately']):
            tags.append('urgent')
        if any(word in text_lower for word in ['please', 'request', 'ask']):
            tags.append('request')
        if any(word in text_lower for word in ['decision', 'agree', 'decided']):
            tags.append('decision')
        
        # Entity type tags
        if '@' in text:
            tags.append('contact_info')
        if '$' in text:
            tags.append('financial')
        if '%' in text:
            tags.append('percentage')
        if 'http' in text_lower:
            tags.append('link')
        
        return tags

    def _identify_content_indicators(self, text: str) -> Dict[str, bool]:
        """Identify various content indicators"""
        text_lower = text.lower()
        
        return {
            'has_question': '?' in text,
            'has_deadline': any(word in text_lower for word in ['deadline', 'due', 'by', 'before']),
            'has_action_item': any(word in text_lower for word in ['action', 'todo', 'task', 'follow up']),
            'has_decision': any(word in text_lower for word in ['decided', 'agree', 'conclude', 'determine']),
            'has_request': any(word in text_lower for word in ['please', 'request', 'ask', 'need']),
            'has_contact': '@' in text or any(word in text_lower for word in ['call', 'email', 'contact']),
            'has_financial': '$' in text or any(word in text_lower for word in ['budget', 'cost', 'price', 'money']),
            'has_technical': any(word in text_lower for word in ['technical', 'code', 'system', 'api', 'database'])
        }
