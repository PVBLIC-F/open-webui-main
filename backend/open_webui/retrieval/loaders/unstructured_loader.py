"""
Unstructured.io Unified Loader for Open WebUI

This module provides a unified file processing solution using Unstructured.io
that replaces multiple extraction engines with a single, high-quality solution.

Features:
- Comprehensive file type support (20+ formats)
- Built-in text cleaning and normalization
- Semantic chunking that preserves context
- Consistent metadata extraction
- Performance optimizations
"""

from typing import List, Optional, Dict, Any
import logging
import os
from pathlib import Path

from langchain_core.documents import Document
from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements
from unstructured.cleaners.core import (
    clean_extra_whitespace,
    clean_dashes,
    clean_bullets,
    clean_ordered_bullets,
    clean_non_ascii_chars,
    clean_trailing_punctuation,
)

log = logging.getLogger(__name__)


class UnstructuredUnifiedLoader:
    """
    Unified file loader using Unstructured.io for all file types.
    
    This loader provides:
    1. Consistent text extraction across all file types
    2. Professional-grade text cleaning
    3. Semantic chunking that preserves document structure
    4. Rich metadata extraction
    5. Performance optimizations
    """
    
    def __init__(
        self,
        file_path: str,
        strategy: str = "fast",  # Changed from "hi_res" for better performance
        include_metadata: bool = True,
        clean_text: bool = True,
        chunk_by_semantic: bool = True,
        max_characters: int = 1000,
        chunk_overlap: int = 200,
        cleaning_level: str = "minimal",  # Changed from "standard" for better performance
        **kwargs
    ):
        self.file_path = file_path
        self.strategy = strategy
        self.include_metadata = include_metadata
        self.clean_text = clean_text
        self.chunk_by_semantic = chunk_by_semantic
        self.max_characters = max_characters
        self.chunk_overlap = chunk_overlap
        self.cleaning_level = cleaning_level
        self.kwargs = kwargs
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    def load(self) -> List[Document]:
        """
        Load and process the file using Unstructured.io
        
        Returns:
            List[Document]: Processed documents with cleaned text and metadata
        """
        try:
            log.info(f"Processing file with Unstructured.io: {self.file_path}")
            
            # Step 1: Partition the document (extract structured elements)
            elements = self._partition_document()
            
            # Step 2: Clean the extracted text
            if self.clean_text:
                elements = self._clean_elements(elements)
            
            # Step 3: Chunk by semantic boundaries
            if self.chunk_by_semantic:
                chunks = self._chunk_semantically(elements)
            else:
                chunks = elements
            
            # Step 4: Convert to LangChain Documents
            documents = self._convert_to_documents(chunks)
            
            log.info(f"Successfully processed {len(documents)} chunks from {self.file_path}")
            return documents
            
        except Exception as e:
            log.error(f"Error processing file {self.file_path} with Unstructured: {e}")
            raise
    
    def _partition_document(self):
        """Partition the document using Unstructured.io"""
        try:
            # Determine file type for optimal strategy
            file_ext = Path(self.file_path).suffix.lower()
            
            # Use appropriate strategy based on file type
            strategy = self._get_optimal_strategy(file_ext)
            
            log.debug(f"Using strategy '{strategy}' for file type '{file_ext}'")
            
            # Partition the document with performance optimizations
            elements = partition(
                filename=self.file_path,
                strategy=strategy,
                include_metadata=self.include_metadata,
                # Add performance optimizations for faster processing
                infer_table_structure=False,  # Disable table structure inference for speed
                extract_images_in_pdf=False,  # Disable image extraction for speed
                **self.kwargs
            )
            
            # Log processing results
            log.info(f"Successfully partitioned document: {len(elements)} elements extracted")
            
            return elements
            
        except Exception as e:
            log.error(f"Error partitioning document {self.file_path}: {e}")
            
            # If the error is related to complex processing, try a simpler strategy
            if "list index out of range" in str(e).lower() or "timeout" in str(e).lower():
                log.warning(f"Complex processing failed, trying 'fast' strategy: {e}")
                try:
                    elements = partition(
                        filename=self.file_path,
                        strategy="fast",
                        include_metadata=self.include_metadata,
                        infer_table_structure=False,
                        extract_images_in_pdf=False,
                        **self.kwargs
                    )
                    log.info(f"Fallback strategy successful: {len(elements)} elements extracted")
                    return elements
                except Exception as fallback_error:
                    log.error(f"Fallback strategy also failed: {fallback_error}")
            
            raise
    
    def _get_optimal_strategy(self, file_ext: str) -> str:
        """
        Select optimal processing strategy based on file type and size.
        Prioritizes speed over quality for better user experience.
        """
        file_size = os.path.getsize(self.file_path)
        
        # PDFs: Use fast strategy for better performance
        # hi_res is very slow and resource intensive
        if file_ext == ".pdf":
            # Only use hi_res for small PDFs if explicitly set in self.strategy
            if self.strategy == "hi_res" and file_size < 5_000_000:  # < 5MB
                return "hi_res"
            return "fast"
        
        # Office documents work well with fast strategy
        elif file_ext in [".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls"]:
            return "fast"
        
        # Text-based formats don't need OCR
        elif file_ext in [".txt", ".md", ".html", ".csv", ".json", ".xml"]:
            return "fast"
        
        # Images need OCR
        elif file_ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            return "ocr_only"
        
        # Default to fast for unknown types (faster than "auto")
        return "fast"
    
    def _clean_elements(self, elements):
        """Apply comprehensive text cleaning to elements"""
        # Skip cleaning entirely if level is "none" for maximum speed
        if self.cleaning_level == "none":
            return elements
            
        cleaned_elements = []
        
        for i, element in enumerate(elements):
            try:
                # Get original text safely - handle different element types
                text = None
                if hasattr(element, 'text'):
                    text = element.text if element.text else None
                
                # If no text attribute or it's None, try converting to string
                if text is None:
                    try:
                        text = str(element)
                    except Exception:
                        log.debug(f"Element {i} has no text content, skipping")
                        continue
                
                # Skip empty or very short text
                if not text or len(text.strip()) < 2:
                    continue
                
                # Apply cleaning based on level
                try:
                    if self.cleaning_level == "minimal":
                        text = self._minimal_cleaning(text)
                    elif self.cleaning_level == "standard":
                        text = self._standard_cleaning(text)
                    elif self.cleaning_level == "aggressive":
                        text = self._aggressive_cleaning(text)
                except Exception as clean_err:
                    log.warning(f"Error in cleaning function for element {i}: {clean_err}")
                    # Continue with uncleaned text rather than failing
                
                # Update element text if it has a text attribute
                if hasattr(element, 'text'):
                    element.text = text
                cleaned_elements.append(element)
                
            except Exception as e:
                log.warning(f"Error processing element {i}: {e}")
                # Keep original element if any processing fails
                try:
                    cleaned_elements.append(element)
                except Exception:
                    pass  # Skip this element entirely if we can't add it
        
        log.debug(f"Cleaned {len(cleaned_elements)} elements from {len(elements)} total")
        return cleaned_elements
    
    def _minimal_cleaning(self, text: str) -> str:
        """Minimal cleaning - just essential whitespace fixes"""
        text = clean_extra_whitespace(text)
        text = clean_trailing_punctuation(text)
        return text.strip()
    
    def _standard_cleaning(self, text: str) -> str:
        """Standard cleaning - good balance of cleaning and preservation"""
        text = clean_extra_whitespace(text)
        text = clean_dashes(text)
        text = clean_bullets(text)
        text = clean_ordered_bullets(text)
        text = clean_trailing_punctuation(text)
        return text.strip()
    
    def _aggressive_cleaning(self, text: str) -> str:
        """Aggressive cleaning - maximum cleaning, may remove some context"""
        text = clean_extra_whitespace(text)
        text = clean_dashes(text)
        text = clean_bullets(text)
        text = clean_ordered_bullets(text)
        text = clean_trailing_punctuation(text)
        text = clean_non_ascii_chars(text)
        return text.strip()
    
    def _chunk_semantically(self, elements):
        """
        Chunk elements using basic character-based chunking.
        Uses chunk_elements for better performance than semantic chunking.
        """
        try:
            chunks = chunk_elements(
                elements,
                max_characters=self.max_characters,
                overlap=self.chunk_overlap,
                new_after_n_chars=self.max_characters,
            )
            
            log.debug(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            log.warning(f"Error in chunking: {e}")
            # If chunking fails, return elements as-is
            log.info(f"Returning {len(elements)} elements without chunking")
            return elements
    
    def _convert_to_documents(self, chunks) -> List[Document]:
        """Convert Unstructured chunks to LangChain Documents"""
        documents = []
        total_chunks = len(chunks)  # Pre-calculate to avoid repeated len() calls
        
        for i, chunk in enumerate(chunks):
            try:
                # Extract content
                content = str(chunk)
                
                # Skip empty content
                if not content or not content.strip():
                    continue
                
                # Extract metadata
                metadata = {}
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    metadata = chunk.metadata.to_dict()
                
                # Add additional metadata (optimized for performance)
                metadata.update({
                    "source": self.file_path,
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "processing_engine": "unstructured",
                    "strategy": self.strategy,
                    "cleaning_level": self.cleaning_level,
                })
                
                # Remove large/unnecessary fields to improve performance
                fields_to_remove = [
                    "orig_elements",  # Very large compressed JSON, not needed for retrieval
                    "name",           # Redundant with filename
                    "source",         # Redundant with filename (we set our own source)
                ]
                for field in fields_to_remove:
                    if field in metadata:
                        del metadata[field]
                
                # Create document
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
                
            except Exception as e:
                log.warning(f"Error converting chunk {i} to document: {e}")
                continue
        
        return documents


def create_unstructured_loader(
    file_path: str,
    config: Dict[str, Any]
) -> UnstructuredUnifiedLoader:
    """
    Factory function to create UnstructuredUnifiedLoader with configuration
    
    Args:
        file_path: Path to the file to process
        config: Configuration dictionary with Unstructured settings
        
    Returns:
        UnstructuredUnifiedLoader instance
    """
    return UnstructuredUnifiedLoader(
        file_path=file_path,
        strategy=config.get("UNSTRUCTURED_STRATEGY", "fast"),  # Default to fast for performance
        include_metadata=config.get("UNSTRUCTURED_INCLUDE_METADATA", True),
        clean_text=config.get("UNSTRUCTURED_CLEAN_TEXT", True),
        chunk_by_semantic=config.get("UNSTRUCTURED_SEMANTIC_CHUNKING", True),
        max_characters=config.get("CHUNK_SIZE", 1000),
        chunk_overlap=config.get("CHUNK_OVERLAP", 200),
        cleaning_level=config.get("UNSTRUCTURED_CLEANING_LEVEL", "minimal"),  # Default to minimal for performance
    )
