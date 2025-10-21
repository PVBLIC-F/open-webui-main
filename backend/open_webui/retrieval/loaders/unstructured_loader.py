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
from unstructured.chunking.title import chunk_by_title
from unstructured.chunking.basic import chunk_elements
from unstructured.cleaners.core import (
    clean_extra_whitespace,
    clean_dashes,
    clean_bullets,
    clean_ordered_bullets,
    clean_non_ascii_chars,
    clean_trailing_punctuation,
    clean,
)
from unstructured.staging.base import elements_to_json

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
        strategy: str = "hi_res",
        include_metadata: bool = True,
        clean_text: bool = True,
        chunk_by_semantic: bool = True,
        max_characters: int = 1000,
        chunk_overlap: int = 200,
        cleaning_level: str = "standard",  # minimal, standard, aggressive
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
            
            # Partition the document
            elements = partition(
                filename=self.file_path,
                strategy=strategy,
                include_metadata=self.include_metadata,
                **self.kwargs
            )
            
            return elements
            
        except Exception as e:
            log.error(f"Error partitioning document {self.file_path}: {e}")
            raise
    
    def _get_optimal_strategy(self, file_ext: str) -> str:
        """
        Select optimal processing strategy based on file type and size
        """
        file_size = os.path.getsize(self.file_path)
        
        # PDFs benefit from hi_res strategy for better layout detection
        if file_ext == ".pdf":
            if file_size > 10_000_000:  # > 10MB
                return "fast"  # Large PDFs should use fast mode
            return "hi_res"
        
        # Office documents work well with fast strategy
        elif file_ext in [".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls"]:
            return "fast"
        
        # Text-based formats don't need OCR
        elif file_ext in [".txt", ".md", ".html", ".csv", ".json"]:
            return "fast"
        
        # Images need OCR
        elif file_ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            return "ocr_only"
        
        # Default to auto for unknown types
        return "auto"
    
    def _clean_elements(self, elements):
        """Apply comprehensive text cleaning to elements"""
        cleaned_elements = []
        
        for element in elements:
            try:
                # Get original text
                text = str(element)
                
                # Apply cleaning based on level
                if self.cleaning_level == "minimal":
                    text = self._minimal_cleaning(text)
                elif self.cleaning_level == "standard":
                    text = self._standard_cleaning(text)
                elif self.cleaning_level == "aggressive":
                    text = self._aggressive_cleaning(text)
                
                # Update element text
                element.text = text
                cleaned_elements.append(element)
                
            except Exception as e:
                log.warning(f"Error cleaning element: {e}")
                cleaned_elements.append(element)  # Keep original if cleaning fails
        
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
        """Chunk elements using semantic boundaries"""
        try:
            # Use title-based chunking for better semantic coherence
            chunks = chunk_by_title(
                elements,
                max_characters=self.max_characters,
                overlap=self.chunk_overlap,
                combine_text_under_n_chars=100,  # Combine very small chunks
            )
            
            log.debug(f"Created {len(chunks)} semantic chunks")
            return chunks
            
        except Exception as e:
            log.warning(f"Error in semantic chunking, falling back to basic chunking: {e}")
            # Fallback to basic chunking
            return chunk_elements(
                elements,
                max_characters=self.max_characters,
                overlap=self.chunk_overlap,
            )
    
    def _convert_to_documents(self, chunks) -> List[Document]:
        """Convert Unstructured chunks to LangChain Documents"""
        documents = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Extract content
                content = str(chunk)
                
                # Extract metadata
                metadata = {}
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    metadata = chunk.metadata.to_dict()
                
                # Add additional metadata
                metadata.update({
                    "source": self.file_path,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "processing_engine": "unstructured",
                    "strategy": self.strategy,
                    "cleaning_level": self.cleaning_level,
                })
                
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
        strategy=config.get("UNSTRUCTURED_STRATEGY", "hi_res"),
        include_metadata=config.get("UNSTRUCTURED_INCLUDE_METADATA", True),
        clean_text=config.get("UNSTRUCTURED_CLEAN_TEXT", True),
        chunk_by_semantic=config.get("UNSTRUCTURED_SEMANTIC_CHUNKING", True),
        max_characters=config.get("CHUNK_SIZE", 1000),
        chunk_overlap=config.get("CHUNK_OVERLAP", 200),
        cleaning_level=config.get("UNSTRUCTURED_CLEANING_LEVEL", "standard"),
    )
