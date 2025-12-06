
import os
import io
import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd
import numpy as np
#import PymuPDF
import fitz 
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import hashlib
from datetime import datetime

@dataclass
class RawDocument:
    """Represents a raw, messy document from various sources"""
    id: str
    content: bytes
    filename: str
    file_type: str
    metadata: Dict[str, Any]
    upload_timestamp: datetime
    
class DocumentIngestor:
    """Handles ingestion of messy real-world documents"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def ingest_document(self, file_path: str) -> RawDocument:
        """Ingest document from various formats with error handling"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Generating unique ID
            file_hash = hashlib.sha256(content).hexdigest()
            doc_id = f"doc_{file_hash[:16]}"
            
            # Extracting metadata
            metadata = self._extract_metadata(file_path, content)
            
            # Processing based on file type
            processed_content = self._process_content(content, metadata['file_type'])
            
            return RawDocument(
                id=doc_id,
                content=processed_content,
                filename=os.path.basename(file_path),
                file_type=metadata['file_type'],
                metadata=metadata,
                upload_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to ingest {file_path}: {str(e)}")
            raise
    
    def _extract_metadata(self, file_path: str, content: bytes) -> Dict:
        """Extract metadata from messy files"""
        file_ext = file_path.split('.')[-1].lower()
        file_type = file_ext if file_ext in ['pdf', 'jpg', 'jpeg', 'png'] else 'unknown'
        if file_type in ['jpg', 'jpeg', 'png']:
            file_type = 'image'

        metadata = {
            'size_bytes': len(content),
            'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)),
            'file_type': file_type,
            'has_handwriting': False,
            'is_scanned': False,
            'quality_score': 0.0
        }
        
        # Detecting if scanned/handwritten (in simplified logic)
        if metadata['file_type'] in ['pdf', 'image']:
            metadata.update(self._analyze_document_quality(content))
            
        return metadata

    def _analyze_document_quality(self, content: bytes) -> Dict:
        """Mock quality analysis"""
        return {'quality_score': 0.85, 'is_scanned': True}
    
    def _process_content(self, content: bytes, file_type: str) -> bytes:
        """Process different document types"""
        processors = {
            'pdf': self._process_pdf,
            'image': self._process_image,
            # 'docx': self._process_docx, # Placeholder for possible future expansion
        }
        
        processor = processors.get(file_type, self._process_unknown)
        return processor(content)
    
    def _process_image(self, content: bytes) -> bytes:
        """Handling image OCR"""
        try:
            image = Image.open(io.BytesIO(content))
            text = pytesseract.image_to_string(image)
            return text.encode()
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return b""

    def _process_unknown(self, content: bytes) -> bytes:
        """Fallback for unknown types"""
        return content

    def _process_pdf(self, content: bytes) -> bytes:
        """Handle messy PDFs - scanned, digital and mixed"""
        try:
            # Tring to extract text directly first
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                
                if text_parts:
                    return "\n\n".join(text_parts).encode()
            
            # If no text, then trying for OCR
            doc = fitz.open(stream=content, filetype="pdf")
            ocr_text = []
            for page_num in range(len(doc)):
                pix = doc[page_num].get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
                ocr_text.append(text)
            
            return "\n\n".join(ocr_text).encode()
            
        except Exception as e:
            self.logger.warning(f"PDF processing failed: {str(e)}")
            return content  # Return original if processing fails