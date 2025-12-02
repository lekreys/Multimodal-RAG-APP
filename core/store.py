import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.logger_config import get_logger
import time

logger = get_logger(__name__)

@dataclass
class ChromaConfig:
    openai_api_key: str
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY")
    text_collection_name: str = os.getenv("TEXT_COLLECTION_NAME")
    image_collection_name: str = os.getenv("IMAGE_COLLECTION_NAME")
    table_collection_name: str = os.getenv("TABLE_COLLECTION_NAME")
    embedding_model: str = os.getenv("EMBEDDING_MODEL_NAME")


class MultimodalChromaStore:
    
    def __init__(self, config: ChromaConfig):
        self.config = config
        
        logger.info("Initializing ChromaDB store...")
        
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_api_key
        )
        
        self.text_store = Chroma(
            collection_name=config.text_collection_name,
            embedding_function=self.embeddings,
            persist_directory=config.chroma_persist_directory
        )
        
        self.image_store = Chroma(
            collection_name=config.image_collection_name,
            embedding_function=self.embeddings,  
            persist_directory=config.chroma_persist_directory
        )
        
        self.table_store = Chroma(
            collection_name=config.table_collection_name,
            embedding_function=self.embeddings,
            persist_directory=config.chroma_persist_directory
        )
        
        logger.info(f"ChromaDB initialized at: {config.chroma_persist_directory}")
        logger.info(f"Embedding model: {config.embedding_model}")
        logger.debug(f"Collections: text={config.text_collection_name}, images={config.image_collection_name}, tables={config.table_collection_name}")
    
    
    def store_text_chunks(self, text_chunks: List[Dict[str, Any]], source_metadata: Dict[str, Any] = None) -> List[str]:

        if not text_chunks:
            logger.warning("No text chunks to store")
            return []
        
        logger.info(f"Storing {len(text_chunks)} text chunks...")
        
        documents = []
        
        for chunk in text_chunks:
            content = chunk.get("content", "")
            
            if not content.strip():
                logger.debug(f"Skipping empty chunk: {chunk.get('id', 'unknown')}")
                continue
            
            metadata = {
                "type": "text",
                "chunk_id": chunk.get("id", ""),
            }
            
            if source_metadata:
                metadata.update(source_metadata)
            
            if "source_pdf_url" in chunk:
                metadata["source_pdf_url"] = chunk["source_pdf_url"]
            if "source_pdf_path" in chunk:
                metadata["source_pdf_path"] = chunk["source_pdf_path"]
            
            if "metadata" in chunk:
                chunk_meta = chunk["metadata"]
                for key, value in chunk_meta.items():
                    if value is not None:
                        if isinstance(value, (str, int, float, bool)):
                            metadata[f"chunk_{key}"] = value
                        elif isinstance(value, (list, dict)):
                            metadata[f"chunk_{key}"] = str(value)
            
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        if documents:
            ids = self.text_store.add_documents(documents)
            logger.info(f"Stored {len(ids)} text chunks successfully")
            return ids
        else:
            logger.warning("No valid text chunks to store after filtering")
        
        return []
    
    
    def store_images(self, images: List[Dict[str, Any]], source_metadata: Dict[str, Any] = None) -> List[str]:
       
        if not images:
            logger.warning("No images to store")
            return []
        
        logger.info(f"Storing {len(images)} images...")
        
        documents = []
        ai_description_count = 0
        
        for img in images:
            content = img.get("content", "")
            
            if not content.strip():
                page = img.get("metadata", {}).get("page_number", "unknown")
                content = f"Image from page {page}"
                logger.debug(f"Using fallback description for image: {img.get('id', 'unknown')}")
            
            if img.get("ai_generated_description"):
                ai_description_count += 1
            
            metadata = {
                "type": "image",
                "image_id": img.get("id", ""),
                "supabase_url": img.get("supabase_url", ""),
                "storage_path": img.get("storage_path", ""),
                "ai_generated_description": img.get("ai_generated_description", False),
            }
            
            if source_metadata:
                metadata.update(source_metadata)
            
            if "source_pdf_url" in img:
                metadata["source_pdf_url"] = img["source_pdf_url"]
            if "source_pdf_path" in img:
                metadata["source_pdf_path"] = img["source_pdf_path"]
            
            if "metadata" in img:
                img_meta = img["metadata"]
                for key, value in img_meta.items():
                    if value is not None:
                        if isinstance(value, (str, int, float, bool)):
                            metadata[f"img_{key}"] = value
                        elif isinstance(value, (list, dict)):
                            metadata[f"img_{key}"] = str(value)
        
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        if documents:
            ids = self.image_store.add_documents(documents)
            logger.info(f"Stored {len(ids)} images successfully")
            if ai_description_count > 0:
                logger.info(f"  {ai_description_count}/{len(ids)} images have AI-generated descriptions")
            return ids
        
        return []
    
    
    def store_tables(self, tables: List[Dict[str, Any]], source_metadata: Dict[str, Any] = None) -> List[str]:
  
        if not tables:
            logger.warning("No tables to store")
            return []
        
        logger.info(f"Storing {len(tables)} tables...")
        
        documents = []
        html_count = 0
        
        for table in tables:
            content = table.get("table_html") or table.get("table_text", "")
            
            if not content.strip():
                logger.debug(f"Skipping empty table: {table.get('id', 'unknown')}")
                continue
            
            has_html = bool(table.get("table_html"))
            if has_html:
                html_count += 1
            
            metadata = {
                "type": "table",
                "table_id": table.get("id", ""),
                "has_html": has_html,
            }
            
            if source_metadata:
                metadata.update(source_metadata)
            
            if "source_pdf_url" in table:
                metadata["source_pdf_url"] = table["source_pdf_url"]
            if "source_pdf_path" in table:
                metadata["source_pdf_path"] = table["source_pdf_path"]
            
            if "metadata" in table:
                table_meta = table["metadata"]
                for key, value in table_meta.items():
                    if value is not None:
                        if isinstance(value, (str, int, float, bool)):
                            metadata[f"table_{key}"] = value
                        elif isinstance(value, (list, dict)):
                            metadata[f"table_{key}"] = str(value)
            
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        if documents:
            ids = self.table_store.add_documents(documents)
            logger.info(f"Stored {len(ids)} tables successfully")
            if html_count > 0:
                logger.info(f"  {html_count}/{len(ids)} tables have HTML format")
            return ids
        
        return []
    
    
    def get_stats(self) -> Dict[str, int]:
        logger.debug("Retrieving database statistics...")
        
        try:
            text_count = len(self.text_store.get()['ids']) if self.text_store.get() else 0
        except Exception as e:
            logger.error(f"Error getting text count: {str(e)}")
            text_count = 0
        
        try:
            image_count = len(self.image_store.get()['ids']) if self.image_store.get() else 0
        except Exception as e:
            logger.error(f"Error getting image count: {str(e)}")
            image_count = 0
        
        try:
            table_count = len(self.table_store.get()['ids']) if self.table_store.get() else 0
        except Exception as e:
            logger.error(f"Error getting table count: {str(e)}")
            table_count = 0
        
        stats = {
            "text_count": text_count,
            "image_count": image_count,
            "table_count": table_count,
            "total": text_count + image_count + table_count
        }
        
        logger.debug(f"Database stats: {stats}")
        return stats


def store_to_chroma(extraction_result: Dict[str, Any], config: ChromaConfig, store_text: bool = True, store_images: bool = True, store_tables: bool = True) -> Dict[str, Any]:
   
    start_time = time.time()
    
    logger.info("="*80)
    logger.info("Starting storage to ChromaDB...")
    logger.info(f"Store text: {store_text}, Store images: {store_images}, Store tables: {store_tables}")
    logger.info("="*80)
    
    store = MultimodalChromaStore(config)
    
    source_metadata = {
        "source_file": extraction_result.get("metadata", {}).get("source", "unknown"),
        "total_elements": extraction_result.get("metadata", {}).get("total_elements", 0),
        "image_bucket": extraction_result.get("metadata", {}).get("image_bucket", ""),
        "document_bucket": extraction_result.get("metadata", {}).get("document_bucket", ""),
    }
    
    if "pdf_url" in extraction_result:
        source_metadata["pdf_url"] = extraction_result["pdf_url"]
        source_metadata["pdf_storage_path"] = extraction_result.get("pdf_storage_path", "")
        logger.debug(f"Source PDF URL: {extraction_result['pdf_url'][:80]}...")
    
    if "metadata" in extraction_result and "source_pdf" in extraction_result["metadata"]:
        pdf_info = extraction_result["metadata"]["source_pdf"]
        source_metadata["pdf_original_filename"] = pdf_info.get("original_filename", "")
        source_metadata["pdf_file_size"] = pdf_info.get("file_size", 0)
        logger.debug(f"PDF metadata: {pdf_info.get('original_filename', 'unknown')} ({pdf_info.get('file_size', 0)} bytes)")
    
    result = {
        "text_ids": [],
        "image_ids": [],
        "table_ids": [],
        "pdf_info": {}
    }
    
    if "pdf_url" in extraction_result:
        result["pdf_info"] = {
            "url": extraction_result["pdf_url"],
            "storage_path": extraction_result.get("pdf_storage_path", ""),
            "bucket": extraction_result.get("metadata", {}).get("document_bucket", "")
        }
    
    if store_text:
        text_data = (
            extraction_result.get("text_chunks_semantic") or 
            extraction_result.get("text_chunks", [])
        )
        if text_data:
            logger.info(f"Processing {len(text_data)} text chunks...")
            result["text_ids"] = store.store_text_chunks(text_data, source_metadata)
        else:
            logger.warning("No text chunks found in extraction result")
    
    if store_images:
        images = extraction_result.get("images", [])
        if images:
            logger.info(f"Processing {len(images)} images...")
            result["image_ids"] = store.store_images(images, source_metadata)
        else:
            logger.warning("No images found in extraction result")
    
    if store_tables:
        tables = extraction_result.get("tables", [])
        if tables:
            logger.info(f"Processing {len(tables)} tables...")
            result["table_ids"] = store.store_tables(tables, source_metadata)
        else:
            logger.warning("No tables found in extraction result")
    
    duration = time.time() - start_time
    
    logger.info("="*80)
    logger.info("Storage Summary:")
    logger.info(f"  Text chunks: {len(result['text_ids'])}")
    logger.info(f"  Images: {len(result['image_ids'])}")
    logger.info(f"  Tables: {len(result['table_ids'])}")
    logger.info(f"  Total stored: {len(result['text_ids']) + len(result['image_ids']) + len(result['table_ids'])}")
    logger.info(f"  Storage time: {duration:.2f}s")

    if result["pdf_info"]:
        logger.info("Source PDF:")
        logger.info(f"  URL: {result['pdf_info']['url'][:80]}...")
        logger.info(f"  Bucket: {result['pdf_info']['bucket']}")
    
    stats = store.get_stats()
    logger.info("Database Statistics:")
    logger.info(f"  Text collection: {stats['text_count']} documents")
    logger.info(f"  Image collection: {stats['image_count']} documents")
    logger.info(f"  Table collection: {stats['table_count']} documents")
    logger.info(f"  Total in database: {stats['total']} documents")
    logger.info("="*80)
    
    result["stats"] = stats
    
    return result