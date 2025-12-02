import os
from typing import List, Dict, Any, Literal, Optional
from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.logger_config import get_logger
import time

logger = get_logger(__name__)

@dataclass
class RetrievalConfig:
    openai_api_key: str
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY")
    text_collection_name: str = os.getenv("TEXT_COLLECTION_NAME")
    image_collection_name: str = os.getenv("IMAGE_COLLECTION_NAME")
    table_collection_name: str = os.getenv("TABLE_COLLECTION_NAME")
    embedding_model: str = os.getenv("EMBEDDING_MODEL_NAME")


class MultimodalRetriever:
    def __init__(self, config: RetrievalConfig):
        self.config = config

        logger.info("Initializing Multimodal Retriever...")
        
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

        logger.info(f"Retriever initialized from: {config.chroma_persist_directory}")
        logger.debug(f"Embedding model: {config.embedding_model}")

    def retrieve_all(self, query: str, k_text: int = 5, k_images: int = 3, k_tables: int = 3, filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, List[Document]]:
        
        logger.info("Retrieving from ALL sources...")
        logger.info(f"Query: '{query}'")
        logger.debug(f"Parameters: k_text={k_text}, k_images={k_images}, k_tables={k_tables}")

        results = {
            "text": [],
            "images": [],
            "tables": []
        }
        
        try:
            if filter_metadata:
                logger.debug(f"Applying metadata filter: {filter_metadata}")
                results["text"] = self.text_store.similarity_search(
                    query, k=k_text, filter=filter_metadata
                )
            else:
                results["text"] = self.text_store.similarity_search(query, k=k_text)
            logger.info(f"Found {len(results['text'])} text chunks")
        except Exception as e:
            logger.error(f"Text retrieval failed: {str(e)}", exc_info=True)
        
        try:
            if filter_metadata:
                results["images"] = self.image_store.similarity_search(
                    query, k=k_images, filter=filter_metadata
                )
            else:
                results["images"] = self.image_store.similarity_search(query, k=k_images)
            logger.info(f"Found {len(results['images'])} images")
        except Exception as e:
            logger.error(f"Image retrieval failed: {str(e)}", exc_info=True)
        
        try:
            if filter_metadata:
                results["tables"] = self.table_store.similarity_search(
                    query, k=k_tables, filter=filter_metadata
                )
            else:
                results["tables"] = self.table_store.similarity_search(query, k=k_tables)
            logger.info(f"Found {len(results['tables'])} tables")
        except Exception as e:
            logger.error(f"Table retrieval failed: {str(e)}", exc_info=True)

        return results


    def retrieve_with_scores(self,query: str,k_text: int = 5,k_images: int = 3,k_tables: int = 3) -> Dict[str, List[tuple[Document, float]]]:
        
        logger.info("Retrieving with similarity scores...")
        logger.debug(f"Query: '{query}'")

        results = {
            "text": [],
            "images": [],
            "tables": []
        }
        
        try:
            results["text"] = self.text_store.similarity_search_with_score(query, k=k_text)
            logger.info(f"Text: {len(results['text'])} results with scores")
        except Exception as e:
            logger.error(f"Text retrieval failed: {str(e)}", exc_info=True)
        
        try:
            results["images"] = self.image_store.similarity_search_with_score(query, k=k_images)
            logger.info(f"Images: {len(results['images'])} results with scores")
        except Exception as e:
            logger.error(f"Image retrieval failed: {str(e)}", exc_info=True)
        
        try:
            results["tables"] = self.table_store.similarity_search_with_score(query, k=k_tables)
            logger.info(f"Tables: {len(results['tables'])} results with scores")
        except Exception as e:
            logger.error(f"Table retrieval failed: {str(e)}", exc_info=True)

        return results

    def retrieve_hybrid_ranked(self,query: str,k: int = 10,text_weight: float = 0.5,image_weight: float = 0.25,table_weight: float = 0.25) -> List[tuple[Document, float, str]]: 
        
        logger.info("Hybrid retrieval with weighted ranking...")
        logger.debug(f"Weights: text={text_weight}, image={image_weight}, table={table_weight}")

        all_results = []
        
        try:
            text_results = self.text_store.similarity_search_with_score(query, k=k)
            for doc, score in text_results:
                weighted_score = score * text_weight
                all_results.append((doc, weighted_score, "text"))
            logger.debug(f"Added {len(text_results)} text results")
        except Exception as e:
            logger.error(f"Text retrieval failed: {str(e)}", exc_info=True)
        
        try:
            image_results = self.image_store.similarity_search_with_score(query, k=k)
            for doc, score in image_results:
                weighted_score = score * image_weight
                all_results.append((doc, weighted_score, "image"))
            logger.debug(f"Added {len(image_results)} image results")
        except Exception as e:
            logger.error(f"Image retrieval failed: {str(e)}", exc_info=True)
        
        try:
            table_results = self.table_store.similarity_search_with_score(query, k=k)
            for doc, score in table_results:
                weighted_score = score * table_weight
                all_results.append((doc, weighted_score, "table"))
            logger.debug(f"Added {len(table_results)} table results")
        except Exception as e:
            logger.error(f"Table retrieval failed: {str(e)}", exc_info=True)

        all_results.sort(key=lambda x: x[1])
        top_results = all_results[:k]

        logger.info(f"Returned top {len(top_results)} ranked results")
        for i, (doc, score, type_) in enumerate(top_results[:5], 1):
            logger.debug(f"  {i}. [{type_.upper()}] score={score:.4f}")

        return top_results


    def retrieve_mmr(self,query: str,k: int = 10,fetch_k: int = 30,lambda_mult: float = 0.5,include_text: bool = True,include_images: bool = True,include_tables: bool = True) -> Dict[str, List[Document]]:
        
        logger.info("MMR retrieval (diversity-focused)...")
        logger.debug(f"Parameters: k={k}, fetch_k={fetch_k}, lambda={lambda_mult}")
        logger.debug(f"Include: text={include_text}, images={include_images}, tables={include_tables}")

        results = {
            "text": [],
            "images": [],
            "tables": []
        }
        
        if include_text:
            try:
                results["text"] = self.text_store.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
                )
                logger.info(f"Text: {len(results['text'])} diverse results")
            except Exception as e:
                logger.error(f"Text MMR failed: {str(e)}", exc_info=True)
        
        if include_images:
            try:
                results["images"] = self.image_store.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
                )
                logger.info(f"Images: {len(results['images'])} diverse results")
            except Exception as e:
                logger.error(f"Image MMR failed: {str(e)}", exc_info=True)
        
        if include_tables:
            try:
                results["tables"] = self.table_store.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
                )
                logger.info(f"Tables: {len(results['tables'])} diverse results")
            except Exception as e:
                logger.error(f"Table MMR failed: {str(e)}", exc_info=True)

        return results


    def retrieve_by_type_only(self,query: str,type_: Literal["text", "image", "table"],k: int = 5) -> List[Document]:
        
        logger.info(f"Retrieving {type_} only...")
        logger.debug(f"Query: '{query}', k={k}")

        if type_ == "text":
            results = self.text_store.similarity_search(query, k=k)
        elif type_ == "image":
            results = self.image_store.similarity_search(query, k=k)
        elif type_ == "table":
            results = self.table_store.similarity_search(query, k=k)
        else:
            logger.error(f"Invalid type: {type_}")
            raise ValueError(f"Invalid type: {type_}. Must be 'text', 'image', or 'table'")

        logger.info(f"Found {len(results)} {type_} results")
        return results


def retrieve_multimodal(query: str,config: RetrievalConfig,method: Literal["all", "hybrid", "mmr", "text_only", "image_only", "table_only"] = "all",k: int = 5,**kwargs) -> Dict[str, Any]:
   
    start_time = time.time()
    
    logger.info("="*80)
    logger.info(f"Starting multimodal retrieval")
    logger.info(f"Query: '{query}'")
    logger.info(f"Method: {method}, k: {k}")
    logger.info("="*80)
    
    retriever = MultimodalRetriever(config)
    
    result = {
        "query": query,
        "method": method,
        "results": None
    }

    if method == "all":
        results = retriever.retrieve_all(
            query,
            k_text=kwargs.get("k_text", k),
            k_images=kwargs.get("k_images", max(2, k // 2)),
            k_tables=kwargs.get("k_tables", max(2, k // 2)),
            filter_metadata=kwargs.get("filter_metadata")
        )
        result["results"] = results
        result["total_results"] = (len(results["text"]) + len(results["images"]) + len(results["tables"]))

    elif method == "hybrid":
        ranked_results = retriever.retrieve_hybrid_ranked(
            query,
            k=k,
            text_weight=kwargs.get("text_weight", 0.5),
            image_weight=kwargs.get("image_weight", 0.25),
            table_weight=kwargs.get("table_weight", 0.25)
        )

        results = {"text": [], "images": [], "tables": []}
        for doc, score, type_ in ranked_results:
            if type_ == "text":
                results["text"].append(doc)
            elif type_ == "image":
                results["images"].append(doc)
            elif type_ == "table":
                results["tables"].append(doc)

        result["results"] = results
        result["ranked_results"] = ranked_results  
        result["total_results"] = len(ranked_results)

    elif method == "mmr":
        results = retriever.retrieve_mmr(
            query,
            k=k,
            fetch_k=kwargs.get("fetch_k", k * 3),
            lambda_mult=kwargs.get("lambda_mult", 0.5),
            include_text=kwargs.get("include_text", True),
            include_images=kwargs.get("include_images", True),
            include_tables=kwargs.get("include_tables", True)
        )
        result["results"] = results
        result["total_results"] = (
            len(results["text"]) +
            len(results["images"]) +
            len(results["tables"])
        )
        
    elif method == "text_only":
        docs = retriever.retrieve_by_type_only(query, "text", k=k)
        result["results"] = {"text": docs, "images": [], "tables": []}
        result["total_results"] = len(docs)

    elif method == "image_only":
        docs = retriever.retrieve_by_type_only(query, "image", k=k)
        result["results"] = {"text": [], "images": docs, "tables": []}
        result["total_results"] = len(docs)

    elif method == "table_only":
        docs = retriever.retrieve_by_type_only(query, "table", k=k)
        result["results"] = {"text": [], "images": [], "tables": docs}
        result["total_results"] = len(docs)
    else:
        logger.error(f"Unknown retrieval method: {method}")
        raise ValueError(f"Unknown method: {method}")

    duration = time.time() - start_time
    
    logger.info("="*80)
    logger.info("Retrieval Summary:")
    logger.info(f"  Method: {method}")
    logger.info(f"  Total results: {result['total_results']}")
    if result.get("results"):
        logger.info(f"  Text: {len(result['results'].get('text', []))}")
        logger.info(f"  Images: {len(result['results'].get('images', []))}")
        logger.info(f"  Tables: {len(result['results'].get('tables', []))}")
    logger.info(f"  Retrieval time: {duration:.2f}s")
    logger.info("="*80)

    return result


def format_result_with_sources(result: Dict[str, Any]) -> str:
    
    logger.debug("Formatting retrieval results with sources...")
    
    output = []
    output.append(f"\n{'='*60}")
    output.append(f"QUERY: {result['query']}")
    output.append(f"METHOD: {result['method']}")
    output.append(f"TOTAL RESULTS: {result['total_results']}")
    output.append(f"{'='*60}\n")
    
    results = result.get("results", {})
    
    if results.get("text"):
        output.append(f"TEXT CHUNKS ({len(results['text'])}):")
        for i, doc in enumerate(results["text"], 1):
            content_preview = doc.page_content[:100] if len(doc.page_content) > 100 else doc.page_content
            output.append(f"\n{i}. {content_preview}")
            
            if "source_pdf_url" in doc.metadata or "pdf_url" in doc.metadata:
                pdf_url = doc.metadata.get("source_pdf_url") or doc.metadata.get("pdf_url", "")
                if pdf_url:
                    output.append(f"   Source: {pdf_url[:80]}...")
            if "chunk_page_number" in doc.metadata:
                output.append(f"   Page: {doc.metadata['chunk_page_number']}")
    
    if results.get("images"):
        output.append(f"\nIMAGES ({len(results['images'])}):")
        for i, doc in enumerate(results["images"], 1):
            output.append(f"\n{i}. {doc.page_content[:100]}...")
            
            if "supabase_url" in doc.metadata:
                output.append(f"   Image URL: {doc.metadata['supabase_url'][:80]}...")
            if "source_pdf_url" in doc.metadata or "pdf_url" in doc.metadata:
                pdf_url = doc.metadata.get("source_pdf_url") or doc.metadata.get("pdf_url", "")
                if pdf_url:
                    output.append(f"   Source: {pdf_url[:80]}...")
            
            if doc.metadata.get("ai_generated_description"):
                output.append(f"   [AI-generated description]")
    
    if results.get("tables"):
        output.append(f"\nTABLES ({len(results['tables'])}):")
        for i, doc in enumerate(results["tables"], 1):
            content_preview = doc.page_content[:100] if len(doc.page_content) > 100 else doc.page_content
            output.append(f"\n{i}. {content_preview}")
            
            if "source_pdf_url" in doc.metadata or "pdf_url" in doc.metadata:
                pdf_url = doc.metadata.get("source_pdf_url") or doc.metadata.get("pdf_url", "")
                if pdf_url:
                    output.append(f"   Source: {pdf_url[:80]}...")
            
            if "table_page_number" in doc.metadata:
                output.append(f"   Page: {doc.metadata['table_page_number']}")
    
    logger.debug("Result formatting completed")
    return "\n".join(output)


def get_unique_source_pdfs(result: Dict[str, Any]) -> List[Dict[str, str]]:
    
    logger.debug("Extracting unique source PDFs...")
    
    pdfs = {}
    results = result.get("results", {})
    
    for type_key in ["text", "images", "tables"]:
        if results.get(type_key):
            for doc in results[type_key]:
                pdf_url = (
                    doc.metadata.get("source_pdf_url") or 
                    doc.metadata.get("pdf_url")
                )
                pdf_path = (
                    doc.metadata.get("source_pdf_path") or 
                    doc.metadata.get("pdf_storage_path")
                )
                
                if pdf_url and pdf_url not in pdfs:
                    pdfs[pdf_url] = {
                        "url": pdf_url,
                        "storage_path": pdf_path or "",
                        "filename": doc.metadata.get("pdf_original_filename", ""),
                        "bucket": doc.metadata.get("document_bucket", "")
                    }
    
    logger.debug(f"Found {len(pdfs)} unique source PDFs")
    return list(pdfs.values())