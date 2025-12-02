import os
import tempfile
import shutil
import uuid
import traceback
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Body, status, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from enum import Enum
import uvicorn

from clients.supabase_client import supabase
from langchain_core.documents import Document

from core.extraction import extract_pdf_multimodal_with_supabase
from core.store import ChromaConfig, store_to_chroma
from core.retrieval import (
    RetrievalConfig, 
    retrieve_multimodal,
    format_result_with_sources,
    get_unique_source_pdfs
)
from core.generation import GenerationConfig, generate_answer
from config.logger_config import setup_logger, get_logger
import time

load_dotenv()

logger = setup_logger(name="api", log_level="INFO")
request_logger = get_logger("api.requests")


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chromaa_multi")
    SUPABASE_IMAGE_BUCKET: str = os.getenv("SUPABASE_IMAGE_BUCKET", "rag-images")
    SUPABASE_DOCUMENT_BUCKET: str = os.getenv("SUPABASE_DOCUMENT_BUCKET", "rag-documents")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    VISION_MODEL: str = os.getenv("VISION_MODEL", "gpt-4o")
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
    USE_VISION: bool = os.getenv("USE_VISION", "true").lower() == "true"
    UPLOAD_SOURCE_PDF: bool = os.getenv("UPLOAD_SOURCE_PDF", "true").lower() == "true"
    
    def validate(self):
        errors = []
        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        return True


settings = Settings()
settings.validate()


try:
    logger.info("Initializing application services...")
    
    chroma_config = ChromaConfig(
        openai_api_key=settings.OPENAI_API_KEY,
        chroma_persist_directory=settings.CHROMA_PERSIST_DIR,
        embedding_model=settings.EMBEDDING_MODEL
    )
    
    retrieval_config = RetrievalConfig(
        openai_api_key=settings.OPENAI_API_KEY,
        chroma_persist_directory=settings.CHROMA_PERSIST_DIR,
        embedding_model=settings.EMBEDDING_MODEL
    )
    
    generation_config = GenerationConfig(
        openai_api_key=settings.OPENAI_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.7,
        max_tokens=2000,
        use_vision=settings.USE_VISION,
        vision_model=settings.VISION_MODEL
    )
    
    logger.info("All services initialized successfully")
    
except Exception as e:
    logger.exception("Initialization failed")
    exit(1)


class RetrievalMethod(str, Enum):
    all = "all"
    hybrid = "hybrid"
    mmr = "mmr"
    text_only = "text_only"
    image_only = "image_only"
    table_only = "table_only"


class GenerationMethod(str, Enum):
    simple = "simple"
    citations = "citations"
    structured = "structured"


class Language(str, Enum):
    indonesian = "Indonesian"
    english = "English"


class ExtractionStrategy(str, Enum):
    fast = "fast"
    hi_res = "hi_res"
    ocr_only = "ocr_only"


class VisionModel(str, Enum):
    gpt_4o = "gpt-4o"
    gpt_4o_mini = "gpt-4o-mini"


class UploadConfig(BaseModel):
    extract_images: bool = Field(default=True, description="Extract images from PDF")
    extract_tables: bool = Field(default=True, description="Extract tables from PDF")
    generate_image_descriptions: bool = Field(default=True, description="Generate AI descriptions")
    description_language: Language = Field(default=Language.indonesian)
    vision_model: VisionModel = Field(default=VisionModel.gpt_4o_mini)
    store_to_vectordb: bool = Field(default=True, description="Store to vector DB")
    strategy: ExtractionStrategy = Field(default=ExtractionStrategy.hi_res)
    upload_source_pdf: bool = Field(default=True, description="Upload PDF source to storage")
    custom_pdf_filename: Optional[str] = Field(default=None, description="Custom filename for PDF")
    
    class Config:
        use_enum_values = True


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    retrieval_method: RetrievalMethod = Field(default=RetrievalMethod.all)
    generation_method: GenerationMethod = Field(default=GenerationMethod.simple)
    language: Language = Field(default=Language.indonesian)
    k: int = Field(default=5, ge=1, le=20, description="Number of results per type")
    k_text: Optional[int] = Field(default=None, ge=1, le=20)
    k_images: Optional[int] = Field(default=None, ge=1, le=10)
    k_tables: Optional[int] = Field(default=None, ge=1, le=10)
    text_weight: float = Field(default=0.5, ge=0, le=1)
    image_weight: float = Field(default=0.25, ge=0, le=1)
    table_weight: float = Field(default=0.25, ge=0, le=1)
    lambda_mult: float = Field(default=0.5, ge=0, le=1, description="0=diversity, 1=relevance")
    include_sources: bool = Field(default=True)
    include_source_pdfs: bool = Field(default=True, description="Include source PDF URLs")
    use_vision: Optional[bool] = Field(default=None, description="Override global vision setting")
    
    @validator('k_text', 'k_images', 'k_tables', always=True)
    def set_default_k(cls, v, values):
        if v is None and 'k' in values:
            return values['k']
        return v
    
    class Config:
        use_enum_values = True


class DocumentsRequest(BaseModel):
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    bucket: Optional[str] = Field(default=None, description="Bucket name (images/documents)")


class DeleteCollectionRequest(BaseModel):
    collection_name: str = Field(..., description="Collection to delete")
    confirm: bool = Field(default=False, description="Must be true to delete")


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, bool]
    models: Dict[str, str]
    storage: Dict[str, str]
    features: Dict[str, Any]


class UploadResponse(BaseModel):
    status: str
    message: str
    document_id: str
    filename: str
    stats: Dict[str, int]
    storage_result: Optional[Dict[str, Any]] = None
    processing_time_seconds: float
    ai_descriptions_generated: int
    vision_used: bool
    pdf_uploaded: bool
    pdf_url: Optional[str] = None
    pdf_storage_path: Optional[str] = None


class QueryResponse(BaseModel):
    status: str
    query: str
    answer: str
    retrieval_method: str
    generation_method: str
    language: str
    vision_used: bool
    sources_count: Dict[str, int]
    sources: Optional[List[Dict[str, Any]]] = None
    source_pdfs: Optional[List[Dict[str, str]]] = None
    total_results: int
    processing_time_seconds: float


class StatsResponse(BaseModel):
    status: str
    timestamp: str
    collections: Dict[str, int]
    storage: Dict[str, str]


class DocumentsResponse(BaseModel):
    status: str
    count: int
    documents: List[Dict[str, Any]]
    pagination: Dict[str, int]
    bucket: str


class ErrorResponse(BaseModel):
    status: str = "error"
    error: str
    detail: Optional[str] = None
    timestamp: str


app = FastAPI(
    title="Multimodal RAG API with Vision & PDF Source Tracking",
    description="Production-ready Multimodal RAG API with GPT-4 Vision and PDF source document management",
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    responses={
        500: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse}
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    request_logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    request_logger.exception("Unhandled exception")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if os.getenv("DEBUG") == "true" else None,
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )


def cleanup_temp_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup temp file {file_path}: {str(e)}")


def validate_file(file: UploadFile) -> tuple[bool, str]:
    if file.content_type != "application/pdf":
        return False, f"Invalid file type. Expected PDF, got {file.content_type}"
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        return False, f"File too large. Max: {settings.MAX_FILE_SIZE_MB}MB, got: {file_size / (1024*1024):.1f}MB"
    
    if not file.filename or file.filename.strip() == "":
        return False, "Invalid filename"
    
    return True, "OK"


def get_collection_count(collection_name: str) -> int:
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        
        result = store.get()
        count = len(result['ids']) if result and 'ids' in result else 0
        logger.debug(f"Collection {collection_name}: {count} documents")
        return count
    except Exception as e:
        logger.error(f"Failed to get count for {collection_name}: {str(e)}")
        return 0


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "Multimodal RAG API with Vision & PDF Tracking",
        "version": "3.1.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "multimodal_extraction": True,
            "ai_vision_analysis": settings.USE_VISION,
            "pdf_source_tracking": settings.UPLOAD_SOURCE_PDF,
            "vision_model": settings.VISION_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.LLM_MODEL
        },
        "endpoints": {
            "health": "GET /health",
            "upload": "POST /api/v1/upload",
            "query": "POST /api/v1/query",
            "stats": "GET /api/v1/stats",
            "documents": "POST /api/v1/documents",
            "delete_collection": "DELETE /api/v1/collections",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    logger.debug("Health check requested")
    
    services_status = {
        "api": True,
        "supabase": False,
        "chromadb": False,
        "openai": False
    }
    
    try:
        supabase.storage.from_(settings.SUPABASE_IMAGE_BUCKET).list(path="", options={"limit": 1})
        supabase.storage.from_(settings.SUPABASE_DOCUMENT_BUCKET).list(path="", options={"limit": 1})
        services_status["supabase"] = True
        logger.debug("Supabase connection OK")
    except Exception as e:
        logger.error(f"Supabase health check failed: {str(e)}")
    
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
        Chroma(
            collection_name="documents_text",
            embedding_function=embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        services_status["chromadb"] = True
        logger.debug("ChromaDB connection OK")
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {str(e)}")
    
    try:
        if settings.OPENAI_API_KEY and len(settings.OPENAI_API_KEY) > 10:
            services_status["openai"] = True
    except:
        pass
    
    health_status = "healthy" if all(services_status.values()) else "degraded"
    logger.info(f"Health check result: {health_status}")
    
    return HealthResponse(
        status=health_status,
        timestamp=datetime.utcnow().isoformat(),
        services=services_status,
        models={
            "embedding": settings.EMBEDDING_MODEL,
            "llm": settings.LLM_MODEL,
            "vision": settings.VISION_MODEL
        },
        storage={
            "chromadb": settings.CHROMA_PERSIST_DIR,
            "image_bucket": settings.SUPABASE_IMAGE_BUCKET,
            "document_bucket": settings.SUPABASE_DOCUMENT_BUCKET
        },
        features={
            "vision_enabled": settings.USE_VISION,
            "pdf_upload_enabled": settings.UPLOAD_SOURCE_PDF,
            "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
            "max_query_length": settings.MAX_QUERY_LENGTH
        }
    )


@app.post("/api/v1/upload", response_model=UploadResponse, tags=["Documents"], status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload"),
    config: str = Form(..., description="JSON configuration string")
):
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    request_logger.info(f"[{request_id}] Upload request received: {file.filename}")
    
    try:
        config_dict = json.loads(config)
        upload_config = UploadConfig(**config_dict)
        request_logger.debug(f"[{request_id}] Upload config: {config_dict}")
    except json.JSONDecodeError:
        request_logger.error(f"[{request_id}] Invalid JSON in config")
        raise HTTPException(status_code=400, detail="Invalid JSON in config parameter")
    except Exception as e:
        request_logger.error(f"[{request_id}] Invalid config: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")
    
    is_valid, error_msg = validate_file(file)
    if not is_valid:
        request_logger.warning(f"[{request_id}] File validation failed: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    
    temp_pdf = None
    document_id = str(uuid.uuid4())
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix=f"upload_{document_id}_") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_pdf = tmp.name
        
        request_logger.info(f"[{request_id}] Processing document: {file.filename}")
        request_logger.debug(f"[{request_id}] Document ID: {document_id}")
        request_logger.debug(f"[{request_id}] Vision: {upload_config.generate_image_descriptions}")
        request_logger.debug(f"[{request_id}] PDF Upload: {upload_config.upload_source_pdf}")
        
        extraction_result = extract_pdf_multimodal_with_supabase(
            pdf_path=temp_pdf,
            supabase=supabase,
            openai_api_key=settings.OPENAI_API_KEY,
            image_bucket_name=settings.SUPABASE_IMAGE_BUCKET,
            document_bucket_name=settings.SUPABASE_DOCUMENT_BUCKET,
            extract_images=upload_config.extract_images,
            extract_tables=upload_config.extract_tables,
            strategy=upload_config.strategy,
            chunk_content=True,
            generate_image_descriptions=upload_config.generate_image_descriptions,
            description_language=upload_config.description_language,
            vision_model=upload_config.vision_model,
            upload_source_pdf=upload_config.upload_source_pdf,
            custom_pdf_filename=upload_config.custom_pdf_filename
        )
        
        ai_descriptions_count = sum(
            1 for img in extraction_result.get("images", [])
            if img.get("ai_generated_description", False)
        )
        
        pdf_url = extraction_result.get("pdf_url")
        pdf_storage_path = extraction_result.get("pdf_storage_path")
        
        storage_result = None
        if upload_config.store_to_vectordb:
            request_logger.info(f"[{request_id}] Storing to vector database...")
            storage_result = store_to_chroma(
                extraction_result=extraction_result,
                config=chroma_config,
                store_text=True,
                store_images=upload_config.extract_images,
                store_tables=upload_config.extract_tables
            )
        
        background_tasks.add_task(cleanup_temp_file, temp_pdf)
        
        processing_time = time.time() - start_time
        
        request_logger.info(f"[{request_id}] Upload completed in {processing_time:.2f}s")
        if pdf_url:
            request_logger.debug(f"[{request_id}] PDF URL: {pdf_url[:80]}...")
        
        return UploadResponse(
            status="success",
            message=f"Document '{file.filename}' processed successfully",
            document_id=document_id,
            filename=file.filename,
            stats={
                "text_chunks": len(extraction_result.get("text_chunks_semantic", [])),
                "images": len(extraction_result.get("images", [])),
                "tables": len(extraction_result.get("tables", [])),
                "total_elements": extraction_result.get("metadata", {}).get("total_elements", 0)
            },
            storage_result=storage_result,
            processing_time_seconds=round(processing_time, 2),
            ai_descriptions_generated=ai_descriptions_count,
            vision_used=upload_config.generate_image_descriptions,
            pdf_uploaded=upload_config.upload_source_pdf and pdf_url is not None,
            pdf_url=pdf_url,
            pdf_storage_path=pdf_storage_path
        )
    
    except HTTPException:
        if temp_pdf and os.path.exists(temp_pdf):
            os.remove(temp_pdf)
        raise
    
    except Exception as e:
        if temp_pdf and os.path.exists(temp_pdf):
            os.remove(temp_pdf)
        
        request_logger.exception(f"[{request_id}] Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest = Body(...)):
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    request_logger.info(f"[{request_id}] Query request: {request.query[:100]}...")
    
    if len(request.query) > settings.MAX_QUERY_LENGTH:
        request_logger.warning(f"[{request_id}] Query too long")
        raise HTTPException(
            status_code=400,
            detail=f"Query too long. Max: {settings.MAX_QUERY_LENGTH} chars"
        )
    
    try:
        request_logger.debug(f"[{request_id}] Retrieval method: {request.retrieval_method}")
        request_logger.debug(f"[{request_id}] Generation method: {request.generation_method}")
        request_logger.debug(f"[{request_id}] Vision: {request.use_vision if request.use_vision is not None else settings.USE_VISION}")
        
        retrieval_kwargs = {"k": request.k}
        
        if request.retrieval_method == "all":
            retrieval_kwargs.update({
                "k_text": request.k_text,
                "k_images": request.k_images,
                "k_tables": request.k_tables
            })
        elif request.retrieval_method == "hybrid":
            retrieval_kwargs.update({
                "text_weight": request.text_weight,
                "image_weight": request.image_weight,
                "table_weight": request.table_weight
            })
        elif request.retrieval_method == "mmr":
            retrieval_kwargs.update({
                "lambda_mult": request.lambda_mult,
                "fetch_k": request.k * 3
            })
        
        retrieval_result = retrieve_multimodal(
            query=request.query,
            config=retrieval_config,
            method=request.retrieval_method,
            **retrieval_kwargs
        )
        
        results = retrieval_result.get("results", {})
        total_found = (
            len(results.get("text", [])) +
            len(results.get("images", [])) +
            len(results.get("tables", []))
        )
        
        request_logger.info(f"[{request_id}] Retrieved {total_found} documents")
        request_logger.debug(f"[{request_id}] Text: {len(results.get('text', []))}, Images: {len(results.get('images', []))}, Tables: {len(results.get('tables', []))}")
        
        source_pdfs = None
        if request.include_source_pdfs:
            source_pdfs = get_unique_source_pdfs(retrieval_result)
            if source_pdfs:
                request_logger.debug(f"[{request_id}] Found {len(source_pdfs)} source PDFs")
        
        if total_found == 0:
            request_logger.info(f"[{request_id}] No results found")
            return QueryResponse(
                status="success",
                query=request.query,
                answer="Maaf, tidak ada informasi relevan yang ditemukan untuk pertanyaan Anda.",
                retrieval_method=request.retrieval_method,
                generation_method=request.generation_method,
                language=request.language,
                vision_used=False,
                sources_count={"text": 0, "images": 0, "tables": 0},
                sources=[],
                source_pdfs=source_pdfs,
                total_results=0,
                processing_time_seconds=round(time.time() - start_time, 2)
            )
        
        use_vision = request.use_vision if request.use_vision is not None else settings.USE_VISION
        
        gen_config = GenerationConfig(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL,
            temperature=0.7,
            max_tokens=2000,
            use_vision=use_vision,
            vision_model=settings.VISION_MODEL
        )
        
        request_logger.info(f"[{request_id}] Generating answer...")
        answer_result = generate_answer(
            query=request.query,
            retrieval_results=results,
            config=gen_config,
            method=request.generation_method,
            language=request.language,
            include_sources=request.include_sources
        )
        
        processing_time = time.time() - start_time
        
        request_logger.info(f"[{request_id}] Query completed in {processing_time:.2f}s")
        
        return QueryResponse(
            status="success",
            query=request.query,
            answer=answer_result.get("answer", ""),
            retrieval_method=request.retrieval_method,
            generation_method=request.generation_method,
            language=request.language,
            vision_used=use_vision and len(results.get("images", [])) > 0,
            sources_count=answer_result.get("sources_count", {"text": 0, "images": 0, "tables": 0}),
            sources=answer_result.get("sources") if request.include_sources else None,
            source_pdfs=source_pdfs,
            total_results=total_found,
            processing_time_seconds=round(processing_time, 2)
        )
    
    except Exception as e:
        request_logger.exception(f"[{request_id}] Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_statistics():
    logger.info("Statistics requested")
    
    try:
        text_count = get_collection_count("documents_text")
        images_count = get_collection_count("documents_images")
        tables_count = get_collection_count("documents_tables")
        
        return StatsResponse(
            status="success",
            timestamp=datetime.utcnow().isoformat(),
            collections={
                "text": text_count,
                "images": images_count,
                "tables": tables_count,
                "total": text_count + images_count + tables_count
            },
            storage={
                "persist_directory": settings.CHROMA_PERSIST_DIR,
                "image_bucket": settings.SUPABASE_IMAGE_BUCKET,
                "document_bucket": settings.SUPABASE_DOCUMENT_BUCKET
            }
        )
    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.post("/api/v1/documents", response_model=DocumentsResponse, tags=["Documents"])
async def list_documents(request: DocumentsRequest = Body(...)):
    logger.info(f"Document list requested: limit={request.limit}, offset={request.offset}")
    
    try:
        bucket_name = request.bucket or settings.SUPABASE_IMAGE_BUCKET
        if bucket_name not in [settings.SUPABASE_IMAGE_BUCKET, settings.SUPABASE_DOCUMENT_BUCKET]:
            bucket_name = settings.SUPABASE_IMAGE_BUCKET
        
        logger.debug(f"Listing documents from bucket: {bucket_name}")
        
        result = supabase.storage.from_(bucket_name).list(
            path="",
            options={"limit": request.limit, "offset": request.offset}
        )
        
        documents = []
        for file in result:
            documents.append({
                "name": file.get("name"),
                "id": file.get("id"),
                "size_bytes": file.get("metadata", {}).get("size", 0),
                "size_mb": round(file.get("metadata", {}).get("size", 0) / (1024 * 1024), 2),
                "created_at": file.get("created_at"),
                "updated_at": file.get("updated_at"),
                "url": supabase.storage.from_(bucket_name).get_public_url(file.get("name"))
            })
        
        logger.info(f"Found {len(documents)} documents")
        
        return DocumentsResponse(
            status="success",
            count=len(documents),
            documents=documents,
            pagination={
                "limit": request.limit,
                "offset": request.offset,
                "total": len(documents)
            },
            bucket=bucket_name
        )
    
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.delete("/api/v1/collections", tags=["Admin"])
async def delete_collection(request: DeleteCollectionRequest = Body(...)):
    logger.warning(f"Collection deletion requested: {request.collection_name}")
    
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to delete collection"
        )
    
    valid_collections = ["documents_text", "documents_images", "documents_tables"]
    if request.collection_name not in valid_collections:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid collection. Must be one of: {', '.join(valid_collections)}"
        )
    
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        
        client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        count_before = get_collection_count(request.collection_name)
        
        client.delete_collection(name=request.collection_name)
        
        logger.warning(f"Deleted collection '{request.collection_name}' ({count_before} documents)")
        
        return {
            "status": "success",
            "message": f"Collection '{request.collection_name}' deleted successfully",
            "documents_deleted": count_before
        }
    
    except Exception as e:
        logger.error(f"Failed to delete collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")


@app.delete("/api/v1/storage/{bucket}/{filename}", tags=["Admin"])
async def delete_file(bucket: str, filename: str):
    logger.warning(f"File deletion requested: {bucket}/{filename}")
    
    try:
        if bucket not in ["images", "documents"]:
            raise HTTPException(
                status_code=400,
                detail="Bucket must be 'images' or 'documents'"
            )
        
        bucket_name = settings.SUPABASE_IMAGE_BUCKET if bucket == "images" else settings.SUPABASE_DOCUMENT_BUCKET
        
        supabase.storage.from_(bucket_name).remove([filename])
        
        logger.info(f"Deleted file: {filename} from {bucket_name}")
        
        return {
            "status": "success",
            "message": f"File '{filename}' deleted successfully from {bucket}",
            "bucket": bucket_name
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@app.on_event("startup")
async def startup_event():
    logger.info("="*70)
    logger.info("Multimodal RAG API with Vision & PDF Tracking - Starting...")
    logger.info("="*70)
    logger.info(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"LLM Model: {settings.LLM_MODEL}")
    logger.info(f"Vision Model: {settings.VISION_MODEL} {'(ENABLED)' if settings.USE_VISION else '(DISABLED)'}")
    logger.info(f"ChromaDB: {settings.CHROMA_PERSIST_DIR}")
    logger.info(f"Supabase Image Bucket: {settings.SUPABASE_IMAGE_BUCKET}")
    logger.info(f"Supabase Document Bucket: {settings.SUPABASE_DOCUMENT_BUCKET}")
    logger.info(f"Max File Size: {settings.MAX_FILE_SIZE_MB}MB")
    logger.info(f"PDF Upload: {'ENABLED' if settings.UPLOAD_SOURCE_PDF else 'DISABLED'}")
    logger.info("="*70)
    
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    
    logger.info("Testing connections...")
    try:
        supabase.storage.from_(settings.SUPABASE_IMAGE_BUCKET).list(path="", options={"limit": 1})
        supabase.storage.from_(settings.SUPABASE_DOCUMENT_BUCKET).list(path="", options={"limit": 1})
        logger.info("Supabase connection OK (both buckets)")
    except Exception as e:
        logger.error(f"Supabase connection failed: {str(e)}")
    
    logger.info("API Ready!")
    logger.info("Docs: http://localhost:8000/docs")
    logger.info("ReDoc: http://localhost:8000/redoc")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Multimodal RAG API...")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )