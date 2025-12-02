import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from PIL import Image
import uuid
import tempfile
import shutil
import base64
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from config.logger_config import get_logger
import time

from dotenv import load_dotenv
from clients.supabase_client import supabase

load_dotenv()

logger = get_logger(__name__)


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_image_description(image_path: str, openai_api_key: str,model: str = "gpt-4o",language: str = "Indonesian") -> Optional[str]:

    model = os.getenv("MODEL_NAME_IMAGE_DESCRIPTION")
   
    try:
        logger.info(f"Generating description for image: {Path(image_path).name}")
        
        llm = ChatOpenAI(
            model=model,
            openai_api_key=openai_api_key,
            max_tokens=500,
            temperature=0.3
        )
        base64_image = encode_image_to_base64(image_path)
        
        if language == "Indonesian":
            prompt = """Analisis gambar ini dengan detail. Jelaskan:
                                1. Jenis konten (chart, diagram, foto, ilustrasi, dll)
                                2. Elemen-elemen utama yang ada
                                3. Data atau informasi penting yang ditampilkan
                                4. Konteks atau tujuan gambar

                                Berikan deskripsi yang lengkap dan informatif dalam bahasa Indonesia."""
        else:
            prompt = """Analyze this image in detail. Describe:
                                1. Type of content (chart, diagram, photo, illustration, etc.)
                                2. Main elements present
                                3. Important data or information displayed
                                4. Context or purpose of the image

                                Provide a comprehensive and informative description."""
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        )

        response = llm.invoke([message])
        description = response.content
        
        logger.info(f"Description generated successfully ({len(description)} chars)")
        return description
    
    except Exception as e:
        logger.error(f"Failed to generate description: {str(e)}", exc_info=True)
        return None


def upload_image_to_supabase(local_image_path: str, supabase, bucket_name: str = "rag-images") -> Dict[str, str]:
    try:
        if not os.path.exists(local_image_path):
            logger.error(f"Image not found: {local_image_path}")
            raise FileNotFoundError(f"Image not found: {local_image_path}")

        file_ext = Path(local_image_path).suffix
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        
        logger.debug(f"Uploading image: {Path(local_image_path).name} -> {unique_filename}")

        with open(local_image_path, 'rb') as f:
            image_data = f.read()

        response = supabase.storage.from_(bucket_name).upload(
            path=unique_filename,
            file=image_data,
            file_options={"content-type": f"image/{file_ext.lstrip('.')}"}
        )

        public_url = supabase.storage.from_(bucket_name).get_public_url(unique_filename)
        
        logger.info(f"Image uploaded successfully: {unique_filename}")
        
        return {
            "public_url": public_url,
            "storage_path": unique_filename,
            "bucket": bucket_name
        }
    except Exception as e:
        logger.error(f"Error uploading image {local_image_path}: {str(e)}", exc_info=True)
        return None


def upload_pdf_to_supabase(local_pdf_path: str,supabase,bucket_name: str = "rag-documents",custom_filename: Optional[str] = None) -> Optional[Dict[str, str]]:

    try:
        if not os.path.exists(local_pdf_path):
            logger.error(f"PDF not found: {local_pdf_path}")
            raise FileNotFoundError(f"PDF not found: {local_pdf_path}")

        if custom_filename:
            unique_filename = f"{uuid.uuid4()}_{custom_filename}"
        else:
            original_name = Path(local_pdf_path).name
            unique_filename = f"{uuid.uuid4()}_{original_name}"

        file_size_mb = os.path.getsize(local_pdf_path) / 1024 / 1024
        logger.info(f"Uploading PDF to Supabase: {Path(local_pdf_path).name} ({file_size_mb:.2f} MB)")

        with open(local_pdf_path, 'rb') as f:
            pdf_data = f.read()

        response = supabase.storage.from_(bucket_name).upload(
            path=unique_filename,
            file=pdf_data,
            file_options={"content-type": "application/pdf"}
        )
        public_url = supabase.storage.from_(bucket_name).get_public_url(unique_filename)

        result = {
            "public_url": public_url,
            "storage_path": unique_filename,
            "bucket": bucket_name,
            "original_filename": Path(local_pdf_path).name,
            "file_size": os.path.getsize(local_pdf_path)
        }

        logger.info(f"PDF uploaded successfully: {unique_filename}")
        logger.debug(f"PDF URL: {public_url[:80]}...")

        return result

    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}", exc_info=True)
        return None


def extract_pdf_multimodal_with_supabase(pdf_path: str,supabase,openai_api_key: str,image_bucket_name: str = "rag-images",document_bucket_name: str = "rag-documents",extract_images: bool = True,extract_tables: bool = True,
    strategy: str = "hi_res",chunk_content: bool = True,generate_image_descriptions: bool = True,description_language: str = "Indonesian",vision_model: str = "gpt-4o",upload_source_pdf: bool = True,custom_pdf_filename: Optional[str] = None
) -> Dict[str, Any]:

    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    temp_dir = tempfile.mkdtemp(prefix="pdf_extraction_")
    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    start_time = time.time()

    try:
        logger.info("="*80)
        logger.info(f"Starting PDF extraction: {pdf_path}")
        logger.info(f"Strategy: {strategy}, Extract images: {extract_images}, Extract tables: {extract_tables}")
        logger.info(f"Generate descriptions: {generate_image_descriptions}, Language: {description_language}")
        logger.info("="*80)
        
        pdf_upload_result = None
        if upload_source_pdf:
            pdf_upload_result = upload_pdf_to_supabase(local_pdf_path=pdf_path,supabase=supabase,bucket_name=document_bucket_name,custom_filename=custom_pdf_filename)
        
        logger.info("Extracting PDF elements...")
        elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy,
            extract_images_in_pdf=extract_images,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_output_dir=images_dir,
            infer_table_structure=extract_tables,
            chunking_strategy=None,
            max_characters=10000,
            overlap=200,
        )
        
        logger.info(f"Extracted {len(elements)} elements from PDF")

        result = {
            "text_chunks": [],
            "tables": [],
            "images": [],
            "formulas": [],
            "metadata": {
                "source": pdf_path,
                "total_elements": len(elements),
                "image_bucket": image_bucket_name,
                "document_bucket": document_bucket_name
            }
        }

        if pdf_upload_result:
            result["metadata"]["source_pdf"] = pdf_upload_result
            result["pdf_url"] = pdf_upload_result["public_url"]
            result["pdf_storage_path"] = pdf_upload_result["storage_path"]
            logger.debug(f"Source PDF stored: {pdf_upload_result['storage_path']}")

        image_count = 0
        table_count = 0
        
        for idx, element in enumerate(elements):
            element_dict = {
                "id": f"{Path(pdf_path).stem}_{idx}",
                "type": element.category,
                "content": str(element),
                "metadata": element.metadata.to_dict() if hasattr(element, 'metadata') else {}
            }

            if pdf_upload_result:
                element_dict["source_pdf_url"] = pdf_upload_result["public_url"]
                element_dict["source_pdf_path"] = pdf_upload_result["storage_path"]

            if element.category == "Image":
                if hasattr(element.metadata, 'image_path') and element.metadata.image_path:
                    local_path = element.metadata.image_path
                    image_count += 1

                    logger.info(f"Processing image {image_count}: {Path(local_path).name}")
                    
                    ai_description = None
                    if generate_image_descriptions:
                        logger.debug("Generating AI description...")
                        ai_description = generate_image_description(
                            image_path=local_path,
                            openai_api_key=openai_api_key,
                            model=vision_model,
                            language=description_language
                        )
                    
                    logger.debug("Uploading image to Supabase...")
                    upload_result = upload_image_to_supabase(local_path, supabase, image_bucket_name)

                    if upload_result:
                        element_dict["supabase_url"] = upload_result["public_url"]
                        element_dict["storage_path"] = upload_result["storage_path"]
                        
                        if ai_description:
                            element_dict["content"] = ai_description
                            element_dict["ai_generated_description"] = True
                        elif element_dict["content"]:
                            element_dict["ai_generated_description"] = False
                        else:
                            page = element_dict["metadata"].get("page_number", "unknown")
                            element_dict["content"] = f"Image from page {page}"
                            element_dict["ai_generated_description"] = False
                        
                        result["images"].append(element_dict)
                        logger.info(f"Image {image_count} processed successfully")
                    else:
                        logger.warning(f"Failed to upload image {image_count}")

            elif element.category == "Table":
                table_count += 1
                logger.debug(f"Processing table {table_count}")
                
                if hasattr(element.metadata, 'text_as_html'):
                    element_dict["table_html"] = element.metadata.text_as_html
                else:
                    element_dict["table_html"] = None

                element_dict["table_text"] = str(element)
                result["tables"].append(element_dict)

            elif element.category == "Formula":
                result["formulas"].append(element_dict)

            else:
                result["text_chunks"].append(element_dict)

        if chunk_content and result["text_chunks"]:
            logger.info("Performing semantic chunking on text elements...")
            text_elements = [el for el in elements if el.category not in ["Table", "Image", "Formula"]]

            if text_elements:
                chunked = chunk_by_title(
                    text_elements,
                    max_characters=1000,
                    combine_text_under_n_chars=200,
                    new_after_n_chars=800,
                )

                result["text_chunks_semantic"] = []
                for i, chunk in enumerate(chunked):
                    chunk_dict = {
                        "id": f"{Path(pdf_path).stem}_chunk_{i}",
                        "content": str(chunk),
                        "metadata": chunk.metadata.to_dict() if hasattr(chunk, 'metadata') else {}
                    }
                    
                    if pdf_upload_result:
                        chunk_dict["source_pdf_url"] = pdf_upload_result["public_url"]
                        chunk_dict["source_pdf_path"] = pdf_upload_result["storage_path"]
                    
                    result["text_chunks_semantic"].append(chunk_dict)
                
                logger.info(f"Created {len(result['text_chunks_semantic'])} semantic chunks")

        duration = time.time() - start_time
        
        logger.info("="*80)
        logger.info("Extraction Summary:")
        logger.info(f"  Text chunks: {len(result['text_chunks'])}")
        logger.info(f"  Images: {len(result['images'])}")
        if generate_image_descriptions:
            ai_desc_count = sum(1 for img in result['images'] if img.get('ai_generated_description'))
            logger.info(f"  Images with AI descriptions: {ai_desc_count}/{len(result['images'])}")
        logger.info(f"  Tables: {len(result['tables'])}")
        logger.info(f"  Formulas: {len(result['formulas'])}")
        logger.info(f"  Total elements: {result['metadata']['total_elements']}")
        logger.info(f"  Processing time: {duration:.2f}s")

        if pdf_upload_result:
            logger.debug(f"Source PDF URL: {pdf_upload_result['public_url'][:80]}...")
        
        logger.info("="*80)

        return result

    except Exception as e:
        logger.exception(f"Extraction failed for {pdf_path}")
        raise

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.debug("Cleaned up temporary directory")