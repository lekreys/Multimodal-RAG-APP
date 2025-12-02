import os
import re
import base64
import requests
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from config.logger_config import get_logger
import time

logger = get_logger(__name__)


@dataclass
class GenerationConfig:
    openai_api_key: str
    model: str = os.getenv("MODEL_NAME_GENERATION")
    temperature: float = 0.7
    max_tokens: int = 2000
    use_vision: bool = True 
    vision_model: str = os.getenv("VISION_MODEL") 


class MultimodalGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config

        logger.info("Initializing Multimodal Generator...")
        
        self.llm = ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=config.openai_api_key
        )

        if config.use_vision:
            self.vision_llm = ChatOpenAI(
                model=config.vision_model,
                temperature=0.3,
                max_tokens=1000,
                openai_api_key=config.openai_api_key
            )

        logger.info(f"Generator initialized - Text model: {config.model}")
        if config.use_vision:
            logger.info(f"Vision model: {config.vision_model} (ENABLED)")
        else:
            logger.info("Vision analysis: DISABLED")


    def _fetch_image_as_base64(self, url: str) -> Optional[str]:
        
        try:
            logger.debug(f"Fetching image from URL: {url[:80]}...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            image_data = response.content
            base64_image = base64.b64encode(image_data).decode('utf-8')

            logger.debug(f"Image fetched successfully ({len(image_data)} bytes)")
            return base64_image

        except Exception as e:
            logger.error(f"Failed to fetch image from {url[:80]}: {str(e)}", exc_info=True)
            return None


    def _analyze_image_with_vision(self,image_url: str,query: str,language: str = "Indonesian") -> Optional[str]:
      
        try:
            logger.info("Analyzing image with vision model...")
            logger.debug(f"Image URL: {image_url[:80]}...")
            logger.debug(f"Query context: {query[:100]}...")

            base64_image = self._fetch_image_as_base64(image_url)
            if not base64_image:
                logger.warning("Failed to fetch image for vision analysis")
                return None

            if language.lower() == "indonesian":
                vision_prompt = f"""Analisis gambar ini dalam konteks pertanyaan: "{query}"

                                    Jelaskan secara detail:
                                    1. Apa yang ditampilkan dalam gambar (chart, diagram, foto, dll)
                                    2. Data atau informasi penting yang relevan dengan pertanyaan
                                    3. Insight atau kesimpulan yang bisa diambil

                                    Berikan analisis yang komprehensif dan fokus pada informasi yang relevan dengan pertanyaan."""
            else:
                vision_prompt = f"""Analyze this image in the context of the question: "{query}"

                                    Explain in detail:
                                    1. What is shown in the image (chart, diagram, photo, etc.)
                                    2. Important data or information relevant to the question
                                    3. Insights or conclusions that can be drawn

                                    Provide comprehensive analysis focused on information relevant to the question."""

            message = HumanMessage(
                content=[
                    {"type": "text", "text": vision_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            )
            
            response = self.vision_llm.invoke([message])
            analysis = response.content

            logger.info(f"Vision analysis completed ({len(analysis)} chars)")
            return analysis

        except Exception as e:
            logger.error(f"Vision analysis failed: {str(e)}", exc_info=True)
            return None


    def _escape_curly_braces(self, text: str) -> str:
        if not text:
            return text
        text = text.replace("{", "{{").replace("}", "}}")
        return text


    def _format_multimodal_context(self,results: Dict[str, List[Document]],query: str,use_vision: bool = True,language: str = "Indonesian") -> str:
       
        logger.info("Formatting multimodal context...")
        
        context_parts = []
        text_count = len(results.get("text", []))
        image_count = len(results.get("images", []))
        table_count = len(results.get("tables", []))
        
        logger.debug(f"Context sources: {text_count} text, {image_count} images, {table_count} tables")

        if results.get("text"):
            context_parts.append("=== TEXT CONTENT ===\n")
            for i, doc in enumerate(results["text"], 1):
                page = doc.metadata.get("chunk_page_number", "N/A")
                content = self._escape_curly_braces(doc.page_content)
                context_parts.append(
                    f"[TEXT-{i}] (Page {page})\n{content}\n"
                )

        if results.get("images"):
            context_parts.append("\n=== IMAGES ===\n")
            for i, doc in enumerate(results["images"], 1):
                url = doc.metadata.get("supabase_url", "")
                page = doc.metadata.get("img_page_number", "N/A")

                context_parts.append(f"[IMAGE-{i}] (Page {page})\n")

                if use_vision and url and self.config.use_vision:
                    logger.info(f"Processing IMAGE-{i} with vision model...")

                    vision_analysis = self._analyze_image_with_vision(
                        image_url=url,
                        query=query,
                        language=language
                    )

                    if vision_analysis:
                        context_parts.append(f"Visual Analysis (Real-time): {vision_analysis}\n")
                        logger.debug(f"IMAGE-{i}: Using real-time vision analysis")
                    else:
                        stored_desc = self._escape_curly_braces(doc.page_content)
                        context_parts.append(f"Visual Analysis (Stored): {stored_desc}\n")
                        logger.debug(f"IMAGE-{i}: Falling back to stored description")
                else:
                    stored_desc = self._escape_curly_braces(doc.page_content)
                    context_parts.append(f"Visual Analysis: {stored_desc}\n")
                    logger.debug(f"IMAGE-{i}: Using stored description (vision disabled)")

                context_parts.append(f"URL: {url}\n")

        if results.get("tables"):
            context_parts.append("\n=== TABLES ===\n")
            for i, doc in enumerate(results["tables"], 1):
                page = doc.metadata.get("table_page_number", "N/A")
                has_html = doc.metadata.get("has_html", False)
                content = self._escape_curly_braces(doc.page_content)
                context_parts.append(
                    f"[TABLE-{i}] (Page {page}, Format: {'HTML' if has_html else 'Plain Text'})\n"
                    f"{content}\n"
                )

        context = "\n".join(context_parts)
        logger.info(f"Context formatted: {len(context)} chars total")
        return context

    def generate_simple(self,query: str,retrieval_results: Dict[str, List[Document]],include_sources: bool = True,language: str = "Indonesian") -> Dict[str, Any]:
        
        start_time = time.time()
        
        logger.info("="*80)
        logger.info("Generating answer (simple method)")
        logger.info(f"Query: '{query}'")
        logger.info(f"Language: {language}")
        logger.info(f"Vision: {'ENABLED' if self.config.use_vision else 'DISABLED'}")
        logger.info("="*80)

        context = self._format_multimodal_context(
            retrieval_results,
            query,
            use_vision=self.config.use_vision,
            language=language
        )
        
        if not context.strip():
            logger.warning("No context available for generation")
            return {
                "answer": "Maaf, saya tidak menemukan informasi yang relevan untuk menjawab pertanyaan Anda.",
                "has_context": False,
                "sources": []
            }
        
        if language.lower() == "indonesian":
            system_prompt = """Kamu adalah AI assistant yang ahli dalam menganalisis dokumen multimodal.

                                Konteks yang diberikan berisi:
                                - **TEXT**: Bagian text dari dokumen
                                - **IMAGES**: Analisis visual real-time dari gambar/chart/diagram menggunakan vision AI
                                - **TABLES**: Tabel dengan data terstruktur

                                INSTRUKSI PENTING:
                                1. Jawab pertanyaan user berdasarkan konteks yang diberikan
                                2. Jika menyebutkan sumber, gunakan format: [TEXT-1], [IMAGE-2], [TABLE-3]
                                3. Untuk IMAGES: Gambar telah dianalisis secara real-time. Gunakan analisis ini untuk memberikan insight yang akurat
                                4. Jika ada data numerik atau tren dalam gambar, sebutkan dengan spesifik
                                5. Jika konteks tidak cukup untuk menjawab, katakan dengan jelas
                                6. Berikan jawaban yang lengkap, informatif, dan mudah dipahami
                                7. Hindari frasa seperti "berdasarkan konteks" - langsung jawab saja"""

        else:
            system_prompt = """You are an AI assistant expert in analyzing multimodal documents.

                                The provided context contains:
                                - **TEXT**: Text portions from the document
                                - **IMAGES**: Real-time visual analysis of images/charts/diagrams using vision AI
                                - **TABLES**: Tables with structured data

                                IMPORTANT INSTRUCTIONS:
                                1. Answer the user's question based on the provided context
                                2. When citing sources, use format: [TEXT-1], [IMAGE-2], [TABLE-3]
                                3. For IMAGES: Images have been analyzed in real-time. Use this analysis for accurate insights
                                4. If there's numerical data or trends in images, mention them specifically
                                5. If context is insufficient to answer, state this clearly
                                6. Provide complete, informative, and easy-to-understand answers
                                7. Avoid phrases like "based on context" - just answer directly"""

        escaped_query = self._escape_curly_braces(query)

        user_template = """Konteks:
                           {context}

                           Pertanyaan: {query}

                           Jawaban:"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_template)
        ])

        try:
            logger.info("Invoking text generation model...")
            chain = prompt | self.llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "query": escaped_query
            })
            
            duration = time.time() - start_time
            logger.info(f"Answer generated successfully ({len(answer)} chars, {duration:.2f}s)")
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            return {
                "answer": f"Maaf, terjadi error saat generate jawaban: {str(e)}",
                "has_context": True,
                "error": str(e)
            }

        result = {
            "answer": answer,
            "has_context": True,
            "model": self.config.model,
            "vision_model": self.config.vision_model if self.config.use_vision else None,
            "language": language,
            "sources_count": {
                "text": len(retrieval_results.get("text", [])),
                "images": len(retrieval_results.get("images", [])),
                "tables": len(retrieval_results.get("tables", []))
            }
        }

        if include_sources:
            result["sources"] = self._extract_sources(retrieval_results)
            logger.debug(f"Extracted {len(result['sources'])} sources")

        logger.info("="*80)
        return result


    def generate_with_citations(self,query: str,retrieval_results: Dict[str, List[Document]],language: str = "Indonesian") -> Dict[str, Any]:
        
        start_time = time.time()
        
        logger.info("="*80)
        logger.info("Generating answer with citations")
        logger.info(f"Query: '{query}'")
        logger.info("="*80)

        context = self._format_multimodal_context(
            retrieval_results,
            query,
            use_vision=self.config.use_vision,
            language=language
        )

        if not context.strip():
            logger.warning("No context available for generation")
            return {
                "answer": "Maaf, saya tidak menemukan informasi yang relevan.",
                "has_context": False,
                "sources": []
            }

        if language.lower() == "indonesian":
            system_prompt = """Kamu adalah AI assistant yang memberikan jawaban dengan citations yang tepat.

                                ATURAN CITATIONS:
                                - Untuk text: [TEXT-1], [TEXT-2], dll
                                - Untuk gambar: [IMAGE-1], [IMAGE-2], dll
                                - Untuk tabel: [TABLE-1], [TABLE-2], dll

                                PENTING untuk IMAGES:
                                - Setiap IMAGE telah dianalisis secara real-time menggunakan vision AI
                                - Gunakan analisis visual ini untuk memberikan insight akurat
                                - SELALU tambahkan citation [IMAGE-X] setelah menjelaskan gambar

                                JANGAN lupa tambahkan citation setelah SETIAP klaim/fakta!"""

        else:
            system_prompt = """You are an AI assistant that provides answers with accurate citations.

                                CITATION RULES:
                                - For text: [TEXT-1], [TEXT-2], etc.
                                - For images: [IMAGE-1], [IMAGE-2], etc.
                                - For tables: [TABLE-1], [TABLE-2], etc.

                                IMPORTANT for IMAGES:
                                - Each IMAGE has been analyzed in real-time using vision AI
                                - Use this visual analysis for accurate insights
                                - ALWAYS add citation [IMAGE-X] after explaining the image

                                DON'T forget to add citation after EVERY claim/fact!"""

        escaped_query = self._escape_curly_braces(query)

        user_template = """Konteks:
                            {context}

                            Pertanyaan: {query}

                            Jawab dengan inline citations:"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_template)
        ])

        try:
            logger.info("Invoking text generation model with citations...")
            chain = prompt | self.llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "query": escaped_query
            })
            
            duration = time.time() - start_time
            logger.info(f"Answer with citations generated ({len(answer)} chars, {duration:.2f}s)")
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            return {
                "answer": f"Error: {str(e)}",
                "has_context": True,
                "error": str(e)
            }

        logger.info("="*80)
        return {
            "answer": answer,
            "has_context": True,
            "model": self.config.model,
            "vision_model": self.config.vision_model if self.config.use_vision else None,
            "language": language,
            "sources": self._extract_sources_with_ids(retrieval_results)
        }


    def generate_structured(self,query: str,retrieval_results: Dict[str, List[Document]],language: str = "Indonesian") -> Dict[str, Any]:
        
        start_time = time.time()
        
        logger.info("="*80)
        logger.info("Generating structured answer")
        logger.info(f"Query: '{query}'")
        logger.info("="*80)

        context = self._format_multimodal_context(
            retrieval_results,
            query,
            use_vision=self.config.use_vision,
            language=language
        )

        if not context.strip():
            logger.warning("No context available for generation")
            return {
                "answer": "Tidak ada informasi yang ditemukan.",
                "has_context": False
            }

        if language.lower() == "indonesian":
            system_prompt = """Kamu adalah AI assistant yang memberikan jawaban terstruktur.

                                Format jawaban:

                                **RINGKASAN EKSEKUTIF**
                                [Jawaban langsung dalam 2-3 kalimat]

                                **TEMUAN UTAMA**
                                1. [Poin pertama]
                                2. [Poin kedua]
                                3. [Dst...]

                                **ANALISIS VISUAL**
                                [Jelaskan gambar/chart yang relevan berdasarkan analisis real-time]

                                **DATA PENDUKUNG**
                                [Detail dari text/tabel]"""

        else:
            system_prompt = """You are an AI assistant providing structured answers.

                                Answer format:

                                **EXECUTIVE SUMMARY**
                                [Direct answer in 2-3 sentences]

                                **KEY FINDINGS**
                                1. [First point]
                                2. [Second point]
                                3. [Etc...]

                                **VISUAL ANALYSIS**
                                [Explain relevant images/charts based on real-time analysis]

                                **SUPPORTING DATA**
                                [Details from text/tables]"""

        escaped_query = self._escape_curly_braces(query)

        user_template = """Konteks:
                            {context}

                            Pertanyaan: {query}

                            Jawaban terstruktur:"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_template)
        ])

        try:
            logger.info("Invoking text generation model for structured output...")
            chain = prompt | self.llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "query": escaped_query
            })
            
            duration = time.time() - start_time
            logger.info(f"Structured answer generated ({len(answer)} chars, {duration:.2f}s)")
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            return {
                "answer": f"Error: {str(e)}",
                "has_context": True,
                "error": str(e)
            }

        logger.info("="*80)
        return {
            "answer": answer,
            "has_context": True,
            "model": self.config.model,
            "vision_model": self.config.vision_model if self.config.use_vision else None,
            "language": language,
            "sources": self._extract_sources(retrieval_results)
        }


    def _extract_sources(self, results: Dict[str, List[Document]]) -> List[Dict[str, Any]]:
        logger.debug("Extracting sources from retrieval results...")
        sources = []

        for i, doc in enumerate(results.get("text", []), 1):
            sources.append({
                "type": "text",
                "id": f"TEXT-{i}",
                "content_preview": doc.page_content[:200] + "...",
                "page": doc.metadata.get("chunk_page_number", "N/A"),
                "metadata": doc.metadata
            })

        for i, doc in enumerate(results.get("images", []), 1):
            sources.append({
                "type": "image",
                "id": f"IMAGE-{i}",
                "description": doc.page_content,
                "url": doc.metadata.get("supabase_url", ""),
                "page": doc.metadata.get("img_page_number", "N/A"),
                "analyzed_with_vision": self.config.use_vision,
                "metadata": doc.metadata
            })

        for i, doc in enumerate(results.get("tables", []), 1):
            sources.append({
                "type": "table",
                "id": f"TABLE-{i}",
                "content_preview": doc.page_content[:300] + "...",
                "page": doc.metadata.get("table_page_number", "N/A"),
                "has_html": doc.metadata.get("has_html", False),
                "metadata": doc.metadata
            })

        logger.debug(f"Extracted {len(sources)} sources")
        return sources


    def _extract_sources_with_ids(self, results: Dict[str, List[Document]]) -> List[Dict[str, Any]]:
        return self._extract_sources(results)


def generate_answer(query: str,retrieval_results: Dict[str, List[Document]],config: GenerationConfig,method: Literal["simple", "citations", "structured"] = "simple",language: str = "Indonesian",include_sources: bool = True) -> Dict[str, Any]:

    logger.info("Initializing answer generation...")
    logger.info(f"Method: {method}, Language: {language}")
    
    generator = MultimodalGenerator(config)

    try:
        if method == "simple":
            return generator.generate_simple(
                query,
                retrieval_results,
                include_sources=include_sources,
                language=language
            )

        elif method == "citations":
            return generator.generate_with_citations(
                query,
                retrieval_results,
                language=language
            )

        elif method == "structured":
            return generator.generate_structured(
                query,
                retrieval_results,
                language=language
            )

        else:
            logger.error(f"Unknown generation method: {method}")
            raise ValueError(f"Unknown method: {method}")

    except Exception as e:
        logger.exception(f"Answer generation failed")
        return {
            "answer": f"Maaf, terjadi error: {str(e)}",
            "has_context": False,
            "error": str(e)
        }
        