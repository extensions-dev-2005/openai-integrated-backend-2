import asyncio
import io
import logging
import os
import tempfile
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import hashlib
from concurrent.futures import ThreadPoolExecutor
import functools

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# New imports for ffmpeg
import ffmpeg
import subprocess

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nova_ai.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Global variables
openai_client = None
thread_pool = ThreadPoolExecutor(max_workers=4)
transcription_cache: Dict[str, Dict] = {}  # Simple in-memory cache with TTL
client_session = None
CACHE_MAX_SIZE = 100  # Limit cache size to prevent memory issues

# Enhanced Request/Response models
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    language: Optional[str] = Field(None, description="Optional language hint")

class TranscriptionResponse(BaseModel):
    text: str
    duration: Optional[float] = None
    language: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: float
    cached: bool = False

class SummaryResponse(BaseModel):
    summary: str
    key_points: list[str] = []
    action_items: list[str] = []
    processing_time: float

class ResponseSuggestionResponse(BaseModel):
    suggestion: str
    context_identified: str
    confidence: Optional[str] = None
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    message: str
    models_loaded: Dict[str, bool]
    uptime: float
    version: str = "2.2.0"  # Updated version

# Cache utilities
def get_cache_key(content: bytes) -> str:
    """Generate cache key from audio content hash"""
    return hashlib.md5(content).hexdigest()

def cache_transcription(key: str, result: dict, ttl: int = 3600):
    """Cache transcription result with TTL, evict oldest if over max size"""
    if len(transcription_cache) >= CACHE_MAX_SIZE:
        # Evict oldest entry
        oldest_key = min(transcription_cache, key=lambda k: transcription_cache[k]['timestamp'])
        del transcription_cache[oldest_key]
        logger.info(f"🧹 Evicted oldest cache entry: {oldest_key}")

    transcription_cache[key] = {
        'result': result,
        'timestamp': time.time(),
        'ttl': ttl
    }

def get_cached_transcription(key: str) -> Optional[dict]:
    """Retrieve cached transcription if valid"""
    if key not in transcription_cache:
        return None

    cached_data = transcription_cache[key]
    if time.time() - cached_data['timestamp'] > cached_data['ttl']:
        del transcription_cache[key]
        return None

    return cached_data['result']

# Startup/Shutdown context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global openai_client, client_session

    logger.info("🚀 Starting Nova.AI Backend v2.2.0 with OpenAI...")
    start_time = time.time()

    try:
        # Initialize HTTP client
        client_session = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),  # Increased timeout for robustness
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )

        # Initialize OpenAI client
        await initialize_openai_client()

        # Clear old cache entries on startup
        cleanup_cache()

        app.state.start_time = start_time
        logger.info(f"✅ Backend initialized successfully in {time.time() - start_time:.2f}s")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        app.state.start_time = start_time
        app.state.startup_error = str(e)

    yield  # Application runs here

    # Cleanup
    logger.info("🔄 Shutting down Nova.AI Backend...")
    if client_session:
        await client_session.aclose()
    thread_pool.shutdown(wait=True)
    transcription_cache.clear()
    logger.info("✅ Shutdown complete")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Nova.AI Backend",
    version="2.2.0",
    description="High-performance meeting transcription and AI assistant with OpenAI",
    lifespan=lifespan
)

# Enhanced middleware stack
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"]
)

async def initialize_openai_client():
    """Enhanced OpenAI client initialization with validation"""
    global openai_client

    try:
        from openai import OpenAI

        # Multiple API key sources
        api_key = (
            os.getenv("OPENAI_API_KEY") or
            os.getenv("OPENAI_KEY") or
            _get_colab_key()
        )

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        # Validate API key format
        if not api_key.startswith('sk-'):
            raise ValueError("Invalid OpenAI API key format")

        openai_client = OpenAI(api_key=api_key)

        # Test API connectivity
        await test_openai_connection()
        logger.info("✅ OpenAI client initialized and tested successfully")

    except Exception as e:
        logger.error(f"❌ OpenAI initialization failed: {e}")
        openai_client = None
        raise

def _get_colab_key() -> Optional[str]:
    """Get API key from Google Colab if available"""
    try:
        from google.colab import userdata
        return userdata.get("OPENAI_API_KEY")
    except:
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def test_openai_connection():
    """Test OpenAI API connectivity with retry"""
    if not openai_client:
        raise ValueError("OpenAI client not initialized")

    try:
        # Test with minimal request using OpenAI's chat completions
        response = openai_client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4o-mini",
            max_tokens=5
        )
        return True
    except Exception as e:
        logger.error(f"OpenAI connection test failed: {e}")
        raise

def cleanup_cache():
    """Remove expired cache entries"""
    current_time = time.time()
    expired_keys = [
        key for key, data in list(transcription_cache.items())
        if current_time - data['timestamp'] > data['ttl']
    ]
    for key in expired_keys:
        del transcription_cache[key]

    if expired_keys:
        logger.info(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")

# Enhanced audio validation
def validate_audio_file(content_type: str, file_size: int, filename: str = "") -> tuple[bool, str]:
    """Enhanced audio file validation for OpenAI Whisper"""

    # Size validation (25MB for OpenAI Whisper, but buffer for safety)
    max_size = 25 * 1024 * 1024 - 1024  # 25MB - 1KB buffer
    if file_size > max_size:
        return False, f"File too large: {file_size/1024/1024:.1f}MB (max: 25MB)"

    if file_size < 100:  # Minimum viable audio (reduced for edge cases like short clips)
        return False, "File too small (minimum: 100 bytes)"

    # Since we use ffmpeg, we can support almost any audio/video format as input
    # But output to supported OpenAI formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
    return True, "Valid for processing"

def convert_audio_optimized(audio_bytes: bytes, target_format: str = "mp3") -> bytes:
    """Optimized audio conversion using ffmpeg-python"""
    try:
        # Use ffmpeg to convert to 16kHz mono mp3 (good compression, supported by OpenAI)
        process = (
            ffmpeg
            .input('pipe:0')
            .output('pipe:1', format=target_format, ar=16000, ac=1, audio_bitrate='64k')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )

        stdout, stderr = process.communicate(input=audio_bytes)
        
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {stderr.decode('utf-8')}")

        converted = stdout
        logger.info(f"Audio converted: {len(audio_bytes)} -> {len(converted)} bytes")
        return converted

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg subprocess error: {e}")
        raise
    except Exception as e:
        logger.warning(f"Conversion failed: {e}, using original if possible")
        # Fallback: if conversion fails, try to use original if it's a supported format
        return audio_bytes

# Async wrapper for CPU-intensive operations
def run_in_thread(func):
    """Decorator to run CPU-intensive functions in thread pool"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(thread_pool, func, *args, **kwargs)
    return wrapper

@run_in_thread
def process_audio_sync(audio_bytes: bytes, content_type: str, filename: str) -> tuple[bytes, str]:
    """Synchronous audio processing for thread execution"""
    try:
        # Validate first
        is_valid, msg = validate_audio_file(content_type, len(audio_bytes), filename)
        if not is_valid:
            raise ValueError(msg)

        # Always convert with ffmpeg for consistency (handle corrupted files better)
        processed_audio = convert_audio_optimized(audio_bytes)
        return processed_audio, "mp3"

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        raise

# API Endpoints
@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API info"""
    return {
        "service": "Nova.AI Backend API",
        "version": "2.2.0",
        "status": "operational",
        "ai_provider": "OpenAI",
        "features": {
            "transcription": True,
            "summarization": True,
            "response_suggestions": True,
            "caching": True,
            "async_processing": True,
            "ffmpeg_processing": True
        },
        "endpoints": ["/health", "/transcribe", "/summarize", "/suggest_response"],
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    uptime = time.time() - getattr(app.state, 'start_time', time.time())
    startup_error = getattr(app.state, 'startup_error', None)

    status = "healthy"
    message = "All systems operational"

    models_loaded = {
        "openai_client": openai_client is not None,
        "whisper_api": openai_client is not None,
        "text_generation": openai_client is not None,
        "thread_pool": not thread_pool._shutdown,
        "cache": bool(transcription_cache)  # Check if cache is usable
    }

    if startup_error:
        status = "degraded"
        message = f"Startup issues: {startup_error}"
    elif openai_client is None:
        status = "degraded" 
        message = "OpenAI client unavailable"
    elif any(not loaded for loaded in models_loaded.values()):
        status = "degraded"
        message = "Some components unavailable"

    return HealthResponse(
        status=status,
        message=message,
        uptime=uptime,
        models_loaded=models_loaded
    )

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file for transcription"),
    language: Optional[str] = Query(None, description="Optional language code for transcription (auto-detect if None)"),
    background_tasks: BackgroundTasks = None
):
    """Enhanced transcription with OpenAI Whisper, caching, and language support"""

    if openai_client is None:
        raise HTTPException(
            status_code=503,
            detail="Transcription service unavailable - OpenAI client not initialized"
        )

    start_time = time.time()

    try:
        # Read and validate audio
        audio_bytes = await audio.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Check cache first
        cache_key = get_cache_key(audio_bytes)
        cached_result = get_cached_transcription(cache_key)

        if cached_result:
            logger.info(f"🎯 Cache hit for transcription")
            cached_result['processing_time'] = time.time() - start_time
            cached_result['cached'] = True
            return TranscriptionResponse(**cached_result)

        logger.info(f"🎵 Processing audio: {audio.filename} ({len(audio_bytes)} bytes)")

        # Process audio in thread pool
        processed_audio, audio_format = await process_audio_sync(
            audio_bytes, 
            audio.content_type or "", 
            audio.filename or ""
        )

        # Create temporary file for OpenAI API
        with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as temp_file:
            temp_file.write(processed_audio)
            temp_file_path = temp_file.name

        try:
            # OpenAI Whisper transcription with retry
            @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), retry_if_exception_type(Exception))
            def transcribe_with_openai():
                logger.info("🔊 Starting OpenAI Whisper transcription...")
                with open(temp_file_path, "rb") as file:
                    return openai_client.audio.transcriptions.create(
                        model="whisper-1",  # Use standard Whisper model
                        file=file,
                        response_format="verbose_json",
                        temperature=0.0,
                        language=language if language else None  # Auto-detect if None
                    )

            transcription_response = transcribe_with_openai()

            # Process transcription result
            raw_text = transcription_response.text.strip() if transcription_response.text else ""

            if not raw_text:
                logger.warning("⚠️ Transcription returned empty text - possible silent audio")
                raw_text = "[No audible speech detected]"

            # Enhanced text cleaning
            cleaned_text = clean_transcription_text(raw_text)

            processing_time = time.time() - start_time

            result = {
                'text': cleaned_text,
                'duration': getattr(transcription_response, 'duration', None),
                'language': getattr(transcription_response, 'language', language or "auto-detected"),
                'confidence': calculate_confidence(cleaned_text),
                'processing_time': processing_time,
                'cached': False
            }

            # Cache result
            cache_transcription(cache_key, result)

            # Schedule cache cleanup
            if background_tasks:
                background_tasks.add_task(cleanup_cache)

            logger.info(f"✅ Transcription completed in {processing_time:.2f}s")
            return TranscriptionResponse(**result)

        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Temp file cleanup failed: {e}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        logger.error(traceback.format_exc())

        # Enhanced error handling
        error_str = str(e).lower()
        if "rate_limit" in error_str:
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Please try again in a moment."
            )
        elif "file_size" in error_str or "too large" in error_str:
            raise HTTPException(
                status_code=413, 
                detail="Audio file too large (max: 25MB)"
            )
        elif "invalid" in error_str or "unsupported" in error_str:
            raise HTTPException(
                status_code=415, 
                detail="Unsupported or invalid audio format"
            )
        elif "no speech" in error_str or len(raw_text) == 0:
            raise HTTPException(
                status_code=400, 
                detail="No speech detected in audio"
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Transcription failed: {str(e)[:200]}"
            )

def clean_transcription_text(text: str) -> str:
    """Enhanced text cleaning for transcriptions"""
    if not text:
        return ""

    # Remove common artifacts
    artifacts = [
        "[Music]", "[Applause]", "[Laughter]", "[Background noise]",
        "(Music)", "(Applause)", "(Laughter)", "♪", "MBC 뉴스",
        "Thanks for watching", "Thank you for watching",
        "ご視聴ありがとうございました", "Please subscribe"
    ]

    cleaned = text
    for artifact in artifacts:
        cleaned = cleaned.replace(artifact, " ")

    # Clean up whitespace and formatting
    import re
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\.{2,}', '.', cleaned)
    cleaned = re.sub(r'^\s*[\[\(].*?[\]\)]\s*', '', cleaned)  # Remove leading brackets
    cleaned = cleaned.strip()

    return cleaned

def calculate_confidence(text: str) -> float:
    """Simple confidence calculation based on text characteristics"""
    if not text or text == "[No audible speech detected]":
        return 0.0

    # Basic heuristics
    word_count = len(text.split())
    if word_count == 0:
        return 0.0

    # Factors that increase confidence
    has_punctuation = any(c in text for c in '.!?')
    avg_word_length = sum(len(word) for word in text.split()) / word_count
    has_varied_content = len(set(text.split())) / word_count > 0.5  # Avoid repetitive text

    confidence = 0.5  # Base confidence

    if has_punctuation:
        confidence += 0.15
    if 3 <= avg_word_length <= 8:
        confidence += 0.15
    if word_count >= 5:
        confidence += 0.1
    if has_varied_content:
        confidence += 0.1

    return min(confidence, 1.0)

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_text(request: TextRequest):
    """Enhanced summarization with OpenAI models"""

    if openai_client is None:
        raise HTTPException(status_code=503, detail="AI service unavailable")

    start_time = time.time()

    try:
        text = request.text.strip()
        if len(text) < 50:  # Minimum text for meaningful summary
            raise HTTPException(
                status_code=400, 
                detail="Text too short for summarization (minimum: 50 characters)"
            )

        # Truncate if too long (approximate token limit for gpt-4o-mini ~128k tokens, but safe)
        max_chars = 100000  
        if len(text) > max_chars:
            text = text[:max_chars] + "... [truncated]"

        logger.info(f"📝 Generating summary for {len(text)} characters")

        # Enhanced prompt for structured output
        messages = [
            {
                "role": "system",
                "content": """You are an expert meeting analyst. Create structured summaries with:
1. A concise 2-3 sentence overview
2. Key discussion points as bullet points  
3. Clear action items with owners if mentioned
4. Important decisions made

Keep summaries professional and actionable."""
            },
            {
                "role": "user", 
                "content": f"""Analyze this meeting transcript and provide a structured summary:

{text}

Format your response as:
SUMMARY: [2-3 sentence overview]
KEY POINTS: [bullet points of main discussion items]
ACTION ITEMS: [specific tasks mentioned with owners if available]
DECISIONS: [any decisions made]"""
            }
        ]

        # Retry for summarization
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), retry_if_exception_type(Exception))
        def summarize_with_openai():
            return openai_client.chat.completions.create(
                messages=messages,
                model="gpt-4o-mini",
                max_tokens=800,
                temperature=0.3,
                top_p=0.9
            )

        response = summarize_with_openai()
        raw_summary = response.choices[0].message.content.strip()

        # Parse structured response
        summary_parts = parse_structured_summary(raw_summary)

        processing_time = time.time() - start_time
        logger.info(f"✅ Summary generated in {processing_time:.2f}s")

        return SummaryResponse(
            summary=summary_parts['summary'],
            key_points=summary_parts['key_points'],
            action_items=summary_parts['action_items'],
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Summarization error: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

def parse_structured_summary(raw_text: str) -> dict:
    """Parse structured summary response"""
    import re

    result = {
        'summary': '',
        'key_points': [],
        'action_items': []
    }

    # Extract sections with more robust regex
    summary_match = re.search(r'SUMMARY:\s*(.*?)(?=KEY POINTS:|ACTION ITEMS:|DECISIONS:|$)', raw_text, re.DOTALL | re.IGNORECASE)
    if summary_match:
        result['summary'] = summary_match.group(1).strip()

    key_points_match = re.search(r'KEY POINTS:\s*(.*?)(?=ACTION ITEMS:|DECISIONS:|$)', raw_text, re.DOTALL | re.IGNORECASE)
    if key_points_match:
        points_text = key_points_match.group(1).strip()
        result['key_points'] = [point.strip('- •*').strip() for point in re.split(r'\n', points_text) if point.strip()]

    action_match = re.search(r'ACTION ITEMS:\s*(.*?)(?=DECISIONS:|$)', raw_text, re.DOTALL | re.IGNORECASE)
    if action_match:
        actions_text = action_match.group(1).strip()
        result['action_items'] = [action.strip('- •*').strip() for action in re.split(r'\n', actions_text) if action.strip()]

    # Fallback to original text if parsing fails
    if not result['summary']:
        result['summary'] = raw_text[:300] + "..." if len(raw_text) > 300 else raw_text

    return result

@app.post("/suggest_response", response_model=ResponseSuggestionResponse)
async def suggest_response(request: TextRequest):
    """Enhanced response suggestions with OpenAI reasoning models"""

    if openai_client is None:
        raise HTTPException(status_code=503, detail="AI service unavailable")

    start_time = time.time()

    try:
        # Get recent context (increased for better analysis)
        text = request.text[-16000:] if len(request.text) > 16000 else request.text  # Increased for gpt-4o-mini

        if len(text.strip()) < 20:
            raise HTTPException(
                status_code=400,
                detail="Insufficient context for response suggestion"
            )

        logger.info(f"💭 Generating response suggestion for {len(text)} characters")

        # Fallback to chat completion (assuming responses API might not be standard)
        messages = [
            {
                "role": "system",
                "content": """You are a professional meeting assistant. Your task is to:
1. Identify the most recent question, request, or discussion point
2. Provide a brief, professional response suggestion
3. Indicate your confidence level (High/Medium/Low)

Keep responses concise but complete (1-3 sentences). Be professional and contextually appropriate."""
            },
            {
                "role": "user",
                "content": f"""Analyze this meeting transcript and suggest a professional response to the most recent query or discussion point:

{text}

Provide your response in this format:
CONTEXT: [what you're responding to]
SUGGESTION: [your suggested response]
CONFIDENCE: [High/Medium/Low]"""
            }
        ]

        # Retry for suggestion
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), retry_if_exception_type(Exception))
        def suggest_with_openai():
            return openai_client.chat.completions.create(
                messages=messages,
                model="gpt-4o-mini",
                max_tokens=400,
                temperature=0.4,
                top_p=0.9
            )

        response = suggest_with_openai()
        raw_response = response.choices[0].message.content.strip()

        parsed_response = parse_response_suggestion(raw_response)

        processing_time = time.time() - start_time
        logger.info(f"✅ Response suggestion generated in {processing_time:.2f}s")

        return ResponseSuggestionResponse(
            suggestion=parsed_response['suggestion'],
            context_identified=parsed_response['context'],
            confidence=parsed_response['confidence'],
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Response suggestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Response suggestion failed: {str(e)}")

def parse_response_suggestion(raw_text: str) -> dict:
    """Parse response suggestion output"""
    import re

    result = {
        'suggestion': '',
        'context': '',
        'confidence': 'Medium'
    }

    # Extract sections with more robust regex
    context_match = re.search(r'CONTEXT:\s*(.*?)(?=SUGGESTION:|CONFIDENCE:|$)', raw_text, re.DOTALL | re.IGNORECASE)
    if context_match:
        result['context'] = context_match.group(1).strip()

    suggestion_match = re.search(r'SUGGESTION:\s*(.*?)(?=CONFIDENCE:|$)', raw_text, re.DOTALL | re.IGNORECASE)
    if suggestion_match:
        result['suggestion'] = suggestion_match.group(1).strip()

    confidence_match = re.search(r'CONFIDENCE:\s*(High|Medium|Low)', raw_text, re.IGNORECASE)
    if confidence_match:
        result['confidence'] = confidence_match.group(1).title()

    # Fallback if parsing fails
    if not result['suggestion']:
        result['suggestion'] = raw_text
        result['context'] = "Could not parse context"

    return result

# Enhanced error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with detailed logging"""
    logger.error(f"🚨 Unhandled exception on {request.method} {request.url}")
    logger.error(f"Exception: {exc}")
    logger.error(traceback.format_exc())

    # Don't expose internal errors in production
    if os.getenv("ENVIRONMENT") == "production":
        detail = "An internal error occurred. Please try again."
    else:
        detail = str(exc)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": detail,
            "timestamp": time.time()
        }
    )

# Add response time header
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(f"{process_time:.3f}")
    return response

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    uvicorn.run(
        "app:app",  # Adjust if file name changes
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        access_log=True,
        reload=os.getenv("ENVIRONMENT") != "production",
        workers=2  # Add workers for better concurrency
    )
