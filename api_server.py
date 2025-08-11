from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi import Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from datetime import datetime, timezone
import uuid
import os
import requests
import base64
from typing import Dict, Optional
from msme_bot import (
    get_scheme_vector_store,
    get_dfl_vector_store,
    process_query_ultra_optimized,
    SessionData,
    cache_manager,
    llm_pool
)
from data import DataManager, STATE_NAME_TO_ID, GENDER_MAPPING
from tts import synthesize
import time
import logging
from fastapi.responses import StreamingResponse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from contextlib import asynccontextmanager

# Optimized executor with more workers
executor = ThreadPoolExecutor(max_workers=8)

# Configure logging for better performance
logging.getLogger("sse_starlette.sse").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(funcName)s — %(message)s",
)

# Lifespan manager for proper startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting optimized MSME Bot API")
    
    # Pre-warm vector stores in background
    asyncio.create_task(warm_up_services())
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down MSME Bot API")
    executor.shutdown(wait=False)

app = FastAPI(
    title="MSME Bot API - Optimized",
    version="2.0.0",
    lifespan=lifespan
)

# Optimized CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Restrict to only needed methods
    allow_headers=["*"],
    expose_headers=["*"],
)

# Connection pooling for external API calls
class OptimizedDataManager(DataManager):
    def __init__(self):
        super().__init__()
        self._http_session = None
    
    async def get_http_session(self):
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=5, connect=2)
            self._http_session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
            )
        return self._http_session

data_manager = OptimizedDataManager()

# Enhanced session cache with TTL
class SessionCache:
    def __init__(self):
        self.sessions: Dict[str, SessionData] = {}
        self.session_timestamps = {}
        self.max_age = 3600  # 1 hour
    
    def get(self, session_id: str) -> Optional[SessionData]:
        if session_id in self.sessions:
            # Check if session is still valid
            if (time.time() - self.session_timestamps.get(session_id, 0)) < self.max_age:
                return self.sessions[session_id]
            else:
                # Clean up expired session
                self.cleanup_session(session_id)
        return None
    
    def set(self, session_id: str, session: SessionData):
        self.sessions[session_id] = session
        self.session_timestamps[session_id] = time.time()
        
        # Cleanup old sessions periodically
        if len(self.sessions) > 1000:
            self.cleanup_old_sessions()
    
    def cleanup_session(self, session_id: str):
        self.sessions.pop(session_id, None)
        self.session_timestamps.pop(session_id, None)
    
    def cleanup_old_sessions(self):
        current_time = time.time()
        expired_sessions = [
            sid for sid, timestamp in self.session_timestamps.items()
            if (current_time - timestamp) > self.max_age
        ]
        
        for sid in expired_sessions:
            self.cleanup_session(sid)

# Initialize optimized session cache
session_cache = SessionCache()

# Environment variables
HQ_BASE_URL = os.getenv("HQ_API_URL", "https://customer-admin-test.haqdarshak.com")
HQ_ENDPOINT = "/person/get/citizen-details"

async def warm_up_services():
    """Pre-warm services in background"""
    try:
        logger.info("🔥 Warming up services...")
        
        # Pre-load vector stores
        asyncio.create_task(
            asyncio.get_event_loop().run_in_executor(executor, get_scheme_vector_store)
        )
        asyncio.create_task(
            asyncio.get_event_loop().run_in_executor(executor, get_dfl_vector_store)
        )
        
        # Test cache connectivity
        if cache_manager.redis_client:
            await cache_manager.redis_client.ping()
            logger.info("✅ Redis cache connected")
        
        logger.info("✅ Services warmed up")
    except Exception as e:
        logger.warning(f"⚠️ Warm up failed: {e}")

async def _load_session_optimized(session_id: str) -> SessionData | None:
    """Optimized session loading with caching"""
    # Check memory cache first
    sess = session_cache.get(session_id)
    if sess:
        return sess

    # Load from database asynchronously
    try:
        loop = asyncio.get_event_loop()
        doc = await asyncio.wait_for(
            loop.run_in_executor(
                executor, 
                data_manager.db.sessions.find_one, 
                {"session_id": session_id}
            ),
            timeout=1.0
        )
        
        if not doc:
            return None

        user = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                data_manager.find_user,
                doc["mobile_number"]
            ),
            timeout=1.0
        )
        
        if not user:
            return None

        sess = SessionData(user=user)
        session_cache.set(session_id, sess)
        return sess
        
    except asyncio.TimeoutError:
        logger.warning(f"Session load timeout for {session_id}")
        return None
    except Exception as e:
        logger.error(f"Session load error: {e}")
        return None

@app.post("/auth/token")
async def auth_token_optimized(payload: Dict[str, str]):
    """Optimized authentication with connection pooling"""
    token = payload.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="token required")
    
    api_url = f"{HQ_BASE_URL}{HQ_ENDPOINT}"
    
    try:
        # Use async HTTP client with connection pooling
        session = await data_manager.get_http_session()
        
        async with session.get(
            api_url,
            headers={"Authorization": f"Bearer {token}"}
        ) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=401, detail="invalid token")
            
            data = await resp.json()
            
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="authentication timeout")
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=500, detail="authentication failed")
    
    if data.get("responseCode") != "OK" or data.get("params", {}).get("status") != "successful":
        raise HTTPException(status_code=401, detail="invalid token")
    
    result = data.get("result", {})
    print(f"result novfal: {result}")
    gender_raw = result.get("gender", "") or ""
    gender = GENDER_MAPPING.get(gender_raw.upper(), gender_raw)
    state_name = result.get("state", "")
    
    user = {
        "fname": result.get("firstName", ""),
        "lname": result.get("lastName", ""),
        "mobile_number": result.get("contactNumber", ""),
        "gender": gender,
        "state_name": state_name,
        "dob": result.get("dob", ""),
        "pincode": result.get("pincode", ""),
        "business_name": result.get("bussinessName", ""),
        "business_category": result.get("employmentType", ""),
        "language": "English",
        "state_id": STATE_NAME_TO_ID.get(state_name, "Unknown"),
        "user_type": result.get("user_type",1),  # Default to 1 if not set
    }
    
    session_id = uuid.uuid4().hex
    
    # Start session in background
    async def run_blocking():
        asyncio.get_event_loop().run_in_executor(
            executor,
            data_manager.start_session,
            user["mobile_number"],
            session_id,
            user
        )
    asyncio.create_task(run_blocking())

    
    # Cache session immediately
    session_data = SessionData(user=user)
    session_cache.set(session_id, session_data)
    
    return {"session_id": session_id, "user": user}

# FIXED ENDPOINT
@app.get("/chat")
async def chat_get_ultra_optimized(session_id: str, query: str):
    """Ultra-optimized chat endpoint with aggressive performance tuning - FIXED VERSION"""
    start_time = time.perf_counter()
    
    # Fast session loading
    sess = await _load_session_optimized(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    
    if not isinstance(sess, SessionData):
        raise HTTPException(500, "invalid session")

    # Add user message (non-blocking)
    user_msg = {
        "role": "user", 
        "content": query, 
        "timestamp": datetime.utcnow().isoformat()
    }
    sess.messages.append(user_msg)
    
    logger.info(f"⚡ Session loaded: {time.perf_counter() - start_time:.3f}s")
    
    # Process query with ultra optimization
    token_stream, audio_task = await process_query_ultra_optimized(
        query,
        session_id,
        sess.user["mobile_number"],
        sess,
        user_language=sess.user.get("language"),
        stream=True,
    )
    
    logger.info(f"⚡ Query processed: {time.perf_counter() - start_time:.3f}s")

    async def ultra_fast_event_generator():
        final_text = ""
        token_count = 0
        stream_start = time.perf_counter()
        
        try:
            # Check if token_stream is actually an async generator
            if hasattr(token_stream, '__aiter__'):
                # Stream response tokens with batching for performance
                batch = ""
                batch_size = 3  # Send multiple tokens at once
                
                async for token in token_stream:
                    final_text += token
                    batch += token
                    token_count += 1
                    
                    # Send in batches for better performance
                    if len(batch) >= batch_size or token_count % 5 == 0:
                        yield {"data": batch}
                        batch = ""
                
                # Send any remaining batch
                if batch:
                    yield {"data": batch}
                    
            else:
                # token_stream is a string, not an async generator
                logger.warning("Token stream is not async generator, sending as complete text")
                final_text = str(token_stream)
                yield {"data": final_text}
            
            logger.info(f"⚡ Streaming: {time.perf_counter() - stream_start:.3f}s ({token_count} tokens)")

            # Audio generation with timeout and error handling
            if audio_task and final_text.strip():
                audio_start = time.perf_counter()
                
                try:
                    # Generate audio script with aggressive timeout
                    script_task = asyncio.create_task(audio_task(final_text))
                    script = await asyncio.wait_for(script_task, timeout=2.5)
                    
                    if script and script.strip():
                        # Generate audio in thread pool (most CPU intensive part)
                        audio_bytes = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                executor, synthesize, script[:200], "Hindi"  # Limit script length
                            ),
                            timeout=3.0
                        )
                        
                        if audio_bytes and len(audio_bytes) > 0:
                            b64_audio = base64.b64encode(audio_bytes).decode()
                            logger.info(f"⚡ Audio: {time.perf_counter() - audio_start:.3f}s ({len(audio_bytes)} bytes)")
                            yield {"event": "audio", "data": b64_audio}
                        
                except asyncio.TimeoutError:
                    logger.warning("Audio generation timeout - skipping")
                except Exception as e:
                    logger.error(f"Audio error: {e}")

            # Update session messages
            assistant_msg = {
                "role": "assistant",
                "content": final_text,
                "timestamp": datetime.utcnow().isoformat()
            }
            sess.messages.append(assistant_msg)

            # Final event
            yield {"event": "done", "data": f"completed in {time.perf_counter() - start_time:.2f}s"}

        except Exception as e:
            logger.error(f"Event generation error: {e}")
            yield {"event": "error", "data": str(e)}

    # Optimized headers for better streaming performance
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Content-Type": "text/event-stream",
    }

    return EventSourceResponse(
        ultra_fast_event_generator(),
        headers=headers,
        ping=30  # Longer ping interval
    )
    

@app.get("/history/{session_id}")
async def get_history_optimized(session_id: str):
    """Optimized history retrieval"""
    session = await _load_session_optimized(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    try:
        # Get conversations asynchronously with timeout
        loop = asyncio.get_event_loop()
        convos = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                data_manager.get_conversations,
                session.user["mobile_number"]
            ),
            timeout=2.0
        )
        
        # Fast message processing
        messages = []
        for conv in reversed(convos[-10:]):  # Limit to last 10 conversations
            conv_msgs = conv.get("messages", [])[-20:]  # Limit messages per conversation
            messages.extend(conv_msgs)

        # Sort by timestamp (keep only essential data)
        def safe_timestamp(msg):
            ts = msg.get("timestamp")
            if isinstance(ts, datetime):
                return ts.timestamp()
            elif isinstance(ts, (int, float)):
                return float(ts)
            elif isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                except ValueError:
                    return 0
            return 0

        messages.sort(key=safe_timestamp)
        
        # Limit response size for better performance
        return {"messages": messages[-50:]}  # Last 50 messages only
        
    except asyncio.TimeoutError:
        logger.warning("History retrieval timeout")
        return {"messages": []}
    except Exception as e:
        logger.error(f"History error: {e}")
        return {"messages": []}

@app.get("/welcome/{session_id}")
async def get_welcome_optimized(session_id: str):
    """Optimized welcome message endpoint"""
    session = await _load_session_optimized(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    mobile = session.user["mobile_number"]
    user_type = session.user["user_type"] 
    try:
        # Quick check for existing conversations
        loop = asyncio.get_event_loop()
        conversations = await asyncio.wait_for(
            loop.run_in_executor(executor, data_manager.get_conversations, mobile),
            timeout=1.0
        )
        
        # Fast welcome check
        has_welcome = any(
            "welcome" in (msg.get("content", "")).lower() 
            for conv in conversations[-3:]  # Check only last 3 conversations
            for msg in conv.get("messages", [])[-5:]  # Check only last 5 messages
        )
        
        if has_welcome:
            return {"welcome": None, "audio": None}

    except asyncio.TimeoutError:
        logger.warning("Welcome check timeout - assuming new user")
    except Exception as e:
        logger.error(f"Welcome check error: {e}")

    # Generate welcome message
    start_time = time.perf_counter()
    
    response_text, audio_task = await process_query_ultra_optimized(
        "welcome",
        session_id,
        mobile,
        session,
        user_language=session.user.get("language"),
        stream=False,
        user_type=user_type
    )
    
    logger.info(f"⚡ Welcome generated: {time.perf_counter() - start_time:.3f}s")

    # Generate audio with timeout
    b64_audio = None
    if audio_task and response_text:
        try:
            audio_start = time.perf_counter()
            script = await asyncio.wait_for(audio_task(response_text), timeout=2.0)
            
            if script and script.strip():
                audio_bytes = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        executor, synthesize, script, "Hindi"
                    ),
                    timeout=3.0
                )
                
                if audio_bytes:
                    b64_audio = base64.b64encode(audio_bytes).decode()
                    logger.info(f"⚡ Welcome audio: {time.perf_counter() - audio_start:.3f}s")
                    
        except asyncio.TimeoutError:
            logger.warning("Welcome audio timeout")
        except Exception as e:
            logger.error(f"Welcome audio error: {e}")

    return {
        "welcome": response_text,
        "audio": b64_audio,
    }

@app.get("/health")
async def health_check():
    """Fast health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "cache_stats": cache_manager.cache_stats,
        "active_sessions": len(session_cache.sessions)
    }

@app.get("/metrics")
async def get_metrics():
    """Performance metrics endpoint"""
    return {
        "cache_stats": cache_manager.cache_stats,
        "active_sessions": len(session_cache.sessions),
        "llm_pool": {
            "available_llms": len(llm_pool.available_llms),
            "available_intent_llms": len(llm_pool.available_intent_llms)
        },
        "session_cache_size": len(session_cache.sessions)
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if hasattr(data_manager, '_http_session') and data_manager._http_session:
        await data_manager._http_session.close()
    
    executor.shutdown(wait=False)

# Error handlers for better performance
@app.exception_handler(asyncio.TimeoutError)
async def timeout_handler(request, exc):
    return HTTPException(status_code=408, detail="Request timeout")

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")

# Optional: Add request middleware for performance monitoring
@app.middleware("http")
async def performance_middleware(request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    
    if process_time > 5.0:  # Log slow requests
        logger.warning(f"Slow request: {request.url.path} took {process_time:.3f}s")
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# UPDATED ENDPOINT with enhanced audio handling
@app.get("/chat")
async def chat_get_with_reliable_audio(session_id: str, query: str):
    """Enhanced chat endpoint with reliable audio generation"""
    start_time = time.perf_counter()
    
    # Fast session loading
    sess = await _load_session_optimized(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    
    if not isinstance(sess, SessionData):
        raise HTTPException(500, "invalid session")

    # Add user message
    user_msg = {
        "role": "user", 
        "content": query, 
        "timestamp": datetime.utcnow().isoformat()
    }
    sess.messages.append(user_msg)
    
    logger.info(f"⚡ Session loaded: {time.perf_counter() - start_time:.3f}s")
    
    # Process query
    token_stream, audio_task = await process_query_optimized(
        query,
        session_id,
        sess.user["mobile_number"],
        sess,
        user_language=sess.user.get("language"),
        stream=True,
    )
    
    logger.info(f"⚡ Query processed: {time.perf_counter() - start_time:.3f}s")

    async def enhanced_event_generator():
        final_text = ""
        token_count = 0
        stream_start = time.perf_counter()
        
        try:
            # Stream response tokens
            if hasattr(token_stream, '__aiter__'):
                batch = ""
                batch_size = 3
                
                async for token in token_stream:
                    final_text += token
                    batch += token
                    token_count += 1
                    
                    if len(batch) >= batch_size or token_count % 5 == 0:
                        yield {"data": batch}
                        batch = ""
                
                if batch:
                    yield {"data": batch}
                    
            else:
                final_text = str(token_stream)
                yield {"data": final_text}
            
            logger.info(f"⚡ Streaming: {time.perf_counter() - stream_start:.3f}s ({token_count} tokens)")

            # Enhanced audio generation with session tracking
            if audio_task and final_text.strip():
                audio_start = time.perf_counter()
                
                try:
                    # Generate audio script with session ID
                    script_task = asyncio.create_task(audio_task(final_text))
                    script = await asyncio.wait_for(script_task, timeout=4.0)
                    
                    if script and script.strip():
                        logger.info(f"🔊 Audio script ready for session {session_id}")
                        
                        # Generate audio with enhanced handling
                        audio_success, b64_audio = await handle_audio_in_endpoint(
                            script, session_id, timeout=5.0
                        )
                        
                        if audio_success and b64_audio:
                            elapsed_audio = time.perf_counter() - audio_start
                            logger.info(f"⚡ Audio complete: {elapsed_audio:.3f}s")
                            yield {"event": "audio", "data": b64_audio}
                        else:
                            logger.warning(f"🔊 Audio generation failed for session {session_id}")
                    else:
                        logger.warning(f"🔊 No audio script generated for session {session_id}")
                        
                except asyncio.TimeoutError:
                    logger.warning(f"🔊 Audio generation timeout for session {session_id}")
                except Exception as e:
                    logger.error(f"🔊 Audio error for session {session_id}: {e}")

            # Update session messages
            assistant_msg = {
                "role": "assistant",
                "content": final_text,
                "timestamp": datetime.utcnow().isoformat()
            }
            sess.messages.append(assistant_msg)

            # Final event with audio stats
            total_time = time.perf_counter() - start_time
            yield {"event": "done", "data": f"completed in {total_time:.2f}s"}

        except Exception as e:
            logger.error(f"Event generation error: {e}")
            yield {"event": "error", "data": str(e)}

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Content-Type": "text/event-stream",
    }

    return EventSourceResponse(
        enhanced_event_generator(),
        headers=headers,
        ping=30
    )

# Monitoring function to check audio system health
async def get_audio_stats():
    """Get audio system statistics for monitoring"""
    return {
        "active_requests": len(audio_tracker.active_requests),
        "stats": audio_tracker.stats,
        "semaphore_available": audio_semaphore._value,
        "thread_pool_active": audio_executor._threads
    }


if __name__ == "__main__":
    import uvicorn
    
    # Optimized uvicorn configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for development, increase for production
        loop="asyncio",
        http="httptools",
        lifespan="on",
        access_log=False,  # Disable for better performance
        server_header=False,
        date_header=False,
    )
