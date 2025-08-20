from fastapi import FastAPI, HTTPException
from fastapi import Query
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from datetime import datetime, timezone
import uuid
import os
import requests
import base64
from typing import Dict
from msme_bot import (
    scheme_vector_store,
    dfl_vector_store,
    process_query_optimized,
    SessionData,
)
from data import DataManager, STATE_NAME_TO_ID, GENDER_MAPPING
from tts import synthesize
import time
import logging
from fastapi.responses import StreamingResponse
import asyncio
from concurrent.futures import ThreadPoolExecutor


executor = ThreadPoolExecutor(max_workers=4)


logging.getLogger("sse_starlette.sse").setLevel(logging.INFO)
# Reduce OpenAI + HTTPX debug logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)          # <─ defined exactly once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
)

app = FastAPI()

# Enable CORS so a separate React frontend can call the API without issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

data_manager = DataManager()

# In-memory store for active sessions
sessions: Dict[str, SessionData] = {}

HQ_BASE_URL = os.getenv("HQ_API_URL", "https://customer-admin-test.haqdarshak.com")
HQ_ENDPOINT = "/person/get/citizen-details"

def _load_session(session_id: str) -> SessionData | None:
    """Return SessionData from cache or rebuild it from Mongo if needed."""
    sess = sessions.get(session_id)
    if sess:
        return sess

    doc = data_manager.db.sessions.find_one({"session_id": session_id})
    if not doc:
        return None                       # real 404 – never authenticated

    user = data_manager.find_user(doc["mobile_number"])
    if not user:
        return None                       # stale session record

    sess = SessionData(user=user)
    sessions[session_id] = sess          # cache for next call
    return sess

@app.post("/auth/token")
def auth_token(payload: Dict[str, str]):
    token = payload.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="token required")
    api_url = f"{HQ_BASE_URL}{HQ_ENDPOINT}"
    try:
        resp = requests.get(api_url, headers={"Authorization": f"Bearer {token}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="invalid token")
    data = resp.json()
    if data.get("responseCode") != "OK" or data.get("params", {}).get("status") != "successful":
        raise HTTPException(status_code=401, detail="invalid token")
    result = data.get("result", {})
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
        "user_type":result.get("user_type", 1),
    }
    session_id = uuid.uuid4().hex
    data_manager.start_session(user["mobile_number"], session_id, user)
    sessions[session_id] = SessionData(user=user)
    return {"session_id": session_id, "user": user}


async def chat_get_optimized(session_id: str, query: str):
    """Optimized FastAPI chat endpoint"""
    sess = await _load_session_async(session_id)  # Make this async too
    if not sess:
        raise HTTPException(404, "session not found")
    if not isinstance(sess, SessionData):
        raise HTTPException(500, "Corrupt session object")

    user_msg = {"role": "user", "content": query, "timestamp": uuid.uuid4().hex}
    sess.messages.append(user_msg)
    user_type = sess.user["user_type"]
    start_time = time.perf_counter()
    
    # Use optimized async version
    token_stream, audio_task = await process_query_optimized(
        query,
        scheme_vector_store,
        dfl_vector_store,
        session_id,
        sess.user["mobile_number"],
        sess,
        user_type,
        user_language=sess.user.get("language"),
        stream=True,
    )
    print(f"⏱️ Optimized process_query: {time.perf_counter() - start_time:.3f}s")

    async def event_generator():
        final_text = ""
        stream_start = time.perf_counter()
        
        try:
            # Stream response tokens
            async for token in token_stream:
                final_text += token
                yield {"data": token}
            
            print(f"⏱️ Streaming complete: {time.perf_counter() - stream_start:.3f}s")

            # Generate audio in background (non-blocking)
            if audio_task:
                audio_start = time.perf_counter()
                try:
                    # Start audio generation
                    script_task = asyncio.create_task(audio_task(final_text))
                    
                    # Wait for audio script with timeout
                    script = await asyncio.wait_for(script_task, timeout=3.0)
                    
                    # Generate actual audio in thread pool (CPU intensive)
                    loop = asyncio.get_event_loop()
                    audio_bytes = await loop.run_in_executor(
                        executor, synthesize, script, "Hindi"  # Your synthesize function
                    )
                    
                    b64_audio = base64.b64encode(audio_bytes).decode()
                    print(f"⏱️ Audio generated: {time.perf_counter() - audio_start:.3f}s")
                    logger.info(f"Audio event generated – {len(audio_bytes)} bytes")
                    yield {"event": "audio", "data": b64_audio}
                    
                except asyncio.TimeoutError:
                    logger.warning("Audio generation timeout - continuing without audio")
                except Exception as e:
                    logger.error(f"Audio generation failed: {e}")

            # Add assistant message to session
            assistant_msg = {
                "role": "assistant",
                "content": final_text,
                "timestamp": uuid.uuid4().hex
            }
            sess.messages.append(assistant_msg)

            yield {"event": "done", "data": ""}

        except Exception as e:
            import traceback
            logger.error(f"SSE generation error: {e}\n{traceback.format_exc()}")
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(
        event_generator(),
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# Helper function to load session asynchronously 
async def _load_session_async(session_id: str):
    """Async version of session loading"""
    # If your session loading involves I/O, make it async
    # For now, assuming it's fast and keeping it sync
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _load_session, session_id)

# Add this to your main file or wherever you define your routes
@app.get("/chat")
async def chat_get(session_id: str, query: str):
    return await chat_get_optimized(session_id, query)


@app.get("/history/{session_id}")
def get_history(session_id: str):
    session = _load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    def _to_epoch(t):
        """Return a numeric epoch regardless of the stored type."""
        if isinstance(t, datetime):
            return t.timestamp()
        if isinstance(t, (int, float)):
            return float(t)
        if isinstance(t, str):
            try:
                return datetime.fromisoformat(t).timestamp()
            except ValueError:
                return 0
        return 0

    convos = data_manager.get_conversations(session.user["mobile_number"])
    messages = []
    for conv in reversed(convos):
        conv_msgs = sorted(conv.get("messages", []), key=lambda m: _to_epoch(m.get("timestamp")))
        messages.extend(conv_msgs)

    messages.sort(key=lambda m: _to_epoch(m.get("timestamp")))
    return {"messages": messages}


    total_start = time.perf_counter()

    session_id = payload.get("session_id")
    query = payload.get("query")
    if not session_id or not query:
        raise HTTPException(status_code=400, detail="session_id and query required")

    # Step 1: Load session
    step1 = time.perf_counter()
    session = _load_session(session_id)
    print("⏱️ Session load:", time.perf_counter() - step1)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    # Step 2: Append user message
    user_msg = {"role": "user", "content": query, "timestamp": uuid.uuid4().hex}
    session.messages.append(user_msg)
    user_type = session.user["user_type"]
    # Step 3: Run process_query
    step2 = time.perf_counter()
    stream, audio_task = process_query_optimized(
        query,
        scheme_vector_store,
        dfl_vector_store,
        session_id,
        session.user["mobile_number"],
        session,
        user_type,
        user_language=session.user.get("language"),
        stream=True,
    )
    print("⏱️ process_query:", time.perf_counter() - step2)

    async def event_generator():
        final_text = ""
        stream_start = time.perf_counter()
        try:
            for token in stream:
                final_text += token
                yield {"data": token}
            print("⏱️ Streaming complete:", time.perf_counter() - stream_start)

            # Step 4: Audio
            audio_bytes = None
            if audio_task:
                script = audio_task(final_text)
                audio_bytes = synthesize(script, "Hindi")
                b64_audio = base64.b64encode(audio_bytes).decode()

            # Step 5: Save assistant message
            assistant_msg = {
                "role": "assistant",
                "content": final_text,
                "timestamp": uuid.uuid4().hex
            }
            session.messages.append(assistant_msg)

            if audio_bytes:
                yield {"event": "audio", "data": b64_audio}
            yield {"event": "done", "data": ""}

        except Exception as e:
            print("SSE error:", e)

    sse_headers = {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    total_end = time.perf_counter()
    print("⏱️ Total POST /chat time:", total_end - total_start)

    return EventSourceResponse(event_generator(), headers=sse_headers)


@app.get("/welcome/{session_id}")
async def get_welcome(session_id: str):
    session = _load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    mobile = session.user["mobile_number"]
    user_type = session.user["user_type"]
    conversations = data_manager.get_conversations(mobile)

    if any("welcome" in (msg["content"] or "").lower() for conv in conversations for msg in conv["messages"]):
        return {"welcome": None, "audio": None}

    response_text, audio_task = await process_query_optimized(
        "welcome",
        scheme_vector_store,
        dfl_vector_store,
        session_id,
        mobile,
        session,
        user_type,
        user_language=session.user.get("language"),
        stream=False   
    )

    b64_audio = None
    if audio_task:
        script = await audio_task(response_text)  # FIX: await async task
        audio_bytes = synthesize(script, "Hindi")  # assuming sync
        b64_audio = base64.b64encode(audio_bytes).decode()

    return {
        "welcome": response_text,
        "audio": b64_audio
    }
