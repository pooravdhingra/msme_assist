from fastapi import FastAPI, HTTPException
from fastapi import Query
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
    process_query,
    SessionData,
)
from data import DataManager, STATE_NAME_TO_ID, GENDER_MAPPING
from tts import synthesize

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
    }
    session_id = uuid.uuid4().hex
    data_manager.start_session(user["mobile_number"], session_id, user)
    sessions[session_id] = SessionData(user=user)
    return {"session_id": session_id, "user": user}

@app.get("/history/{session_id}")
def get_history(session_id: str):
    session = sessions.get(session_id)
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
    for conv in reversed(convos):  # reverse conversations: oldest first
        conv_msgs = sorted(conv.get("messages", []), key=lambda m: _to_epoch(m.get("timestamp")))
        messages.extend(conv_msgs)                       # keep the original dict

    messages.sort(key=lambda m: _to_epoch(m.get("timestamp")))
    return {"messages": messages}

@app.post("/chat")
def chat(payload: Dict[str, str]):
    session_id = payload.get("session_id")
    query = payload.get("query")
    if not session_id or not query:
        raise HTTPException(status_code=400, detail="session_id and query required")
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    # append user message
    user_msg = {"role": "user", "content": query, "timestamp": uuid.uuid4().hex}
    session.messages.append(user_msg)

    stream, audio_task = process_query(
        query,
        scheme_vector_store,
        dfl_vector_store,
        session_id,
        session.user["mobile_number"],
        session,
        user_language=session.user.get("language"),
        stream=True,
    )

    async def event_generator():
        final_text = ""
        try:
            for token in stream:
                final_text += token
                yield {"data": token}
            audio_bytes = None
            if audio_task:
                script = audio_task(final_text)
                audio_bytes = synthesize(script, "Hindi")
                b64_audio = base64.b64encode(audio_bytes).decode()
            # after streaming is done
            assistant_msg = {
                "role": "assistant",
                "content": final_text,
                "timestamp": uuid.uuid4().hex
            }
            session.messages.append(user_msg)

            if audio_bytes:
                yield {"event": "audio", "data": b64_audio}
            
            yield {"event": "done", "data": ""}

        except Exception as e:
            # If you want to debug server-side errors
            print("SSE error:", e)

    # Explicit CORS headers here
    sse_headers = {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return EventSourceResponse(event_generator(), headers=sse_headers)

@app.get("/chat")
async def chat_get(session_id: str, query: str):
    sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    if not isinstance(sess, SessionData):
        import inspect, logging
        logging.error(f"sess is {type(sess)}: {sess}  (callable? {callable(sess)})  source: {inspect.getsource(sess) if callable(sess) else ''}")
        raise HTTPException(500, "Corrupt session object")

    # Always push a timestamp
    user_msg = {"role": "user", "content": query, "timestamp": uuid.uuid4().hex}
    sess.messages.append(user_msg)

    token_stream, make_audio = process_query(
        query,
        scheme_vector_store,
        dfl_vector_store,
        session_id,
        sess.user["mobile_number"],
        sess,                                 # <— make sure this is SessionData
        user_language=sess.user.get("language"),
        stream=True,
    )

    async def event_generator():
        final_text = ""
        try:
            for token in token_stream:
                final_text += token
                yield {"data": token}

            # audio
            if make_audio:
                script = make_audio(final_text)  # <- returns Hindi script
                audio_bytes = synthesize(script, "Hindi")
                b64_audio = base64.b64encode(audio_bytes).decode()
                yield {"event": "audio", "data": b64_audio}

            # save assistant message w/ timestamp
            assistant_msg = {
                "role": "assistant",
                "content": final_text,
                "timestamp": uuid.uuid4().hex
            }
            sess.messages.append(assistant_msg)

            # ② make the write idempotent for this session

            yield {"event": "done", "data": ""}

        except Exception as e:
            # Optional: send an error event to client
            import traceback, logging
            logging.error("SSE gen error: %s\n%s", e, traceback.format_exc())

    return EventSourceResponse(
        event_generator(),
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/welcome/{session_id}")
def get_welcome(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    mobile = session.user["mobile_number"]
    conversations = data_manager.get_conversations(mobile)

    if any("welcome" in (msg["content"] or "").lower() for conv in conversations for msg in conv["messages"]):
        return {"welcome": None, "audio": None}

    response_text, audio_task = process_query(
        "welcome",
        scheme_vector_store,
        dfl_vector_store,
        session_id,
        mobile,
        session,
        user_language=session.user.get("language"),
        stream=False
    )

    b64_audio = None
    if audio_task:
        script = audio_task(response_text)
        audio_bytes = synthesize(script, "Hindi")
        b64_audio = base64.b64encode(audio_bytes).decode()

    return {
        "welcome": response_text,
        "audio": b64_audio
    }
