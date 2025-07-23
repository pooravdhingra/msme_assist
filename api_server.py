from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse
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
    convos = data_manager.get_conversations(session.user["mobile_number"])
    messages = []
    for conv in convos:
        for msg in conv.get("messages", []):
            messages.append({"role": msg["role"], "content": msg["content"], "timestamp": msg.get("timestamp")})
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
    session.messages.append({"role": "user", "content": query, "timestamp": uuid.uuid4().hex})

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
        for token in stream:
            final_text += token
            yield {"data": token}
        audio_bytes = None
        if audio_task:
            script = audio_task(final_text)
            audio_bytes = synthesize(script, "Hindi")
            b64_audio = base64.b64encode(audio_bytes).decode()
        session.messages.append({"role": "assistant", "content": final_text})
        if audio_bytes:
            yield {"event": "audio", "data": b64_audio}

    return EventSourceResponse(event_generator())

