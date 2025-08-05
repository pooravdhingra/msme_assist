import sys
import base64
import requests
from sseclient import SSEClient

BASE_URL = "http://127.0.0.1:8000"

if len(sys.argv) < 3:
    sys.exit(1)

TOKEN = sys.argv[1]
QUERY = sys.argv[2]

def authenticate(token):
    resp = requests.post(f"{BASE_URL}/auth/token", json={"token": token})
    resp.raise_for_status()
    data = resp.json()
    return data["session_id"], data["user"]

session_id, user = authenticate(TOKEN)

resp = requests.post(
    f"{BASE_URL}/chat",
    json={"session_id": session_id, "query": QUERY},
    stream=True,
)
client = SSEClient(resp)
full_text = ""
for event in client.events():
    if event.event == "message":
        full_text += event.data
    elif event.event == "audio":
        audio_bytes = base64.b64decode(event.data)
        with open("response.mp3", "wb") as f:
            f.write(audio_bytes)

history_resp = requests.get(f"{BASE_URL}/history/{session_id}")
history_resp.raise_for_status()
