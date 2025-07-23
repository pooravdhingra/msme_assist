import sys
import base64
import requests
from sseclient import SSEClient

BASE_URL = "http://127.0.0.1:8000"

if len(sys.argv) < 3:
    print("Usage: python test_api.py <TOKEN> <QUERY>")
    sys.exit(1)

TOKEN = sys.argv[1]
QUERY = sys.argv[2]

def authenticate(token):
    resp = requests.post(f"{BASE_URL}/auth/token", json={"token": token})
    resp.raise_for_status()
    data = resp.json()
    return data["session_id"], data["user"]

session_id, user = authenticate(TOKEN)
print("Authenticated as", user.get("fname"))

resp = requests.post(
    f"{BASE_URL}/chat",
    json={"session_id": session_id, "query": QUERY},
    stream=True,
)
client = SSEClient(resp)
full_text = ""
for event in client.events():
    if event.event == "message":
        print(event.data, end="", flush=True)
        full_text += event.data
    elif event.event == "audio":
        audio_bytes = base64.b64decode(event.data)
        with open("response.mp3", "wb") as f:
            f.write(audio_bytes)
        print("\n[Audio saved to response.mp3]")

history_resp = requests.get(f"{BASE_URL}/history/{session_id}")
history_resp.raise_for_status()
print("\nConversation history:")
print(history_resp.json())
