# MSME Assist

This repository now exposes a FastAPI backend that can be consumed by a custom React frontend.

*Create Virtual Environment*

python3 -m venv .venv

source .venv/bin/activate

*Install Dependencies*

pip install -r requirements.txt

*Configure Pinecone*

Update `.env` with your `PINECONE_API_KEY`, `PINECONE_SCHEME_HOST`, and `PINECONE_DFL_HOST` values along with the optional `PINECONE_INDEX_NAME` and `PINECONE_DFL_INDEX_NAME` for index creation. An example file is provided in `.env.example`.

The application will create the indexes automatically on the first run using Pinecone's built-in embedding model, so no manual setup is required. All data is stored under the `__default__` namespace.

*Index Data*

Run the application once to automatically download data and populate the Pinecone indexes. Subsequent runs will reuse the stored data.

*Run the API server*

uvicorn api_server:app --reload

*Available Endpoints*

- `POST /auth/token` – authenticate using a provided token and start a new session.
- `GET /history/{session_id}` – retrieve past conversation history for the authenticated user.
- `POST /chat` – process a user query. Tokens are streamed using Server Sent Events and an additional `audio` event is sent once the TTS audio is ready.

*Testing the API*

A helper script `test_api.py` is included for quick verification. First run the server:

```
uvicorn api_server:app --reload
```

In another terminal, install the test dependencies and run:

```
python test_api.py <YOUR_TOKEN> "Hello"
```

This will authenticate, send the query, stream tokens from `/chat`, save the audio to `response.mp3`, and finally print the conversation history.
