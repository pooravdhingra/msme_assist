# MSME Assist

This application provides a Streamlit interface for interacting with the MSME assistant bot.

*Create Virtual Environment*

python3 -m venv .venv

source .venv/bin/activate

*Install Dependencies*

pip install -r requirements.txt

*Configure Pinecone*

Update `.env` with your `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` values. These are required for storing embeddings in Pinecone. An example file is provided in `.env.example`.

The application will create the index automatically on the first run if it does not already exist, so no manual setup is required.

*Index Data*

Run the application once to automatically download data and populate the Pinecone index. Subsequent runs will reuse the stored index.

*Run the app*

streamlit run app.py
