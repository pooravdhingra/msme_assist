import pandas as pd
from pinecone import Pinecone as PineconeClient
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import os
import logging
import gdown  # For downloading from Google Drive
import hashlib
import requests
from utils import get_embeddings
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY:
    logger.warning("PINECONE_API_KEY not set; vector storage will not work")

pc = None
if PINECONE_API_KEY:
    try:
        pc = PineconeClient(api_key=PINECONE_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client: {e}")

__all__ = ["load_rag_data", "load_dfl_data"]


class PineconeRecordRetriever(BaseRetriever):
    """Simple retriever that queries a Pinecone index using text search."""

    def __init__(self, index, state: str | None = None, gender: str | None = None, k: int = 5):
        super().__init__()
        self.index = index
        self.state = state
        self.gender = gender
        self.k = k

    def _get_relevant_documents(self, query: str, *, run_manager):  # type: ignore[override]
        flt = {}
        states = []
        if self.state:
            states.append(self.state)
        states.append("all_states")
        if states:
            flt["Applicability (State)"] = {"$in": states}
        if self.gender:
            flt["gender"] = {"$in": [self.gender, "all_genders"]}

        try:
            res = self.index.search(
                namespace="",
                query={"top_k": self.k, "inputs": {"chunk_text": query}, "filter": flt},
                fields=["*"]
            )
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []

        docs = []
        hits = getattr(getattr(res, "result", None), "hits", [])
        for hit in hits:
            fields = hit.fields or {}
            text = fields.get("chunk_text", "")
            metadata = {
                "scheme_guid": fields.get("scheme_guid"),
                "scheme_name": fields.get("scheme_name"),
                "Applicability (State)": fields.get("Applicability (State)"),
                "Type (Sch/Doc)": fields.get("Type (Sch/Doc)")
            }
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

def load_rag_data(
    google_drive_file_id: str = "13OKwV3G_j4OJdm1Cxt7FAlO6qzrxBMW6",
    index_name: str | None = None,
    version_file: str = "faiss_version.txt",
    cached_file_path: str = "scheme_db_latest.xlsx",
    chunk_size: int = 100,
):
    """Download scheme data and upsert it into a Pinecone index."""
    if index_name is None:
        index_name = PINECONE_INDEX_NAME

    if index_name is None:
        raise ValueError("Pinecone index name not provided")

    # Check if index and cached file hash exist
    if pc and pc.has_index(index_name) and os.path.exists(version_file):
        try:
            with open(version_file, "r") as vf:
                stored_hash = vf.read().strip()
            if os.path.exists(cached_file_path):
                with open(cached_file_path, "rb") as cf:
                    cached_hash = hashlib.md5(cf.read()).hexdigest()
                if cached_hash == stored_hash:
                    logger.info(
                        f"Using existing Pinecone index {index_name} with cached file hash {stored_hash}"
                    )
                    return pc.Index(index_name)
        except Exception as e:
            logger.error(f"Failed to load existing Pinecone index: {str(e)}. Will attempt download.")

    # Download the Google Sheet as an Excel file
    download_url = f"https://docs.google.com/spreadsheets/d/{google_drive_file_id}/export?format=xlsx"
    temp_file_path = cached_file_path
    logger.info(
        f"Downloading Google Sheet from Google Drive (File ID: {google_drive_file_id}) to {temp_file_path}"
    )
    try:
        gdown.download(download_url, temp_file_path, quiet=False)
        logger.info("Download completed")
    except Exception as e:
        logger.error(f"Failed to download file from Google Drive: {str(e)}")
        raise

    # Compute file hash
    with open(temp_file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    logger.info(f"Computed file hash: {file_hash}")

    # If index exists and hash matches, skip processing
    if pc and pc.has_index(index_name) and os.path.exists(version_file):
        try:
            with open(version_file, "r") as vf:
                stored_hash = vf.read().strip()
            if stored_hash == file_hash:
                logger.info(
                    f"Using existing Pinecone index {index_name} with matching hash"
                )
                return pc.Index(index_name)
            else:
                logger.info(
                    f"Hash mismatch: stored {stored_hash}, current {file_hash}. Recreating index."
                )
                pc.delete_index(index_name)
        except Exception as e:
            logger.error(
                f"Failed to load existing Pinecone index: {str(e)}. Recreating."
            )



    # Read Excel file
    try:
        df = pd.read_excel(temp_file_path)
        logger.info(f"Excel file loaded successfully. Rows: {len(df)}")
    except Exception as e:
        logger.error(f"Failed to read Excel file: {str(e)}")
        raise
    finally:
        # Keep the downloaded file for future reuse
        logger.info(f"Cached file available at {temp_file_path}")

    relevant_columns = [
        "Scheme GUID",
        "Scheme Name",
        "parent_scheme_name",
        "Applicability (State)",
        "Central Department Name",
        "State department name",
        "Type (Sch/Doc)",
        "Service Type Name",
        "Scheme description",
        "Scheme Eligibility",
        "Application Process",
        "Benefit"
    ]

    records = []
    logger.info(f"Processing {len(df)} rows in chunks of {chunk_size}")

    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        for _, row in chunk.iterrows():
            parts = []
            for col in df.columns:
                if pd.notna(row.get(col)):
                    parts.append(str(row[col]))
            content = " ".join(parts)
            record = {
                "_id": str(row.get("_id", row.name)),
                "inputs": {"chunk_text": content},
                "metadata": {
                    "scheme_guid": row.get("Scheme GUID", ""),
                    "scheme_name": row.get("Scheme Name", ""),
                    "Applicability (State)": row.get("Applicability (State)", ""),
                    "Type (Sch/Doc)": row.get("Type (Sch/Doc)", ""),
                    "gender": row.get("gender", "")
                },
            }
            records.append(record)

    if pc and not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
        )

    index = pc.Index(index_name)
    for start in range(0, len(records), chunk_size):
        batch = records[start : start + chunk_size]
        index.upsert_records(namespace="", records=batch)

    try:
        with open(version_file, "w") as f:
            f.write(file_hash)
        logger.info(f"Saved file hash to {version_file}")
    except Exception as e:
        logger.error(f"Failed to save version file: {str(e)}")
        raise

    logger.info("Pinecone index updated")
    return index


def load_dfl_data(
    google_drive_file_id: str = "1nHdHze3Za5BthXGsk9KptADCLNM7SN0JW4ZI8eIWJCE",
    index_name: str | None = None,
    version_file: str = "dfl_version.txt",
    chunk_tokens: int = 350,
):

    download_url = f"https://docs.google.com/document/d/{google_drive_file_id}/export?format=txt"
    logger.info(f"Downloading Google Doc from {download_url}")
    try:
        response = requests.get(download_url)
        response.raise_for_status()
        text = response.text
        file_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        logger.info(f"Downloaded document and computed hash {file_hash}")
    except Exception as e:
        logger.error(f"Failed to download Google Doc: {str(e)}")
        raise

    if index_name is None:
        index_name = PINECONE_INDEX_NAME

    if index_name is None:
        raise ValueError("Pinecone index name not provided")

    if pc and pc.has_index(index_name) and os.path.exists(version_file):
        try:
            with open(version_file, "r") as vf:
                stored_hash = vf.read().strip()
            if stored_hash == file_hash:
                logger.info(
                    f"Using existing Pinecone index {index_name} for DFL with matching hash"
                )
                return pc.Index(index_name)
            else:
                logger.info(
                    f"Hash mismatch: stored {stored_hash}, current {file_hash}. Recreating index."
                )
                pc.delete_index(index_name)
        except Exception as e:
            logger.error(f"Failed to load existing DFL index: {str(e)}. Recreating.")

    logger.info("Creating new DFL Pinecone index")
    if pc and not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
        )

    index = pc.Index(index_name)
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    records = []
    for i in range(0, len(tokens), chunk_tokens):
        chunk = enc.decode(tokens[i : i + chunk_tokens])
        records.append({"_id": str(i // chunk_tokens), "inputs": {"chunk_text": chunk}})
    for start in range(0, len(records), 100):
        index.upsert_records(namespace="", records=records[start : start + 100])

    try:
        with open(version_file, "w") as vf:
            vf.write(file_hash)
        logger.info(f"Saved file hash to {version_file}")
    except Exception as e:
        logger.error(f"Failed to save DFL version file: {str(e)}")
        raise

    logger.info("DFL Pinecone index updated")
    return index
