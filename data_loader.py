import pandas as pd
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import Any
import os
import logging
import gdown  # For downloading from Google Drive
import hashlib
import requests
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_DFL_INDEX_NAME = os.getenv("PINECONE_DFL_INDEX_NAME")

if not PINECONE_API_KEY:
    logger.warning("PINECONE_API_KEY not set; vector storage will not work")

pc = None
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client: {e}")

def pinecone_has_index(name: str) -> bool:
    if not pc:
        return False
    try:
        return name in pc.list_indexes().names()
    except Exception as exc:
        logger.error(f"Failed to list Pinecone indexes: {exc}")
        return False

def safe_get(row: pd.Series, column: str, default: str = ""):
    """Return a value from the row, replacing NaN with a default."""
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return value

__all__ = ["load_rag_data", "load_dfl_data", "PineconeRecordRetriever"]


class PineconeRecordRetriever(BaseRetriever):
    """Simple retriever that queries a Pinecone index using text search."""

    index: Any
    state: str | None = None
    gender: str | None = None
    k: int = 3

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def __init__(
        self,
        index: Any,
        state: str | None = None,
        gender: str | None = None,
        k: int = 3,
    ) -> None:
        # Ensure BaseModel initialises correctly
        super().__init__(index=index, state=state, gender=gender, k=k)


    def _get_relevant_documents(self, query: str, *, run_manager):  # type: ignore[override]

        logger.debug(f"Pinecone query text: {query}")
        logger.debug(f"Top K: {self.k}")

        try:
            embedding = pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=query,
                parameters={"input_type": "query"},
            ).data[0]["values"]
            logger.debug(f"Query embedding sample: {embedding[:5]}")
            res = self.index.query(
                vector=embedding,
                top_k=self.k,
                namespace="__default__",
                include_metadata=True,
            )
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []

  
        hits = res.get("matches", [])
        logger.debug(f"Number of matches returned: {len(hits)}")
        docs = []
        for hit in hits:
            metadata = getattr(hit, "metadata", {}) or {}
            text = metadata.get("chunk_text", "")
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

def load_rag_data(
    google_drive_file_id: str = "13OKwV3G_j4OJdm1Cxt7FAlO6qzrxBMW6",
    index_name: str | None = None,
    version_file: str = "faiss_version.txt",
    cached_file_path: str = "scheme_db_latest.xlsx",
    chunk_size: int = 50,
):
    """Download scheme data and upsert it into a Pinecone index."""
    if index_name is None:
        index_name = PINECONE_INDEX_NAME

    if index_name is None:
        raise ValueError("Pinecone index name not provided")

    # Check if index and cached file hash exist
    if pc and pinecone_has_index(index_name) and os.path.exists(version_file):
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
    if pc and pinecone_has_index(index_name) and os.path.exists(version_file):
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



    relevant_columns = [
        "scheme_guid",
        "scheme_name",
        "parent_scheme_name",
        "applicability_state",
        "central_department_name",
        "state_department_name",
        "type_sch_doc",
        "service_type_name",
        "scheme_description",
        "scheme_eligibility",
        "application_process",
        "benefit",
    ]

    # Read Excel file with only relevant columns
    try:
        df = pd.read_excel(temp_file_path, usecols=relevant_columns)
        logger.info(f"Excel file loaded successfully. Rows: {len(df)}")
    except Exception as e:
        logger.error(f"Failed to read Excel file: {str(e)}")
        raise
    finally:
        # Keep the downloaded file for future reuse
        logger.info(f"Cached file available at {temp_file_path}")

    metadata_only_columns = [
        "parent_scheme_name",
        "central_department_name",
        "state_department_name",
        "type_sch_doc",
        "scheme_guid",
        "scheme_eligibility",
        "application_process",
        "benefit",
    ]

    records = []
    logger.info(f"Processing {len(df)} rows in chunks of {chunk_size}")

    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        for _, row in chunk.iterrows():
            parts = []
            for col in relevant_columns:
                if col in metadata_only_columns:
                    continue
                value = safe_get(row, col)
                if value:
                    parts.append(str(value))
            content = " ".join(parts)
            scheme_guid = safe_get(row, "scheme_guid", row.name)
            record = {
                "id": str(scheme_guid),
                "chunk_text": content,
                "scheme_guid": safe_get(row, "scheme_guid"),
                "scheme_name": safe_get(row, "scheme_name"),
                "applicability_state": safe_get(row, "applicability_state"),
                "type_sch_doc": safe_get(row, "type_sch_doc"),
                "service_type_name": safe_get(row, "service_type_name"),
                "scheme_eligibility": safe_get(row, "scheme_eligibility"),
                "application_process": safe_get(row, "application_process"),
                "benefit": safe_get(row, "benefit"),
            }
            records.append(record)

    if pc and not pinecone_has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
        )

    index = pc.Index(index_name)
    for start in range(0, len(records), chunk_size):
        batch = records[start : start + chunk_size]
        index.upsert_records("__default__", batch)

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
        index_name = PINECONE_DFL_INDEX_NAME

    if index_name is None:
        raise ValueError("Pinecone index name not provided")

    if pc and pinecone_has_index(index_name) and os.path.exists(version_file):
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
    if pc and not pinecone_has_index(index_name):
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
        records.append({"id": str(i // chunk_tokens), "chunk_text": chunk})
    for start in range(0, len(records), 100):
        batch = records[start : start + 100]
        index.upsert_records("__default__", batch)

    try:
        with open(version_file, "w") as vf:
            vf.write(file_hash)
        logger.info(f"Saved file hash to {version_file}")
    except Exception as e:
        logger.error(f"Failed to save DFL version file: {str(e)}")
        raise

    logger.info("DFL Pinecone index updated")
    return index
