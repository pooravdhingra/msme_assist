import pandas as pd
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import Any
from functools import lru_cache
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
PINECONE_SCHEME_HOST = os.getenv("PINECONE_SCHEME_HOST")
PINECONE_DFL_HOST = os.getenv("PINECONE_DFL_HOST")

if not PINECONE_API_KEY:
    logger.warning("PINECONE_API_KEY not set; vector storage will not work")

pc = None
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client: {e}")

_index_cache: dict[str, Any] = {}

# Cached dataframe for direct lookups
_scheme_df: pd.DataFrame | None = None

# Common scheme keywords for direct search
DIRECT_SCHEME_KEYWORDS = [
    "fssai",
    "udyam",
    "shop act",
    "mudra",
    "mudra yojana",
    "vishwakarma",
    "svanidhi",
    "pmegp",
]


def get_index_by_host(host: str):
    """Return a cached Pinecone index connection for the given host."""
    if host in _index_cache:
        return _index_cache[host]
    if not pc:
        raise ValueError("Pinecone client not initialized")
    logger.info(f"Connecting to Pinecone index at {host}")
    index = pc.Index(host=host)
    _index_cache[host] = index
    return index

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


def get_scheme_dataframe(relevant_columns: list[str] | None = None) -> pd.DataFrame | None:
    """Return the cached scheme dataframe if available."""
    global _scheme_df
    if _scheme_df is not None:
        return _scheme_df
    cached = "scheme_db_latest.xlsx"
    if not os.path.exists(cached):
        return None
    cols = relevant_columns or [
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
    try:
        _scheme_df = pd.read_excel(cached, usecols=cols)
        logger.info("Loaded cached scheme dataframe")
    except Exception as exc:
        logger.error(f"Failed to load cached dataframe: {exc}")
        _scheme_df = None
    return _scheme_df


def find_scheme_by_query(query: str) -> dict | None:
    """Return the first scheme row matching known keywords in the query."""
    df = get_scheme_dataframe()
    if df is None:
        return None
    qlower = query.lower()
    for kw in DIRECT_SCHEME_KEYWORDS:
        if kw in qlower:
            try:
                matches = df[df["scheme_name"].str.contains(kw, case=False, na=False)]
                if not matches.empty:
                    logger.info(f"Direct scheme match for '{kw}'")
                    return matches.iloc[0].to_dict()
            except Exception as exc:
                logger.error(f"Direct lookup failed for {kw}: {exc}")
    return None


def scheme_row_to_text(row: dict) -> str:
    """Construct a text block from a scheme dataframe row."""
    parts = []
    if not row:
        return ""
    desc = row.get("scheme_description")
    if desc:
        parts.append(str(desc))
    elig = row.get("scheme_eligibility")
    if elig:
        parts.append(f"Eligibility: {elig}")
    proc = row.get("application_process")
    if proc:
        parts.append(f"How to apply: {proc}")
    ben = row.get("benefit")
    if ben:
        parts.append(f"Benefits: {ben}")
    return "\n".join(parts)

__all__ = [
    "load_rag_data",
    "load_dfl_data",
    "PineconeRecordRetriever",
    "get_scheme_dataframe",
    "find_scheme_by_query",
    "scheme_row_to_text",
]


class PineconeRecordRetriever(BaseRetriever):
    """Simple retriever that queries a Pinecone index using text search."""

    index: Any
    state: str | None = None
    gender: str | None = None
    k: int = 5

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def __init__(
        self,
        index: Any,
        state: str | None = None,
        gender: str | None = None,
        k: int = 5,
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
            filter_arg = None
            if self.state:
                filter_arg = {
                    "applicability_state": {"$in": [self.state, "ALL_STATES"]}
                }
            res = self.index.query(
                vector=embedding,
                top_k=self.k,
                namespace="__default__",
                include_metadata=True,
                filter=filter_arg,
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
    host: str | None = None,
    index_name: str | None = None,
    version_file: str = "faiss_version.txt",
    cached_file_path: str = "scheme_db_latest.xlsx",
    chunk_size: int = 50,
):
    """Download scheme data and upsert it into a Pinecone index."""
    global _scheme_df
    if host is None:
        host = PINECONE_SCHEME_HOST

    if host is None:
        raise ValueError("Pinecone host not provided")

    if index_name is None:
        index_name = PINECONE_INDEX_NAME

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
                    if _scheme_df is None:
                        try:
                            _scheme_df = pd.read_excel(
                                cached_file_path,
                                usecols=[
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
                                ],
                            )
                        except Exception as exc:
                            logger.error(f"Failed to load cached dataframe: {exc}")
                    return get_index_by_host(host)
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
                return get_index_by_host(host)
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
        _scheme_df = df
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

    if pc and index_name and not pinecone_has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
        )

    index = get_index_by_host(host)
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
    host: str | None = None,
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

    if host is None:
        host = PINECONE_DFL_HOST

    if host is None:
        raise ValueError("Pinecone host not provided")

    if index_name is None:
        index_name = PINECONE_DFL_INDEX_NAME

    if pc and pinecone_has_index(index_name) and os.path.exists(version_file):
        try:
            with open(version_file, "r") as vf:
                stored_hash = vf.read().strip()
            if stored_hash == file_hash:
                logger.info(
                    f"Using existing Pinecone index {index_name} for DFL with matching hash"
                )
                return get_index_by_host(host)
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

    index = get_index_by_host(host)
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
