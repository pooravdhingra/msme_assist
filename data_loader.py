import pandas as pd
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone as PineconeClient
from langchain_core.documents import Document
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

def load_rag_data(
    google_drive_file_id="13OKwV3G_j4OJdm1Cxt7FAlO6qzrxBMW6",
    index_name: str | None = None,
    version_file="faiss_version.txt",
    cached_file_path="scheme_db_latest.xlsx",
    chunk_size=400,
):
    """
    Load scheme_db.xlsx and store documents in a Pinecone index.
    If a valid precomputed index exists, load it; otherwise, process in chunks and create a new index.
    
    Args:
        google_drive_file_id (str): The file ID from the Google Drive shareable link.
        index_name (str): Pinecone index name. Defaults to PINECONE_INDEX_NAME env var.
        version_file (str): File containing the hash of the Excel file used for the index.
    
    Returns:
        Pinecone: Vector store backed by Pinecone index.
    """
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
                    embeddings = get_embeddings()
                    return PineconeVectorStore.from_existing_index(index_name, embeddings)
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
                embeddings = get_embeddings()
                return PineconeVectorStore.from_existing_index(index_name, embeddings)
            else:
                logger.info(
                    f"Hash mismatch: stored {stored_hash}, current {file_hash}. Recreating index."
                )
                if pc:
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
        "_id",
        "scheme_guid",
        "scheme_name",
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

    embeddings = get_embeddings()
    all_documents = []

    def process_rows(rows):
        docs = []
        for _, row in rows.iterrows():
            parts = []
            for col in relevant_columns:
                if col in row and pd.notna(row[col]):
                    clean_col = col.replace('(', ' ').replace(')', '')
                    parts.append(f"{clean_col}: {row[col]}")
            content = "\n".join(parts)
            metadata = {
                "guid": row.get("scheme_guid", ""),
                "name": row.get("scheme_name", ""),
                "applicability": row.get("Applicability (State)", ""),
                "type": row.get("Type (Sch/Doc)", "")
            }
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    logger.info(f"Processing {len(df)} rows in chunks of {chunk_size}")
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        all_documents.extend(process_rows(chunk))

    if pc and not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
        )

    vector_store = PineconeVectorStore.from_documents(all_documents, embeddings, index_name=index_name)

    try:
        with open(version_file, "w") as f:
            f.write(file_hash)
        logger.info(f"Saved file hash to {version_file}")
    except Exception as e:
        logger.error(f"Failed to save version file: {str(e)}")
        raise

    logger.info("Pinecone index updated")
    return vector_store


def load_dfl_data(
    google_drive_file_id="1nHdHze3Za5BthXGsk9KptADCLNM7SN0JW4ZI8eIWJCE",
    index_name: str | None = None,
    version_file="dfl_version.txt",
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
                embeddings = get_embeddings()
                return PineconeVectorStore.from_existing_index(index_name, embeddings)
            else:
                logger.info(
                    f"Hash mismatch: stored {stored_hash}, current {file_hash}. Recreating index."
                )
                if pc:
                    pc.delete_index(index_name)
        except Exception as e:
            logger.error(f"Failed to load existing DFL index: {str(e)}. Recreating.")

    logger.info("Creating new DFL Pinecone index")
    chunk_size = 300
    documents = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            documents.append(Document(page_content=chunk, metadata={"chunk": i // chunk_size}))

    embeddings = get_embeddings()

    if pc and not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
        )

    vector_store = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)

    try:
        with open(version_file, "w") as vf:
            vf.write(file_hash)
        logger.info(f"Saved file hash to {version_file}")
    except Exception as e:
        logger.error(f"Failed to save DFL version file: {str(e)}")
        raise

    logger.info("DFL Pinecone index updated")
    return vector_store
