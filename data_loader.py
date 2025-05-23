import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import logging
import gdown  # For downloading from Google Drive
import tempfile  # For temporary file handling
from utils import get_embeddings

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_rag_data(google_drive_file_id="1MQFFB-TEmKD8ToAyiQk49lQPQDTfedEp"):
    """
    Load scheme_db.xlsx, process in chunks to avoid token limits, and create a FAISS vector store.
    
    Args:
        google_drive_file_id (str): The file ID from the Google Drive shareable link.
    
    Returns:
        FAISS: FAISS vector store with indexed scheme documents.
    """
    # Download the Google Sheet as an Excel file
    download_url = f"https://docs.google.com/spreadsheets/d/{google_drive_file_id}/export?format=xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
        temp_file_path = temp_file.name
        logger.info(f"Downloading Google Sheet from Google Drive (File ID: {google_drive_file_id}) to {temp_file_path}")
        try:
            gdown.download(download_url, temp_file_path, quiet=False)
            logger.info("Download completed")
        except Exception as e:
            logger.error(f"Failed to download file from Google Drive: {str(e)}")
            raise
    
    # Read Excel file
    try:
        df = pd.read_excel(temp_file_path)
        logger.info(f"Excel file loaded successfully. Rows: {len(df)}")
    except Exception as e:
        logger.error(f"Failed to read Excel file: {str(e)}")
        raise
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)
        logger.info(f"Temporary file {temp_file_path} deleted")
    
    # Split data into two chunks
    total_rows = len(df)
    midpoint = total_rows // 2  # ~500 rows per chunk
    chunk1 = df.iloc[:midpoint]  # First chunk
    chunk2 = df.iloc[midpoint:]  # Second chunk
    logger.info(f"Split data into two chunks: Chunk 1 ({len(chunk1)} rows), Chunk 2 ({len(chunk2)} rows)")
    
    # Relevant columns to include in the document
    relevant_columns = [
        "name",
        "applicability",
        "type- SCH/DOC",
        "service type",
        "scheme type",
        "description",
        "objective(Eligibility)",
        "application method",
        "process",
        "benefit value description",
        "benefit amount (description)",
        "tags",
        "beneficiary type"
    ]
    
    # Function to process a chunk into LangChain Documents
    def process_chunk(chunk):
        documents = []
        for _, row in chunk.iterrows():
            # Create text content from relevant columns
            content_parts = []
            for col in relevant_columns:
                if col in row and pd.notna(row[col]):
                    # Clean column name for display (remove parentheses, etc.)
                    clean_col = col.replace('(', ' ').replace(')', '')
                    content_parts.append(f"{clean_col}: {row[col]}")
            content = "\n".join(content_parts)
            
            # Create metadata with scheme details
            metadata = {
                "guid": row["guid"] if pd.notna(row["guid"]) else "",
                "name": row["name"] if pd.notna(row["name"]) else ""
            }
            
            # Create LangChain Document
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        return documents
    
    # Process Chunk 1
    logger.info("Processing Chunk 1...")
    documents1 = process_chunk(chunk1)
    logger.info(f"Created {len(documents1)} documents from Chunk 1")
    
    # Create FAISS vector store for Chunk 1
    try:
        embeddings = get_embeddings()
        vector_store1 = FAISS.from_documents(documents1, embeddings)
        logger.info(f"FAISS vector store for Chunk 1 created with {vector_store1.index.ntotal} documents")
    except Exception as e:
        logger.error(f"Failed to create FAISS vector store for Chunk 1: {str(e)}")
        raise
    
    # Process Chunk 2
    logger.info("Processing Chunk 2...")
    documents2 = process_chunk(chunk2)
    logger.info(f"Created {len(documents2)} documents from Chunk 2")
    
    # Create FAISS vector store for Chunk 2
    try:
        vector_store2 = FAISS.from_documents(documents2, embeddings)
        logger.info(f"FAISS vector store for Chunk 2 created with {vector_store2.index.ntotal} documents")
    except Exception as e:
        logger.error(f"Failed to create FAISS vector store for Chunk 2: {str(e)}")
        raise
    
    # Merge the two vector stores
    logger.info("Merging vector stores...")
    vector_store1.merge_from(vector_store2)
    logger.info(f"Combined FAISS vector store created with {vector_store1.index.ntotal} documents")
    
    return vector_store1