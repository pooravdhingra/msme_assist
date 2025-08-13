import logging
import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
logger = logging.getLogger(__name__)
logging.getLogger("pymongo").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

@lru_cache(maxsize=1)
def get_embeddings():
    """
    Initialize and return the OpenAI embedding model.
    
    Returns:
        OpenAIEmbeddings: Configured embedding model instance.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key
        )
        logger.info("OpenAI embeddings initialized successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI embeddings: {str(e)}")
        raise


def extract_scheme_guid(sources):
    """Return the GUID from the document that generated the response.

    Pinecone records store the scheme identifier under ``scheme_guid`` but
    some earlier versions used ``guid``. This helper fetches the GUID from
    the first document in ``sources`` with either of these keys. The first
    document typically represents the most relevant hit used to produce the
    final answer.
    """

    for doc in sources:
        if not doc or not getattr(doc, "metadata", None):
            continue
        guid = doc.metadata.get("scheme_guid") or doc.metadata.get("guid")
        if guid:
            return str(guid).strip()

    return None


