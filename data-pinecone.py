import os
from pymongo import MongoClient
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from dotenv import load_dotenv
import time
from typing import List, Dict, Any

load_dotenv()

# ========= CONFIG =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_key")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your_pinecone_key")
INDEX_NAME = "hq-bot-msme-nonmsme-scheme"
PINECONE_ENV = "us-east-1"
MONGO_URI = os.getenv("MONGO_URI", "your_mongo_uri")
DB_NAME = os.getenv("MONGO_DB_NAME", "msme_db")
COLLECTION_NAME = "schemes"

# Batch sizes
EMBEDDING_BATCH_SIZE = 100  # OpenAI allows up to 2048 inputs per request
PINECONE_BATCH_SIZE = 100   # Pinecone allows up to 1000 vectors per upsert
MONGO_BATCH_SIZE = 500      # MongoDB cursor batch size

# ========= CLIENTS =========
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# MongoDB connection with timeout and error handling
try:
    print("🔗 Connecting to MongoDB...")
    mongo_client = MongoClient(
        MONGO_URI, 
        serverSelectionTimeoutMS=5000,  # 5 second timeout
        connectTimeoutMS=5000,
        socketTimeoutMS=5000
    )
    # Test the connection
    mongo_client.admin.command('ping')
    print("✅ MongoDB connected successfully!")
    
    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    print("Please check your MONGO_URI in the .env file")
    exit(1)

# Create Pinecone index if not exists
if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    print(f"Creating new Pinecone index '{INDEX_NAME}' with dimension 1024...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,  # Match your existing index dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
else:
    # Check existing index dimension
    index_info = pc.describe_index(INDEX_NAME)
    print(f"Using existing index '{INDEX_NAME}' with dimension {index_info.dimension}")
    if index_info.dimension != 1024:
        print(f"⚠️  Warning: Index dimension is {index_info.dimension}, but we're using 1024. This may cause issues.")

index = pc.Index(INDEX_NAME)

# ========= HELPERS =========
def safe_get(doc: Dict[str, Any], key: str) -> str:
    """Safely get a value from document and convert to string."""
    return str(doc.get(key, "")) if doc.get(key) else ""

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a batch of texts using the model that matches your index dimension."""
    # Clean and validate texts
    cleaned_texts = []
    for text in texts:
        if text is None or text.strip() == "":
            # Use a default placeholder for empty texts
            cleaned_texts.append("No content available")
        else:
            # Truncate very long texts (OpenAI has a ~8191 token limit)
            cleaned_text = str(text).strip()
            if len(cleaned_text) > 30000:  # Rough character limit
                cleaned_text = cleaned_text[:30000] + "..."
            cleaned_texts.append(cleaned_text)
    
    try:
        # Use multilingual-e5-large which has 1536 dimensions by default,
        # but we can specify dimensions=1024 to match your index
        response = client.embeddings.create(
            input=cleaned_texts,
            model="multilingual-e5-large",
            dimensions=1024  # This matches your existing Pinecone index
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Error getting batch embeddings with multilingual-e5-large: {e}")
        print("Trying fallback to text-embedding-ada-002 with truncation...")
        
        try:
            # Fallback: try ada-002 and truncate to 1024 dimensions
            response = client.embeddings.create(
                input=cleaned_texts,
                model="text-embedding-ada-002"
            )
            # Truncate embeddings from 1536 to 1024 dimensions
            return [data.embedding[:1024] for data in response.data]
        except Exception as fallback_error:
            print(f"Error with fallback embeddings: {fallback_error}")
            print(f"Falling back to individual processing for {len(cleaned_texts)} texts...")
            
            # Final fallback to individual processing
            embeddings = []
            for i, text in enumerate(cleaned_texts):
                try:
                    response = client.embeddings.create(
                        input=text,
                        model="multilingual-e5-large",
                        dimensions=1024
                    )
                    embeddings.append(response.data[0].embedding)
                    if i % 10 == 0:  # Progress update every 10 embeddings
                        print(f"  Processed {i+1}/{len(cleaned_texts)} individual embeddings...")
                except Exception as individual_error:
                    print(f"Error with individual embedding {i+1}: {individual_error}")
                    # Use zero vector as fallback with correct dimension
                    embeddings.append([0.0] * 1024)
            return embeddings

def prepare_record(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare a single record for Pinecone upload matching your existing index format."""
    
    # Build comprehensive content for embedding by combining all relevant fields
    content_parts = []
    
    # Add scheme name
    if doc.get("scheme_name"):
        content_parts.append(str(doc.get("scheme_name")))
    
    # Add scheme description (this seems to be the main content)
    if doc.get("scheme_description"):
        content_parts.append(str(doc.get("scheme_description")))
        
    # Add eligibility
    if doc.get("scheme_eligibility"):
        content_parts.append(str(doc.get("scheme_eligibility")))
        
    # Add benefits
    if doc.get("benefit"):
        content_parts.append(str(doc.get("benefit")))
        
    # Add application process
    if doc.get("application_process"):
        content_parts.append(str(doc.get("application_process")))
    
    # Combine all content with spaces
    full_content = " ".join(content_parts).strip() if content_parts else "No content available"
    
    # Get scheme_guid - use the field as it exists in your MongoDB
    scheme_guid = safe_get(doc, "scheme_guid")
    
    # Fallback ID generation if scheme_guid is missing
    if not scheme_guid:
        # Try parent_scheme_guid or generate from _id
        scheme_guid = safe_get(doc, "parent_scheme_guid") or str(doc.get("_id", f"generated_{hash(str(doc))}"))
    
    # Return in the exact format matching your existing index
    return {
        "id": scheme_guid,
        "text": full_content,
        "type": int(doc.get("type")),
        "chunk_text": full_content,
        "scheme_guid": scheme_guid,
        "scheme_name": safe_get(doc, "scheme_name"),
        "applicability_state": safe_get(doc, "applicability_state"),
        "type_sch_doc": safe_get(doc, "type_sch_doc"),
        "service_type_name": safe_get(doc, "service_type_name"),
        "scheme_eligibility": safe_get(doc, "scheme_eligibility"),
        "application_process": safe_get(doc, "application_process"),
        "benefit": safe_get(doc, "benefit"),
    }

def batch_generator(cursor, batch_size: int):
    """Generate batches from MongoDB cursor."""
    batch = []
    for doc in cursor:
        batch.append(doc)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:  # Don't forget the last batch
        yield batch

def upload_batch_to_pinecone(vectors: List[Dict[str, Any]]):
    """Upload a batch of vectors to Pinecone with retry logic."""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            index.upsert(vectors=vectors)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1} after error: {e}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to upload batch after {max_retries} attempts: {e}")
                return False
    return False

def upload_from_mongo_batch(limit: int = None):
    """Upload MongoDB data to Pinecone using batch processing."""
    query = {}  # Add filters if needed
    
    # Get total count with timeout protection
    try:
        print("📊 Counting documents...")
        total_docs = collection.count_documents(query, maxTimeMS=30000)  # 30 second timeout
    except Exception as e:
        print(f"⚠️  Could not count documents: {e}")
        print("Proceeding without total count...")
        total_docs = limit or "unknown"
    
    if limit and isinstance(total_docs, int):
        total_docs = min(total_docs, limit)
    
    print(f"Processing {total_docs} documents...")
    
    cursor = collection.find(query)
    if limit:
        cursor = cursor.limit(limit)
    
    # Use batch size for MongoDB cursor with timeout
    cursor.batch_size(MONGO_BATCH_SIZE).max_time_ms(60000)  # 60 second timeout per batch
    
    processed_count = 0
    failed_count = 0
    batch_number = 0
    
    try:
        # Process in batches
        for mongo_batch in batch_generator(cursor, EMBEDDING_BATCH_SIZE):
            batch_number += 1
            try:
                print(f"📦 Processing batch {batch_number} ({len(mongo_batch)} documents)...")
                
                # Prepare records
                records = [prepare_record(doc) for doc in mongo_batch]
                texts = [rec["text"] for rec in records]
                
                # Get embeddings in batch
                embeddings = get_embeddings_batch(texts)
                
                # Prepare Pinecone vectors
                vectors = []
                for rec, embedding in zip(records, embeddings):
                    vectors.append({
                        "id": rec["id"],
                        "values": embedding,
                        "metadata": {k: v for k, v in rec.items() if k != "id"}
                    })
                
                # Upload to Pinecone in smaller batches if needed
                for i in range(0, len(vectors), PINECONE_BATCH_SIZE):
                    batch_vectors = vectors[i:i + PINECONE_BATCH_SIZE]
                    success = upload_batch_to_pinecone(batch_vectors)
                    
                    if success:
                        processed_count += len(batch_vectors)
                        print(f"✅ Uploaded {len(batch_vectors)} vectors (Total: {processed_count})")
                    else:
                        failed_count += len(batch_vectors)
                        print(f"❌ Failed to upload {len(batch_vectors)} vectors")
                
                # Small delay to avoid rate limits
                time.sleep(0.2)
                
            except Exception as e:
                print(f"❌ Error processing batch {batch_number}: {e}")
                failed_count += len(mongo_batch)
                continue
                
    except Exception as e:
        print(f"❌ Fatal error during processing: {e}")
        print(f"Processed {processed_count} records before error")
    
    print(f"✅ Upload complete!")
    print(f"📊 Successfully processed: {processed_count}")
    print(f"❌ Failed: {failed_count}")
    if processed_count + failed_count > 0:
        print(f"📈 Success rate: {(processed_count / (processed_count + failed_count)) * 100:.1f}%")

# ========= OPTIMIZED UPLOAD FUNCTIONS =========
def upload_with_progress_tracking(limit: int = None):
    """Enhanced upload with detailed progress tracking."""
    query = {}
    
    # Try to get total count, but don't fail if we can't
    try:
        total_docs = collection.count_documents(query, maxTimeMS=10000)  # 10 second timeout
        if limit:
            total_docs = min(total_docs, limit)
        print(f"🚀 Starting batch upload of {total_docs} documents")
    except Exception as e:
        print(f"⚠️  Could not get document count: {e}")
        print(f"🚀 Starting batch upload (limit: {limit or 'all'})")
        total_docs = limit or "unknown"
    
    print(f"⚙️  Embedding batch size: {EMBEDDING_BATCH_SIZE}")
    print(f"⚙️  Pinecone batch size: {PINECONE_BATCH_SIZE}")
    print("-" * 50)
    
    start_time = time.time()
    upload_from_mongo_batch(limit)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"⏱️  Total time: {duration:.2f} seconds")
    if isinstance(total_docs, int):
        print(f"🏃 Average speed: {total_docs / duration:.2f} docs/second")

def check_existing_records():
    """Check how many records already exist in Pinecone."""
    try:
        stats = index.describe_index_stats()
        return stats.total_vector_count
    except Exception as e:
        print(f"Error checking index stats: {e}")
        return 0

# ========= PREVIEW FUNCTIONS =========
def preview_data_transformation(limit: int = 3):
    """Preview how MongoDB data will be transformed for Pinecone."""
    print("🔍 Previewing Data Transformation...")
    print("=" * 60)
    
    try:
        sample_docs = list(collection.find({}).limit(limit))
        
        for i, doc in enumerate(sample_docs, 1):
            print(f"\n--- Sample Document {i} ---")
            print("MongoDB Document Fields:")
            for key, value in doc.items():
                if key == '_id':
                    print(f"  {key}: {value}")
                else:
                    preview_val = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    print(f"  {key}: {preview_val}")
            
            print("\nPinecone Record (after transformation):")
            pinecone_record = prepare_record(doc)
            for key, value in pinecone_record.items():
                if key in ['text', 'chunk_text']:
                    preview_val = str(value)[:150] + "..." if len(str(value)) > 150 else str(value)
                    print(f"  {key}: {preview_val}")
                else:
                    print(f"  {key}: {value}")
            
            print("-" * 60)
            
    except Exception as e:
        print(f"Error during preview: {e}")

# ========= RUN =========
if __name__ == "__main__":
    try:
        # Check existing records
        print("🔍 Checking existing Pinecone records...")
        existing_count = check_existing_records()
        print(f"📋 Existing records in Pinecone: {existing_count}")
        
        # Test MongoDB connection by trying to find one document
        print("🧪 Testing MongoDB connection...")
        test_doc = collection.find_one()
        if test_doc:
            print("✅ MongoDB connection verified - found sample document")
        else:
            print("⚠️  No documents found in collection")
        
        # Preview data transformation
        print("\n" + "="*60)
        preview_data_transformation()
        print("="*60)
        
        # Ask user for confirmation
        response = input("\n🤔 Does the data transformation look correct? (y/n): ").lower().strip()
        if response != 'y':
            print("❌ Upload cancelled. Please review the data transformation.")
            exit(0)
        
        # Run the batch upload
        upload_with_progress_tracking(limit=4875)  # Remove limit to process all records
        
    except KeyboardInterrupt:
        print("\n⏸️  Upload interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("Please check your .env file and network connection")