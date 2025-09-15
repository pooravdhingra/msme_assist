from pinecone import Pinecone
import os
import time
from typing import List, Dict, Any

def get_all_vector_ids(index, batch_size: int = 10000) -> List[str]:
    """
    Get all vector IDs from a Pinecone index using query with top_k.
    This is a workaround since Pinecone doesn't have a direct "list all IDs" method.
    """
    all_ids = []
    
    # Get index stats to understand the data
    stats = index.describe_index_stats()
    total_vectors = stats['total_vector_count']
    print(f"Total vectors in source index: {total_vectors}")
    
    if total_vectors == 0:
        print("No vectors found in source index")
        return []
    
    # Use query to get all IDs (this works by querying with a dummy vector)
    # First, let's get the dimension from index stats
    dimension = stats['dimension']
    
    # Create a dummy vector of zeros
    dummy_vector = [0.0] * dimension
    
    # Query to get all vectors (or as many as possible)
    try:
        # Query with high top_k to get as many IDs as possible
        query_response = index.query(
            vector=dummy_vector,
            top_k=min(batch_size, total_vectors),
            include_values=False,
            include_metadata=False
        )
        
        all_ids = [match['id'] for match in query_response['matches']]
        print(f"Retrieved {len(all_ids)} vector IDs using query method")
        
    except Exception as e:
        print(f"Error getting vector IDs: {e}")
        print("You'll need to provide the vector IDs manually or use an alternative method")
        return []
    
    return all_ids

def migrate_pinecone_index(old_api_key: str, new_api_key: str, index_name: str, batch_size: int = 100):
    """
    Migrate all vectors from one Pinecone index to another with the same name.
    
    Args:
        old_api_key: API key for source Pinecone account
        new_api_key: API key for destination Pinecone account  
        index_name: Name of the index (same in both accounts)
        batch_size: Number of vectors to process in each batch
    """
    
    # Initialize Pinecone clients
    print("Initializing Pinecone clients...")
    old_pc = Pinecone(api_key=old_api_key)
    new_pc = Pinecone(api_key=new_api_key)
    
    # Get index references
    old_index = old_pc.Index(index_name)
    new_index = new_pc.Index(index_name)
    
    print(f"Connected to indexes: {index_name}")
    
    # Get all vector IDs from the old index
    print("Retrieving vector IDs from source index...")
    vector_ids = get_all_vector_ids(old_index)
    
    if not vector_ids:
        print("❌ No vector IDs found. Migration cannot proceed.")
        return
    
    print(f"Found {len(vector_ids)} vectors to migrate")
    
    # Migrate vectors in batches
    total_migrated = 0
    failed_batches = []
    
    for start in range(0, len(vector_ids), batch_size):
        batch_ids = vector_ids[start:start + batch_size]
        batch_num = (start // batch_size) + 1
        total_batches = (len(vector_ids) + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch_ids)} vectors)...")
        
        try:
            # Fetch vectors from old index
            response = old_index.fetch(ids=batch_ids)
            
            if not hasattr(response, 'vectors') or not response.vectors:
                print(f"⚠️  No vectors returned for batch {batch_num}")
                continue
            
            # Prepare vectors for upsert
            vectors_to_upsert = []
            
            # response.vectors is a dict where keys are IDs and values are Vector objects
            for vid, vector_obj in response.vectors.items():
                vector_entry = {
                    'id': vid,
                    'values': vector_obj.values  # Access as attribute, not dict key
                }
                
                # Add metadata if it exists
                if hasattr(vector_obj, 'metadata') and vector_obj.metadata:
                    vector_entry['metadata'] = vector_obj.metadata
                
                vectors_to_upsert.append(vector_entry)
            
            # Upsert to new index
            if vectors_to_upsert:
                upsert_response = new_index.upsert(vectors=vectors_to_upsert)
                total_migrated += len(vectors_to_upsert)
                print(f"✅ Batch {batch_num} completed: {len(vectors_to_upsert)} vectors migrated")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Error processing batch {batch_num}: {e}")
            failed_batches.append(batch_num)
            continue
    
    # Summary
    print("\n" + "="*50)
    print("MIGRATION SUMMARY")
    print("="*50)
    print(f"Total vectors found: {len(vector_ids)}")
    print(f"Total vectors migrated: {total_migrated}")
    print(f"Failed batches: {len(failed_batches)}")
    
    if failed_batches:
        print(f"Failed batch numbers: {failed_batches}")
    
    if total_migrated == len(vector_ids):
        print("✅ Migration completed successfully!")
    else:
        print("⚠️  Migration completed with some issues. Check failed batches above.")

# Usage
if __name__ == "__main__":
    # Your API keys
    OLD_API_KEY = "pcsk_6VEd7N_wHVyxWDAivkgSB5f83AGD8oKb9puRUk4SaMPwLfhm7HviibsfwpKULB1pC7gfi"
    NEW_API_KEY = "pcsk_5vq3i8_Qtsa8soq32VZ23R3nxLzErrKoRcteAproeQntg8hUtPBtxTd1ykwqYFhnDHb8Jg"
    INDEX_NAME = "msme-dfl-chatbot"
    
    # Start migration
    migrate_pinecone_index(
        old_api_key=OLD_API_KEY,
        new_api_key=NEW_API_KEY,
        index_name=INDEX_NAME,
        batch_size=100
    )