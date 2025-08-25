import numpy as np
import time
from pinecone import Pinecone

pinecone_api_key = 'pcsk_6VEd7N_wHVyxWDAivkgSB5f83AGD8oKb9puRUk4SaMPwLfhm7HviibsfwpKULB1pC7gfi'

# Source Pinecone setup
source_pinecone_environment = 'us-east-1'
source_pinecone_index_name = 'msme-discovery'

# Target Pinecone setup
target_pinecone_environment = 'us-east-1'
target_pinecone_index_name = 'hq-chatbot-msme'

pc = Pinecone(api_key=pinecone_api_key)

def transform_vector_dimensions(vector, source_dim, target_dim, method="pad_zeros"):
    """Transform vector dimensions from source to target size"""
    vector = np.array(vector)
    
    if source_dim == target_dim:
        return vector.tolist()
    
    if method == "pad_zeros":
        # Pad with zeros if target is larger
        if target_dim > source_dim:
            padding = np.zeros(target_dim - source_dim)
            transformed = np.concatenate([vector, padding])
        else:
            # Truncate if target is smaller
            transformed = vector[:target_dim]
    
    elif method == "pad_random":
        # Pad with small random values if target is larger
        if target_dim > source_dim:
            # Use small random values (scaled down)
            padding = np.random.normal(0, 0.01, target_dim - source_dim)
            transformed = np.concatenate([vector, padding])
        else:
            transformed = vector[:target_dim]
    
    elif method == "interpolate":
        # Use interpolation to resize
        if target_dim != source_dim:
            from scipy import interpolate
            x_old = np.linspace(0, 1, source_dim)
            x_new = np.linspace(0, 1, target_dim)
            f = interpolate.interp1d(x_old, vector, kind='linear')
            transformed = f(x_new)
        else:
            transformed = vector
    
    elif method == "pca_extend":
        # This would require fitting PCA on the dataset first
        # For now, fall back to pad_zeros
        print("PCA extension not implemented, falling back to pad_zeros")
        return transform_vector_dimensions(vector, source_dim, target_dim, "pad_zeros")
    
    else:
        raise ValueError(f"Unknown resize method: {method}")
    
    return transformed.tolist()

def get_all_ids_from_index(index, num_dimensions, max_attempts=50):
    """Get all vector IDs from an index using random query approach"""
    try:
        stats = index.describe_index_stats()
        namespace_map = stats.get('namespaces', {})
        
        if not namespace_map:
            print("No namespaces found in the source index")
            return {}

        all_ids = {}
        
        for namespace in namespace_map:
            num_vectors = namespace_map[namespace]['vector_count']
            print(f"Processing namespace '{namespace}' with {num_vectors} vectors")
            
            all_ids[namespace] = set()
            attempts = 0
            
            while len(all_ids[namespace]) < num_vectors and attempts < max_attempts:
                try:
                    # Generate random query vector
                    input_vector = np.random.rand(num_dimensions).tolist()
                    ids = get_ids_from_query(index, input_vector, namespace)
                    
                    previous_count = len(all_ids[namespace])
                    all_ids[namespace].update(ids)
                    new_count = len(all_ids[namespace])
                    
                    if new_count > previous_count:
                        print(f"Namespace '{namespace}': {new_count}/{num_vectors} IDs collected")
                    
                    attempts += 1
                    
                    # Add small delay to avoid rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error querying namespace '{namespace}': {e}")
                    attempts += 1
                    time.sleep(1)
            
            if len(all_ids[namespace]) < num_vectors:
                print(f"Warning: Only collected {len(all_ids[namespace])} out of {num_vectors} IDs for namespace '{namespace}'")

        return all_ids
        
    except Exception as e:
        print(f"Error getting IDs from index: {e}")
        return {}

def get_ids_from_query(index, input_vector, namespace):
    """Get vector IDs from a similarity query"""
    try:
        results = index.query(
            vector=input_vector,
            top_k=10000,
            namespace=namespace,
            include_values=False,
            include_metadata=False
        )
        return {result['id'] for result in results.get('matches', [])}
    except Exception as e:
        print(f"Error in query: {e}")
        return set()

def migrate_vectors(source_index, target_index, source_dimensions, target_dimensions, batch_size=100, resize_method="pad_zeros"):
    """Migrate vectors from source to target index with dimension transformation"""
    try:
        print("Getting all vector IDs from source index...")
        all_ids = get_all_ids_from_index(source_index, source_dimensions)
        
        if not all_ids:
            print("No vectors found to migrate")
            return
        
        print('Starting migration...')
        total_vectors = sum(len(ids) for ids in all_ids.values())
        migrated_vectors = 0
        failed_vectors = 0

        for namespace, ids in all_ids.items():
            print(f'\nProcessing namespace: {namespace} ({len(ids)} vectors)')
            ids_list = list(ids)
            
            # Process in batches
            for i in range(0, len(ids_list), batch_size):
                batch_ids = ids_list[i:i + batch_size]
                
                try:
                    # Fetch vectors from source
                    fetch_response = source_index.fetch(ids=batch_ids, namespace=namespace)
                    
                    # Handle different response types
                    vectors_data = {}
                    if hasattr(fetch_response, 'vectors'):
                        # New Pinecone client format
                        vectors_data = fetch_response.vectors or {}
                    elif hasattr(fetch_response, 'to_dict'):
                        # Response has to_dict method
                        response_dict = fetch_response.to_dict()
                        vectors_data = response_dict.get('vectors', {})
                    elif isinstance(fetch_response, dict):
                        # Already a dictionary
                        vectors_data = fetch_response.get('vectors', {})
                    else:
                        # Try to access as attribute
                        try:
                            vectors_data = getattr(fetch_response, 'vectors', {})
                        except:
                            print(f"Unknown fetch response type: {type(fetch_response)}")
                            print(f"Response attributes: {dir(fetch_response)}")
                            continue
                    
                    if not vectors_data:
                        print(f"No vector data returned for batch starting at index {i}")
                        failed_vectors += len(batch_ids)
                        continue
                    
                    # Prepare vectors for upsert with dimension transformation
                    vectors_to_upsert = []
                    for vector_id in batch_ids:
                        if vector_id in vectors_data:
                            vector_item = vectors_data[vector_id]
                            
                            # Handle different vector data formats
                            if hasattr(vector_item, 'values'):
                                # Vector object with .values attribute
                                original_values = vector_item.values
                                metadata = getattr(vector_item, 'metadata', {}) or {}
                            elif isinstance(vector_item, dict):
                                # Dictionary format
                                original_values = vector_item.get('values', [])
                                metadata = vector_item.get('metadata', {}) or {}
                            else:
                                print(f"Unknown vector format for ID {vector_id}: {type(vector_item)}")
                                continue
                            
                            if not original_values:
                                print(f"No values found for vector ID {vector_id}")
                                continue
                            
                            # Transform dimensions if needed
                            if source_dimensions != target_dimensions:
                                transformed_values = transform_vector_dimensions(
                                    original_values, 
                                    source_dimensions, 
                                    target_dimensions, 
                                    resize_method
                                )
                            else:
                                transformed_values = original_values
                            
                            vectors_to_upsert.append((
                                vector_id,
                                transformed_values,
                                metadata
                            ))
                    
                    if vectors_to_upsert:
                        # Upsert to target index
                        response = target_index.upsert(
                            vectors=vectors_to_upsert,
                            namespace=namespace
                        )
                        
                        batch_count = len(vectors_to_upsert)
                        migrated_vectors += batch_count
                        percentage_complete = (migrated_vectors / total_vectors) * 100
                        
                        print(f'✓ Migrated batch: {batch_count} vectors ({migrated_vectors}/{total_vectors}, {percentage_complete:.2f}%)')
                        
                        # Add delay to avoid rate limiting
                        time.sleep(0.5)
                    else:
                        print(f"No valid vectors found in batch starting at index {i}")
                        failed_vectors += len(batch_ids)
                        
                except Exception as e:
                    print(f"Error processing batch starting at index {i}: {e}")
                    failed_vectors += len(batch_ids)
                    time.sleep(2)  # Longer delay on error

        print(f"\nMigration Summary:")
        print(f"Total vectors processed: {total_vectors}")
        print(f"Successfully migrated: {migrated_vectors}")
        print(f"Failed: {failed_vectors}")
        print("Migration completed.")
        
    except Exception as e:
        print(f"Error during migration: {e}")

def verify_migration(source_index, target_index):
    """Verify that migration was successful"""
    try:
        source_stats = source_index.describe_index_stats()
        target_stats = target_index.describe_index_stats()
        
        print("\nMigration Verification:")
        print("Source Index Stats:", source_stats)
        print("Target Index Stats:", target_stats)
        
    except Exception as e:
        print(f"Error verifying migration: {e}")

if __name__ == "__main__":
    try:
        # Initialize indices
        print("Initializing Pinecone indices...")
        source_index = pc.Index(source_pinecone_index_name)
        target_index = pc.Index(target_pinecone_index_name)
        
        # Verify indices are accessible
        print("Verifying source index...")
        source_stats = source_index.describe_index_stats()
        print(f"Source index stats: {source_stats}")
        
        print("Verifying target index...")
        target_stats = target_index.describe_index_stats()
        print(f"Target index stats: {target_stats}")
        
        # Get dimensions from indices
        source_stats = source_index.describe_index_stats()
        target_stats = target_index.describe_index_stats()
        source_dimensions = source_stats['dimension']
        target_dimensions = target_stats['dimension']
        
        print(f"Source dimensions: {source_dimensions}")
        print(f"Target dimensions: {target_dimensions}")
        
        if source_dimensions != target_dimensions:
            print(f"Dimension mismatch detected! Will transform {source_dimensions}D vectors to {target_dimensions}D")
            print("Available transformation methods:")
            print("1. 'pad_zeros' - Pad with zeros (recommended)")
            print("2. 'pad_random' - Pad with small random values")
            print("3. 'interpolate' - Use interpolation (requires scipy)")
            
            resize_method = input("Choose method (default: pad_zeros): ").strip() or "pad_zeros"
        else:
            resize_method = "none"
        
        # Start migration
        migrate_vectors(source_index, target_index, source_dimensions, target_dimensions, resize_method=resize_method)
        
        # Verify migration
        verify_migration(source_index, target_index)
        
    except Exception as e:
        print(f"Critical error: {e}")
        print("Please check your API key and index names")