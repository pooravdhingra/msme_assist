import pandas as pd
from pymongo import MongoClient, TEXT
import datetime
import json

def import_data_and_create_indexes():
    # MongoDB connection
    client = MongoClient("mongodb+srv://pooravdhingra:Poorav123@msmeassist.womwsrv.mongodb.net/")
    db = client["haqdarshak"]
    collection = db["schemes"]
    
    print("🚀 Starting complete import process...")
    
    # STEP 1: Import Data
    print("\n📊 STEP 1: Importing data from Excel...")
    
    # Read Excel
    df = pd.read_excel("scheme_db_latest_new.xlsx")
    df.columns = df.columns.str.strip()
    
    print(f"Found {len(df)} rows to import")
    
    # Convert to records
    records = []
    for index, row in df.iterrows():
        record = {}
        
        for column in df.columns:
            value = row[column]
            
            if pd.isna(value) or value is None:
                record[column] = None
            elif isinstance(value, str):
                cleaned_value = value.strip()
                record[column] = cleaned_value if cleaned_value else None
            else:
                record[column] = value
        
        record['created_at'] = datetime.datetime.utcnow()
        record['updated_at'] = datetime.datetime.utcnow()
        records.append(record)
        
        if (index + 1) % 500 == 0:
            print(f"  Processed {index + 1} records...")
    
    # Insert data in batches
    try:
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            collection.insert_many(batch)
            print(f"  ✅ Inserted batch {i//batch_size + 1}: {len(batch)} records")
        
        print(f"✅ Successfully imported {len(records)} records!")
        
    except Exception as e:
        print(f"❌ Error during data import: {e}")
        client.close()
        return False
    
    # STEP 2: Create Indexes
    print("\n🔍 STEP 2: Creating indexes...")
    
    try:
        # Check if scheme_guid has duplicates before creating unique index
        print("  Checking for duplicate scheme_guid values...")
        pipeline = [
            {"$group": {"_id": "$scheme_guid", "count": {"$sum": 1}}},
            {"$match": {"count": {"$gt": 1}}}
        ]
        duplicates = list(collection.aggregate(pipeline))
        
        if duplicates:
            print(f"  ⚠️  Found {len(duplicates)} duplicate scheme_guid values:")
            for dup in duplicates[:5]:  # Show first 5
                print(f"    - {dup['_id']}: {dup['count']} occurrences")
            
            create_unique = input("  Create unique index anyway? (y/n): ").lower() == 'y'
        else:
            print("  ✅ No duplicates found")
            create_unique = True
        
        # Create indexes
        if create_unique:
            collection.create_index("scheme_guid", unique=True)
            print("  ✅ Created unique index on 'scheme_guid'")
        else:
            collection.create_index("scheme_guid")
            print("  ✅ Created non-unique index on 'scheme_guid'")
        
        # Create other indexes
        indexes_to_create = [
            "type",
            "scheme_status", 
            "applicability_state",
            "central_department_name",
            "state_department_name",
            "Individual/MSME",
            "Service Type GUID",
            "parent_scheme_guid",
            "created_at",
            "updated_at"
        ]
        
        for field in indexes_to_create:
            collection.create_index(field)
            print(f"  ✅ Created index on '{field}'")
        
        # Text search index
        collection.create_index([
            ("scheme_name", TEXT),
            ("scheme_description", TEXT), 
            ("scheme_eligibility", TEXT),
            ("benefit", TEXT)
        ])
        print("  ✅ Created text search index")
        
        # Compound indexes
        compound_indexes = [
            [("type", 1), ("applicability_state", 1)],
            [("scheme_status", 1), ("type", 1)],
            [("Individual/MSME", 1), ("scheme_status", 1)]
        ]
        
        for idx in compound_indexes:
            collection.create_index(idx)
            print(f"  ✅ Created compound index on {[field[0] for field in idx]}")
        
        print("\n🎉 All indexes created successfully!")
        
        # Show final stats
        total_docs = collection.count_documents({})
        total_indexes = len(list(collection.list_indexes()))
        
        print(f"\n📈 Final Statistics:")
        print(f"  📄 Total documents: {total_docs}")
        print(f"  🔍 Total indexes: {total_indexes}")
        
    except Exception as e:
        print(f"❌ Error creating indexes: {e}")
    
    client.close()
    print("\n✨ Import process completed!")
    return True

# Run the complete process
if __name__ == "__main__":
    import_data_and_create_indexes()