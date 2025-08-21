# scheme_mongo_lookup.py
import logging
import os
import time
from typing import Optional, Dict, List, Any
from pymongo import MongoClient
from functools import lru_cache
import re
from dataclasses import dataclass
from langchain.schema import Document


logger = logging.getLogger(__name__)

SCHEME_KEYWORDS: Dict[str, List[str]] = {
    "SH0008BK": ["pradhan mantri mudra yojana", "mudra", "mudra loan", "pmmy"],
    "SH000B51": ["pm vishwakarma scheme", "pm vishwakarma", "vishwakarma yojana", "vishwakarma", "vishwakarma scheme"],
    "SH0009RA": ["prime minister employment generation program", "pmegp"],
    "SH000889": ["stand up india", "standup india", "standup", "standupindia"],
    "SH00088L": ["credit guarantee fund for micro units", "cgfmu"],
    "DC0008R0": ["udyam registration", "udyam"],
    "SH0008C4": ["government e-marketplace", "gem"],
    "SH0008C5": ["trade receivables discounting system", "treds"],
    "SH0008C6": ["59 minutes loan gst", "msme loan 59 minutes gst", "59 minute loan gst"],
    "SH0008C7": ["59 minutes loan", "msme loan 59 minutes", "59 minute loan"],
    "SH0008C9": ["credit guarantee scheme for subordinate debt", "subordinate debt", "cgssd"],
    "SH0008C0": ["zed certification", "zed"],
    "SH0008C1": ["lean manufacturing competitiveness", "lean manufacturing"],
    "SH000893": ["technology and quality upgradation", "tequp"],
    "SH0008BJ": ["raw material assistance", "nsic rma"],
    "SH0008BL": ["credit facilitation", "nsic credit"],
    "SH0008BM": ["bill discounting", "nsic bill discounting"],
    "SH0008BN": ["single point registration", "sprs"],
    "SH0008BO": ["msme global mart", "global mart"],
    "SH0008BQ": ["testing fee reimbursement"],
    "SH0008BR": ["export promotion council membership reimbursement", "epc reimbursement"],
    "SH0008BS": ["bank loan processing reimbursement"],
    "SH0008BT": ["bank guarantee charges reimbursement"],
    "SH0008BU": ["capacity building management fee reimbursement"],
    "SH0008BW": ["special credit linked capital subsidy"],
    "SH0008BX": ["material testing labs", "nsic material testing"],
    "SH0008BY": ["market development assistance", "international cooperation"],
    "SH0008BZ": ["support for participation in domestic fairs", "procurement and marketing support"],
    "SH0008C3": ["msme samadhaan", "samadhaan"],
    "SH0008CA": ["amended technology upgradation fund scheme", "atufs"],
    "SH0009B9": ["marketing support barcode", "bar code"],
    "SH000B94": ["raising and accelerating msme productivity", "ramp"],
    "SH000BEX": ["e-commerce onboarding", "e-commerce", "ecommerce"],
    "SH000A16": ["food aggregator onboarding", "cloud kitchen"],
    "SH000A17": ["swiggy onboarding", "swiggy"],
    "DC00096J": ["fssai registration", "fssai"],
    "SH000AD3": ["fssai state license", "fssai state"],
    "SH000DFP": ["prime minister formalisation of micro food processing enterprises", "pmfme", "pm fme"],
    "SH0009G3": ["seed capital shg", "pm-fme"],
    "SH000DV7": ["iso 9000", "iso 14001"],
    "SH000DVV": ["product certification", "technology quality upgradation support"],
    "SH000DWF": ["iso 18000", "iso 22000", "iso 27000"],
    "SH000A2R": ["chief minister employment generation programme", "cmegp"],
    "SH0008RA": ["maitri portal", "single window registration", "maitri"],
    "SH0008PI": ["interest subsidy ksfc"],
    "SH0008RM": ["interest subsidy sc/st", "karnataka scst interest", "interest subsidy sc", "interest subsidy st"],
    "SH0008RN": ["term loan ksfc"],
    "SH0008RO": ["amara", "marketing related activities"],
    "SH0009ZJ": ["mukhya mantri krishak udyami", "mukhyamantri krishak udyami", "krishak udyami"],
    "SH000DNC": ["mukhyamantri udyami", "mukhya mantri udyami", "udyami"],
}

# Local cache of scheme documents keyed by GUID
SCHEME_DOCS: Dict[str, Document] = {}

@dataclass
class SchemeDocument:
    """Document structure for scheme data"""
    page_content: str
    # metadata: Optional[Dict[str, Any]]

class MongoSchemeRetriever:
    """Fast MongoDB-based scheme retriever for popular schemes"""
    
    def __init__(self, connection_string: str = None, database_name: str = "haqdarshak", collection_name: str = "schemes"):
        self.connection_string = connection_string or os.getenv("mongodb+srv://pooravdhingra:Poorav123@msmeassist.womwsrv.mongodb.net/haqdarshak")
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize MongoDB connection"""
        try:
            if not self.connection_string:
                raise ValueError("MongoDB connection string not provided")
            
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"MongoDB connection established to {self.database_name}.{self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB connection: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if MongoDB connection is available"""
        return self.client is not None
    
    @lru_cache(maxsize=500)
    def find_scheme_guid_by_query(self, query: str) -> Optional[str]:
        """
        Find scheme GUID by query with caching
        Uses text search and fuzzy matching for popular schemes
        """
        if not self.is_available():
            return None
        
        try:
            start_time = time.perf_counter()
            query_lower = query.lower().strip()
            
            
           # Check direct mappings first - iterate through scheme_mappings
            for guid, keywords in SCHEME_KEYWORDS.items():
                for keyword in keywords:
                    if keyword.lower() in query_lower:
                        elapsed = time.perf_counter() - start_time
                        logger.info(f"Found scheme tobi GUID via mapping: {guid} for query: '{query}' (matched: '{keyword}') in {elapsed:.3f}s")
                        return guid
            
            print(f"No direct mapping tobi found for query: '{query}'")
            # If no direct mapping, try MongoDB text search
            search_queries = [
                # Text search with original query
                {"$text": {"$search": query}},
                
                # Regex search on scheme name
                {"scheme_name": {"$regex": re.escape(query), "$options": "i"}},
                
                # Search in description
                {"scheme_description": {"$regex": re.escape(query), "$options": "i"}}
            ]
            
            print(f"Searching MongoDB with {search_queries} queries for: '{query}'")
            for search_query in search_queries:
                try:
                    # Add filters for active schemes
                    full_query = {
                        "$and": [
                            search_query,
                            {"scheme_status": {"$in": ["Active", "active"]}},
                            {"type": {"$in": ["Centrally Sponsored Scheme", "Central Sector Scheme", "State Scheme"]}}
                        ]
                    }
                    
                    # Find with relevance scoring for text search
                    if "$text" in search_query:
                        cursor = self.collection.find(
                            full_query,
                            {"scheme_guid": 1, "scheme_name": 1, "score": {"$meta": "textScore"}}
                        ).sort([("score", {"$meta": "textScore"})]).limit(1)
                    else:
                        cursor = self.collection.find(
                            full_query,
                            {"scheme_guid": 1, "scheme_name": 1}
                        ).limit(1)
                    
                    result = list(cursor)
                    if result:
                        scheme_guid = result[0].get("scheme_guid")
                        scheme_name = result[0].get("scheme_name", "Unknown")
                        elapsed = time.perf_counter() - start_time
                        logger.info(f"Found scheme GUID via MongoDB search: {scheme_guid} ({scheme_name}) for query: '{query}' in {elapsed:.3f}s")
                        return scheme_guid
                        
                except Exception as search_error:
                    logger.warning(f"Search query failed: {search_query}, error: {search_error}")
                    continue
            
            elapsed = time.perf_counter() - start_time
            logger.info(f"No scheme GUID found for query: '{query}' in {elapsed:.3f}s")
            return None
            
        except Exception as e:
            logger.error(f"Error in find_scheme_guid_by_query: {e}")
            return None
    
    def fetch_scheme_docs_by_guid(self, guid: str, limit: int = 10) -> List[SchemeDocument]:
        """
        Fetch scheme documents by GUID from MongoDB
        """
        if not self.is_available():
            return []
        
        try:
            start_time = time.perf_counter()
            
            # Query for the specific scheme
            query = {"scheme_guid": guid}
            

            # Project only needed fields to optimize performance
            projection = {
                "scheme_name": 1,
                "scheme_description": 1,
                "scheme_eligibility": 1,
                "benefit": 1,
                "application_process": 1,
            }
            
            cursor = self.collection.find(query, projection).limit(1)
            results = list(cursor)
            logging.info(f"Fetched {len(results)} documents tobi for GUID: {guid}")
            if not results:
                logger.warning(f"No documents found for scheme GUID: {guid}")
                return []
            
            # Convert MongoDB documents to SchemeDocument format
            scheme_docs = []
            for doc in results:
                # Create page content from key fields
                content_parts = []
                
                if doc.get("scheme_name"):
                    content_parts.append(f"Scheme Name: {doc['scheme_name']}")
                
                if doc.get("scheme_description"):
                    content_parts.append(f"Description: {doc['scheme_description']}")
                
                if doc.get("benefit"):
                    content_parts.append(f"Benefits: {doc['benefit']}")
                
                if doc.get("scheme_eligibility"):
                    content_parts.append(f"Eligibility: {doc['scheme_eligibility']}")
                
                if doc.get("application_process"):
                    content_parts.append(f"Application Process: {doc['application_process']}")
                
                # if doc.get("required_documents"):
                #     content_parts.append(f"Required Documents: {doc['required_documents']}")
                
                page_content = "\n\n".join(content_parts)
                
                # # Create metadata
                # metadata = {
                #     "scheme_name": doc.get("scheme_name"),
                #     "applicability_state": doc.get("applicability_state"),
                #     "target": doc.get("Individual/MSME"),
                #     "status": doc.get("scheme_status"),
                #     "source": "mongodb"
                # }
                
                scheme_docs.append(SchemeDocument(
                    page_content=page_content,
                ))
            elapsed = time.perf_counter() - start_time
            logger.info(f"Retrieved {len(scheme_docs)} tobi documents for GUID {guid} in {elapsed:.3f}s")
            return scheme_docs
            
        except Exception as e:
            logger.error(f"Error fetching scheme docs for GUID {guid}: {e}")
            return []
    
    def search_schemes_by_query(self, query: str, limit: int = 5) -> List[SchemeDocument]:
        """
        Search schemes by query text and return documents
        """
        if not self.is_available():
            return []
        
        try:
            start_time = time.perf_counter()
            
            # Multiple search strategies
            search_pipeline = [
                # Text search with scoring
                {
                    "$match": {
                        "$and": [
                            {"$text": {"$search": query}},
                            {"scheme_status": {"$in": ["Active", "active"]}},
                            {"type": {"$in": ["Centrally Sponsored Scheme", "Central Sector Scheme", "State Scheme"]}}
                        ]
                    }
                },
                {"$addFields": {"score": {"$meta": "textScore"}}},
                {"$sort": {"score": -1}},
                {"$limit": limit}
            ]
            
            results = list(self.collection.aggregate(search_pipeline))
            
            if not results:
                # Fallback to regex search
                fallback_query = {
                    "$and": [
                        {
                            "$or": [
                                {"scheme_name": {"$regex": re.escape(query), "$options": "i"}},
                                {"scheme_description": {"$regex": re.escape(query), "$options": "i"}}
                            ]
                        },
                        {"scheme_status": {"$in": ["Active", "active"]}}
                    ]
                }
                results = list(self.collection.find(fallback_query).limit(limit))
            
            # Convert to SchemeDocument format
            scheme_docs = []
            for doc in results:
                content_parts = []
                
                if doc.get("scheme_name"):
                    content_parts.append(f"Scheme Name: {doc['scheme_name']}")
                
                if doc.get("scheme_description"):
                    content_parts.append(f"Description: {doc['scheme_description']}")
                
                if doc.get("benefit"):
                    content_parts.append(f"Benefits: {doc['benefit']}")
                
                if doc.get("scheme_eligibility"):
                    content_parts.append(f"Eligibility: {doc['scheme_eligibility']}")
                
                page_content = "\n\n".join(content_parts)
                
                # metadata = {
                #     "scheme_guid": doc.get("scheme_guid"),
                #     "scheme_name": doc.get("scheme_name"),
                #     "type": doc.get("type"),
                #     "applicability_state": doc.get("applicability_state"),
                #     "source": "mongodb"
                # }
                
                scheme_docs.append(SchemeDocument(
                    page_content=page_content,
                    # metadata=metadata
                ))
            elapsed = time.perf_counter() - start_time
            logger.info(f"Search returned {len(scheme_docs)} documents for query '{query}' in {elapsed:.3f}s")
            return scheme_docs
            
        except Exception as e:
            logger.error(f"Error searching schemes for query '{query}': {e}")
            return []
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# Initialize global MongoDB retriever
mongo_retriever = None

def initialize_mongo_scheme_retriever():
    """Initialize MongoDB scheme retriever"""
    global mongo_retriever
    try:
        connection_string = "mongodb+srv://pooravdhingra:Poorav123@msmeassist.womwsrv.mongodb.net/haqdarshak"
        if not connection_string:
            logger.error("MONGODB_CONNECTION_STRING environment variable not set")
            return False
        
        mongo_retriever = MongoSchemeRetriever(connection_string)
        if mongo_retriever.is_available():
            logger.info("MongoDB scheme retriever initialized successfully")
            return True
        else:
            logger.error("MongoDB scheme retriever failed to connect")
            return False
            
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB scheme retriever: {e}")
        return False

# Compatibility functions to replace XLSX functions
def find_scheme_guid_by_query_mongo(query: str) -> Optional[str]:
    """Find scheme GUID using MongoDB"""
    if mongo_retriever and mongo_retriever.is_available():
        return mongo_retriever.find_scheme_guid_by_query(query)
    return None

def fetch_scheme_docs_by_guid_mongo(guid: str, vector_store=None, use_mongo: bool = True) -> List:
    """Fetch scheme docs by GUID - MongoDB version"""
    if use_mongo and mongo_retriever and mongo_retriever.is_available():
        docs = mongo_retriever.fetch_scheme_docs_by_guid(guid)
        # Convert to format expected by your existing code
        return [{"page_content": doc.page_content} for doc in docs]
    
    # Fallback to existing Pinecone logic if needed
    if vector_store:
        # Your existing Pinecone fallback code here
        pass
    
    return []

def search_schemes_by_query_mongo(query: str, limit: int = 5) -> List:
    """Search schemes by query - MongoDB version"""
    if mongo_retriever and mongo_retriever.is_available():
        docs = mongo_retriever.search_schemes_by_query(query, limit)
        return [{"page_content": doc.page_content} for doc in docs]
    return []