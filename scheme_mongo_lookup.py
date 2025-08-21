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

SCHEME_KEYWORDS_MSME: Dict[str, List[str]] = {
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

SCHEME_KEYWORDS_NON_MSME: Dict[str, List[str]] = {
"SH00000V": ["savitribai phule scholarship", "savitribai phule", "scholarship maharashtra"],
"SH00000X": ["rajarshi shahu maharaj meritorious scholarship", "vjnt sbc scholarship", "11th 12th scholarship"],
"SH00000Z": ["margin money schemes", "margin money", "maharashtra margin money"],
"SH000010": ["term loan schemes", "sc mpbcdc", "mpbcdc term loan"],
"SH000011": ["mahila samrudhi yojana", "women empowerment scheme", "mahila samrudhi"],
"SH000014": ["education loan mamfdc", "mamfdc education loan", "mamfdc loan"],
"SH000018": ["financial assistance female child", "bocw female child", "construction worker female child"],
"SH00001D": ["financial assistance funeral", "bocw funeral", "construction worker funeral"],
"SH00001E": ["financial assistance widow widower", "bocw death", "construction worker death"],
"SH00001G": ["financial assistance legal heir", "bocw accidental death", "construction worker accidental death"],
"SH00001K": ["financial assistance wedding", "bocw wedding", "construction worker wedding"],
"SH00001M": ["financial assistance disability", "bocw disability", "construction worker disability"],
"SH00001O": ["educational assistance children", "bocw children education", "construction worker children"],
"SH00001R": ["educational assistance 11th 12th", "bocw 11th 12th", "construction worker education"],
"SH000023": ["per drop more crop", "micro irrigation", "pradhan mantri krishi sinchayee yojana"],
"SH000024": ["indira gandhi national disability pension", "igndps", "disability pension"],
"SH000025": ["indira gandhi national widow pension", "ignwps", "widow pension"],
"SH000026": ["sanjay gandhi niradhar anudan yojana", "niradhar anudan", "maharashtra niradhar"],
"SH000028": ["educational assistance medical engineering", "bocw medical engineering", "construction worker degree"],
"SH000029": ["educational assistance 50 percent marks", "bocw 10th 12th", "construction worker scholarship"],
"SH00002B": ["financial assistance childbirth", "bocw childbirth", "construction worker childbirth"],
"SH00002C": ["medical assistance serious ailments", "bocw medical", "construction worker medical"],
"SH00002D": ["financial assistance ms cit", "bocw ms cit", "construction worker ms cit"],
"SH00002E": ["maharashtra agriculture mechanisation", "agriculture mechanisation", "farm mechanisation"],
"SH00002M": ["shravan bal seva rajya nivrutti vetan", "shravan bal seva", "maharashtra pension"],
"SH00003D": ["financial assistance disabled self employment", "disabled self employment", "maharashtra disabled"],
"SH00003I": ["post matric scholarship disabled", "disabled scholarship", "maharashtra disabled scholarship"],
"SH00003N": ["micro finance disabled", "nhfdc micro finance", "disabled nhfdc"],
"SH00003T": ["vasantrao naik tanda basti", "tanda basti development", "maharashtra tanda"],
"SH00003W": ["post matric scholarship st", "st scholarship", "government of india scholarship st"],
"SH00003X": ["maintenance allowance obc vjnt sbc", "professional course allowance", "obc vjnt sbc allowance"],
"SH00003Y": ["post matric scholarship sc obc vjnt sbc", "government of india scholarship", "sc obc scholarship"],
"SH00003Z": ["maintenance allowance st", "st professional course", "st allowance"],
"SH000040": ["post matric scholarship sc", "sc scholarship", "maharashtra sc scholarship"],
"SH000041": ["maintenance allowance sc", "sc professional course", "sc allowance"],
"SH000042": ["apki beti scheme", "rajasthan apki beti", "apki beti"],
"SH00004G": ["mukhya mantri rajshree yojana", "rajshree yojana", "rajasthan rajshree"],
"SH00004R": ["artisan identity card", "artisan card", "central artisan scheme"],
"SH00004Y": ["pre matric scholarship sc", "sc pre matric", "rajasthan sc scholarship"],
"SH00004Z": ["pre matric scholarship obc", "obc pre matric", "rajasthan obc scholarship"],
"SH000050": ["pre matric scholarship sbc mbc", "sbc mbc scholarship", "rajasthan sbc mbc"],
"SH000052": ["ambedkar national relief", "sc atrocities", "central sc victims"],
"SH00008D": ["goat sirohi genetic", "sirohi goat", "rajasthan goat evolution"],
"SH00009F": ["mahatma jyotiba phule jan arogya", "jyotiba phule arogya", "maharashtra health scheme"],
"SH00009J": ["soil health card", "central soil health", "soil card"],
"SH00009S": ["palanhaar scheme", "rajasthan palanhaar", "palanhaar"],
"SH00009W": ["chief minister special abled self employment", "disabled self employment rajasthan", "special abled employment"],
"SH00009X": ["special abled scholarship", "disabled scholarship rajasthan", "special abled rajasthan"],
"SH0000A1": ["chief minister old age pension", "old age pension rajasthan", "cm old age pension"],
"SH0000A3": ["chief minister disability pension", "disability pension rajasthan", "cm disability pension"],
"SH0000A5": ["maharana pratap aid gadia lohars", "gadia lohars construction", "rajasthan gadia lohars"],
"SH0000A6": ["chief minister destitute rehabilitation", "destitute rehabilitation rajasthan", "cm destitute"],
"SH0000A7": ["indira gandhi national old age pension", "old age pension rajasthan", "ignoaps rajasthan"],
"SH0000AS": ["farm pond khet talai", "khet talai construction", "rajasthan farm pond"],
"SH0000AT": ["per drop more crop rajasthan", "micro irrigation rajasthan", "rajasthan micro irrigation"],
"SH0000B4": ["devnarayan anuprati scheme", "devnarayan college", "rajasthan devnarayan"],
"SH0000BC": ["water pump set tribal", "tribal farmers pump", "maharashtra tribal pump"],
"SH0000BN": ["government ashram school st", "st ashram school", "maharashtra ashram school"],
"SH0000BQ": ["eklavya english medium school", "eklavya residential school", "st english medium school"],
"SH0000BR": ["tribal minority scholarship overseas", "st overseas scholarship", "maharashtra st overseas"],
"SH0000BX": ["hostel st students", "st hostel", "maharashtra st hostel"],
"SH0000BZ": ["free education st english medium", "st english medium school", "maharashtra st free education"],
"SH0000CT": ["nmdfc educational loan", "nmdfc credit line 1", "central nmdfc loan"],
"SH0000CU": ["nmdfc educational loan credit line 2", "nmdfc credit line 2", "central nmdfc credit"],
"SH0000D0": ["irrigation pipeline program", "rajasthan irrigation pipeline", "pipeline irrigation"],
"SH0000D1": ["tubewell borewell pump set", "rajasthan tubewell", "borewell scheme"],
"SH0000D6": ["post matric scholarships sc", "central sc scholarship", "sc post matric central"],
"SH0000DD": ["ambedkar post matric scholarship ebc", "ebc scholarship", "central ebc scholarship"],
"SH0000DQ": ["maharashtra fellowship tribal research", "tribal research fellowship", "maharashtra tribal research"],
"SH0000EE": ["state pre metric scholarship disabled", "disabled pre metric", "maharashtra disabled scholarship"],
"SH0000EH": ["scholarship differently abled", "nhfdc scholarship", "central disabled scholarship"],
"SH0000EI": ["swabhimaan scheme tribal", "bpl tribal families", "maharashtra tribal empowerment"],
"SH0000EU": ["indira gandhi national disability pension rajasthan", "igndps rajasthan", "disability pension rajasthan"],
"SH0000EW": ["indira gandhi national widow pension rajasthan", "ignwps rajasthan", "widow pension rajasthan"],
"SH0000F3": ["devnarayan scooty yojana", "devnarayan incentive", "rajasthan scooty scheme"],
"SH0000F8": ["indira gandhi old age pension", "old age pension maharashtra", "ignoaps maharashtra"],
"SH0000FE": ["post matric scholarship transgender", "transgender scholarship", "rajasthan transgender scholarship"],
"SH0000FG": ["deen dayal upadhyay land housing", "housing land purchase", "maharashtra land housing"],
"SH0000FH": ["farm pond programme", "maharashtra farm pond", "demand based farm pond"],
"SH0000FJ": ["abdul kalam amrut aahaar", "amrut aahaar scheme", "maharashtra amrut aahaar"],
"SH0000FN": ["credit linked subsidy mig", "clss mig", "housing subsidy mig"],
"SH0000G3": ["national biogas manure management", "biogas programme", "maharashtra biogas"],
"SH0000G4": ["golden jubilee scholarship st", "st class 1 to 10", "maharashtra st scholarship"],
"SH0000G5": ["ex gratia grant st accidental death", "st ashram school accident", "maharashtra st accident"],
"SH0000G6": ["navinyapurna yojana milch animals", "cow buffalo distribution", "maharashtra milch animals"],
"SH0000GB": ["grant tractor power tiller", "agricultural mechanisation grant", "sub mission agricultural mechanisation"],
"SH0000GC": ["subsidised seeds rice", "paddy production programme", "rashtriya krishi vikas yojana rice"],
"SH0000GD": ["grant rice cultivation equipment", "paddy equipment grant", "rice cultivation grant"],
"SH0000GI": ["district annual scheme milch animal", "sc st milch animal", "maharashtra sc st animal"],
"SH0000GJ": ["navinyapurna scheme goats", "stallfed goats distribution", "maharashtra goat distribution"],
"SH0000GP": ["mukhyamantri yuva swavalamban", "yuva swavalamban yojana", "rajasthan youth self employment"],
"SH0000GT": ["mahila yogyata scholarship", "women scholarship rajasthan", "rajasthan mahila scholarship"],
"SH0000GV": ["urdu scholarship", "rajasthan urdu scholarship", "urdu medium scholarship"],
"SH0000GY": ["scholarship indian military college", "military college dehradun", "central military scholarship"],
"SH0000GZ": ["shubhshakti scheme bocw", "construction worker rajasthan", "rajasthan bocw"],
"SH0000H2": ["student traveling concession", "travel concession rajasthan", "rajasthan student travel"],
"SH0000H6": ["nirman shramik sulabhy awas", "bocw housing rajasthan", "construction worker housing rajasthan"],
"SH0000H7": ["onion storage subsidy", "rajasthan onion storage", "onion subsidy"],
"SH0000H9": ["gopinath munde farmer accident", "farmer accident protection", "maharashtra farmer accident"],
"SH0000HA": ["free travel mentally retarded", "disabled travel rajasthan", "mental retardation travel"],
"SH0000HB": ["pradhan mantri ujjwala yojana", "ujjwala yojana", "central ujjwala"],
"SH0000HC": ["silicosis victims assistance", "silicosis scheme rajasthan", "rajasthan silicosis"],
"SH0000HU": ["district annual scheme goats", "sc st goats distribution", "maharashtra sc st goats"],
"SH0000HV": ["district annual scheme animal husbandry", "sc st animal training", "maharashtra sc st training"],
"SH0000HW": ["navinyapurna scheme poultry farming", "broiler birds rearing", "maharashtra poultry farming"],
"SH0000I6": ["nashik zilla parishad cess", "nashik cess scheme", "maharashtra zilla parishad"],
"SH0000IU": ["chief minister higher education scholarship", "cm higher education", "rajasthan cm scholarship"],
"SH0000IV": ["protected cultivation horticulture", "national horticulture mission", "maharashtra protected cultivation"]
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
    def find_scheme_guid_by_query(self, query: str,user_type: int=1) -> Optional[str]:
        """
        Find scheme GUID by query with caching
        Uses text search and fuzzy matching for popular schemes
        """
        logger.info(f"Finding scheme GUID podu for query: {query} with userType : {user_type}")
        if not self.is_available():
            return None
        
        try:
            start_time = time.perf_counter()
            query_lower = query.lower().strip()
            
            
           # Check direct mappings first - iterate through scheme_mappings
            if user_type == 1:  # MSME user type
                for guid, keywords in SCHEME_KEYWORDS_MSME.items():
                    for keyword in keywords:
                        if keyword.lower() in query_lower:
                            elapsed = time.perf_counter() - start_time
                            logger.info(f"Found scheme tobi GUID via mapping: {guid} for query: '{query}' (matched: '{keyword}') in {elapsed:.3f}s")
                            return guid
            else:  # Non-MSME user type
                for guid, keywords in SCHEME_KEYWORDS_NON_MSME.items():
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
    
    def fetch_scheme_docs_by_guid(self, guid: str, limit: int = 10,user_type:int=1) -> List[SchemeDocument]:
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
                
              
                
                page_content = "\n\n".join(content_parts)
                
                
                
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
def find_scheme_guid_by_query_mongo(query: str,user_type: int) -> Optional[str]:
    """Find scheme GUID using MongoDB"""
    if mongo_retriever and mongo_retriever.is_available():
        return mongo_retriever.find_scheme_guid_by_query(query,user_type)
    return None

def fetch_scheme_docs_by_guid_mongo(guid: str, vector_store=None, use_mongo: bool = True,user_type :int = 1) -> List:
    """Fetch scheme docs by GUID - MongoDB version"""
    if use_mongo and mongo_retriever and mongo_retriever.is_available():
        docs = mongo_retriever.fetch_scheme_docs_by_guid(guid, user_type=user_type)
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