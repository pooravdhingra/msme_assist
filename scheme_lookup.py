# scheme_lookup.py
import logging
import pandas as pd
import os
from typing import Dict, List, Optional
from langchain.schema import Document
from langchain.schema import BaseRetriever
from functools import lru_cache

logger = logging.getLogger(__name__)

# Mapping of scheme GUIDs to search keywords
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

# XLSX data cache
XLSX_DATA_CACHE: Optional[pd.DataFrame] = None

class FastXLSXSchemeManager:
    """Ultra-fast in-memory scheme manager with precomputed indexes"""
    
    def __init__(self, xlsx_file_path: str):
        self.xlsx_file_path = xlsx_file_path
        self.data = None
        self.scheme_index = {}  # GUID -> row data
        self.keyword_index = {}  # keyword -> list of GUIDs
        self.text_index = {}    # text tokens -> list of GUIDs
        self.load_and_index_data()
    
    def load_and_index_data(self):
        """Load data and create all indexes upfront"""
        import time
        start = time.perf_counter()
        
        # Load data
        self.data = pd.read_excel(self.xlsx_file_path)
        self.data.columns = self.data.columns.str.strip().str.lower()
        
        # Build indexes
        self._build_scheme_index()
        self._build_keyword_index()
        self._build_text_index()
        
        logger.info(f"Fast XLSX manager loaded in {time.perf_counter() - start:.3f}s")
    
    def _build_scheme_index(self):
        """Build GUID -> data mapping"""
        for _, row in self.data.iterrows():
            for col_name in ['scheme_guid', 'guid', 'id', 'scheme_id']:
                if col_name in self.data.columns and pd.notna(row[col_name]):
                    guid = str(row[col_name])
                    self.scheme_index[guid] = row.to_dict()
                    break
    
    def _build_keyword_index(self):
        """Build keyword -> GUID mappings"""
        for guid, keywords in SCHEME_KEYWORDS.items():
            for keyword in keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(guid)
    
    def _build_text_index(self):
        """Build text search index"""
        search_columns = ['scheme_name', 'name', 'title', 'scheme_description', 'description','scheme_eligibility','application_process','benefit']
        
        for guid, row_data in self.scheme_index.items():
            text_tokens = set()
            for col in search_columns:
                if col in row_data and pd.notna(row_data[col]):
                    tokens = str(row_data[col]).lower().split()
                    text_tokens.update(tokens)
            
            for token in text_tokens:
                if token not in self.text_index:
                    self.text_index[token] = []
                self.text_index[token].append(guid)

    @lru_cache(maxsize=1)
    def load_data(self):
        """Load and cache XLSX data"""
        try:
            if not os.path.exists(self.xlsx_file_path):
                raise FileNotFoundError(f"XLSX file not found: {self.xlsx_file_path}")
            
            # Read the XLSX file
            self.data = pd.read_excel(self.xlsx_file_path)
            
            # Clean column names (remove extra spaces, convert to lowercase)
            self.data.columns = self.data.columns.str.strip().str.lower()
            
            # Log available columns for debugging
            logger.info(f"Loaded XLSX data with columns: {list(self.data.columns)}")
            logger.info(f"Total rows in XLSX: {len(self.data)}")
            
            # Cache the data globally
            global XLSX_DATA_CACHE
            XLSX_DATA_CACHE = self.data
            
            # Register scheme documents from XLSX
            self._register_xlsx_docs()
            
            return self.data
            
        except Exception as e:
            logger.error(f"Failed to load XLSX data: {str(e)}")
            raise
    
    def _register_xlsx_docs(self):
        """Register scheme documents from XLSX data"""
        if self.data is None:
            return
        
        for _, row in self.data.iterrows():
            try:
                # Try to get GUID from different possible column names
                guid = None
                for col_name in ['scheme_guid', 'guid', 'id', 'scheme_id']:
                    if col_name in self.data.columns:
                        guid = str(row[col_name]) if pd.notna(row[col_name]) else None
                        if guid:
                            break
                
                if not guid:
                    continue
                
                # Build text content from available columns
                text_parts = []
                content_columns = [
                    'scheme_name', 'name', 'title',
                    'scheme_description', 'description', 'desc',
                    'benefit', 'benefits',
                    'application_process', 'process', 'how_to_apply',
                    'scheme_eligibility', 'eligibility', 'eligible',
                    'documents_required', 'documents', 'docs',
                    'amount', 'loan_amount', 'subsidy',
                    'interest_rate', 'rate',
                    'details', 'full_details'
                ]
                
                for col in content_columns:
                    if col in self.data.columns and pd.notna(row[col]):
                        text_parts.append(str(row[col]))
                
                if text_parts:
                    text = " ".join(text_parts)
                    
                    # Create metadata from row data
                    metadata = {}
                    for col in self.data.columns:
                        if pd.notna(row[col]):
                            metadata[col] = row[col]
                    
                    # Create and store document
                    SCHEME_DOCS[guid] = Document(
                        page_content=text,
                        metadata=metadata
                    )
                    # logger.debug(f"Registered scheme document for GUID: {guid}")
                    
            except Exception as e:
                logger.error(f"Failed to register scheme from row: {str(e)}")
                continue
    
    def get_scheme_by_guid(self, guid: str) -> Optional[Dict]:
        """O(1) GUID lookup"""
        return self.scheme_index.get(str(guid))
    
    def search_schemes_fast(self, query: str, limit: int = 5) -> List[Dict]:
        """Fast search using precomputed indexes"""
        query_lower = query.lower()
        candidate_guids = set()
        
        # 1. Exact keyword matches (highest priority)
        for keyword, guids in self.keyword_index.items():
            if keyword in query_lower:
                candidate_guids.update(guids)
        
        # 2. Text token matches
        query_tokens = query_lower.split()
        for token in query_tokens:
            if token in self.text_index:
                candidate_guids.update(self.text_index[token])
        
        # Return top matches
        results = []
        for guid in list(candidate_guids)[:limit]:
            scheme_data = self.scheme_index.get(guid)
            if scheme_data:
                results.append(scheme_data)
        
        return results

# Global fast manager
fast_xlsx_manager: Optional[FastXLSXSchemeManager] = None

def initialize_fast_xlsx_manager(xlsx_file_path: str):
    """Initialize the fast XLSX manager"""
    global fast_xlsx_manager
    fast_xlsx_manager = FastXLSXSchemeManager(xlsx_file_path)
    logger.info(f"Initialized fast XLSX manager")

def add_scheme_keywords(guid: str, keywords: List[str]) -> None:
    """Add additional keywords for a scheme guid."""
    kw_list = SCHEME_KEYWORDS.setdefault(guid, [])
    for kw in keywords:
        kw_lower = kw.lower().strip()
        if kw_lower not in kw_list:
            kw_list.append(kw_lower)

def register_scheme_docs(records: List[Dict]) -> None:
    """Populate the local scheme document cache from record dictionaries."""
    for rec in records:
        guid = str(rec.get("scheme_guid") or rec.get("id"))
        text_parts = [
            rec.get("scheme_name", ""),
            rec.get("scheme_description", ""),
            rec.get("benefit", ""),
            rec.get("application_process", ""),
            rec.get("scheme_eligibility", ""),
        ]
        text = " ".join(part for part in text_parts if part) or rec.get("chunk_text", "")
        if text:
            SCHEME_DOCS[guid] = Document(page_content=text, metadata=rec)

def find_scheme_guid_by_query(query: str) -> Optional[str]:
    """Return scheme guid if query matches keywords."""
    text = query.lower()
       # Direct exact matches first (fastest)
    direct_matches = {
        "mudra": "SH0008BK",
        "pmegp": "SH0009RA", 
        "fssai": "DC00096J",
        "udyam": "DC0008R0",
        "vishwakarma": "SH000B51",
        "pmfme": "SH000DFP"
    }
    
    for key, guid in direct_matches.items():
        if key in text:
            logger.debug(f"Direct match: '{key}' -> {guid}")
            return guid
            
    for guid, keywords in SCHEME_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                # logger.debug(f"Matched keyword '{kw}' for guid {guid}")
                return guid
    return None



def fetch_scheme_docs_by_guid(guid: str, index=None, use_xlsx: bool = True):
    """Optimized GUID fetch with detailed logging"""
    
    logger.info(f"Fetching scheme docs for GUID: {guid}, use_xlsx: {use_xlsx}")
    
    # Fast XLSX lookup (O(1) time)
    if use_xlsx and fast_xlsx_manager:
        import time
        start_time = time.perf_counter()
        
        logger.debug(f"Using fast XLSX manager for GUID: {guid}")
        scheme_data = fast_xlsx_manager.get_scheme_by_guid(guid)
        
        fetch_time = time.perf_counter() - start_time
        logger.info(f"XLSX fetch time for GUID {guid}: {fetch_time:.3f}s")
        
        if scheme_data:
            logger.info(f"Found scheme data in XLSX for GUID: {guid}")
            
            # Log the fields being fetched
            available_fields = [k for k, v in scheme_data.items() if pd.notna(v) and str(v).strip() != '']
            logger.info(f"Available fields for GUID {guid}: {available_fields}")
            
            # Log specific field values for debugging
            key_fields = ['scheme_name', 'scheme_description', 'benefit', 'application_process', 'scheme_eligibility']
            field_data = {}
            for field in key_fields:
                if field in scheme_data and pd.notna(scheme_data[field]):
                    value = str(scheme_data[field])
                    field_data[field] = f"{len(value)} chars" if len(value) > 50 else value
            logger.info(f"Key field data for GUID {guid}: {field_data}")
            
            # Quick text building
            text_build_start = time.perf_counter()
            text_parts = [
                str(scheme_data.get('scheme_name', '')),
                str(scheme_data.get('scheme_description', '')),
                str(scheme_data.get('benefit', '')),
                str(scheme_data.get('application_process', '')),
                str(scheme_data.get('scheme_eligibility', '')),
            ]
            text = " ".join(p for p in text_parts if p and p != 'nan')
            text_build_time = time.perf_counter() - text_build_start
            
            logger.info(f"Text building time for GUID {guid}: {text_build_time:.3f}s")
            
            if text:
                logger.info(f"Built document text for GUID: {guid}, length: {len(text)} characters")
                doc = Document(page_content=text, metadata=scheme_data)
                
                total_time = time.perf_counter() - start_time
                logger.info(f"Total XLSX processing time for GUID {guid}: {total_time:.3f}s")
                
                return [doc]
            else:
                logger.warning(f"No valid text content found for GUID: {guid}")
        else:
            logger.warning(f"No scheme data found in XLSX for GUID: {guid} (fetch time: {fetch_time:.3f}s)")
    elif use_xlsx:
        logger.warning("Fast XLSX manager not available, falling back to cache")
    
    # Fallback to cache/Pinecone (existing logic)
    logger.debug(f"Checking document cache for GUID: {guid}")
    doc = SCHEME_DOCS.get(str(guid))
    if doc:
        logger.info(f"Found document in cache for GUID: {guid}")
        return [doc]
    
    logger.warning(f"No documents found for GUID: {guid} in any source")
    return []

def search_schemes_by_query(query: str, limit: int = 5) -> List[Document]:
    """Optimized search using fast manager with detailed logging"""
    import time
    
    logger.info(f"Searching schemes for query: '{query}', limit: {limit}")
    
    if not fast_xlsx_manager:
        logger.error("Fast XLSX manager not initialized - cannot perform search")
        return []
    
    try:
        search_start = time.perf_counter()
        
        # Use fast search
        matches = fast_xlsx_manager.search_schemes_fast(query, limit)
        
        search_time = time.perf_counter() - search_start
        logger.info(f"XLSX search time for query '{query}': {search_time:.3f}s, found {len(matches)} raw matches")
        
        if matches:
            # Log details of found matches
            match_details = []
            for i, match in enumerate(matches):
                scheme_name = match.get('scheme_name', 'Unknown')
                scheme_guid = match.get('scheme_guid', 'N/A')
                available_fields = [k for k, v in match.items() if pd.notna(v) and str(v).strip() != '']
                match_details.append(f"Match {i+1}: {scheme_name} (GUID: {scheme_guid}, Fields: {len(available_fields)})")
            
            logger.info(f"Match details: {'; '.join(match_details)}")
        
        doc_build_start = time.perf_counter()
        documents = []
        
        for i, match in enumerate(matches):
            # Log fields being processed for each match
            scheme_guid = match.get('scheme_guid', 'N/A')
            text_fields = ['scheme_name', 'scheme_description', 'benefit', 'application_process','scheme_eligibility']
            
            field_info = {}
            text_parts = []
            for field in text_fields:
                if field in match and pd.notna(match[field]):
                    value = str(match[field])
                    text_parts.append(value)
                    field_info[field] = f"{len(value)} chars" if len(value) > 50 else "present"
                else:
                    field_info[field] = "missing"
            
            logger.debug(f"Processing match {i+1} (GUID: {scheme_guid}), field status: {field_info}")
            
            if text_parts:
                text = " ".join(text_parts)
                doc = Document(page_content=text, metadata=match)
                documents.append(doc)
                logger.debug(f"Built document for match {i+1}: {len(text)} chars")
        
        doc_build_time = time.perf_counter() - doc_build_start
        total_time = time.perf_counter() - search_start
        
        logger.info(f"Document building time: {doc_build_time:.3f}s, Total search processing time: {total_time:.3f}s")
        logger.info(f"Successfully converted {len(documents)} matches to documents for query: '{query}'")
        
        return documents
        
    except Exception as e:
        logger.error(f"Fast search failed for query '{query}': {str(e)}")
        return []

class DocumentListRetriever(BaseRetriever):
    """A simple retriever that returns a fixed list of documents."""

    docs: List[Document]

    def __init__(self, docs: List[Document]):
        super().__init__(docs=docs)

    def _get_relevant_documents(self, query: str, *, run_manager=None):  # type: ignore[override]
        return self.docs

class XLSXSchemeRetriever(BaseRetriever):
    """A retriever that searches schemes in XLSX data"""
    
    def __init__(self, limit: int = 5):
        super().__init__()
        self.limit = limit
    
    def _get_relevant_documents(self, query: str, *, run_manager=None):  # type: ignore[override]
        return search_schemes_by_query(query, self.limit)