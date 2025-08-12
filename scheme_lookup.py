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

class XLSXSchemeDataManager:
    """Manages scheme data from XLSX file"""
    
    def __init__(self, xlsx_file_path: str):
        self.xlsx_file_path = xlsx_file_path
        self.data = None
        self.load_data()
    
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
                    logger.debug(f"Registered scheme document for GUID: {guid}")
                    
            except Exception as e:
                logger.error(f"Failed to register scheme from row: {str(e)}")
                continue
    
    def get_scheme_by_guid(self, guid: str) -> Optional[Dict]:
        """Get scheme data by GUID from XLSX"""
        if self.data is None:
            return None
        
        # Try different column names for GUID
        for col_name in ['scheme_guid', 'guid', 'id', 'scheme_id']:
            if col_name in self.data.columns:
                match = self.data[self.data[col_name].astype(str) == str(guid)]
                if not match.empty:
                    return match.iloc[0].to_dict()
        
        return None
    
    def search_schemes_by_keywords(self, keywords: List[str], limit: int = 5) -> List[Dict]:
        """Search schemes by keywords in XLSX data"""
        if self.data is None:
            return []
        
        matches = []
        search_columns = [
            'scheme_name', 'name', 'title',
            'scheme_description', 'description', 'desc',
            'keywords', 'tags'
        ]
        
        for _, row in self.data.iterrows():
            row_text = ""
            for col in search_columns:
                if col in self.data.columns and pd.notna(row[col]):
                    row_text += str(row[col]).lower() + " "
            
            # Check if any keyword matches
            for keyword in keywords:
                if keyword.lower() in row_text:
                    matches.append(row.to_dict())
                    break
        
        return matches[:limit]

# Global XLSX manager instance
xlsx_manager: Optional[XLSXSchemeDataManager] = None

def initialize_xlsx_manager(xlsx_file_path: str):
    """Initialize the global XLSX manager"""
    global xlsx_manager
    xlsx_manager = XLSXSchemeDataManager(xlsx_file_path)
    logger.info(f"Initialized XLSX manager with file: {xlsx_file_path}")

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
    for guid, keywords in SCHEME_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                logger.debug(f"Matched keyword '{kw}' for guid {guid}")
                return guid
    return None

def fetch_scheme_docs_by_guid(guid: str, index=None, use_xlsx: bool = True):
    """Return scheme document from XLSX, local cache, or Pinecone."""
    
    # First, try XLSX data if available and use_xlsx is True
    if use_xlsx and xlsx_manager:
        try:
            scheme_data = xlsx_manager.get_scheme_by_guid(guid)
            if scheme_data:
                # Build text content from scheme data
                text_parts = []
                content_fields = [
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
                
                for field in content_fields:
                    if field in scheme_data and pd.notna(scheme_data[field]):
                        text_parts.append(str(scheme_data[field]))
                
                if text_parts:
                    text = " ".join(text_parts)
                    doc = Document(page_content=text, metadata=scheme_data)
                    logger.info(f"Found scheme data for GUID {guid} in XLSX")
                    return [doc]
        except Exception as e:
            logger.error(f"Failed to fetch scheme from XLSX for GUID {guid}: {e}")
    
    # Fallback to local cache
    doc = SCHEME_DOCS.get(str(guid))
    if doc:
        logger.info(f"Found scheme data for GUID {guid} in local cache")
        return [doc]
    
    # Fallback to Pinecone
    logger.warning(f"Scheme document for GUID {guid} not found in XLSX or local cache")
    if index is not None:
        try:
            res = index.fetch(ids=[str(guid)], namespace="__default__")
            record = res.records.get(str(guid))
            if record and record.metadata:
                text = record.metadata.get("chunk_text", "")
                if text:
                    doc = Document(page_content=text, metadata=record.metadata)
                    SCHEME_DOCS[str(guid)] = doc
                    logger.info(f"Found scheme data for GUID {guid} in Pinecone")
                    return [doc]
        except Exception as exc:
            logger.error(f"Failed to fetch GUID {guid} from Pinecone: {exc}")
    
    logger.warning(f"No scheme data found for GUID {guid}")
    return []

def search_schemes_by_query(query: str, limit: int = 5) -> List[Document]:
    print(f"Searching schemes by query: {query} with limit {limit}")
    """Search schemes by query in XLSX data"""
    if not xlsx_manager:
        logger.warning("XLSX manager not initialized")
        return []
    
    try:
        # Extract keywords from query
        query_words = query.lower().split()
        
        # Search using XLSX manager
        matches = xlsx_manager.search_schemes_by_keywords(query_words, limit)
        
        documents = []
        for match in matches:
            # Build text content
            text_parts = []
            content_fields = [
                'scheme_name', 'name', 'title',
                'scheme_description', 'description', 'desc',
                'benefit', 'benefits',
                'application_process', 'process', 'how_to_apply',
                'scheme_eligibility', 'eligibility', 'eligible'
            ]
            
            for field in content_fields:
                if field in match and pd.notna(match[field]):
                    text_parts.append(str(match[field]))
            
            if text_parts:
                text = " ".join(text_parts)
                doc = Document(page_content=text, metadata=match)
                documents.append(doc)
        
        logger.info(f"Found {len(documents)} schemes matching query: {query}")
        return documents
        
    except Exception as e:
        logger.error(f"Failed to search schemes by query: {str(e)}")
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