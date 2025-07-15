import logging
from typing import Dict, List, Optional
from langchain.schema import Document
from langchain.schema import BaseRetriever

logger = logging.getLogger(__name__)

# Mapping of scheme GUIDs to search keywords
SCHEME_KEYWORDS: Dict[str, List[str]] = {
    "SH0008BK": ["pradhan mantri mudra yojana", "mudra", "mudra loan", "pmmy"],
    "SH000B51": ["pm vishwakarma scheme", "pm vishwakarma", "vishwakarma yojana", "vishwakarma", "vishwakarma scheme"],
    "SH000BGH": ["prime minister employment generation program", "pmegp"],
    "SH000889": ["stand up india", "standup india", "standup", "standupindia"],
    "SH00088L": ["credit guarantee fund for micro units", "cgfmu"],
    "SH0008VH": ["udyam registration", "udyam"],
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
    "SH000A2E": ["fssai registration", "fssai"],
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


def fetch_scheme_docs_by_guid(guid: str, index=None):
    """Return scheme document from the local cache or Pinecone."""
    doc = SCHEME_DOCS.get(str(guid))
    if doc:
        return [doc]
    logger.warning(f"Scheme document for GUID {guid} not found in local cache")
    if index is not None:
        try:
            res = index.fetch(ids=[str(guid)], namespace="__default__")
            record = res.records.get(str(guid))
            if record and record.metadata:
                text = record.metadata.get("chunk_text", "")
                if text:
                    doc = Document(page_content=text, metadata=record.metadata)
                    SCHEME_DOCS[str(guid)] = doc
                    return [doc]
        except Exception as exc:
            logger.error(f"Failed to fetch GUID {guid} from Pinecone: {exc}")
    return []


class DocumentListRetriever(BaseRetriever):
    """A simple retriever that returns a fixed list of documents."""

    docs: List[Document]

    def __init__(self, docs: List[Document]):
        super().__init__(docs=docs)

    def _get_relevant_documents(self, query: str, *, run_manager=None):  # type: ignore[override]
        return self.docs
