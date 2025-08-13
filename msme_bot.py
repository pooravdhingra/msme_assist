# msme_bot.py
import logging
import time
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from data_loader import load_rag_data, load_dfl_data, PineconeRecordRetriever
from scheme_lookup import (
    find_scheme_guid_by_query,
    fetch_scheme_docs_by_guid,
    DocumentListRetriever,
    initialize_fast_xlsx_manager,  # Add this import
    search_schemes_by_query,  # Add this import
    XLSXSchemeRetriever,     # Add this import
)
from utils import extract_scheme_guid
from data import DataManager
import re
import os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, AsyncGenerator, Tuple
import redis.asyncio as redis

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
logger = logging.getLogger(__name__)
logging.getLogger("pymongo").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=8)

# Enhanced Cache Manager
class CacheManager:
    def __init__(self):
        try:
            self.redis_client = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379"),
                decode_responses=True
            )
        except:
            self.redis_client = None
            logger.warning("Redis not available, using memory cache only")
        
        self.memory_cache = {}
        self.max_memory_cache_size = 1000
    
    def _cleanup_memory_cache(self):
        """Keep memory cache size under control"""
        if len(self.memory_cache) > self.max_memory_cache_size:
            # Remove oldest half of entries
            items = list(self.memory_cache.items())
            items_to_keep = items[len(items)//2:]
            self.memory_cache = dict(items_to_keep)
    
    @lru_cache(maxsize=500)
    def _generate_cache_key(self, query: str, context: str = "") -> str:
        """Generate consistent cache keys"""
        key_data = f"{query}:{context}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get_intent_cache(self, query: str, conversation_history: str = "") -> Optional[str]:
        """Get cached intent with memory fallback"""
        cache_key = self._generate_cache_key(query, conversation_history)
        
        # Try Redis first
        if self.redis_client:
            try:
                cached = await self.redis_client.get(f"intent:{cache_key}")
                if cached:
                    return cached
            except:
                pass
        
        # Fallback to memory cache
        return self.memory_cache.get(f"intent:{cache_key}")
    
    async def set_intent_cache(self, query: str, intent: str, conversation_history: str = "", ttl: int = 3600):
        """Set intent cache with Redis and memory"""
        cache_key = self._generate_cache_key(query, conversation_history)
        
        # Set in Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(f"intent:{cache_key}", ttl, intent)
            except:
                pass
        
        # Set in memory cache
        self.memory_cache[f"intent:{cache_key}"] = intent
        self._cleanup_memory_cache()
    
    async def get_rag_cache(self, query: str, user_context: dict = None) -> Optional[dict]:
        """Get cached RAG response"""
        context_str = json.dumps(user_context or {}, sort_keys=True)
        cache_key = self._generate_cache_key(query, context_str)
        
        if self.redis_client:
            try:
                cached = await self.redis_client.get(f"rag:{cache_key}")
                return json.loads(cached) if cached else None
            except:
                pass
        
        return self.memory_cache.get(f"rag:{cache_key}")
    
    async def set_rag_cache(self, query: str, data: dict, user_context: dict = None, ttl: int = 1800):
        """Set RAG cache"""
        context_str = json.dumps(user_context or {}, sort_keys=True)
        cache_key = self._generate_cache_key(query, context_str)
        
        if self.redis_client:
            try:
                await self.redis_client.setex(f"rag:{cache_key}", ttl, json.dumps(data))
            except:
                pass
        
        # Don't store large RAG responses in memory cache
        if len(str(data)) < 10000:  # Only cache smaller responses in memory
            self.memory_cache[f"rag:{cache_key}"] = data
            self._cleanup_memory_cache()

# Initialize cache manager
cache_manager = CacheManager()


def init_fast_xlsx_scheme_manager():
    """Initialize fast XLSX scheme manager"""
    try:
        xlsx_path = os.getenv("SCHEME_XLSX_PATH")
        if not xlsx_path:
            logger.error("SCHEME_XLSX_PATH environment variable not set")
            return False
        
        if not os.path.exists(xlsx_path):
            logger.error(f"XLSX file not found at path: {xlsx_path}")
            return False
            
        initialize_fast_xlsx_manager(xlsx_path)
        logger.info(f"Fast XLSX manager initialized successfully with file: {xlsx_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize fast XLSX manager: {e}")
        return False

# Initialize at module level
FAST_XLSX_AVAILABLE = init_fast_xlsx_scheme_manager()


# Initialize DataManager - make it async capable
class AsyncDataManager(DataManager):
    async def get_conversations_async(self, mobile_number: str):
        """Async version of get_conversations"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self.get_conversations, mobile_number)
    
    async def save_conversation_async(self, session_id: str, mobile_number: str, messages: list):
        """Async version of save_conversation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self.save_conversation, session_id, mobile_number, messages)

# Initialize async data manager
data_manager = AsyncDataManager()

# Initialize cached resources with async support
@lru_cache(maxsize=1)
def init_llm():
    """Initialize the default LLM client for all tasks except intent classification."""
    logger.info("Initializing LLM client")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    llm = ChatOpenAI(
        model="gpt-4.1-mini-2025-04-14",
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        temperature=0
    )
    logger.info("LLM initialized")
    return llm

@lru_cache(maxsize=1)
def init_intent_llm():
    """Initialize a dedicated LLM client for intent classification."""
    logger.info("Initializing Intent LLM client")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    intent_llm = ChatOpenAI(
        model="gpt-4.1",
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        temperature=0
    )
    logger.info("Intent LLM initialized")
    return intent_llm

@lru_cache(maxsize=1)
def init_vector_store():
    logger.info("Loading vector store")
    index_host = os.getenv("PINECONE_SCHEME_HOST")
    if not index_host:
        raise ValueError("PINECONE_SCHEME_HOST environment variable not set")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    try:
        vector_store = load_rag_data(host=index_host, index_name=index_name, version_file="faiss_version.txt")
    except Exception as e:
        logger.error(f"Failed to load scheme index: {e}")
        raise
    logger.info("Vector store loaded")
    return vector_store

@lru_cache(maxsize=1)
def init_dfl_vector_store():
    logger.info("Loading DFL vector store")
    google_drive_file_id = os.getenv("DFL_GOOGLE_DOC_ID")
    if not google_drive_file_id:
        raise ValueError("DFL_GOOGLE_DOC_ID environment variable not set")
    index_host = os.getenv("PINECONE_DFL_HOST")
    if not index_host:
        raise ValueError("PINECONE_DFL_HOST environment variable not set")
    index_name = os.getenv("PINECONE_DFL_INDEX_NAME")
    try:
        vector_store = load_dfl_data(google_drive_file_id, host=index_host, index_name=index_name)
    except Exception as e:
        logger.error(f"Failed to load DFL index: {e}")
        raise
    logger.info("DFL vector store loaded")
    return vector_store

llm = init_llm()
intent_llm = init_intent_llm()
scheme_vector_store = init_vector_store()
dfl_vector_store = init_dfl_vector_store()

# Keep your existing dataclasses and helper functions unchanged
@dataclass
class UserContext:
    name: str
    state_id: str
    state_name: str
    business_name: str
    business_category: str
    gender: str

class SessionData:
    """Simple container for per-session information."""
    def __init__(self, user=None):
        self.user = user
        self.messages = []
        self.rag_cache = {}
        self.dfl_rag_cache = {}

def get_user_context(session_state):
    try:
        user = session_state.user
        return UserContext(
            name=user["fname"],
            state_id=user.get("state_id", "Unknown"),
            state_name=user.get("state_name", "Unknown"),
            business_name=user.get("business_name", "Unknown"),
            business_category=user.get("business_category", "Unknown"),
            gender=user.get("gender", "Unknown"),
        )
    except AttributeError:
        logger.error("User data not found in session state")
        return None

def detect_language(query):
    """Keep your existing language detection logic"""
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    if devanagari_pattern.search(query):
        return "Hindi"
    
    hindi_words = [
        "kya", "kaise", "ke", "mein", "hai", "kaun", "kahan", "kab",
        "batao", "sarkari", "yojana", "paise", "karobar", "dukaan", "nayi", "naye", "chahiye", "madad", "karo",
        "dikhao", "samjhao", "tarika", "aur", "arey", "bhi", "kya", "hai", "hoga", "hogi", "ho", "hoon", "magar", "lekin", "par", 
        "toh", "ab", "phir", "kuch", "thoda", "zyada", "sab", "koi", "kuchh", "aap", "tum", "main",
        "hum", "unhe", "unko", "unse", "yeh", "woh", "aisa", "aisi", "aise", "bataiye", "achha", "acha", "accha", "theek", "theekh", 
        "thik", "thikk", "idhar", "udhar", "yahan", "wahan", "waha", "bhai", "bhaiya", "bhaiyya", 
        "bhiya", "bahut", "bahot", "bohot", "bahuut", "zara", "jara", "mat", "maat", "matlab", "matlb", "fir", "phirr", "phhir", "phir", 
        "main", "aap", "aapke", "yojanaen", "liye", "kar", "sakte", "hain", "tak"
    ]
    query_lower = query.lower()
    hindi_word_count = sum(1 for word in hindi_words if word in query_lower)
    total_words = len(query_lower.split())
    
    if total_words > 0 and hindi_word_count / total_words > 0.15:
        return "Hinglish"
    
    return "English"

def get_system_prompt(language, user_name="User", word_limit=200):

    """Return tone and style instructions."""

    system_rules = f"""1. **Language Handling**:
       - The query language is provided as {language} (English, Hindi, or Hinglish).
       - For Hindi queries, respond in Devanagari script using simple, clear words suitable for micro business owners with low Hindi proficiency.
       - For Hinglish queries, use a natural mix of simple English and Hindi words in Roman script, prioritizing hindi words in the mix.
       - For English queries, respond in simple English.
       
       2. **Response Guidelines**:
       - Scope: Only respond to queries about government schemes, digital/financial literacy, or business growth.
       - Tone and Style: Use simple, clear words, short sentences, friendly tone, relatable examples.
       - Give structured responses with formatting like bullets or headings/subheadings. Do not give long paragraphs of text.
       - Response must be <={word_limit} words.

       - Never mention agent fees unless specified in RAG Response for scheme queries.
       - Never repeat user query or bring up ambiguity in the response, proceed directly to answering.
       - Never mention technical terms like RAG, LLM, Database etc. to the user.
       - Use scheme names exactly as provided in the RAG Response without paraphrasing (underscores may be replaced with spaces).
       - Start the response with 'Hi {user_name}!' (English), 'Namaste {user_name}!' (Hinglish), or 'नमस्ते {user_name}!' (Hindi) unless Out_of_Scope."""

    system_prompt = system_rules.format(language=language, user_name=user_name)
    return system_prompt


# Build conversation history from stored messages for intent classification
def build_conversation_history(messages):
    conversation_history = ""
    session_messages = []
    for msg in messages[-10:]:
        if msg["role"] == "assistant" and "Welcome" in msg["content"]:
            continue
        session_messages.append((msg["role"], msg["content"], msg["timestamp"]))
    session_messages = sorted(session_messages, key=lambda x: x[2], reverse=True)[:5]
    for role, content, _ in session_messages:
        conversation_history += f"{role.capitalize()}: {content}\n"
    return conversation_history

# Welcome user
def welcome_user(state_name, user_name, query_language):
    """Generate a welcome message in the user's chosen language."""
    prompt = f"""You are a helpful assistant for Haqdarshak, supporting small business owners in India with government schemes, digital/financial literacy, and business growth. The user is a new user named {user_name} from {state_name}.

    **Input**:
    - Query Language: {query_language}

    **Instructions**:
    - Generate a welcome message for a new user in the specified language ({query_language}).
    - For Hindi, use Devanagari script with simple, clear words suitable for micro business owners with low Hindi proficiency.
    - For English, use simple English with a friendly tone.
    - The message should welcome the user, and offer assistance with schemes and documents applicable to their state and all central government schemes or help with digital/financial literacy and business growth.
    - Response must be ≤70 words.
    - Start the response with 'Hi {user_name}!' (English) or 'नमस्ते {user_name}!' (Hindi).

    **Output**:
    - Return only the welcome message in the specified language.
    """

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        generated_response = response.content.strip()
        logger.info(f"Generated welcome message in {query_language}: {generated_response}")
        return generated_response
    except Exception as e:
        logger.error(f"Failed to generate welcome message: {str(e)}")
        # Fallback to default messages
        if query_language == "Hindi":
            return f"नमस्ते {user_name}! हकदर्शक MSME चैटबॉट में स्वागत है। आप {state_name} से हैं, मैं आपकी राज्य और केंद्रीय योजनाओं में मदद करूँगा। आज कैसे सहायता करूँ?"
        return f"Hi {user_name}! Welcome to Haqdarshak MSME Chatbot! Since you're from {state_name}, I'll help with schemes and documents applicable to your state and all central government schemes. How can I assist you today?"

def generate_interaction_id(query, timestamp):
    return f"{query[:500]}_{timestamp.strftime('%Y%m%d%H%M%S')}"

# NEW: Async versions of your core functions
async def classify_intent_async(query: str, conversation_history: str = "") -> str:
    """Async version of classify_intent with caching"""
    # Check cache first
    cached_intent = await cache_manager.get_intent_cache(query, conversation_history)
    if cached_intent:
        logger.info(f"Intent cache hit: {cached_intent}")
        return cached_intent
    
    prompt = f"""You are an assistant for Haqdarshak. Classify the user's intent.

    **Input**:
    - Query: {query}
    - Conversation History: {conversation_history}

    **Instructions**:
    Return only one label from the following:
       - Schemes_Know_Intent - General queries enquiring about schemes or loans without specific names (e.g., 'show me schemes', 'mere liye schemes dikhao', 'loan', 'Schemes for credit?', 'MSME ke liye schemes kya hain?', 'क्रेडिट के लिए योजनाएं?', 'loan chahiye', 'scheme dikhao' etc.)
       - DFL_Intent - Digital/financial literacy queries (e.g., 'Current account', 'How to use UPI?', 'डिजिटल भुगतान कैसे करें?', 'Opening Bank Account', 'Why get Insurance', 'Why take loans', 'Online Safety', 'Setting up internet banking', 'Benefits of internet for business' etc.)
       - Specific_Scheme_Know_Intent - Queries that mention specific scheme names. Generally asking for loan or scheme is NOT specific. (e.g., 'What is FSSAI?', 'PMFME ke baare mein batao', 'एफएसएसएआई क्या है?', 'Pashu Kisan Credit Scheme ke baare mein bataiye', 'Udyam', 'Mudra Yojana', 'pmegp' etc.)
       - Specific_Scheme_Apply_Intent - Queries about applying for specific schemes (e.g., 'Apply', 'Apply kaise karna hai', 'How to apply for FSSAI?', 'FSSAI kaise apply karu?', 'एफएसएसआईएआई के लिए आवेदन कैसे करें?' etc.)
       - Specific_Scheme_Eligibility_Intent - Queries about eligibility for specific schemes (e.g., 'Eligibility', 'Eligibility batao', 'Am I eligible for FSSAI?', 'FSSAI eligibility?', 'एफएसएसआईएआई की पात्रता क्या है?' etc.)
       - Out_of_Scope - Queries that are not relevant to business growth or digital literacy or financial literacy (e.g., 'What's the weather?', 'Namaste', 'मौसम कैसा है?', 'Time?' etc.)
       - Contextual_Follow_Up - Follow-up queries (e.g., 'Tell me more', 'Aur batao', 'और बताएं', 'iske baare mein aur jaankaari chahiye' etc.)
       - Confirmation_New_RAG - Confirmation for initiating another RAG search (Only to be chosen when user query is confirmation for initating another RAG search ("Yes", "Haan batao", "Haan dikhao", "Yes search again") AND previous assistant response says that the bot needs to fetch more details about some scheme. ('I need to fetch more details about [scheme name]. Please confirm if this is the scheme you meant.'))
       - Gratitude_Intent - User expresses thanks or acknowledgement (e.g., 'ok thanks', 'got it', 'theek hai', 'accha', 'thank you', 'शुक्रिया', 'धन्यवाद' etc.)

    **Tips**:
       - Use rule-based checks for Out_of_Scope (keywords: 'hello', 'hi', 'hey', 'weather', 'time', 'namaste', 'mausam', 'samay').
       - Single word queries with scheme names like 'pmegp', 'fssai', 'udyam' are in scope and should be classified as Specific_Scheme_Know_Intent.
       - For Contextual_Follow_Up, prioritise the most recent query-response pair from the conversation history to check if the query is a follow-up.
       - Use conversation history for context but intent should be determined solely by the current query.
       - To distinguish between Specific_Scheme_Know_Intent and Scheme_Know_Intent, check for whether query is asking for information about specific scheme or general information about schemes.
       - If some scheme name is mentioned in the query, then classify it as Specific_Scheme_Know_Intent.
    """
    
    try:
        response = await intent_llm.ainvoke([{"role": "user", "content": prompt}])
        intent = response.content.strip()
        
        # Cache the result
        await cache_manager.set_intent_cache(query, intent, conversation_history)
        
        return intent
    except Exception as e:
        logger.error(f"Failed to classify intent: {str(e)}")
        return "Out_of_Scope"

async def get_rag_response_async(query, vector_store, state="ALL_STATES", gender=None, business_category=None):
    """Async version of get_rag_response"""
    try:
        details = []
        if state:
            details.append(f"state: {state}")
        if gender:
            details.append(f"gender: {gender}")
        if business_category:
            details.append(f"business category: {business_category}")

        full_query = query
        if details:
            full_query = f"{full_query}. {' '.join(details)}"

        # logger.debug(f"Processing query: {full_query}")
        
        # Run retrieval in thread pool since it's CPU intensive
        loop = asyncio.get_event_loop()
        
        def run_retrieval():
            retriever = PineconeRecordRetriever(
                index=vector_store, state=state, gender=gender, k=5
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
            return qa_chain.invoke({"query": full_query})
        
        result = await loop.run_in_executor(executor, run_retrieval)
        response = result["result"]
        sources = result["source_documents"]
        
        logger.info(f"RAG response generated: {response}")
        return {"text": response, "sources": sources}
    except Exception as e:
        logger.error(f"RAG retrieval failed: {str(e)}")
        return {"text": "Error retrieving scheme information.", "sources": []}

async def get_scheme_response_async(
    query,
    vector_store,
    state="ALL_STATES",
    gender=None,
    business_category=None,
    include_mudra=False,
    intent=None,
    use_xlsx=True,  # Add this parameter
):
    """Async version of get_scheme_response with XLSX support"""
    logger.info("Querying scheme dataset")

    # Check cache first
    cache_context = {
        "state": state,
        "gender": gender,
        "business_category": business_category,
        "include_mudra": include_mudra,
        "intent": intent,
         "use_xlsx": True
    }
    
    cached_response = await cache_manager.get_rag_cache(query, cache_context)
    if cached_response:
        logger.info("Scheme response cache hit")
        return cached_response

    guid = None
    rag = None
    
    # For specific scheme queries, try direct XLSX lookup first
    if intent == "Specific_Scheme_Know_Intent" and FAST_XLSX_AVAILABLE:
        guid_start = time.perf_counter()
        logger.info(f"Using XLSX for specific scheme lookup - Query: {query}")
        
        # First, try to find GUID
        loop = asyncio.get_event_loop()
        guid = await loop.run_in_executor(executor, find_scheme_guid_by_query, query)

        guid_time = time.perf_counter() - guid_start
        logger.info(f"GUID lookup time: {guid_time:.3f}s, found: {guid}")
        
        if guid:
            fetch_start = time.perf_counter()
            logger.info(f"Found popular scheme GUID: {guid} for query: '{query}'")
            # Fetch docs directly from XLSX
            docs = await loop.run_in_executor(
                executor, 
                fetch_scheme_docs_by_guid, 
                guid, 
                None, 
                True
            )
            fetch_time = time.perf_counter() - fetch_start
            
            if docs:
                llm_start = time.perf_counter()
                logger.info(f"Retrieved {len(docs)} documents from XLSX for GUID: {guid} (fetch time: {fetch_time:.3f}s)")
                def run_qa_chain():
                    retriever = DocumentListRetriever(docs)
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                    )
                    return qa_chain.invoke({"query": query})
                
                result = await loop.run_in_executor(executor, run_qa_chain)
                llm_time = time.perf_counter() - llm_start
               
                rag = {"text": result["result"], "sources": result["source_documents"]}
                logger.info(f"LLM processing time: {llm_time:.3f}s, response length: {len(rag['text'])} chars")
                logger.info("Successfully retrieved scheme data from XLSX")
            else:
                logger.warning("No documents found in XLSX for GUID; falling back to search")
    
            # If direct GUID lookup didn't work, try XLSX search
        if not rag and use_xlsx and FAST_XLSX_AVAILABLE:
                search_start = time.perf_counter()
                logger.info("Using XLSX search for scheme lookup")
                
                loop = asyncio.get_event_loop()
                
                def run_xlsx_search():
                    # Search schemes in XLSX
                    logger.info(f"Searching schemes by query: {query} with limit 5")
                    docs = search_schemes_by_query(query, limit=5)
                    
                    if docs:
                        retriever = DocumentListRetriever(docs)
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            return_source_documents=True,
                        )
                        return qa_chain.invoke({"query": query})
                    return None
                
                result = await loop.run_in_executor(executor, run_xlsx_search)
                search_time = time.perf_counter() - search_start
                
                if result:
                    rag = {"text": result["result"], "sources": result["source_documents"]}
                    logger.info(f"XLSX search + LLM time: {search_time:.3f}s, response length: {len(rag['text'])} chars")
                    logger.info("Successfully retrieved scheme data from XLSX search")
    
    # Fallback to Pinecone if XLSX didn't work
    if not rag:
        logger.info("Falling back to Pinecone search")
        if guid:
            # Try Pinecone with known GUID
            docs = await loop.run_in_executor(
                executor, 
                fetch_scheme_docs_by_guid, 
                guid, 
                vector_store,
                False  # use_xlsx=False for Pinecone fallback
            )
            
            if docs:
                def run_qa_chain():
                    retriever = DocumentListRetriever(docs)
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                    )
                    return qa_chain.invoke({"query": query})
                
                result = await loop.run_in_executor(executor, run_qa_chain)
                rag = {"text": result["result"], "sources": result["source_documents"]}
        else:
            # Regular Pinecone search
            rag = await get_rag_response_async(
                query,
                vector_store,
                state=state,
                gender=gender,
                business_category=business_category,
            )

    # Handle Mudra inclusion (if needed)
    if include_mudra:
        logger.info("Including Pradhan Mantri Mudra Yojana details")
        loop = asyncio.get_event_loop()
        
        mudra_guid = await loop.run_in_executor(
            executor, 
            find_scheme_guid_by_query, 
            "pradhan mantri mudra yojana"
        ) or "SH0008BK"
        
        # Try XLSX first for Mudra
        mudra_docs = None
        if use_xlsx and FAST_XLSX_AVAILABLE:
            mudra_docs = await loop.run_in_executor(
                executor, 
                fetch_scheme_docs_by_guid, 
                mudra_guid, 
                None, 
                True
            )
        
        # Fallback to Pinecone for Mudra if needed
        if not mudra_docs:
            mudra_docs = await loop.run_in_executor(
                executor, 
                fetch_scheme_docs_by_guid, 
                mudra_guid, 
                vector_store,
                False
            )

        if mudra_docs:
            def run_mudra_qa():
                retriever = DocumentListRetriever(mudra_docs)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                )
                return qa_chain.invoke({"query": "Pradhan Mantri Mudra Yojana"})
            
            result = await loop.run_in_executor(executor, run_mudra_qa)
            mudra_rag = {"text": result["result"], "sources": result["source_documents"]}
        else:
            logger.warning("Mudra documents not found; skipping")
            mudra_rag = {"text": "", "sources": []}

        if not isinstance(rag, dict):
            rag = {"text": str(rag), "sources": []}

        rag["text"] = f"{rag.get('text', '')}\n{mudra_rag.get('text', '')}"
        rag["sources"] = rag.get("sources", []) + mudra_rag.get("sources", [])

    total_rag_time = time.perf_counter() - rag_start_time
    logger.info(f"Total RAG processing time: {total_rag_time:.3f}s")    

    # Cache the result
    await cache_manager.set_rag_cache(query, rag, cache_context)
    
    return rag

async def get_dfl_response_async(query, vector_store, state=None, gender=None, business_category=None):
    """Async wrapper for DFL dataset retrieval"""
    logger.info("Querying DFL dataset")
    return await get_rag_response_async(
        query,
        vector_store,
        state=None,
        gender=gender,
        business_category=business_category,
    )

# @lru_cache(maxsize=100)
async def generate_response_async(
    intent: str, 
    rag_response: str, 
    user_info: UserContext, 
    language: str, 
    context: str, 
    query: str, 
    scheme_guid: str = None, 
    stream: bool = False
):
    """Async version of generate_response with proper streaming support"""
    print(f"Generating response for intent: {intent}, language: {language}, query: {query} and rag_response: {rag_response}...")
    # Handle non-streaming cases first (these return strings)
    if intent == "Out_of_Scope":
        if language == "Hindi":
            return "क्षमा करें, मैं केवल सरकारी योजनाओं, डिजिटल या वित्तीय साक्षरता और व्यावसायिक वृद्धि पर मदद कर सकता हूँ।"
        if language == "Hinglish":
            return "Maaf kijiye, main sirf sarkari yojanaon, digital ya financial literacy aur business growth mein madad kar sakta hoon."
        return "Sorry, I can only help with government schemes, digital/financial literacy or business growth."
        
        if stream:
            async def stream_response():
                for char in response:
                    yield char
            return stream_response()
        return response

    if intent == "Gratitude_Intent":
        gratitude_prompt = f"""You are a friendly assistant for Haqdarshak. The user {user_info.name} has thanked you.

        **Instructions**:
        - Respond briefly in the same language ({language}) acknowledging the thanks and offering further help.
        - Use Devanagari script for Hindi and a natural mix of Hindi and English words in Roman script for Hinglish.
        - Keep the message under 30 words.

        **Output**:
        - Only the acknowledgement message in the user's language."""
        
        try:
            if stream:
                async def stream_gratitude():
                    buffer = ""
                    async for chunk in llm.astream([{"role": "user", "content": gratitude_prompt}]):
                        token = chunk.content or ""
                        buffer += token
                        if token:
                            yield token
                return stream_gratitude()
            else:
                response = await llm.ainvoke([{"role": "user", "content": gratitude_prompt}])
                return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate gratitude response: {str(e)}")
            fallback_response = ""
            if language == "Hindi":
                fallback_response = "धन्यवाद! क्या मैं और मदद कर सकता हूँ?"
            elif language == "Hinglish":
                fallback_response = "Thanks! Kya main aur madad kar sakta hoon?"
            else:
                fallback_response = "You're welcome! Let me know if you need anything else."
            
            if stream:
                async def stream_fallback():
                    for char in fallback_response:
                        yield char
                return stream_fallback()
            return fallback_response

    # Build the main prompt for other intents
    word_limit = 150 if intent == "Schemes_Know_Intent" else 100
    tone_prompt = get_system_prompt(language, user_info.name, word_limit)
    base_prompt = f"""You are a helpful assistant for Haqdarshak, supporting small business owners in India with government schemes, digital/financial literacy, and business growth.

    **Input**:
    - Intent: {intent}
    - RAG Response: {rag_response}
    - Current Query: {query}
    - User Name: {user_info.name}
    - State: {user_info.state_name} ({user_info.state_id})
    - Gender: {user_info.gender}
    - Business Name: {user_info.business_name}
    - Business Category: {user_info.business_category}
    - Conversation Context: {context}
    - Language: {language}"""
    if scheme_guid:
        base_prompt += f"\n    - Scheme GUID: {scheme_guid}"

    base_prompt += f"""

    **Language Handling and Tone Instructions**:
    {tone_prompt}

    **Task**:
    Use any user-provided scheme details to pick relevant schemes from retrieved data and personalise the scheme information wherever applicable.
    Prioritise the **Current Query** over the **Conversation Context** when determining the response.
    """

    special_schemes = ["Udyam", "FSSAI", "Shop Act", "GST", "Mudra", "PMEGP", "PMFME", "CMEGP", "Yuva Udyami", "PMSBY", "PMJJBY", "PMJAY (Ayushman Bharat)"]
    link = "https://haqdarshak.com/contact"

    if intent == "Specific_Scheme_Know_Intent":
        intent_prompt = (
            "Share scheme name, purpose, benefits and other fetched relevant details in a structured format from **RAG Response**. "
            "Ask: 'Want details on eligibility or how to apply?' "
            "(English), 'Eligibility ya apply karne ke baare mein jaanna chahte hain?' "
            "(Hinglish), or 'पात्रता या आवेदन करने के बारे में जानना चाहते हैं?' (Hindi)."
        )
        intent_prompt += (
            f" For {', '.join(special_schemes)}, add: 'Haqdarshak can help you apply for this document. "
            f"Please book in the app.' (English), 'Haqdarshak aapko yeh document dilaane mein madad kar sakta hai. "
            f"Kripya app mein book karein.' (Hinglish), or 'हकदर्शक आपको यह दस्तावेज़ "
            f"दिलाने में मदद कर सकता है। कृपया ऐप में बुक करें' (Hindi)."
        )
    elif intent == "Specific_Scheme_Apply_Intent":
        intent_prompt = (
            "Share application process from **RAG Response**."
        )
        intent_prompt += (
            f" For {', '.join(special_schemes)}, add: 'Haqdarshak can help you apply for this document. "
            f"Please book in the app.' (English), 'Haqdarshak aapko yeh document dilaane mein madad kar sakta hai. "
            f"Kripya app mein book karein.' (Hinglish), or 'हकदर्शक आपको यह दस्तावेज़ "
            f"दिलाने में मदद कर सकता है। कृपया ऐप में बुक करें' (Hindi)."
        )
    elif intent == "Specific_Scheme_Eligibility_Intent":
        intent_prompt = (
            "Summarize eligibility rules from **RAG Response** and provide a link "
            f"to check eligibility: https://customer.haqdarshak.com/check-eligibility/{scheme_guid}. "
            "Ask the user to verify their eligibility there."
        )
        intent_prompt += (
            f" For {', '.join(special_schemes)}, add: 'Haqdarshak can help you apply for this document. "
            f"Please book in the app.' (English), 'Haqdarshak aapko yeh document dilaane mein madad kar sakta hai. "
            f"Kripya app mein book karein.' (Hinglish), or 'हकदर्शक आपको यह दस्तावेज़ "
            f"दिलाने में मदद कर सकता है। कृपया ऐप में बुक करें' (Hindi)."
        )
    elif intent == "Schemes_Know_Intent":
        intent_prompt = (
            "List 3-4 schemes from **RAG Response** with a short one-line description for each. "
            "Always include Pradhan Mantri Mudra Yojana as one of the schemes. "
            "Use any user provided scheme details to choose the most relevant schemes. "
            "If no close match is found, still list the top schemes applicable to the user in their state or CSS. "
            "Finally Ask: 'Want more details on any scheme?' (English), 'Kisi yojana ke baare mein aur jaanna chahte hain?' (Hinglish), or "
            "'किसी योजना के बारे में और जानना चाहते हैं?' (Hindi)."
        )
        intent_prompt += (
            f" For {', '.join(special_schemes)}, add: 'Haqdarshak can help you apply for this document. "
            f"Please book in the app.' (English), 'Haqdarshak aapko yeh document dilaane mein madad kar sakta hai. "
            f"Kripya app mein book karein.' (Hinglish), or 'हकदर्शक आपको यह दस्तावेज़ "
            f"दिलाने में मदद कर सकता है। कृपया ऐप में बुक करें' (Hindi). Add this only in the description for the applicable scheme/s, not under the entire list."
        )
    elif intent == "DFL_Intent":
        intent_prompt = (
            "Use the **RAG Response** if available, augmenting with your own knowledge "
            "where relevant. If the RAG Response is empty or not relevant, do not mention that to user and provide a helpful answer "
            "smoothly from your own knowledge in simple language "
            "with helpful examples."
        )
    elif intent == "Contextual_Follow_Up":
        intent_prompt = (
            "Use the Previous Assistant Response and Conversation Context to identify the topic. "
            "If the RAG Response does not match the referenced scheme, indicate a new RAG search "
            "is needed. Provide a relevant follow-up response using the RAG Response, "
            "filtering for schemes where 'applicability' includes state_id or 'scheme type' is "
            "'Centrally Sponsored Scheme' (CSS). If unclear, ask for clarification (e.g., "
            "'Could you specify which scheme?' or 'Kaunsi scheme ke baare mein?' or 'कौन सी योजना के बारे में?')."
        )
    elif intent == "Confirmation_New_RAG":
        intent_prompt = (
            "If the user confirms to initiate a new RAG search, respond with the details of the "
            "scheme they are interested in, refer to conversation context for details."
        )
    else:
        intent_prompt = ""

    output_prompt = """
    **Output**:
       - Return only the final response in the query's language (no intent label or intermediate steps). If a new RAG search is needed for schemes, indicate with: 'I need to fetch more details about [scheme name]. Please confirm if this is the scheme you meant.' (English), 'Mujhe [scheme name] ke baare mein aur jaankari leni hogi. Kya aap isi scheme ki baat kar rahe hain?' (Hinglish), or 'मुझे [scheme name] के बारे में और जानकारी लेनी होगी। क्या आप इसी योजना की बात कर रहे हैं?' (Hindi).
       - If RAG Response is empty or 'No relevant scheme information found,' and the query is a Contextual_Follow_Up referring to a specific scheme, indicate a new RAG search is needed. Otherwise, say: 'I don't have information on this right now.' (English), 'Mujhe iske baare mein abhi jaankari nahi hai.' (Hinglish), or 'मुझे इसके बारे में अभी जानकारी नहीं है।' (Hindi).
       - Do not mention any other scheme when a specific scheme is being talked about.
       - When intent is Schemes_Know, do not mention other schemes from past conversation, only the current relevant ones.
       - No need to mention user profile details in every response, only include where contextually relevant.
       - Scheme answers must come only from scheme data. For DFL answers, use the DFL document supplemented by your own knowledge when possible, but rely on your own knowledge if nothing relevant is found.
    """

    prompt = f"{base_prompt}{intent_prompt}\n{output_prompt}"

    try:
        if stream:
            async def stream_main_response():
                buffer = ""
                try:
                    async for chunk in llm.astream([{"role": "user", "content": prompt}]):
                        token = chunk.content or ""
                        buffer += token
                        if token:
                            yield token
                
                     # Add eligibility link for specific intent after streaming
                    if intent == "Specific_Scheme_Eligibility_Intent" and scheme_guid:
                        screening_link = f"https://customer.haqdarshak.com/check-eligibility/{scheme_guid}"
                        if screening_link not in buffer:
                            link_text = f"\n{screening_link}"
                            for char in link_text:
                             yield char
                            
                except Exception as e:
                    logger.error(f"Failed to stream response: {str(e)}")
                    error_msg = "Sorry, I couldn't process your query."
                    for char in error_msg:
                        yield char
            
            return stream_main_response()
        else:
            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            final_text = response.content.strip()
            
            if intent == "Specific_Scheme_Eligibility_Intent" and scheme_guid:
                screening_link = f"https://customer.haqdarshak.com/check-eligibility/{scheme_guid}"
                if screening_link not in final_text:
                    final_text += f"\n{screening_link}"
            
            return final_text
            
    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}")
        error_response = ""
        if language == "Hindi":
            error_response = "क्षमा करें, मैं आपका प्रश्न संसाधित नहीं कर सका।"
        elif language == "Hinglish":
            error_response = "Sorry, main aapka query process nahi kar saka."
        else:
            error_response = "Sorry, I couldn't process your query."
        
        if stream:
            async def stream_error():
                for char in error_response:
                    yield char
            return stream_error()
        return error_response


def generate_hindi_audio_script(
    original_response: str,
    user_info: UserContext,
    rag_response: str = "",
) -> str:
    """
    Generates a summarized, human-like Hindi script for text-to-speech from the original bot response.
    The script should avoid punctuation marks and focus on natural flow.
    """
    prompt = f"""You are an assistant for Haqdarshak. Your task is to summarize the provided text into a concise, human-like script
    in natural Hindi (Devanagari script) for a text-to-speech system.
    
    **Instructions**:
    - Summarize the core information from the provided 'Final Response' and 'RAG Response'.
    - Ensure the summary flows naturally as if spoken by a human.
    - Translate the summary into clear and simple Hindi (Devanagari script) using simple hindi words.
    - Focus on the main points and keep the summary concise, between 50-100 words, to ensure a smooth audio experience.
    - The response should be purely the Hindi script, with no introductory or concluding remarks.
    - For number ranges like "10%-20%", use "10 se 20" in Hindi.
    - Do NOT use any english words. 
    - Do NOT translate Smileys or emoticons.
    - Always use simpler alternatives wherever the words are in complex hindi e.g. Instead of "vyavyasay" say "business", instead of "vanijya" say "finance"
    - Do NOT include urls and web links. 

    **Final Response**:
    {original_response}

    **RAG Response**:
    {rag_response}

    **Output**:
    """
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        hindi_script = response.content.strip()
        logger.info(f"Generated Hindi audio script: {hindi_script}")
        return hindi_script
    except Exception as e:
        logger.error(f"Failed to generate Hindi audio script: {str(e)}")
        try:
            translation_prompt = f"Translate the following text into simple Hindi (Devanagari script), removing all punctuation and hyphens for a smooth audio output: {original_response}"
            translation_response = llm.invoke([{"role": "user", "content": translation_prompt}])
            hindi_script = translation_response.content.strip()
            logger.warning(f"Falling back to direct translation for Hindi audio script: {hindi_script}")
            return hindi_script
        except Exception as inner_e:
            logger.error(f"Failed to fall back to direct translation: {str(inner_e)}")
            return "ऑडियो स्क्रिप्ट उत्पन्न करने में त्रुटि हुई है।"

# Background task functions
async def save_conversation_background(
    session_id: str, 
    mobile_number: str, 
    query: str, 
    response: str,
    hindi_script: str = ""
):
    """Save conversation in background without blocking response"""
    try:
        interaction_id = generate_interaction_id(query, datetime.utcnow())
        messages_to_save = [
            {"role": "user", "content": query, "timestamp": datetime.utcnow(), "interaction_id": interaction_id},
            {"role": "assistant", "content": response, "timestamp": datetime.utcnow(), "interaction_id": interaction_id, "audio_script": hindi_script},
        ]
        
        await data_manager.save_conversation_async(session_id, mobile_number, messages_to_save)
        logger.info(f"Background save completed for session {session_id} (Interaction ID: {interaction_id})")
    except Exception as e:
        logger.error(f"Background save failed for session {session_id}: {str(e)}")

async def generate_audio_script_background(response: str, user_info: UserContext, rag_response: str = "") -> str:
    """Generate hindi audio script in background"""
    try:
        loop = asyncio.get_event_loop()
        hindi_script = await loop.run_in_executor(
            executor, 
            generate_hindi_audio_script,
            response,
            user_info,
            rag_response
        )
        return hindi_script
    except Exception as e:
        logger.error(f"Background audio script generation failed: {str(e)}")
        return "ऑडियो स्क्रिप्ट उत्पन्न करने में त्रुटि हुई है।"

async def get_popular_scheme_response_fast(query: str, intent: str) -> Optional[dict]:
    """Ultra-fast response for popular schemes (1-2 seconds)"""
    if intent != "Specific_Scheme_Know_Intent":
        return None
    
    # Step 1: Quick GUID lookup (< 100ms)
    guid = find_scheme_guid_by_query(query)
    if not guid:
        logger.info(f"No popular scheme GUID found for query: '{query}' - using regular path")
        return None
    
    logger.info(f"Found popular scheme GUID: {guid} for query: '{query}' - using fast path")
    
    # Step 2: Fast XLSX fetch (< 200ms)  
    if not FAST_XLSX_AVAILABLE:
        logger.warning("XLSX not available - falling back to regular path")
        return None
    
    try:
        loop = asyncio.get_event_loop()
        
        # Single fast operation - no parallel tasks overhead
        docs = await loop.run_in_executor(
            executor, 
            fetch_scheme_docs_by_guid, 
            guid, 
            None, 
            True
        )
        
        if not docs:
            logger.warning(f"No docs found for popular scheme GUID: {guid}")
            return None
        
        # Step 3: Fast QA chain (< 1000ms)
        def run_fast_qa():
            retriever = DocumentListRetriever(docs)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff", 
                retriever=retriever,
                return_source_documents=True,
            )
            return qa_chain.invoke({"query": query})
        
        result = await loop.run_in_executor(executor, run_fast_qa)
        rag_response = {"text": result["result"], "sources": result["source_documents"]}
        print(f"Fast path RAG kittu response: {rag_response['text']}...")  # Log first 100 chars
        logger.info(f"Fast path completed for popular scheme: {guid}")
        return rag_response
        
    except Exception as e:
        logger.error(f"Fast path failed for GUID {guid}: {str(e)} - falling back")
        return None

def create_audio_task_background(response: str, user_info: UserContext, rag_response: str = ""):
    """Create a background audio task that returns a coroutine"""
    async def audio_task(final_text: str = None) -> str:
        text_to_use = final_text or response
        return await generate_audio_script_background(text_to_use, user_info, rag_response)
    return audio_task

# Performance tracking context manager
class PerformanceTracker:
    def __init__(self):
        self.timings = {}
        self.start_time = time.perf_counter()
    
    def start_timer(self, operation_name: str):
        self.timings[f"{operation_name}_start"] = time.perf_counter()
    
    def end_timer(self, operation_name: str):
        start_key = f"{operation_name}_start"
        if start_key in self.timings:
            elapsed = time.perf_counter() - self.timings[start_key]
            self.timings[operation_name] = elapsed
            del self.timings[start_key]
    
    def log_summary(self):
        total = time.perf_counter() - self.start_time
        operations = {k: v for k, v in self.timings.items() if not k.endswith('_start')}
        summary = "\n".join(f"{k}: {v:.3f}s" for k, v in operations.items())
        summary += f"\nTotal: {total:.3f}s"
        logger.info(f"Performance summary:\n{summary}")

# MAIN OPTIMIZED ASYNC FUNCTION
async def process_query_optimized(
    query: str,
    scheme_vector_store,
    dfl_vector_store,
    session_id: str,
    mobile_number: str,
    session_data: SessionData,
    user_language: str = None,
    stream: bool = False
) -> Tuple[Any, callable]:
    """
    Optimized async version of process_query with parallel processing and caching
    Expected improvement: 12.81s -> 3-4s (70% improvement)
    """
    tracker = PerformanceTracker()
    logger.info(f"Starting optimized query processing for: {query}")

    # Step 1: Get user context (fast, local operation)
    tracker.start_timer("user_context")
    user_info = get_user_context(session_data)
    tracker.end_timer("user_context")
    
    if not user_info:
        tracker.log_summary()
        return "Error: User not logged in.", None

    # Step 2: Language detection (fast, local operation)
    tracker.start_timer("language_detection")
    query_language = user_language if query.lower() == "welcome" and user_language else detect_language(query)
    tracker.end_timer("language_detection")
    
    logger.info(f"Using query language: {query_language}")

    # Step 3: Start background conversation fetch early
    tracker.start_timer("fetch_conversations")
    conversations_task = asyncio.create_task(data_manager.get_conversations_async(mobile_number))

    # Step 4: Handle welcome query (early return)
    if query.lower() == "welcome":
        conversations = await conversations_task
        tracker.end_timer("fetch_conversations")
        user_type = "returning" if conversations else "new"
        
        if user_type == "new":
            response = welcome_user(user_info.state_name, user_info.name, query_language)
            
            # Create background task for saving welcome message
            async def save_welcome():
                try:
                    interaction_id = generate_interaction_id(response, datetime.utcnow())
                    messages = [{"role": "assistant", "content": response, "timestamp": datetime.utcnow(), "interaction_id": interaction_id}]
                    await data_manager.save_conversation_async(session_id, mobile_number, messages)
                    logger.info(f"Saved welcome message for new user in session {session_id}")
                except Exception as e:
                    logger.error(f"Failed to save welcome message: {str(e)}")
            
            # Start background save but don't wait
            asyncio.create_task(save_welcome())
            
            audio_task = create_audio_task_background(response, user_info)
            tracker.log_summary()
            
            if stream:
                async def gen():
                    for ch in response:
                        yield ch
                return gen(), audio_task
            return response, audio_task
        else:
            tracker.log_summary()
            return None, None

    # Step 5: Build conversation history (fast, local)
    conversation_history = build_conversation_history(session_data.messages)
    
    # Step 6: Start intent classification early (parallel with conversation fetch)
    tracker.start_timer("intent_classification")
    intent_task = asyncio.create_task(classify_intent_async(query, conversation_history))
    
    # Step 7: Wait for conversations and get user type
    conversations = await conversations_task
    tracker.end_timer("fetch_conversations")
    user_type = "returning" if conversations else "new"

    # Step 8: Get intent result
    intent = await intent_task
    tracker.end_timer("intent_classification")
    logger.info(f"Classified intent: {intent}")

    # Step 9: Determine context and prepare for RAG
    follow_up_intents = {
        "Contextual_Follow_Up",
        "Specific_Scheme_Eligibility_Intent", 
        "Specific_Scheme_Apply_Intent",
        "Confirmation_New_RAG",
    }
    follow_up = intent in follow_up_intents
    
    # Get recent conversation context only for follow-ups
    recent_query = None
    recent_response = None
    if follow_up and session_data.messages:
        for msg in reversed(session_data.messages):
            if msg["role"] == "assistant" and "Welcome" not in msg["content"]:
                recent_response = msg["content"]
                msg_index = session_data.messages.index(msg)
                if msg_index > 0 and session_data.messages[msg_index - 1]["role"] == "user":
                    recent_query = session_data.messages[msg_index - 1]["content"]
                break
    
    context_pair = f"User: {recent_query}\nAssistant: {recent_response}" if follow_up and recent_query and recent_response else ""
    


    scheme_intents = {"Specific_Scheme_Know_Intent", "Specific_Scheme_Apply_Intent", "Specific_Scheme_Eligibility_Intent", "Schemes_Know_Intent", "Contextual_Follow_Up", "Confirmation_New_RAG"}
    dfl_intents = {"DFL_Intent", "Non_Scheme_Know_Intent"}

    logger.info(f"Processing query: '{query}' with intent: {intent}")

    rag_response = None
    if intent in scheme_intents:
        tracker.start_timer("rag_retrieval")
        print(f"Retrieving RAG response for kittu intent: {intent}")
        # TRY FAST PATH FIRST for popular schemes (1-2 seconds)
        if intent == "Specific_Scheme_Know_Intent":
            rag_response = await get_popular_scheme_response_fast(query, intent)
        
        # FALLBACK to full pipeline if fast path didn't work
        if not rag_response:
            logger.info("Using full scheme response pipeline")
            include_mudra = intent == "Schemes_Know_Intent"
            
            rag_response = await get_scheme_response_async(
                query=query,
                vector_store=scheme_vector_store,
                state=user_info.state_id,
                gender=user_info.gender,
                business_category=user_info.business_category,
                include_mudra=include_mudra,
                intent=intent,
                use_xlsx=True
            )
        
        tracker.end_timer("rag_retrieval")

    elif intent in dfl_intents:
        tracker.start_timer("dfl_retrieval")
        
        rag_response = await get_dfl_response_async(
            query=query,
            vector_store=dfl_vector_store,
            state=user_info.state_id,
            gender=user_info.gender,
            business_category=user_info.business_category
        )
        
        tracker.end_timer("dfl_retrieval")

    # Step 11: Generate response (this is where the fix is important)
    tracker.start_timer("generate_response")
    rag_text = rag_response.get("text") if isinstance(rag_response, dict) else rag_response
    if intent == "DFL_Intent" and (rag_text is None or "No relevant" in rag_text):
        rag_text = ""
    scheme_guid = None
    if isinstance(rag_response, dict) and intent == "Specific_Scheme_Eligibility_Intent":
        scheme_guid = extract_scheme_guid(rag_response.get("sources", []))
    
    response_result = await generate_response_async(
        intent,
        rag_text or "",
        user_info,
        query_language,
        context_pair,
        query,
        scheme_guid=scheme_guid,
        stream=stream,
    )
    tracker.end_timer("generate_response")

    # Step 12: Handle streaming vs non-streaming audio tasks
    if stream:
        # For streaming, response_result is an async generator
        def create_streaming_audio_task():
            async def streaming_audio_task(final_text: str) -> str:
                try:
                    hindi_script = await generate_audio_script_background(final_text, user_info, rag_text or "")
                    # Fire and forget the save task
                    asyncio.create_task(
                        save_conversation_background(session_id, mobile_number, query, final_text, hindi_script)
                    )
                    return hindi_script
                except Exception as e:
                    logger.error(f"Audio script generation failed: {e}")
                    return "ऑडियो स्क्रिप्ट उत्पन्न करने में त्रुटि हुई है।"
            
            return streaming_audio_task
    
        audio_task = create_streaming_audio_task()
    else:
        # For non-streaming, response_result is a string
        response_text = response_result
        
        async def background_audio_task(final_text: str = None) -> str:
            text_to_use = final_text or response_text
            
            try:
                hindi_script = await generate_audio_script_background(text_to_use, user_info, rag_text or "")
            except Exception as e:
                logger.error(f"Audio script generation failed: {e}")
                hindi_script = "ऑडियो स्क्रिप्ट उत्पन्न करने में त्रुटि हुई है।"
            
            # Start save task in fire-and-forget mode
            asyncio.create_task(
                save_conversation_background(session_id, mobile_number, query, text_to_use, hindi_script)
            )
            
            return hindi_script
        
        audio_task = background_audio_task

    tracker.log_summary()
    logger.info(f"Query processing completed for: {query}")

    return response_result, audio_task


def run_qa_chain_fast(docs, query):
    """Fast QA chain execution"""
    retriever = DocumentListRetriever(docs)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain.invoke({"query": query})
    return {"text": result["result"], "sources": result["source_documents"]}