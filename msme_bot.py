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
)
from utils import extract_scheme_guid
from data import DataManager
import re
import os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, AsyncGenerator, Tuple, List
import redis.asyncio as redis
import aiohttp
from collections import deque
import threading
from asyncio import Semaphore


# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
logger = logging.getLogger(__name__)
logging.getLogger("pymongo").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=6)

# Enhanced Cache Manager with connection pooling
class CacheManager:
    def __init__(self):
        try:
            self.redis_client = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379"),
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                socket_connect_timeout=1,
                socket_timeout=1
            )
        except:
            self.redis_client = None
            logger.warning("Redis not available, using memory cache only")
        
        self.memory_cache = {}
        self.max_memory_cache_size = 2000
        self.cache_stats = {"hits": 0, "misses": 0}
    
    def _cleanup_memory_cache(self):
        """Keep memory cache size under control"""
        if len(self.memory_cache) > self.max_memory_cache_size:
            # Remove oldest quarter of entries (LRU-like)
            items = list(self.memory_cache.items())
            items_to_keep = items[len(items)//4:]
            self.memory_cache = dict(items_to_keep)
    
    @lru_cache(maxsize=1000)
    def _generate_cache_key(self, query: str, context: str = "") -> str:
        """Generate consistent cache keys"""
        key_data = f"{query}:{context}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]  # Shorter keys
    
    async def get_intent_cache(self, query: str, conversation_history: str = "") -> Optional[str]:
        """Get cached intent with memory fallback"""
        cache_key = self._generate_cache_key(query, conversation_history)
        
        # Check memory cache first (faster)
        memory_result = self.memory_cache.get(f"intent:{cache_key}")
        if memory_result:
            self.cache_stats["hits"] += 1
            return memory_result
        
        # Try Redis with timeout
        if self.redis_client:
            try:
                cached = await asyncio.wait_for(
                    self.redis_client.get(f"intent:{cache_key}"), 
                    timeout=0.1
                )
                if cached:
                    # Update memory cache
                    self.memory_cache[f"intent:{cache_key}"] = cached
                    self.cache_stats["hits"] += 1
                    return cached
            except (asyncio.TimeoutError, Exception):
                pass
        
        self.cache_stats["misses"] += 1
        return None
    
    async def set_intent_cache(self, query: str, intent: str, conversation_history: str = "", ttl: int = 3600):
        """Set intent cache with Redis and memory"""
        cache_key = self._generate_cache_key(query, conversation_history)
        
        # Set in memory cache first (immediate)
        self.memory_cache[f"intent:{cache_key}"] = intent
        self._cleanup_memory_cache()
        
        # Set in Redis asynchronously (fire and forget)
        if self.redis_client:
            asyncio.create_task(self._set_redis_async(f"intent:{cache_key}", intent, ttl))
    
    async def _set_redis_async(self, key: str, value: str, ttl: int):
        """Helper to set Redis values asynchronously"""
        try:
            await asyncio.wait_for(
                self.redis_client.setex(key, ttl, value),
                timeout=0.1
            )
        except (asyncio.TimeoutError, Exception):
            pass
    
    async def get_rag_cache(self, query: str, user_context: dict = None) -> Optional[dict]:
        """Get cached RAG response"""
        context_str = json.dumps(user_context or {}, sort_keys=True)
        cache_key = self._generate_cache_key(query, context_str)
        
        # Memory first
        memory_result = self.memory_cache.get(f"rag:{cache_key}")
        if memory_result:
            self.cache_stats["hits"] += 1
            return memory_result
        
        # Redis with timeout
        if self.redis_client:
            try:
                cached = await asyncio.wait_for(
                    self.redis_client.get(f"rag:{cache_key}"),
                    timeout=0.15
                )
                if cached:
                    result = json.loads(cached)
                    # Cache in memory if small
                    if len(str(result)) < 10000:
                        self.memory_cache[f"rag:{cache_key}"] = result
                    self.cache_stats["hits"] += 1
                    return result
            except (asyncio.TimeoutError, Exception):
                pass
        
        self.cache_stats["misses"] += 1
        return None
    
    async def set_rag_cache(self, query: str, data: dict, user_context: dict = None, ttl: int = 1800):
        """Set RAG cache"""
        context_str = json.dumps(user_context or {}, sort_keys=True)
        cache_key = self._generate_cache_key(query, context_str)
        
        # Store small responses in memory cache
        if len(str(data)) < 10000:
            self.memory_cache[f"rag:{cache_key}"] = data
            self._cleanup_memory_cache()
        
        # Store in Redis asynchronously
        if self.redis_client:
            asyncio.create_task(self._set_redis_rag_async(f"rag:{cache_key}", data, ttl))
    
    async def _set_redis_rag_async(self, key: str, data: dict, ttl: int):
        """Helper to set RAG data in Redis"""
        try:
            await asyncio.wait_for(
                self.redis_client.setex(key, ttl, json.dumps(data)),
                timeout=0.2
            )
        except (asyncio.TimeoutError, Exception):
            pass

# Initialize cache manager
cache_manager = CacheManager()

# Optimized DataManager with connection pooling
class AsyncDataManager(DataManager):
    def __init__(self):
        super().__init__()
        self._executor_pool = ThreadPoolExecutor(max_workers=3)
    
    async def get_conversations_async(self, mobile_number: str):
        """Async version of get_conversations with caching"""
        # Simple in-memory cache for recent conversations
        cache_key = f"conv_{mobile_number}"
        
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor_pool, self.get_conversations, mobile_number),
                timeout=1.0
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Conversation fetch timeout for {mobile_number}")
            return []
    
    async def save_conversation_async(self, session_id: str, mobile_number: str, messages: list):
        """Async version of save_conversation"""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                self._executor_pool, 
                self.save_conversation, 
                session_id, 
                mobile_number, 
                messages
            )
        except Exception as e:
            logger.error(f"Save conversation failed: {e}")

# Initialize async data manager
data_manager = AsyncDataManager()

# LLM Connection Pool
class LLMPool:
    def __init__(self, pool_size=3):
        self.pool_size = pool_size
        self.available_llms = deque()
        self.available_intent_llms = deque()
        self._initialize_pools()
    
    def _initialize_pools(self):
        """Initialize LLM pools"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Main LLM pool
        for _ in range(self.pool_size):
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Faster model
                api_key=api_key,
                base_url="https://api.openai.com/v1",
                temperature=0,
                max_tokens=500,  # Limit response length
                timeout=10  # Faster timeout
            )
            self.available_llms.append(llm)
        
        # Intent LLM pool (smaller, faster)
        for _ in range(self.pool_size):
            intent_llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=api_key,
                base_url="https://api.openai.com/v1",
                temperature=0,
                max_tokens=50,  # Very short responses
                timeout=5  # Very fast timeout
            )
            self.available_intent_llms.append(intent_llm)
    
    async def get_llm(self):
        """Get an available LLM from the pool"""
        while not self.available_llms:
            await asyncio.sleep(0.01)  # Wait briefly
        return self.available_llms.popleft()
    
    def return_llm(self, llm):
        """Return LLM to the pool"""
        self.available_llms.append(llm)
    
    async def get_intent_llm(self):
        """Get an available intent LLM from the pool"""
        while not self.available_intent_llms:
            await asyncio.sleep(0.01)
        return self.available_intent_llms.popleft()
    
    def return_intent_llm(self, llm):
        """Return intent LLM to the pool"""
        self.available_intent_llms.append(llm)

# Initialize LLM pool
llm_pool = LLMPool(pool_size=4)

# Audio-specific thread pool with proper sizing
audio_executor = ThreadPoolExecutor(
    max_workers=2,  # Reduced to prevent TTS service overload
    thread_name_prefix="audio_synthesis"
)

# Audio generation semaphore to limit concurrent requests
audio_semaphore = Semaphore(2)  # Max 2 concurrent audio generations

# Audio request queue and tracking
class AudioRequestTracker:
    def __init__(self):
        self.active_requests = {}
        self.request_queue = deque()
        self.lock = asyncio.Lock()
        self.stats = {"generated": 0, "failed": 0, "cancelled": 0}
    
    async def add_request(self, session_id: str, request_id: str):
        async with self.lock:
            self.active_requests[session_id] = {
                "request_id": request_id,
                "start_time": time.time(),
                "status": "pending"
            }
    
    async def complete_request(self, session_id: str, success: bool):
        async with self.lock:
            if session_id in self.active_requests:
                if success:
                    self.stats["generated"] += 1
                else:
                    self.stats["failed"] += 1
                del self.active_requests[session_id]
    
    async def cancel_old_requests(self, session_id: str):
        """Cancel any pending audio requests for this session"""
        async with self.lock:
            if session_id in self.active_requests:
                self.active_requests[session_id]["status"] = "cancelled"
                self.stats["cancelled"] += 1
                logger.info(f"🔊 Cancelled old audio request for session {session_id}")

# Global audio tracker
audio_tracker = AudioRequestTracker()

# Initialize cached resources with lazy loading
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

# Lazy load vector stores
scheme_vector_store = None
dfl_vector_store = None

def get_scheme_vector_store():
    global scheme_vector_store
    if scheme_vector_store is None:
        scheme_vector_store = init_vector_store()
    return scheme_vector_store

def get_dfl_vector_store():
    global dfl_vector_store
    if dfl_vector_store is None:
        dfl_vector_store = init_dfl_vector_store()
    return dfl_vector_store

# Keep your existing dataclasses
@dataclass
class UserContext:
    name: str
    state_id: str
    state_name: str
    business_name: str
    business_category: str
    gender: str

class SessionData:
    """Enhanced session container with caching."""
    def __init__(self, user=None):
        self.user = user
        self.messages = []
        self.rag_cache = {}  # Session-level cache
        self.dfl_rag_cache = {}
        self.last_intent = None
        self.context_cache = {}

# Keep your existing helper functions (optimized where possible)
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

# Optimized language detection
@lru_cache(maxsize=500)
def detect_language(query):
    """Cached language detection"""
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    if devanagari_pattern.search(query):
        return "Hindi"
    
    # Simplified hindi word detection
    hindi_keywords = {"kya", "kaise", "mein", "hai", "kaun", "batao", "yojana", "madad"}
    query_words = set(query.lower().split())
    
    if hindi_keywords.intersection(query_words):
        return "Hinglish"
    
    return "English"

def get_system_prompt(language, user_name="User", word_limit=200):
    """Optimized system prompt generation"""
    return f"""Language: {language}. Response limit: {word_limit} words.
    
For {language} queries:
- Hindi: Use Devanagari script, simple words
- Hinglish: Mix Hindi/English in Roman script
- English: Simple, clear English

Guidelines:
- Government schemes/digital literacy/business growth only
- Structured responses with bullets/headings
- Start with greeting: Hi/Namaste {user_name}!
- Use scheme names exactly as provided
- Never mention technical terms like RAG/LLM"""

def build_conversation_history(messages):
    """Optimized conversation history building"""
    if not messages:
        return ""
    
    # Get last 5 non-welcome messages
    relevant_msgs = []
    for msg in reversed(messages[-10:]):
        if msg["role"] == "assistant" and "Welcome" in msg.get("content", ""):
            continue
        relevant_msgs.append((msg["role"], msg["content"][:200]))  # Truncate long messages
        if len(relevant_msgs) >= 5:
            break
    
    return "\n".join(f"{role}: {content}" for role, content in reversed(relevant_msgs))

# def welcome_user(state_name, user_name, query_language,user_type):
#     """Quick welcome message generation"""
#     if query_language == "Hindi":
#         return f"नमस्ते {user_name}! हकदर्शक MSME चैटबॉट में स्वागत है। मैं {state_name} की योजनाओं में मदद करूँगा। आज कैसे सहायता करूँ?"
#     return f"Hi {user_name}! Welcome to Haqdarshak MSME Chatbot! I'll help with schemes from {state_name}. How can I assist you today?"

def welcome_user(state_name, user_name, query_language, user_type):
    """Optimized quick welcome message generation based on user type"""

    # Scheme type based on user_type
    if query_language == "Hindi":
        scheme_type = "MSME योजनाओं" if user_type == 1 else "सामान्य योजनाओं"
        return f"नमस्ते {user_name}! हकदर्शक चैटबॉट में आपका स्वागत है। मैं {state_name} की {scheme_type} में मदद करूँगा। आज कैसे सहायता करूँ?"
    else:
        scheme_type = "MSME schemes" if user_type == 1 else "general schemes"
        return f"Hi {user_name}! Welcome to Haqdarshak Chatbot! I'll help you with {scheme_type} from {state_name}. How can I assist you today?"

def generate_interaction_id(query, timestamp):
    return hashlib.md5(f"{query[:100]}_{timestamp.strftime('%Y%m%d%H%M%S')}".encode()).hexdigest()[:12]

# OPTIMIZED ASYNC FUNCTIONS

async def classify_intent_async(query: str, conversation_history: str = "") -> str:
    """Ultra-fast intent classification with aggressive caching"""
    # Check cache first
    cached_intent = await cache_manager.get_intent_cache(query, conversation_history)
    if cached_intent:
        logger.debug(f"Intent cache hit: {cached_intent}")
        return cached_intent

    # PRE-PROCESSING: Handle common abbreviations and short queries
    query_lower = query.lower().strip()
    
    # Common scheme abbreviations - expand them for better classification
    scheme_expansions = {
        "pmmy": "pradhan mantri mudra yojana",
        "mudra": "pradhan mantri mudra yojana", 
        "pmegp": "prime minister employment generation programme",
        "cgtmse": "credit guarantee trust micro small enterprises",
        "msme": "micro small medium enterprises schemes",
        "startup": "startup india scheme",
        "pmkvy": "pradhan mantri kaushal vikas yojana",
        "pmfme": "pradhan mantri formalization micro food processing",
        "stand up": "stand up india scheme"
    }
    
    # Expand abbreviations
    expanded_query = query
    for abbr, full_form in scheme_expansions.items():
        if abbr in query_lower:
            expanded_query = f"{query} {full_form}"
            logger.info(f"🔍 Expanded query: '{query}' → '{expanded_query}'")
            break
    
    # For very short queries (< 4 chars), assume they're scheme-related
    if len(query_lower) < 4 and query_lower.isalpha():
        logger.info(f"🔍 Short query detected: '{query}', defaulting to Specific_Scheme_Know_Intent")
        intent = "Specific_Scheme_Know_Intent"
        await cache_manager.set_intent_cache(query, intent, conversation_history)
        return intent
    
    # Simplified prompt for faster processing
    prompt = f"""Classify intent for: "{query}"
    

Return ONE label:
- Schemes_Know_Intent - General scheme queries
- DFL_Intent - Digital/financial literacy  
- Specific_Scheme_Know_Intent - Specific scheme mentioned (mudra, pmegp, startup, pmmy, msme, etc.)
- Specific_Scheme_Apply_Intent - Apply for scheme
- Specific_Scheme_Eligibility_Intent - Eligibility questions
- Out_of_Scope - ONLY for clearly irrelevant topics (weather, sports, movies, personal chat)
- Contextual_Follow_Up - Follow-up query
- Confirmation_New_RAG - Confirmation request
- Gratitude_Intent - Thanks/acknowledgement

IMPORTANT: 
- Treat abbreviations like "pmmy", "mudra", "pmegp" as Specific_Scheme_Know_Intent
- When in doubt between schemes vs out_of_scope, choose schemes
- Only use Out_of_Scope for clearly non-business topics

Examples:
- "pmmy" → Specific_Scheme_Know_Intent
- "mudra" → Specific_Scheme_Know_Intent
- "schemes" → Schemes_Know_Intent
- "weather" → Out_of_Scope

Label:"""

    
    llm = await llm_pool.get_intent_llm()
    try:
        response = await asyncio.wait_for(
            llm.ainvoke([{"role": "user", "content": prompt}]),
            timeout=3.0
        )
        
        full_response = response.content.strip()
        intent = full_response.split()[0]  # Take first word only
        
        logger.info(f"🔍 Query: '{query}' → Intent: '{intent}'")
        
        # Validate intent - if invalid, default to scheme intent for business-like queries
        valid_intents = {
            "Schemes_Know_Intent", "DFL_Intent", "Specific_Scheme_Know_Intent",
            "Specific_Scheme_Apply_Intent", "Specific_Scheme_Eligibility_Intent", 
            "Out_of_Scope", "Contextual_Follow_Up", "Confirmation_New_RAG", "Gratitude_Intent"
        }
        
        if intent not in valid_intents:
            logger.warning(f"🔍 Invalid intent '{intent}', defaulting to Specific_Scheme_Know_Intent")
            intent = "Specific_Scheme_Know_Intent"
        
        # Cache the result
        await cache_manager.set_intent_cache(query, intent, conversation_history)
        
        return intent
    except (asyncio.TimeoutError, Exception) as e:
        logger.error(f"Intent classification failed: {str(e)}")
        # For short queries that might be scheme abbreviations, default to scheme intent
        if len(query.strip()) < 10:
            return "Specific_Scheme_Know_Intent"
        return "Schemes_Know_Intent"  # Default fallback
    finally:
        llm_pool.return_intent_llm(llm)

async def get_rag_response_parallel(query, vector_store, state="ALL_STATES", gender=None, business_category=None, k=3):
    """Parallel RAG response with reduced retrieval count"""
    try:
        # Build query with context
        context_parts = []
        if state != "ALL_STATES":
            context_parts.append(f"state: {state}")
        if gender:
            context_parts.append(f"gender: {gender}")
        if business_category:
            context_parts.append(f"category: {business_category}")

        full_query = f"{query}. {' '.join(context_parts)}" if context_parts else query
        
        logger.debug(f"RAG query: {full_query}")
        
        def run_retrieval():
            retriever = PineconeRecordRetriever(
                index=vector_store, 
                state=state, 
                gender=gender, 
                k=k  # Reduced from 5 to 3 for speed
            )
            
            # Get LLM from pool
            llm = llm_pool.available_llms.popleft() if llm_pool.available_llms else None
            if not llm:
                # Fallback - create temporary LLM
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=300, timeout=8)
            
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                )
                result = qa_chain.invoke({"query": full_query})
                return result
            finally:
                if llm in [l for l in llm_pool.available_llms]:
                    llm_pool.return_llm(llm)
        
        # Run with timeout
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(executor, run_retrieval),
            timeout=8.0
        )
        
        response = result["result"]
        sources = result["source_documents"]
        
        logger.info(f"RAG response generated: {len(response)} chars")
        return {"text": response, "sources": sources}
        
    except asyncio.TimeoutError:
        logger.error("RAG retrieval timeout")
        return {"text": "Information temporarily unavailable.", "sources": []}
    except Exception as e:
        logger.error(f"RAG retrieval failed: {str(e)}")
        return {"text": "Error retrieving information.", "sources": []}

async def get_scheme_response_optimized(
    query,
    vector_store,
    state="ALL_STATES",
    gender=None,
    business_category=None,
    include_mudra=False,
    intent=None,
):
    """Highly optimized scheme response with parallel processing"""
    logger.info("Querying scheme dataset")

    # Check cache first
    cache_context = {
        "state": state,
        "gender": gender,
        "business_category": business_category,
        "include_mudra": include_mudra,
        "intent": intent
    }
    
    cached_response = await cache_manager.get_rag_cache(query, cache_context)
    if cached_response:
        logger.info("Scheme response cache hit")
        return cached_response

    # Parallel processing for GUID lookup and regular RAG
    tasks = []
    
    # Task 1: Main RAG response
    # main_rag_task = asyncio.create_task(
    #     get_rag_response_parallel(
    #         query, vector_store, state=state, 
    #         gender=gender, business_category=business_category, k=3
    #     )
    # )
    # tasks.append(("main_rag", main_rag_task))
    
    # # Task 2: GUID lookup (if specific scheme)
    # guid_task = None
    # if intent == "Specific_Scheme_Know_Intent":
    #     guid_task = asyncio.create_task(
    #         asyncio.get_event_loop().run_in_executor(
    #             executor, find_scheme_guid_by_query, query
    #         )
    #     )
    #     tasks.append(("guid", guid_task))

    main_rag_task = asyncio.create_task(
        get_rag_response_parallel(
            query,
            vector_store,
            state=state,
            gender=gender,
            business_category=business_category,
            k=3
        )
    )
    tasks.append(("main_rag", main_rag_task))


    # Task 2: GUID lookup (if specific scheme)
    guid_task = None

    async def run_guid_lookup():
        return await asyncio.get_event_loop().run_in_executor(
            executor, find_scheme_guid_by_query, query
        )

    if intent == "Specific_Scheme_Know_Intent":
        guid_task = asyncio.create_task(run_guid_lookup())
        tasks.append(("guid", guid_task))

    
    # Task 3: Mudra data (if needed)
    mudra_task = None
    if include_mudra:
        async def run_mudra_task():
            return await asyncio.get_event_loop().run_in_executor(
            executor, find_scheme_guid_by_query, "pradhan mantri mudra yojana"
            )

        mudra_task = asyncio.create_task(run_mudra_task())
        tasks.append(("mudra_guid", mudra_task))
    
    # Wait for all tasks with timeout
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
            timeout=6.0
        )
        
        # Process results
        task_results = {}
        for i, (name, _) in enumerate(tasks):
            if i < len(results) and not isinstance(results[i], Exception):
                task_results[name] = results[i]
        
        # Get main RAG result
        rag = task_results.get("main_rag", {"text": "Error retrieving information.", "sources": []})
        
        # Handle GUID-based retrieval if available
        if "guid" in task_results and task_results["guid"]:
            guid = task_results["guid"]
            logger.info(f"Using GUID-based retrieval: {guid}")
            
            # Quick GUID-based document fetch
            try:
                docs = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        executor, fetch_scheme_docs_by_guid, guid, vector_store
                    ),
                    timeout=2.0
                )
                
                if docs:
                    # Override with GUID-specific data
                    def run_guid_qa():
                        retriever = DocumentListRetriever(docs)
                        llm = llm_pool.available_llms.popleft() if llm_pool.available_llms else None
                        if llm:
                            try:
                                qa_chain = RetrievalQA.from_chain_type(
                                    llm=llm, chain_type="stuff", retriever=retriever,
                                    return_source_documents=True,
                                )
                                return qa_chain.invoke({"query": query})
                            finally:
                                llm_pool.return_llm(llm)
                        return None
                    
                    guid_result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(executor, run_guid_qa),
                        timeout=3.0
                    )
                    
                    if guid_result:
                        rag = {"text": guid_result["result"], "sources": guid_result["source_documents"]}
            except asyncio.TimeoutError:
                logger.warning("GUID-based retrieval timeout, using main RAG")
        
        # Add Mudra information if requested and available
        if include_mudra and "mudra_guid" in task_results:
            mudra_guid = task_results["mudra_guid"] or "SH0008BK"
            
            try:
                mudra_docs = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        executor, fetch_scheme_docs_by_guid, mudra_guid, vector_store
                    ),
                    timeout=1.5
                )
                
                if mudra_docs:
                    def run_mudra_qa():
                        retriever = DocumentListRetriever(mudra_docs)
                        llm = llm_pool.available_llms.popleft() if llm_pool.available_llms else None
                        if llm:
                            try:
                                qa_chain = RetrievalQA.from_chain_type(
                                    llm=llm, chain_type="stuff", retriever=retriever,
                                    return_source_documents=True,
                                )
                                return qa_chain.invoke({"query": "Pradhan Mantri Mudra Yojana"})
                            finally:
                                llm_pool.return_llm(llm)
                        return {"result": "", "source_documents": []}
                    
                    mudra_result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(executor, run_mudra_qa),
                        timeout=2.0
                    )
                    
                    if mudra_result and mudra_result["result"]:
                        if not isinstance(rag, dict):
                            rag = {"text": str(rag), "sources": []}
                        
                        rag["text"] = f"{rag.get('text', '')}\n\n{mudra_result['result']}"
                        rag["sources"] = rag.get("sources", []) + mudra_result.get("source_documents", [])
            except asyncio.TimeoutError:
                logger.warning("Mudra retrieval timeout, skipping")
        
        # Cache the result
        await cache_manager.set_rag_cache(query, rag, cache_context)
        return rag
        
    except asyncio.TimeoutError:
        logger.error("Scheme retrieval timeout")
        return {"text": "Information temporarily unavailable.", "sources": []}

async def get_dfl_response_async(query, vector_store, state=None, gender=None, business_category=None):
    """Optimized DFL response"""
    logger.info("Querying DFL dataset")
    return await get_rag_response_parallel(
        query, vector_store, state=None, 
        gender=gender, business_category=business_category, k=3
    )

async def generate_response_optimizeds(
    intent: str, 
    rag_response: str, 
    user_info: UserContext, 
    language: str, 
    context: str, 
    query: str, 
    scheme_guid: str = None, 
    stream: bool = False
) -> str:
    """Optimized response generation"""
    # Quick responses for simple cases
    if intent == "Out_of_Scope":
        quick_responses = {
            "Hindi": "क्षमा करें, मैं केवल सरकारी योजनाओं और व्यावसायिक मदद कर सकता हूँ।",
            "Hinglish": "Sorry, main sirf sarkari yojana aur business growth mein madad kar sakta hoon.",
            "English": "Sorry, I can only help with government schemes and business growth."
        }
        return quick_responses.get(language, quick_responses["English"])

    if intent == "Gratitude_Intent":
        gratitude_responses = {
            "Hindi": "धन्यवाद! और कुछ मदद चाहिए?",
            "Hinglish": "Welcome! Aur kuch madad chahiye?",
            "English": "You're welcome! Need anything else?"
        }
        return gratitude_responses.get(language, gratitude_responses["English"])

    # Build optimized prompt
    word_limit = 120 if intent == "Schemes_Know_Intent" else 80
    
    prompt = f"""Task: Answer in {language} for {user_info.name} from {user_info.state_name}.
    
Intent: {intent}
Query: {query}
Data: {rag_response}
Limit: {word_limit} words

Format: Greeting + structured answer + follow-up question.
Use scheme names exactly as provided."""

    llm = await llm_pool.get_llm()
    try:
        if stream:
            async def token_generator():
                buffer = ""
                try:
                    async for chunk in llm.astream([{"role": "user", "content": prompt}]):
                        token = chunk.content or ""
                        buffer += token
                        if token:
                            yield token
                except Exception as e:
                    logger.error(f"Stream generation failed: {str(e)}")
                    return

                # Add eligibility link if needed
                if intent == "Specific_Scheme_Eligibility_Intent" and scheme_guid:
                    screening_link = f"https://customer.haqdarshak.com/check-eligibility/{scheme_guid}"
                    if screening_link not in buffer:
                        for char in f"\n{screening_link}":
                            yield char

            return token_generator()
        else:
            response = await asyncio.wait_for(
                llm.ainvoke([{"role": "user", "content": prompt}]),
                timeout=8.0
            )
            final_text = response.content.strip()
            
            # Add eligibility link if needed
            if intent == "Specific_Scheme_Eligibility_Intent" and scheme_guid:
                screening_link = f"https://customer.haqdarshak.com/check-eligibility/{scheme_guid}"
                if screening_link not in final_text:
                    final_text += f"\n{screening_link}"
            
            return final_text
    except asyncio.TimeoutError:
        logger.error("Response generation timeout")
        fallback_responses = {
            "Hindi": "क्षमा करें, कृपया अपना प्रश्न दोबारा पूछें।",
            "Hinglish": "Sorry, kripya apna question dobara poochiye.",
            "English": "Sorry, please ask your question again."
        }
        return fallback_responses.get(language, fallback_responses["English"])
    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}")
        return "I'm having technical difficulties. Please try again."
    finally:
        llm_pool.return_llm(llm)

def generate_hindi_audio_script_sync(original_response: str, user_info: UserContext, rag_response: str = "") -> str:
    """Synchronous Hindi audio script generation for thread pool"""
    prompt = f"""Summarize in simple Hindi (50-80 words) for TTS:

Response: {original_response[:500]}  
Data: {rag_response[:300]}

Requirements:
- Natural Hindi (Devanagari)  
- Simple words only
- No English words
- No URLs
- Smooth for audio

Summary:"""
    
    try:
        # Use a simple LLM instance for this task
        from langchain_openai import ChatOpenAI
        temp_llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0, 
            max_tokens=150,
            timeout=5
        )
        response = temp_llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Hindi audio script failed: {str(e)}")
        return "ऑडियो उत्पन्न करने में समस्या हुई।"

# Background task functions
async def save_conversation_background(
    session_id: str, 
    mobile_number: str, 
    query: str, 
    response: str,
    hindi_script: str = ""
):
    """Non-blocking conversation save"""
    try:
        interaction_id = generate_interaction_id(query, datetime.utcnow())
        messages_to_save = [
            {"role": "user", "content": query, "timestamp": datetime.utcnow(), "interaction_id": interaction_id},
            {"role": "assistant", "content": response, "timestamp": datetime.utcnow(), 
             "interaction_id": interaction_id, "audio_script": hindi_script},
        ]
        
        await data_manager.save_conversation_async(session_id, mobile_number, messages_to_save)
        logger.debug(f"Background save completed for session {session_id}")
    except Exception as e:
        logger.error(f"Background save failed: {str(e)}")

async def generate_audio_script_background(response: str, user_info: UserContext, rag_response: str = "") -> str:
    """Fast audio script generation"""
    try:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(
                executor, 
                generate_hindi_audio_script_sync,
                response[:300],  # Truncate for speed
                user_info,
                rag_response[:200]  # Truncate for speed
            ),
            timeout=3.0
        )
    except asyncio.TimeoutError:
        logger.warning("Audio script generation timeout")
        return "ऑडियो उत्पन्न करने में समस्या हुई।"
    except Exception as e:
        logger.error(f"Audio script generation failed: {str(e)}")
        return "ऑडियो उत्पन्न करने में समस्या हुई।"

def create_audio_task_background(response: str, user_info: UserContext, rag_response: str = ""):
    """Create optimized background audio task"""
    async def audio_task(final_text: str = None) -> str:
        text_to_use = final_text or response
        return await generate_audio_script_background(text_to_use[:400], user_info, rag_response[:200])
    return audio_task

# Performance tracking with reduced overhead
class PerformanceTracker:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.checkpoints = {}
    
    def checkpoint(self, name: str):
        self.checkpoints[name] = time.perf_counter() - self.start_time
    
    def log_summary(self):
        total = time.perf_counter() - self.start_time
        summary = " | ".join(f"{k}: {v:.2f}s" for k, v in self.checkpoints.items())
        logger.info(f"Performance: {summary} | Total: {total:.2f}s")


async def generate_response_optimized(
    intent: str, 
    rag_response: str, 
    user_info: UserContext, 
    language: str, 
    context: str, 
    query: str, 
    scheme_guid: str = None, 
    stream: bool = False
) -> str:
    """Optimized response generation - FIXED VERSION"""
    # Quick responses for simple cases
    if intent == "Out_of_Scope":
        logger.warning(f"Out_of_Scope detected for: '{query}', treating as Schemes_Know_Intent")
        quick_responses = {
            "Hindi": "क्षमा करें, मैं केवल सरकारी योजनाओं और व्यावसायिक मदद कर सकता हूँ।",
            "Hinglish": "Sorry, main sirf sarkari yojana aur business growth mein madad kar sakta hoon.",
            "English": "Sorry, I can only help with you novfal government schemes and business growth."
        }
        response_text = quick_responses.get(language, quick_responses["English"])
        
        if stream:
            async def quick_generator():
                for char in response_text:
                    yield char
                    await asyncio.sleep(0.01)  # Small delay for streaming effect
            return quick_generator()
        return response_text

    if intent == "Gratitude_Intent":
        gratitude_responses = {
            "Hindi": "धन्यवाद! और कुछ मदद चाहिए?",
            "Hinglish": "Welcome! Aur kuch madad chahiye?",
            "English": "You're welcome! Need anything else?"
        }
        response_text = gratitude_responses.get(language, gratitude_responses["English"])
        
        if stream:
            async def gratitude_generator():
                for char in response_text:
                    yield char
                    await asyncio.sleep(0.01)
            return gratitude_generator()
        return response_text

    # Build optimized prompt
    word_limit = 120 if intent == "Schemes_Know_Intent" else 80
    
    prompt = f"""Task: Answer in {language} for {user_info.name} from {user_info.state_name}.
    
Intent: {intent}
Query: {query}
Data: {rag_response}
Limit: {word_limit} words

Format: Greeting + structured answer + follow-up question.
Use scheme names exactly as provided."""

    llm = await llm_pool.get_llm()
    try:
        if stream:
            async def token_generator():
                buffer = ""
                try:
                    async for chunk in llm.astream([{"role": "user", "content": prompt}]):
                        token = chunk.content or ""
                        if token:
                            buffer += token
                            yield token
                            await asyncio.sleep(0.001)  # Tiny delay for better UX
                except Exception as e:
                    logger.error(f"Stream generation failed: {str(e)}")
                    error_msg = "Error generating response."
                    for char in error_msg:
                        yield char
                    return

                # Add eligibility link if needed
                if intent == "Specific_Scheme_Eligibility_Intent" and scheme_guid:
                    screening_link = f"\nhttps://customer.haqdarshak.com/check-eligibility/{scheme_guid}"
                    if screening_link not in buffer:
                        for char in screening_link:
                            yield char
                            await asyncio.sleep(0.001)

            return token_generator()
        else:
            response = await asyncio.wait_for(
                llm.ainvoke([{"role": "user", "content": prompt}]),
                timeout=8.0
            )
            final_text = response.content.strip()
            
            # Add eligibility link if needed
            if intent == "Specific_Scheme_Eligibility_Intent" and scheme_guid:
                screening_link = f"\nhttps://customer.haqdarshak.com/check-eligibility/{scheme_guid}"
                if screening_link not in final_text:
                    final_text += screening_link
            
            return final_text
    except asyncio.TimeoutError:
        logger.error("Response generation timeout")
        fallback_responses = {
            "Hindi": "क्षमा करें, कृपया अपना प्रश्न दोबारा पूछें।",
            "Hinglish": "Sorry, kripya apna question dobara poochiye.",
            "English": "Sorry, please ask your question again."
        }
        fallback_text = fallback_responses.get(language, fallback_responses["English"])
        
        if stream:
            async def fallback_generator():
                for char in fallback_text:
                    yield char
                    await asyncio.sleep(0.01)
            return fallback_generator()
        return fallback_text
    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}")
        error_text = "I'm having technical difficulties. Please try again."
        
        if stream:
            async def error_generator():
                for char in error_text:
                    yield char
                    await asyncio.sleep(0.01)
            return error_generator()
        return error_text
    finally:
        llm_pool.return_llm(llm)


# FIXED MAIN ULTRA-OPTIMIZED FUNCTION
async def process_query_ultra_optimized(
    query: str,
    session_id: str,
    mobile_number: str,
    session_data: SessionData,
    user_language: str = None,
    stream: bool = False,
    user_type: int = 1
) -> Tuple[Any, callable]:
    """
    Ultra-optimized query processing - FIXED VERSION
    Target: 12.81s -> 3-4s (75%+ improvement)
    """
    tracker = PerformanceTracker()
    logger.info(f"Processing query: {query[:50]}...")

    # Step 1: Fast user context and language detection (parallel)
    user_info = get_user_context(session_data)
    if not user_info:
        error_msg = "Error: User not logged in."
        if stream:
            async def error_gen():
                for char in error_msg:
                    yield char
            return error_gen(), None
        return error_msg, None

    query_language = user_language if query.lower() == "welcome" and user_language else detect_language(query)
    tracker.checkpoint("setup")

    # Step 2: Handle welcome (early return)
    if query.lower() == "welcome":
        print(f"Welcome query detected kittu for {mobile_number} in {query_language} type {user_type}")
        # Start background conversation check
        conversations_task = asyncio.create_task(
            data_manager.get_conversations_async(mobile_number)
        )
        
        # Don't wait for it - assume new user for speed
        response = welcome_user(user_info.state_name, user_info.name, query_language,user_type)
        
        # Fire-and-forget welcome save
        asyncio.create_task(save_conversation_background(session_id, mobile_number, "welcome", response))
        
        # audio_task = create_audio_task_background(response, user_info)
        audio_task = create_optimized_audio_task(response, user_info, session_id)

        tracker.log_summary()
        
        if stream:
            async def welcome_generator():
                for char in response:
                    yield char
                    await asyncio.sleep(0.01)
            return welcome_generator(), audio_task
        return response, audio_task

    # Step 3: Parallel processing - Intent + Context + User Type
    conversation_history = build_conversation_history(session_data.messages)
    
    # Start all parallel tasks
    tasks = {
        "intent": asyncio.create_task(classify_intent_async(query, conversation_history)),
        "conversations": asyncio.create_task(data_manager.get_conversations_async(mobile_number))
    }
    
    # Wait for critical tasks only (intent)
    intent = await tasks["intent"]
    tracker.checkpoint("intent")
    
    logger.info(f"Intent novfal: {intent}")

    # Step 4: Determine if we need RAG (early filtering)
    no_rag_intents = {"Out_of_Scope", "Gratitude_Intent"}
    
    if intent in no_rag_intents:
        # Skip RAG entirely for these intents
        response_result = await generate_response_optimized(
            intent, "", user_info, query_language, "", query, stream=stream
        )
        
        # Create audio task for both streaming and non-streaming
        # audio_task = create_audio_task_background("", user_info)
        audio_task = create_optimized_audio_task(response, user_info, rag_text, session_id)

        
        if stream:
            # response_result is already an async generator
            return response_result, audio_task
        else:
            # response_result is a string
            # Fire-and-forget save
            asyncio.create_task(save_conversation_background(session_id, mobile_number, query, response_result))
            return response_result, audio_task

    # Step 5: Context determination (optimized)
    follow_up_intents = {
        "Contextual_Follow_Up", "Specific_Scheme_Eligibility_Intent", 
        "Specific_Scheme_Apply_Intent", "Confirmation_New_RAG"
    }
    follow_up = intent in follow_up_intents
    
    context_pair = ""
    if follow_up and session_data.messages:
        # Quick context extraction (last 2 messages max)
        for i, msg in enumerate(reversed(session_data.messages[-4:])):
            if msg["role"] == "assistant" and "Welcome" not in msg.get("content", ""):
                if i > 0:
                    prev_msg = list(reversed(session_data.messages[-4:]))[i-1]
                    if prev_msg["role"] == "user":
                        context_pair = f"User: {prev_msg['content'][:100]}\nAssistant: {msg['content'][:100]}"
                break
    
    tracker.checkpoint("context")

    # Step 6: RAG Processing (with aggressive caching)
    scheme_intents = {
        "Specific_Scheme_Know_Intent", "Specific_Scheme_Apply_Intent", 
        "Specific_Scheme_Eligibility_Intent", "Schemes_Know_Intent", 
        "Contextual_Follow_Up", "Confirmation_New_RAG"
    }
    dfl_intents = {"DFL_Intent", "Non_Scheme_Know_Intent"}
    
    rag_response = None
    
    # Check session cache first (fastest)
    cache_key = query if not follow_up else f"{query}_{context_pair[:50]}"
    
    if intent in scheme_intents:
        if cache_key in session_data.rag_cache:
            rag_response = session_data.rag_cache[cache_key]
            logger.info("Session cache hit - scheme")
        else:
            rag_response = await get_scheme_response_optimized(
                query,
                get_scheme_vector_store(),
                state=user_info.state_id,
                gender=user_info.gender,
                business_category=user_info.business_category,
                include_mudra=(intent == "Schemes_Know_Intent"),
                intent=intent,
            )
            session_data.rag_cache[cache_key] = rag_response
            
    elif intent in dfl_intents:
        if cache_key in session_data.dfl_rag_cache:
            rag_response = session_data.dfl_rag_cache[cache_key]
            logger.info("Session cache hit - DFL")
        else:
            rag_response = await get_dfl_response_async(
                query,
                get_dfl_vector_store(),
                state=user_info.state_id,
                gender=user_info.gender,
                business_category=user_info.business_category,
            )
            session_data.dfl_rag_cache[cache_key] = rag_response
    
    tracker.checkpoint("rag")

    # Step 7: Response Generation
    rag_text = rag_response.get("text") if isinstance(rag_response, dict) else rag_response
    if intent == "DFL_Intent" and (rag_text is None or "No relevant" in str(rag_text)):
        rag_text = ""
    
    scheme_guid = None
    if isinstance(rag_response, dict) and intent == "Specific_Scheme_Eligibility_Intent":
        scheme_guid = extract_scheme_guid(rag_response.get("sources", []))
    
    response_result = await generate_response_optimized(
        intent, rag_text or "", user_info, query_language,
        context_pair, query, scheme_guid=scheme_guid, stream=stream
    )
    
    tracker.checkpoint("response")

    # Step 8: Background Tasks (fire-and-forget for non-streaming)
    if stream:
        def create_streaming_audio_task():
            async def streaming_audio_task(final_text: str) -> str:
                # Start both tasks but only wait for audio
                audio_task = asyncio.create_task(
                    generate_audio_script_background(final_text, user_info, rag_text or "")
                )
                
                # Fire-and-forget save
                asyncio.create_task(
                    save_conversation_background(session_id, mobile_number, query, final_text)
                )
                
                return await audio_task
            
            return streaming_audio_task
        
        audio_task = create_streaming_audio_task()
        # response_result is already an async generator
        return response_result, audio_task
    else:
        response_text = response_result
        
        async def background_audio_task(final_text: str = None) -> str:
            text_to_use = final_text or response_text
            
            # Start both tasks in parallel
            audio_task = asyncio.create_task(
                generate_audio_script_background(text_to_use, user_info, rag_text or "")
            )
            save_task = asyncio.create_task(
                save_conversation_background(session_id, mobile_number, query, text_to_use)
            )
            
            # Wait for audio, let save complete in background
            try:
                return await audio_task
            except Exception as e:
                logger.error(f"Audio task failed: {e}")
                return "ऑडियो उत्पन्न करने में समस्या हुई।"
        
        audio_task = background_audio_task
        return response_result, audio_task

    tracker.checkpoint("background")
    tracker.log_summary()
    
    logger.info(f"Query processing completed: {query[:50]}...")
    return response_result, audio_task

def synthesize_with_retry(text: str, language: str = "Hindi", max_retries: int = 2):
    """Enhanced synthesize function with retry logic and better error handling"""
    
    # Validate input
    if not text or not text.strip():
        logger.warning("🔊 Empty text provided for synthesis")
        return None
    
    # Limit text length for faster processing
    text = text.strip()[:300]  # Max 300 chars
    
    for attempt in range(max_retries + 1):
        try:
            # Add attempt info for debugging
            logger.info(f"🔊 Synthesis attempt {attempt + 1} for text: '{text[:50]}...'")
            
            # Call your original synthesize function
            result = synthesize(text, language)
            
            if result and len(result) > 0:
                logger.info(f"🔊 Synthesis successful: {len(result)} bytes")
                return result
            else:
                logger.warning(f"🔊 Synthesis returned empty result on attempt {attempt + 1}")
                
        except Exception as e:
            logger.error(f"🔊 Synthesis failed on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
            
    logger.error("🔊 All synthesis attempts failed")
    return None

async def generate_audio_script_enhanced(
    response: str, 
    user_info: UserContext, 
    rag_response: str = "",
    session_id: str = None
) -> str:
    """Enhanced audio script generation with session tracking"""
    
    if not session_id:
        session_id = "unknown"
        
    # Cancel any pending audio requests for this session
    await audio_tracker.cancel_old_requests(session_id)
    
    # Generate unique request ID
    request_id = hashlib.md5(f"{session_id}_{time.time()}".encode()).hexdigest()[:8]
    await audio_tracker.add_request(session_id, request_id)
    
    try:
        # Use semaphore to limit concurrent script generation
        async with audio_semaphore:
            logger.info(f"🔊 Starting audio script generation for session {session_id}")
            
            # Check if request was cancelled while waiting
            if session_id in audio_tracker.active_requests:
                if audio_tracker.active_requests[session_id]["status"] == "cancelled":
                    logger.info(f"🔊 Audio request cancelled for session {session_id}")
                    return ""
            
            loop = asyncio.get_event_loop()
            script = await asyncio.wait_for(
                loop.run_in_executor(
                    audio_executor,
                    generate_hindi_audio_script_sync,
                    response[:200],  # Limit length
                    user_info,
                    rag_response[:150]  # Limit length
                ),
                timeout=4.0  # Increased timeout
            )
            
            await audio_tracker.complete_request(session_id, True)
            return script
            
    except asyncio.TimeoutError:
        logger.warning(f"🔊 Audio script generation timeout for session {session_id}")
        await audio_tracker.complete_request(session_id, False)
        return ""
    except Exception as e:
        logger.error(f"🔊 Audio script generation failed for session {session_id}: {str(e)}")
        await audio_tracker.complete_request(session_id, False)
        return ""

async def generate_audio_with_queue(
    script: str, 
    language: str = "Hindi",
    session_id: str = None,
    max_wait_time: float = 8.0
) -> bytes:
    """Enhanced audio generation with proper queuing and resource management"""
    
    if not script or not script.strip():
        logger.warning("🔊 No script provided for audio generation")
        return None
    
    if not session_id:
        session_id = f"temp_{time.time()}"
    
    logger.info(f"🔊 Starting audio synthesis for session {session_id}")
    
    try:
        # Use semaphore to limit concurrent audio synthesis
        async with audio_semaphore:
            # Check if this is still the active request for this session
            if session_id in audio_tracker.active_requests:
                if audio_tracker.active_requests[session_id]["status"] == "cancelled":
                    logger.info(f"🔊 Audio synthesis cancelled for session {session_id}")
                    return None
            
            # Generate audio with dedicated executor and timeout
            start_time = time.time()
            
            loop = asyncio.get_event_loop()
            audio_bytes = await asyncio.wait_for(
                loop.run_in_executor(
                    audio_executor,
                    synthesize_with_retry,
                    script,
                    language
                ),
                timeout=max_wait_time
            )
            
            elapsed_time = time.time() - start_time
            
            if audio_bytes and len(audio_bytes) > 0:
                logger.info(f"🔊 Audio generated successfully in {elapsed_time:.2f}s: {len(audio_bytes)} bytes")
                return audio_bytes
            else:
                logger.warning(f"🔊 Audio generation returned empty result for session {session_id}")
                return None
                
    except asyncio.TimeoutError:
        logger.error(f"🔊 Audio synthesis timeout after {max_wait_time}s for session {session_id}")
        return None
    except Exception as e:
        logger.error(f"🔊 Audio synthesis failed for session {session_id}: {str(e)}")
        return None

# Enhanced audio task creation
def create_optimized_audio_task(response: str, user_info: UserContext, rag_response: str = "", session_id: str = None):
    """Create optimized audio task with proper session handling"""
    
    async def optimized_audio_task(final_text: str = None) -> str:
        text_to_use = final_text or response
        
        if not text_to_use or not text_to_use.strip():
            logger.warning("🔊 No text provided for audio task")
            return ""
        
        try:
            # Step 1: Generate script
            script = await generate_audio_script_enhanced(
                text_to_use, 
                user_info, 
                rag_response, 
                session_id
            )
            
            if not script or not script.strip():
                logger.warning("🔊 No audio script generated")
                return ""
            
            return script
            
        except Exception as e:
            logger.error(f"🔊 Audio task failed: {str(e)}")
            return ""
    
    return optimized_audio_task

# Enhanced endpoint audio handling
async def handle_audio_in_endpoint(
    script: str, 
    session_id: str,
    timeout: float = 6.0
) -> tuple[bool, str]:
    """Handle audio generation in the endpoint with proper error handling"""
    
    if not script or not script.strip():
        return False, ""
    
    try:
        # Generate audio with session tracking
        audio_bytes = await generate_audio_with_queue(
            script, 
            "Hindi", 
            session_id, 
            timeout
        )
        
        if audio_bytes and len(audio_bytes) > 0:
            import base64
            b64_audio = base64.b64encode(audio_bytes).decode()
            logger.info(f"🔊 Audio ready for session {session_id}: {len(audio_bytes)} bytes")
            return True, b64_audio
        else:
            logger.warning(f"🔊 No audio generated for session {session_id}")
            return False, ""
            
    except Exception as e:
        logger.error(f"🔊 Audio handling failed for session {session_id}: {str(e)}")
        return False, ""


# Export the optimized function
process_query_ultra_optimized = process_query_ultra_optimized