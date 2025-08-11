# import logging
# import time
# import asyncio
# import json
# import hashlib
# from datetime import datetime, timedelta
# from dataclasses import dataclass
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain.chains import RetrievalQA
# from data_loader import load_rag_data, load_dfl_data, PineconeRecordRetriever
# from scheme_lookup import (
#     find_scheme_guid_by_query,
#     fetch_scheme_docs_by_guid,
#     DocumentListRetriever,
# )
# from utils import extract_scheme_guid
# from data import DataManager
# import re
# import os
# from functools import lru_cache
# from concurrent.futures import ThreadPoolExecutor
# from typing import Optional, Dict, Any, AsyncGenerator, Tuple
# import redis.asyncio as redis
# import aiocache

# # Set up logging
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     force=True)
# logger = logging.getLogger(__name__)
# logging.getLogger("pymongo").setLevel(logging.WARNING)

# # Load environment variables
# load_dotenv()

# # Optimized Thread pool with more workers for better parallelism
# executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="msme-worker")

# # Ultra-Enhanced Cache Manager with aggressive caching
# class UltraCacheManager:
#     def __init__(self):
#         try:
#             self.redis_client = redis.Redis.from_url(
#                 os.getenv("REDIS_URL", "redis://localhost:6379"),
#                 decode_responses=True,
#                 socket_connect_timeout=1,  # Fast connection timeout
#                 socket_timeout=1,          # Fast read timeout
#                 retry_on_timeout=True,
#                 health_check_interval=30
#             )
#         except:
#             self.redis_client = None
#             logger.warning("Redis not available, using memory cache only")
        
#         # Larger memory caches with TTL
#         self.memory_cache = {}
#         self.memory_ttl = {}
#         self.max_memory_cache_size = 5000  # Increased from 1000
        
#         # Pre-computed cache keys for common patterns
#         self._key_cache = {}
    
#     def _cleanup_memory_cache(self):
#         """Optimized cleanup with TTL awareness"""
#         current_time = time.time()
        
#         # Remove expired entries
#         expired_keys = [k for k, ttl in self.memory_ttl.items() if ttl < current_time]
#         for key in expired_keys:
#             self.memory_cache.pop(key, None)
#             self.memory_ttl.pop(key, None)
        
#         # If still over limit, remove oldest
#         if len(self.memory_cache) > self.max_memory_cache_size:
#             items = list(self.memory_cache.items())
#             items_to_keep = items[len(items)//3:]  # Keep more items
#             self.memory_cache = dict(items_to_keep)
#             # Update TTL cache accordingly
#             self.memory_ttl = {k: v for k, v in self.memory_ttl.items() if k in self.memory_cache}
    
#     def _generate_cache_key(self, query: str, context: str = "", cache_type: str = "") -> str:
#         """Optimized cache key generation with pre-computation"""
#         key_data = f"{cache_type}:{query[:200]}:{context[:100]}"  # Truncate for performance
#         key_hash = key_data if len(key_data) < 50 else hashlib.md5(key_data.encode()).hexdigest()
        
#         # Cache the key computation for repeated queries
#         if key_data not in self._key_cache:
#             if len(self._key_cache) > 1000:
#                 self._key_cache.clear()  # Simple cleanup
#             self._key_cache[key_data] = key_hash
        
#         return self._key_cache[key_data]
    
#     async def get_cache(self, cache_type: str, query: str, context: str = "") -> Optional[Any]:
#         """Unified cache getter with fast fallback"""
#         cache_key = self._generate_cache_key(query, context, cache_type)
        
#         # Check memory first (fastest)
#         if cache_key in self.memory_cache:
#             if cache_key in self.memory_ttl and self.memory_ttl[cache_key] > time.time():
#                 return self.memory_cache[cache_key]
#             else:
#                 # Expired
#                 self.memory_cache.pop(cache_key, None)
#                 self.memory_ttl.pop(cache_key, None)
        
#         # Try Redis with short timeout
#         if self.redis_client:
#             try:
#                 cached = await asyncio.wait_for(
#                     self.redis_client.get(cache_key), 
#                     timeout=0.1  # Very short timeout
#                 )
#                 if cached:
#                     if cache_type in ['intent', 'rag']:
#                         return json.loads(cached) if cache_type == 'rag' else cached
#                     return cached
#             except (asyncio.TimeoutError, Exception):
#                 pass  # Fail silently and continue
        
#         return None
    
#     async def set_cache(self, cache_type: str, query: str, data: Any, context: str = "", ttl: int = 1800):
#         """Unified cache setter with async Redis operations"""
#         cache_key = self._generate_cache_key(query, context, cache_type)
        
#         # Always set in memory cache first
#         self.memory_cache[cache_key] = data
#         self.memory_ttl[cache_key] = time.time() + ttl
        
#         # Background Redis set (fire and forget)
#         if self.redis_client:
#             asyncio.create_task(self._set_redis_background(cache_key, data, ttl, cache_type))
        
#         # Cleanup if needed
#         if len(self.memory_cache) > self.max_memory_cache_size:
#             self._cleanup_memory_cache()
    
#     async def _set_redis_background(self, key: str, data: Any, ttl: int, cache_type: str):
#         """Background Redis setter"""
#         try:
#             serialized = json.dumps(data) if cache_type == 'rag' else str(data)
#             await asyncio.wait_for(
#                 self.redis_client.setex(key, ttl, serialized),
#                 timeout=0.5
#             )
#         except Exception:
#             pass  # Fail silently

# # Initialize ultra cache manager
# cache_manager = UltraCacheManager()

# # Connection pooling for external services
# class ConnectionPool:
#     def __init__(self):
#         self.llm_pool = []
#         self.intent_llm_pool = []
#         self.pool_size = 3
        
#     async def get_llm(self, for_intent=False):
#         """Get LLM from pool or create new one"""
#         pool = self.intent_llm_pool if for_intent else self.llm_pool
#         if pool:
#             return pool.pop()
#         return init_intent_llm() if for_intent else init_llm()
    
#     async def return_llm(self, llm, for_intent=False):
#         """Return LLM to pool"""
#         pool = self.intent_llm_pool if for_intent else self.llm_pool
#         if len(pool) < self.pool_size:
#             pool.append(llm)

# connection_pool = ConnectionPool()

# # Pre-warming cache for common queries
# COMMON_QUERIES = [
#     "pradhan mantri mudra yojana",
#     "business loan schemes",
#     "women entrepreneur schemes",
#     "startup schemes",
#     "digital payment methods"
# ]

# async def prewarm_cache():
#     """Pre-warm cache with common queries"""
#     logger.info("Pre-warming cache...")
#     tasks = []
#     for query in COMMON_QUERIES:
#         task = asyncio.create_task(
#             cache_manager.set_cache("common", query, {"prewarmed": True}, "", 7200)
#         )
#         tasks.append(task)
#     await asyncio.gather(*tasks, return_exceptions=True)
#     logger.info("Cache pre-warming completed")

# # Initialize async data manager (unchanged but with connection pooling)
# class AsyncDataManager(DataManager):
#     def __init__(self):
#         super().__init__()
#         self._connection_semaphore = asyncio.Semaphore(10)  # Limit concurrent DB operations
    
#     async def get_conversations_async(self, mobile_number: str):
#         async with self._connection_semaphore:
#             loop = asyncio.get_event_loop()
#             return await loop.run_in_executor(executor, self.get_conversations, mobile_number)
    
#     async def save_conversation_async(self, session_id: str, mobile_number: str, messages: list):
#         async with self._connection_semaphore:
#             loop = asyncio.get_event_loop()
#             return await loop.run_in_executor(executor, self.save_conversation, session_id, mobile_number, messages)

# # Initialize async data manager
# data_manager = AsyncDataManager()

# # Cached LLM initialization with connection pooling
# @lru_cache(maxsize=1)
# def init_llm():
#     """Initialize the default LLM client with optimizations."""
#     logger.info("Initializing LLM client")
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise ValueError("OPENAI_API_KEY environment variable not set")
    
#     llm = ChatOpenAI(
#         model="gpt-4o-mini",  # Faster model
#         api_key=api_key,
#         base_url="https://api.openai.com/v1",
#         temperature=0,
#         max_tokens=300,  # Limit tokens for faster response
#         timeout=5,       # Shorter timeout
#         max_retries=1    # Fewer retries
#     )
#     logger.info("LLM initialized")
#     return llm

# @lru_cache(maxsize=1)
# def init_intent_llm():
#     """Initialize a dedicated LLM client for intent classification."""
#     logger.info("Initializing Intent LLM client")
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise ValueError("OPENAI_API_KEY environment variable not set")
    
#     intent_llm = ChatOpenAI(
#         model="gpt-4o-mini",  # Faster model for intent
#         api_key=api_key,
#         base_url="https://api.openai.com/v1",
#         temperature=0,
#         max_tokens=50,   # Very limited for intent classification
#         timeout=3,       # Even shorter timeout for intent
#         max_retries=1
#     )
#     logger.info("Intent LLM initialized")
#     return intent_llm

# # Initialize resources
# llm = init_llm()
# intent_llm = init_intent_llm()

# # Lazy loading with async initialization
# _scheme_vector_store = None
# _dfl_vector_store = None

# async def get_scheme_vector_store():
#     global _scheme_vector_store
#     if _scheme_vector_store is None:
#         logger.info("Loading scheme vector store")
#         loop = asyncio.get_event_loop()
#         _scheme_vector_store = await loop.run_in_executor(executor, init_vector_store)
#     return _scheme_vector_store

# async def get_dfl_vector_store():
#     global _dfl_vector_store
#     if _dfl_vector_store is None:
#         logger.info("Loading DFL vector store")
#         loop = asyncio.get_event_loop()
#         _dfl_vector_store = await loop.run_in_executor(executor, init_dfl_vector_store)
#     return _dfl_vector_store

# def init_vector_store():
#     """Initialize scheme vector store"""
#     index_host = os.getenv("PINECONE_SCHEME_HOST")
#     if not index_host:
#         raise ValueError("PINECONE_SCHEME_HOST environment variable not set")
#     index_name = os.getenv("PINECONE_INDEX_NAME")
#     try:
#         vector_store = load_rag_data(host=index_host, index_name=index_name, version_file="faiss_version.txt")
#     except Exception as e:
#         logger.error(f"Failed to load scheme index: {e}")
#         raise
#     return vector_store

# def init_dfl_vector_store():
#     """Initialize DFL vector store"""
#     google_drive_file_id = os.getenv("DFL_GOOGLE_DOC_ID")
#     if not google_drive_file_id:
#         raise ValueError("DFL_GOOGLE_DOC_ID environment variable not set")
#     index_host = os.getenv("PINECONE_DFL_HOST")
#     if not index_host:
#         raise ValueError("PINECONE_DFL_HOST environment variable not set")
#     index_name = os.getenv("PINECONE_DFL_INDEX_NAME")
#     try:
#         vector_store = load_dfl_data(google_drive_file_id, host=index_host, index_name=index_name)
#     except Exception as e:
#         logger.error(f"Failed to load DFL index: {e}")
#         raise
#     return vector_store

# # Keep your existing dataclasses and helper functions unchanged
# @dataclass
# class UserContext:
#     name: str
#     state_id: str
#     state_name: str
#     business_name: str
#     business_category: str
#     gender: str

# class SessionData:
#     """Enhanced session container with caching."""
#     def __init__(self, user=None):
#         self.user = user
#         self.messages = []
#         self.rag_cache = {}  # Increased cache size
#         self.dfl_rag_cache = {}
#         self.intent_cache = {}
#         self.last_access = time.time()

# # Session cleanup for memory management
# active_sessions = {}
# SESSION_TTL = 3600  # 1 hour

# def cleanup_sessions():
#     """Background task to cleanup old sessions"""
#     current_time = time.time()
#     expired = [sid for sid, sess in active_sessions.items() 
#                if current_time - sess.last_access > SESSION_TTL]
#     for sid in expired:
#         active_sessions.pop(sid, None)
#     logger.info(f"Cleaned up {len(expired)} expired sessions")

# # Your existing helper functions (kept the same for brevity)
# async def get_user_context_async(session_data):
#     # ... (same as before)
#     try:
#         user = session_data.user
#         return UserContext(
#             name=user["fname"],
#             state_id=user.get("state_id", "Unknown"),
#             state_name=user.get("state_name", "Unknown"),
#             business_name=user.get("business_name", "Unknown"),
#             business_category=user.get("business_category", "Unknown"),
#             gender=user.get("gender", "Unknown"),
#         )
#     except AttributeError:
#         logger.error("User data not found in session state")
#         return None

# async def detect_language_async(query):
#     # ... (same as before but cached)
#     cache_key = f"lang:{query[:50]}"
#     if cache_key in cache_manager.memory_cache:
#         return cache_manager.memory_cache[cache_key]
    
#     devanagari_pattern = re.compile(r'[\u0900-\u097F]')
#     if devanagari_pattern.search(query):
#         result = "Hindi"
#     else:
#         hindi_words = [
#             "kya", "kaise", "ke", "mein", "hai", "kaun", "kahan", "kab",
#             "batao", "sarkari", "yojana", "paise", "karobar", "dukaan", "nayi", "naye", "chahiye", "madad", "karo",
#         ]
#         query_lower = query.lower()
#         hindi_word_count = sum(1 for word in hindi_words if word in query_lower)
#         total_words = len(query_lower.split())
        
#         if total_words > 0 and hindi_word_count / total_words > 0.15:
#             result = "Hinglish"
#         else:
#             result = "English"
    
#     # Cache the result
#     cache_manager.memory_cache[cache_key] = result
#     cache_manager.memory_ttl[cache_key] = time.time() + 300  # 5 minutes
    
#     return result

# def get_system_prompt(language, user_name="User", word_limit=200):
#     # ... (same as before)
#     system_rules = f"""1. **Language Handling**:
#        - The query language is provided as {language} (English, Hindi, or Hinglish).
#        - For Hindi queries, respond in Devanagari script using simple, clear words suitable for micro business owners with low Hindi proficiency.
#        - For Hinglish queries, use a natural mix of simple English and Hindi words in Roman script, prioritizing hindi words in the mix.
#        - For English queries, respond in simple English.
       
#        2. **Response Guidelines**:
#        - Scope: Only respond to queries about government schemes, digital/financial literacy, or business growth.
#        - Tone and Style: Use simple, clear words, short sentences, friendly tone, relatable examples.
#        - Give structured responses with formatting like bullets or headings/subheadings. Do not give long paragraphs of text.
#        - Response must be <={word_limit} words.

#        - Never mention agent fees unless specified in RAG Response for scheme queries.
#        - Never repeat user query or bring up ambiguity in the response, proceed directly to answering.
#        - Never mention technical terms like RAG, LLM, Database etc. to the user.
#        - Use scheme names exactly as provided in the RAG Response without paraphrasing (underscores may be replaced with spaces).
#        - Start the response with 'Hi {user_name}!' (English), 'Namaste {user_name}!' (Hinglish), or 'नमस्ते {user_name}!' (Hindi) unless Out_of_Scope."""

#     return system_rules.format(language=language, user_name=user_name)

# def build_conversation_history(messages):
#     # ... (same as before)
#     conversation_history = ""
#     session_messages = []
#     for msg in messages[-10:]:
#         if msg["role"] == "assistant" and "Welcome" in msg["content"]:
#             continue
#         session_messages.append((msg["role"], msg["content"], msg["timestamp"]))
#     session_messages = sorted(session_messages, key=lambda x: x[2], reverse=True)[:5]
#     for role, content, _ in session_messages:
#         conversation_history += f"{role.capitalize()}: {content}\n"
#     return conversation_history

# def welcome_user(state_name, user_name, query_language):
#     # ... (same as before)
#     if query_language == "Hindi":
#         return f"नमस्ते {user_name}! हकदर्शक MSME चैटबॉट में स्वागत है। आप {state_name} से हैं, मैं आपकी राज्य और केंद्रीय योजनाओं में मदद करूँगा। आज कैसे सहायता करूँ?"
#     return f"Hi {user_name}! Welcome to Haqdarshak MSME Chatbot! Since you're from {state_name}, I'll help with schemes and documents applicable to your state and all central government schemes. How can I assist you today?"

# def generate_interaction_id(query, timestamp):
#     return f"{query[:500]}_{timestamp.strftime('%Y%m%d%H%M%S')}"

# # ULTRA-OPTIMIZED ASYNC FUNCTIONS
# async def classify_intent_ultra_async(query: str, conversation_history: str = "") -> str:
#     """Ultra-optimized intent classification with aggressive caching and fast models"""
    
#     # Check cache first
#     cached_intent = await cache_manager.get_cache("intent", query, conversation_history)
#     if cached_intent:
#         return cached_intent
    
#     # Simplified prompt for faster processing
#     simplified_prompt = f"""Classify intent for: "{query[:200]}"
# Context: {conversation_history[:300]}

# Return only ONE label:
# Schemes_Know_Intent, DFL_Intent, Specific_Scheme_Know_Intent, Specific_Scheme_Apply_Intent, 
# Specific_Scheme_Eligibility_Intent, Out_of_Scope, Contextual_Follow_Up, Confirmation_New_RAG, Gratitude_Intent"""
    
#     try:
#         # Use connection pool
#         llm = await connection_pool.get_llm(for_intent=True)
        
#         # Set shorter timeout for intent classification
#         response = await asyncio.wait_for(
#             llm.ainvoke([{"role": "user", "content": simplified_prompt}]),
#             timeout=3.0
#         )
        
#         intent = response.content.strip()
        
#         # Return LLM to pool and cache result
#         asyncio.create_task(connection_pool.return_llm(llm, for_intent=True))
#         asyncio.create_task(cache_manager.set_cache("intent", query, intent, conversation_history, 7200))
        
#         return intent
#     except asyncio.TimeoutError:
#         logger.warning("Intent classification timeout - using fallback")
#         return "Schemes_Know_Intent"  # Safe fallback
#     except Exception as e:
#         logger.error(f"Failed to classify intent: {str(e)}")
#         return "Out_of_Scope"

# async def get_rag_response_ultra_async(query, vector_store, state="ALL_STATES", gender=None, business_category=None):
#     """Ultra-optimized RAG response with parallel processing"""
#     try:
#         # Build query details in parallel
#         # details_task = asyncio.create_task(asyncio.coroutine(lambda: [
#         #     detail for detail in [
#         #         f"state: {state}" if state else None,
#         #         f"gender: {gender}" if gender else None,
#         #         f"business category: {business_category}" if business_category else None
#         #     ] if detail
#         # ])())
        
#         # details = await details_task
#         details = [
#             detail for detail in [
#                 f"state: {state}" if state else None,
#                 f"gender: {gender}" if gender else None, 
#                 f"business category: {business_category}" if business_category else None
#             ] if detail
#          ]
#         full_query = f"{query}. {' '.join(details)}" if details else query

#         logger.debug(f"Processing query: {full_query}")
        
#         # Optimized retrieval with smaller k for faster response
#         def run_retrieval():
#             retriever = PineconeRecordRetriever(
#                 index=vector_store, state=state, gender=gender, k=3  # Reduced from 5
#             )
#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=llm,
#                 chain_type="stuff",
#                 retriever=retriever,
#                 return_source_documents=True,
#             )
#             return qa_chain.invoke({"query": full_query})
        
#         # Run with timeout
#         result = await asyncio.wait_for(
#             asyncio.get_event_loop().run_in_executor(executor, run_retrieval),
#             timeout=8.0  # Shorter timeout
#         )
        
#         response = result["result"]
#         sources = result["source_documents"]
        
#         logger.info(f"RAG response generated: {response[:100]}...")
#         return {"text": response, "sources": sources}
        
#     except asyncio.TimeoutError:
#         logger.warning("RAG retrieval timeout - using cached response")
#         return {"text": "I apologize, but I'm experiencing delays. Please try again.", "sources": []}
#     except Exception as e:
#         logger.error(f"RAG retrieval failed: {str(e)}")
#         return {"text": "Error retrieving information.", "sources": []}

# async def get_scheme_response_ultra_async(
#     query,
#     vector_store,
#     state="ALL_STATES",
#     gender=None,
#     business_category=None,
#     include_mudra=False,
#     intent=None,
# ):
#     """Ultra-optimized scheme response with parallel processing and smart caching"""
#     logger.info("Querying scheme dataset with ultra optimization")

#     # Enhanced cache context
#     cache_context = json.dumps({
#         "state": state,
#         "gender": gender,
#         "business_category": business_category,
#         "include_mudra": include_mudra,
#         "intent": intent
#     }, sort_keys=True)
    
#     # Check cache first
#     cached_response = await cache_manager.get_cache("rag", query, cache_context)
#     if cached_response:
#         logger.info("Scheme response cache hit")
#         return cached_response

#     guid = None
#     if intent == "Specific_Scheme_Know_Intent":
#         # Run GUID lookup with timeout
#         try:
#             loop = asyncio.get_event_loop()
#             guid = await asyncio.wait_for(
#                 loop.run_in_executor(executor, find_scheme_guid_by_query, query),
#                 timeout=2.0
#             )
#         except asyncio.TimeoutError:
#             logger.warning("GUID lookup timeout - falling back to search")

#     # Parallel processing for scheme and mudra data
#     tasks = []
    
#     # Main scheme task
#     if guid:
#         async def fetch_by_guid():
#             try:
#                 loop = asyncio.get_event_loop()
#                 docs = await asyncio.wait_for(
#                     loop.run_in_executor(executor, fetch_scheme_docs_by_guid, guid, vector_store),
#                     timeout=3.0
#                 )
#                 if docs:
#                     def run_qa_chain():
#                         retriever = DocumentListRetriever(docs)
#                         qa_chain = RetrievalQA.from_chain_type(
#                             llm=llm, chain_type="stuff", retriever=retriever,
#                             return_source_documents=True,
#                         )
#                         return qa_chain.invoke({"query": query})
                    
#                     result = await asyncio.wait_for(
#                         loop.run_in_executor(executor, run_qa_chain),
#                         timeout=5.0
#                     )
#                     return {"text": result["result"], "sources": result["source_documents"]}
#                 return None
#             except asyncio.TimeoutError:
#                 logger.warning("Fetch by GUID timeout")
#                 return None
        
#         tasks.append(fetch_by_guid())
#     else:
#         # Regular RAG search
#         tasks.append(get_rag_response_ultra_async(
#             query, vector_store, state=state, gender=gender, business_category=business_category,
#         ))

#     # Mudra task (if needed)
#     mudra_task = None
#     if include_mudra:
#         async def fetch_mudra():
#             try:
#                 logger.info("Fetching Pradhan Mantri Mudra Yojana details")
#                 loop = asyncio.get_event_loop()
#                 mudra_guid = await asyncio.wait_for(
#                     loop.run_in_executor(executor, find_scheme_guid_by_query, "pradhan mantri mudra yojana"),
#                     timeout=1.0
#                 ) or "SH0008BK"
                
#                 mudra_docs = await asyncio.wait_for(
#                     loop.run_in_executor(executor, fetch_scheme_docs_by_guid, mudra_guid, vector_store),
#                     timeout=2.0
#                 )

#                 if mudra_docs:
#                     def run_mudra_qa():
#                         retriever = DocumentListRetriever(mudra_docs)
#                         qa_chain = RetrievalQA.from_chain_type(
#                             llm=llm, chain_type="stuff", retriever=retriever,
#                             return_source_documents=True,
#                         )
#                         return qa_chain.invoke({"query": "Pradhan Mantri Mudra Yojana"})
                    
#                     result = await asyncio.wait_for(
#                         loop.run_in_executor(executor, run_mudra_qa),
#                         timeout=3.0
#                     )
#                     return {"text": result["result"], "sources": result["source_documents"]}
#                 return {"text": "", "sources": []}
#             except asyncio.TimeoutError:
#                 logger.warning("Mudra fetch timeout")
#                 return {"text": "", "sources": []}
        
#         mudra_task = asyncio.create_task(fetch_mudra())

#     # Execute main scheme task
#     try:
#         rag_results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=6.0)
#         rag = rag_results[0] if rag_results and not isinstance(rag_results[0], Exception) else {"text": "", "sources": []}
#     except asyncio.TimeoutError:
#         logger.warning("Main RAG timeout")
#         rag = {"text": "Information retrieval timeout. Please try again.", "sources": []}

#     # Handle mudra results if available
#     if mudra_task:
#         try:
#             mudra_rag = await asyncio.wait_for(mudra_task, timeout=2.0)
#             if isinstance(mudra_rag, dict) and mudra_rag.get("text"):
#                 rag["text"] = f"{rag.get('text', '')}\n{mudra_rag.get('text', '')}"
#                 rag["sources"] = rag.get("sources", []) + mudra_rag.get("sources", [])
#         except asyncio.TimeoutError:
#             logger.warning("Mudra task timeout - proceeding without it")

#     # Cache the result
#     asyncio.create_task(cache_manager.set_cache("rag", query, rag, cache_context, 1800))
    
#     return rag

# # Background task functions remain similar but with timeouts
# # async def generate_response_ultra_async(
# #     intent: str, 
# #     rag_response: str, 
# #     user_info: UserContext, 
# #     language: str, 
# #     context: str, 
# #     query: str, 
# #     scheme_guid: str = None, 
# #     stream: bool = False
# # ) -> str:
# #     """Ultra-optimized response generation"""
    
# #     # Fast returns for simple intents
# #     if intent == "Out_of_Scope":
# #         responses = {
# #             "Hindi": "क्षमा करें, मैं केवल सरकारी योजनाओं, डिजिटल या वित्तीय साक्षरता और व्यावसायिक वृद्धि पर मदद कर सकता हूँ।",
# #             "Hinglish": "Maaf kijiye, main sirf sarkari yojanaon, digital ya financial literacy aur business growth mein madad kar sakta hoon.",
# #             "English": "Sorry, I can only help with government schemes, digital/financial literacy or business growth."
# #         }
# #         return responses.get(language, responses["English"])

# #     if intent == "Gratitude_Intent":
# #         responses = {
# #             "Hindi": "धन्यवाद! क्या मैं और मदद कर सकता हूँ?",
# #             "Hinglish": "Thanks! Kya main aur madad kar sakta hoon?",
# #             "English": "You're welcome! Let me know if you need anything else."
# #         }
# #         return responses.get(language, responses["English"])

# #     # Shortened prompts for faster processing
# #     word_limit = 120 if intent == "Schemes_Know_Intent" else 80  # Reduced limits
# #     tone_prompt = get_system_prompt(language, user_info.name, word_limit)

# #     # Optimized prompt building
# #     base_prompt = f"""Assistant for Haqdarshak MSME chatbot.

# # Input: Intent={intent}, Query="{query}", User={user_info.name}, State={user_info.state_name}, Language={language}
# # RAG: {rag_response[:500]}...

# # {tone_prompt}

# # Task: Provide helpful response about government schemes/business growth."""
    
# #     if scheme_guid:
# #         base_prompt += f" SchemeGUID: {scheme_guid}"

# #     # Intent-specific optimizations
# #     if intent == "Specific_Scheme_Know_Intent":
# #         base_prompt += " Focus: Share scheme details, ask about eligibility/application."
# #     elif intent == "Schemes_Know_Intent":
# #         base_prompt += " Focus: List 3-4 relevant schemes with brief descriptions."

# #     try:
# #         if stream:
# #             # Get LLM from pool
# #             llm_instance = await connection_pool.get_llm()
            
# #             async def token_generator():
# #                 buffer = ""
# #                 try:
# #                     async for chunk in asyncio.wait_for(
# #                         llm_instance.astream([{"role": "user", "content": base_prompt}]),
# #                         timeout=10.0
# #                     ):
# #                         token = chunk.content or ""
# #                         buffer += token
# #                         if token:
# #                             yield token
                            
# #                     # Return LLM to pool
# #                     await connection_pool.return_llm(llm_instance)
                    
# #                     # Add screening link if needed
# #                     if intent == "Specific_Scheme_Eligibility_Intent" and scheme_guid:
# #                         screening_link = f"https://customer.haqdarshak.com/check-eligibility/{scheme_guid}"
# #                         if screening_link not in buffer:
# #                             for char in f"\n{screening_link}":
# #                                 yield char
                                
# #                 except asyncio.TimeoutError:
# #                     await connection_pool.return_llm(llm_instance)
# #                     for char in "Response timeout. Please try again.":
# #                         yield char
# #                 except Exception as e:
# #                     await connection_pool.return_llm(llm_instance)
# #                     logger.error(f"Streaming error: {e}")
# #                     for char in "Error generating response.":
# #                         yield char

# #             return token_generator()
# #         else:
# #             # Non-streaming with timeout
# #             llm_instance = await connection_pool.get_llm()
            
# #             response = await asyncio.wait_for(
# #                 llm_instance.ainvoke([{"role": "user", "content": base_prompt}]),
# #                 timeout=8.0
# #             )
            
# #             await connection_pool.return_llm(llm_instance)
            
# #             final_text = response.content.strip()
            
# #             if intent == "Specific_Scheme_Eligibility_Intent" and scheme_guid:
# #                 screening_link = f"https://customer.haqdarshak.com/check-eligibility/{scheme_guid}"
# #                 if screening_link not in final_text:
# #                     final_text += f"\n{screening_link}"
            
# #             return final_text
            
# #     except asyncio.TimeoutError:
# #         logger.warning("Response generation timeout")
# #         if language == "Hindi":
# #             return "समय सीमा समाप्त। कृपया पुनः प्रयास करें।"
# #         elif language == "Hinglish":
# #             return "Timeout ho gaya. Please try again."
# #         return "Response timeout. Please try again."
# #     except Exception as e:
# #         logger.error(f"Failed to generate response: {str(e)}")
# #         if language == "Hindi":
# #             return "क्षमा करें, मैं आपका प्रश्न संसाधित नहीं कर सका।"
# #         elif language == "Hinglish":
# #             return "Sorry, main aapka query process nahi kar saka."
# #         return "Sorry, I couldn't process your query."

# async def generate_response_ultra_async(
#     intent: str, 
#     rag_response: str, 
#     user_info: UserContext, 
#     language: str, 
#     context: str, 
#     query: str, 
#     scheme_guid: str = None, 
#     stream: bool = False
# ) -> str:  # Always return string now
#     """Ultra-optimized response generation - always returns string"""
    
#     # Fast returns for simple intents
#     if intent == "Out_of_Scope":
#         responses = {
#             "Hindi": "क्षमा करें, मैं केवल सरकारी योजनाओं, डिजिटल या वित्तीय साक्षरता और व्यावसायिक वृद्धि पर मदद कर सकता हूँ।",
#             "Hinglish": "Maaf kijiye, main sirf sarkari yojanaon, digital ya financial literacy aur business growth mein madad kar sakta hoon.",
#             "English": "Sorry, I can only help with government schemes, digital/financial literacy or business growth."
#         }
#         return responses.get(language, responses["English"])

#     if intent == "Gratitude_Intent":
#         responses = {
#             "Hindi": "धन्यवाद! क्या मैं और मदद कर सकता हूँ?",
#             "Hinglish": "Thanks! Kya main aur madad kar sakta hoon?",
#             "English": "You're welcome! Let me know if you need anything else."
#         }
#         return responses.get(language, responses["English"])

#     # Build prompt (same as before)
#     word_limit = 120 if intent == "Schemes_Know_Intent" else 80
#     tone_prompt = get_system_prompt(language, user_info.name, word_limit)

#     base_prompt = f"""Assistant for Haqdarshak MSME chatbot.

# Input: Intent={intent}, Query="{query}", User={user_info.name}, State={user_info.state_name}, Language={language}
# RAG: {rag_response[:500]}...

# {tone_prompt}

# Task: Provide helpful response about government schemes/business growth."""
    
#     if scheme_guid:
#         base_prompt += f" SchemeGUID: {scheme_guid}"

#     # Intent-specific optimizations
#     if intent == "Specific_Scheme_Know_Intent":
#         base_prompt += " Focus: Share scheme details, ask about eligibility/application."
#     elif intent == "Schemes_Know_Intent":
#         base_prompt += " Focus: List 3-4 relevant schemes with brief descriptions."

#     try:
#         # Always use non-streaming approach for simplicity
#         llm_instance = await connection_pool.get_llm()
        
#         response = await asyncio.wait_for(
#             llm_instance.ainvoke([{"role": "user", "content": base_prompt}]),
#             timeout=8.0
#         )
        
#         await connection_pool.return_llm(llm_instance)
        
#         final_text = response.content.strip()
        
#         # Add eligibility link if needed
#         if intent == "Specific_Scheme_Eligibility_Intent" and scheme_guid:
#             screening_link = f"https://customer.haqdarshak.com/check-eligibility/{scheme_guid}"
#             if screening_link not in final_text:
#                 final_text += f"\n{screening_link}"
        
#         return final_text
        
#     except asyncio.TimeoutError:
#         logger.warning("Response generation timeout")
#         if language == "Hindi":
#             return "समय सीमा समाप्त। कृपया पुनः प्रयास करें।"
#         elif language == "Hinglish":
#             return "Timeout ho gaya. Please try again."
#         return "Response timeout. Please try again."
#     except Exception as e:
#         logger.error(f"Failed to generate response: {str(e)}")
#         if language == "Hindi":
#             return "क्षमा करें, मैं आपका प्रश्न संसाधित नहीं कर सका।"
#         elif language == "Hinglish":
#             return "Sorry, main aapka query process nahi kar saka."
#         return "Sorry, I couldn't process your query."

# # If you need streaming functionality, create a separate function:
# async def generate_streaming_response(response_text: str):
#     """Convert a regular response to a streaming response"""
#     for char in response_text:
#         yield char
#         await asyncio.sleep(0.01)  # Small delay for streaming effect

# # Alternative fix if you want to keep the dual return type:
# # Create a wrapper that handles both cases

# # async def handle_response_result(response_result, is_streaming=False):
# #     """Handle both streaming and non-streaming responses"""
# #     if is_streaming:
# #         # response_result should be an async generator
# #         if hasattr(response_result, '__aiter__'):
# #             full_text = ""
# #             async for chunk in response_result:
# #                 full_text += chunk
# #                 yield chunk
# #             return full_text
# #         else:
# #             # Fallback: convert string to streaming
# #             for char in str(response_result):
# #                 yield char
# #     else:
# #         # Non-streaming: just return the string
# #         if hasattr(response_result, '__aiter__'):
# #             # If it's accidentally an async generator, collect all chunks
# #             full_text = ""
# #             async for chunk in response_result:
# #                 full_text += chunk
# #             return full_text
# #         else:
# #             return str(response_result)

# # Optimized background tasks with better error handling
# async def save_conversation_background(
#     session_id: str, 
#     mobile_number: str, 
#     query: str, 
#     response: str,
#     hindi_script: str = ""
# ):
#     """Ultra-fast background save with timeout"""
#     try:
#         interaction_id = f"{hash(query)}_{int(time.time())}"  # Faster ID generation
#         messages_to_save = [
#             {"role": "user", "content": query, "timestamp": datetime.utcnow(), "interaction_id": interaction_id},
#             {"role": "assistant", "content": response, "timestamp": datetime.utcnow(), "interaction_id": interaction_id, "audio_script": hindi_script},
#         ]
        
#         await asyncio.wait_for(
#             data_manager.save_conversation_async(session_id, mobile_number, messages_to_save),
#             timeout=5.0
#         )
#         logger.info(f"Background save completed for session {session_id}")
#     except asyncio.TimeoutError:
#         logger.warning(f"Background save timeout for session {session_id}")
#     except Exception as e:
#         logger.error(f"Background save failed for session {session_id}: {str(e)}")

# def generate_hindi_audio_script_fast(
#     original_response: str,
#     user_info: UserContext,
#     rag_response: str = "",
# ) -> str:
#     """Faster Hindi audio script generation with caching"""
#     # Check cache first
#     cache_key = f"audio:{hash(original_response[:200])}"
#     if cache_key in cache_manager.memory_cache:
#         return cache_manager.memory_cache[cache_key]
    
#     # Simplified prompt for faster processing
#     prompt = f"""Convert to simple Hindi audio script (50-80 words):
# {original_response[:300]}

# Requirements: Clear Hindi, no English words, no URLs, simple vocabulary."""
    
#     try:
#         response = llm.invoke([{"role": "user", "content": prompt}])
#         hindi_script = response.content.strip()
        
#         # Cache the result
#         cache_manager.memory_cache[cache_key] = hindi_script
#         cache_manager.memory_ttl[cache_key] = time.time() + 1800
        
#         return hindi_script
#     except Exception as e:
#         logger.error(f"Fast Hindi script generation failed: {str(e)}")
#         return "ऑडियो स्क्रिप्ट त्रुटि।"

# async def generate_audio_script_background(response: str, user_info: UserContext, rag_response: str = "") -> str:
#     """Ultra-fast background audio script generation"""
#     try:
#         loop = asyncio.get_event_loop()
#         hindi_script = await asyncio.wait_for(
#             loop.run_in_executor(
#                 executor, 
#                 generate_hindi_audio_script_fast,
#                 response,
#                 user_info,
#                 rag_response
#             ),
#             timeout=3.0
#         )
#         return hindi_script
#     except asyncio.TimeoutError:
#         logger.warning("Audio script generation timeout")
#         return "ऑडियो स्क्रिप्ट त्रुटि।"
#     except Exception as e:
#         logger.error(f"Background audio script generation failed: {str(e)}")
#         return "ऑडियो स्क्रिप्ट त्रुटि।"

# # Performance tracking with detailed metrics
# class UltraPerformanceTracker:
#     def __init__(self):
#         self.timings = {}
#         self.start_time = time.perf_counter()
#         self.checkpoint_times = {}
    
#     def checkpoint(self, name: str):
#         """Mark a checkpoint"""
#         self.checkpoint_times[name] = time.perf_counter() - self.start_time
    
#     def get_duration_since_checkpoint(self, checkpoint_name: str) -> float:
#         """Get time since a checkpoint"""
#         if checkpoint_name in self.checkpoint_times:
#             return time.perf_counter() - self.start_time - self.checkpoint_times[checkpoint_name]
#         return 0.0
    
#     def log_summary(self):
#         total = time.perf_counter() - self.start_time
#         checkpoints = "\n".join(f"{k}: {v:.3f}s" for k, v in self.checkpoint_times.items())
#         logger.info(f"Ultra Performance Summary:\n{checkpoints}\nTotal: {total:.3f}s")
        
#         # Performance warnings
#         if total > 5.0:
#             logger.warning(f"Slow response detected: {total:.3f}s")

# # MAIN ULTRA-OPTIMIZED FUNCTION
# async def process_query_ultra_optimized(
#     query: str,
#     scheme_vector_store,
#     dfl_vector_store,
#     session_id: str,
#     mobile_number: str,
#     session_data: SessionData,
#     user_language: str = None,
#     stream: bool = False
# ) -> Tuple[Any, callable]:
#     """
#     Ultra-optimized async version targeting 3s total response time
#     Key optimizations:
#     1. Parallel execution wherever possible
#     2. Aggressive caching with memory + Redis
#     3. Connection pooling for LLMs
#     4. Shorter timeouts and faster models
#     5. Smart early returns
#     6. Background processing for non-critical operations
#     """
#     tracker = UltraPerformanceTracker()
#     logger.info(f"🚀 Ultra-optimized processing: {query[:50]}...")

#     # Update session access time
#     session_data.last_access = time.time()
    
#     # Step 1: Parallel initialization (user context + language detection)
#     tracker.checkpoint("init_start")
    
#     user_info_task = asyncio.create_task(get_user_context_async(session_data))
    
#     # Early language detection
#     if user_language and query.lower() != "welcome":
#         query_language = user_language
#         language_task = None
#     else:
#         language_task = asyncio.create_task(detect_language_async(query))
    
#     # Get user context
#     user_info = await user_info_task
#     if not user_info:
#         tracker.log_summary()
#         return "Error: User not logged in.", None

#     # Get language if not already set
#     if language_task:
#         query_language = await language_task
    
#     tracker.checkpoint("init_complete")
#     logger.info(f"Language: {query_language}, Init: {tracker.get_duration_since_checkpoint('init_start'):.3f}s")

#     # Step 2: Early return for welcome (fastest path)
#     if query.lower() == "welcome":
#         # Start background conversation fetch but don't wait
#         conversations_task = asyncio.create_task(data_manager.get_conversations_async(mobile_number))
        
#         try:
#             conversations = await asyncio.wait_for(conversations_task, timeout=1.0)
#             user_type = "returning" if conversations else "new"
#         except asyncio.TimeoutError:
#             user_type = "new"  # Assume new user if DB is slow
        
#         if user_type == "new":
#             response = welcome_user(user_info.state_name, user_info.name, query_language)
            
#             # Background save (fire and forget)
#             asyncio.create_task(save_conversation_background(
#                 session_id, mobile_number, "welcome", response
#             ))
            
#             # Create audio task
#             async def welcome_audio_task(final_text: str = None) -> str:
#                 text_to_use = final_text or response
#                 return await generate_audio_script_background(text_to_use, user_info)
            
#             tracker.log_summary()
            
#             if stream:
#                 async def gen():
#                     for ch in response:
#                         yield ch
#                 return gen(), welcome_audio_task
#             return response, welcome_audio_task
#         else:
#             tracker.log_summary()
#             return None, None

#     # Step 3: Parallel conversation fetch and intent classification
#     tracker.checkpoint("parallel_start")
    
#     # Start both tasks in parallel
#     conversation_history = build_conversation_history(session_data.messages)
    
#     conversations_task = asyncio.create_task(data_manager.get_conversations_async(mobile_number))
#     intent_task = asyncio.create_task(classify_intent_ultra_async(query, conversation_history))
    
#     # Wait for both with timeout
#     try:
#         conversations, intent = await asyncio.wait_for(
#             asyncio.gather(conversations_task, intent_task),
#             timeout=4.0
#         )
#     except asyncio.TimeoutError:
#         logger.warning("Parallel fetch timeout - using defaults")
#         conversations = []
#         intent = "Schemes_Know_Intent"  # Safe default
    
#     tracker.checkpoint("parallel_complete")
#     logger.info(f"Intent: {intent}, Parallel: {tracker.get_duration_since_checkpoint('parallel_start'):.3f}s")

#     # Step 4: Context determination (fast)
#     follow_up_intents = {
#         "Contextual_Follow_Up", "Specific_Scheme_Eligibility_Intent", 
#         "Specific_Scheme_Apply_Intent", "Confirmation_New_RAG",
#     }
#     follow_up = intent in follow_up_intents
    
#     context_pair = ""
#     if follow_up and session_data.messages:
#         for msg in reversed(session_data.messages[-6:]):  # Check fewer messages
#             if msg["role"] == "assistant" and "Welcome" not in msg["content"]:
#                 recent_response = msg["content"]
#                 msg_index = session_data.messages.index(msg)
#                 if msg_index > 0 and session_data.messages[msg_index - 1]["role"] == "user":
#                     recent_query = session_data.messages[msg_index - 1]["content"]
#                     context_pair = f"User: {recent_query}\nAssistant: {recent_response}"
#                 break

#     # Step 5: RAG retrieval with smart caching and parallel processing
#     tracker.checkpoint("rag_start")
    
#     scheme_intents = {"Specific_Scheme_Know_Intent", "Specific_Scheme_Apply_Intent", 
#                      "Specific_Scheme_Eligibility_Intent", "Schemes_Know_Intent", 
#                      "Contextual_Follow_Up", "Confirmation_New_RAG"}
#     dfl_intents = {"DFL_Intent", "Non_Scheme_Know_Intent"}
    
#     rag_response = None
    
#     if intent in scheme_intents:
#         # Check session cache first (fastest)
#         cache_key = query if not follow_up else f"{query}_{hash(context_pair)}"
        
#         if cache_key in session_data.rag_cache:
#             rag_response = session_data.rag_cache[cache_key]
#             logger.info("Session cache hit for scheme response")
#         else:
#             # Enhanced query for follow-ups
#             augmented_query = query
#             if follow_up and context_pair:
#                 augmented_query = f"{query}. Context: {context_pair[:200]}"
            
#             # Get with ultra optimization
#             rag_response = await get_scheme_response_ultra_async(
#                 augmented_query,
#                 scheme_vector_store,
#                 state=user_info.state_id,
#                 gender=user_info.gender,
#                 business_category=user_info.business_category,
#                 include_mudra=intent == "Schemes_Know_Intent",
#                 intent=intent,
#             )
            
#             # Cache in session
#             session_data.rag_cache[cache_key] = rag_response
            
#             # Limit session cache size
#             if len(session_data.rag_cache) > 20:
#                 # Remove oldest entries
#                 keys_to_remove = list(session_data.rag_cache.keys())[:-15]
#                 for k in keys_to_remove:
#                     session_data.rag_cache.pop(k, None)
                    
#     elif intent in dfl_intents:
#         cache_key = query if not follow_up else f"{query}_{hash(context_pair)}"
        
#         if cache_key in session_data.dfl_rag_cache:
#             rag_response = session_data.dfl_rag_cache[cache_key]
#             logger.info("Session cache hit for DFL response")
#         else:
#             rag_response = await get_rag_response_ultra_async(
#                 query, dfl_vector_store,
#                 state=user_info.state_id, gender=user_info.gender,
#                 business_category=user_info.business_category,
#             )
#             session_data.dfl_rag_cache[cache_key] = rag_response
            
#             if len(session_data.dfl_rag_cache) > 10:
#                 keys_to_remove = list(session_data.dfl_rag_cache.keys())[:-8]
#                 for k in keys_to_remove:
#                     session_data.dfl_rag_cache.pop(k, None)

#     tracker.checkpoint("rag_complete")
#     logger.info(f"RAG: {tracker.get_duration_since_checkpoint('rag_start'):.3f}s")

#     # Step 6: Response generation with ultra optimization
#     tracker.checkpoint("response_start")
    
#     rag_text = rag_response.get("text") if isinstance(rag_response, dict) else rag_response
#     if intent == "DFL_Intent" and (rag_text is None or "No relevant" in rag_text):
#         rag_text = ""
    
#     scheme_guid = None
#     if isinstance(rag_response, dict) and intent == "Specific_Scheme_Eligibility_Intent":
#         scheme_guid = extract_scheme_guid(rag_response.get("sources", []))
    
#     # response_result = await generate_response_ultra_async(
#     #     intent, rag_text or "", user_info, query_language,
#     #     context_pair, query, scheme_guid=scheme_guid, stream=stream,
#     # )
#     response_result = await generate_response_ultra_async(
#         intent, rag_text or "", user_info, query_language,
#         context_pair, query, scheme_guid=scheme_guid, stream=False  # Always False now
#     )
    
#     tracker.checkpoint("response_complete")
#     logger.info(f"Response: {tracker.get_duration_since_checkpoint('response_start'):.3f}s")

#     # Step 7: Create ultra-fast background tasks
#     if stream:
#         def create_ultra_streaming_audio_task():
#             async def ultra_streaming_audio_task(final_text: str) -> str:
#                 # Start background save immediately (fire and forget)
#                 asyncio.create_task(save_conversation_background(
#                     session_id, mobile_number, query, final_text
#                 ))
                
#                 # Generate audio script with timeout
#                 try:
#                     return await asyncio.wait_for(
#                         generate_audio_script_background(final_text, user_info, rag_text or ""),
#                         timeout=2.0
#                     )
#                 except asyncio.TimeoutError:
#                     logger.warning("Audio generation timeout in streaming")
#                     return "ऑडियो त्रुटि।"
            
#             return ultra_streaming_audio_task
        
#         audio_task = create_ultra_streaming_audio_task()
#     else:
#         response_text = response_result
        
#         async def ultra_background_audio_task(final_text: str = None) -> str:
#             text_to_use = final_text or response_text
            
#             # Start save task immediately (fire and forget)
#             asyncio.create_task(save_conversation_background(
#                 session_id, mobile_number, query, text_to_use
#             ))
            
#             # Generate audio with timeout
#             try:
#                 return await asyncio.wait_for(
#                     generate_audio_script_background(text_to_use, user_info, rag_text or ""),
#                     timeout=2.0
#                 )
#             except asyncio.TimeoutError:
#                 logger.warning("Audio generation timeout")
#                 return "ऑडियो त्रुटि।"
        
#         audio_task = ultra_background_audio_task

#     tracker.log_summary()
#     total_time = time.perf_counter() - tracker.start_time
    
#     if total_time <= 3.0:
#         logger.info(f"🎯 SUCCESS: Ultra-optimized processing completed in {total_time:.3f}s")
#     else:
#         logger.warning(f"⚠️ Target missed: {total_time:.3f}s (target: 3.0s)")

#     return response_result, audio_task

# # Background cleanup task
# async def periodic_cleanup():
#     """Periodic cleanup of caches and sessions"""
#     while True:
#         try:
#             await asyncio.sleep(300)  # Every 5 minutes
#             cleanup_sessions()
#             cache_manager._cleanup_memory_cache()
#             logger.info("Periodic cleanup completed")
#         except Exception as e:
#             logger.error(f"Cleanup error: {e}")

# # Initialize background tasks
# async def initialize_ultra_optimizations():
#     """Initialize all ultra optimizations"""
#     try:
#         # Pre-warm cache
#         await prewarm_cache()
        
#         # Start cleanup task
#         asyncio.create_task(periodic_cleanup())
        
#         # Pre-load vector stores
#         scheme_task = asyncio.create_task(get_scheme_vector_store())
#         dfl_task = asyncio.create_task(get_dfl_vector_store())
        
#         await asyncio.gather(scheme_task, dfl_task)
        
#         logger.info("🚀 Ultra optimizations initialized successfully")
        
#     except Exception as e:
#         logger.error(f"Optimization initialization failed: {e}")

# # Export the main function
# __all__ = ["process_query_ultra_optimized", "initialize_ultra_optimizations"]

# # if stream:
# #     async def streaming_generator():
# #         for char in response_text:
# #             yield char
# #             await asyncio.sleep(0.001)  # Small delay for streaming effect
# #     return streaming_generator(), audio_task
# # else:
# #     return response_text, audio_task

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

def welcome_user(state_name, user_name, query_language):
    """Quick welcome message generation"""
    if query_language == "Hindi":
        return f"नमस्ते {user_name}! हकदर्शक MSME चैटबॉट में स्वागत है। मैं {state_name} की योजनाओं में मदद करूँगा। आज कैसे सहायता करूँ?"
    return f"Hi {user_name}! Welcome to Haqdarshak MSME Chatbot! I'll help with schemes from {state_name}. How can I assist you today?"

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

# MAIN ULTRA-OPTIMIZED FUNCTION
# async def process_query_ultra_optimized(
#     query: str,
#     session_id: str,
#     mobile_number: str,
#     session_data: SessionData,
#     user_language: str = None,
#     stream: bool = False
# ) -> Tuple[Any, callable]:
#     """
#     Ultra-optimized query processing
#     Target: 12.81s -> 3-4s (75%+ improvement)
#     """
#     tracker = PerformanceTracker()
#     logger.info(f"Processing query: {query[:50]}...")

#     # Step 1: Fast user context and language detection (parallel)
#     user_info = get_user_context(session_data)
#     if not user_info:
#         return "Error: User not logged in.", None

#     query_language = user_language if query.lower() == "welcome" and user_language else detect_language(query)
#     tracker.checkpoint("setup")

#     # Step 2: Handle welcome (early return)
#     if query.lower() == "welcome":
#         # Start background conversation check
#         conversations_task = asyncio.create_task(
#             data_manager.get_conversations_async(mobile_number)
#         )
        
#         # Don't wait for it - assume new user for speed
#         response = welcome_user(user_info.state_name, user_info.name, query_language)
        
#         # Fire-and-forget welcome save
#         asyncio.create_task(save_conversation_background(session_id, mobile_number, "welcome", response))
        
#         audio_task = create_audio_task_background(response, user_info)
#         tracker.log_summary()
        
#         if stream:
#             async def gen():
#                 for ch in response:
#                     yield ch
#             return gen(), audio_task
#         return response, audio_task

#     # Step 3: Parallel processing - Intent + Context + User Type
#     conversation_history = build_conversation_history(session_data.messages)
    
#     # Start all parallel tasks
#     tasks = {
#         "intent": asyncio.create_task(classify_intent_async(query, conversation_history)),
#         "conversations": asyncio.create_task(data_manager.get_conversations_async(mobile_number))
#     }
    
#     # Wait for critical tasks only (intent)
#     intent = await tasks["intent"]
#     tracker.checkpoint("intent")
    
#     logger.info(f"Intent: {intent}")

#     # Step 4: Determine if we need RAG (early filtering)
#     no_rag_intents = {"Out_of_Scope", "Gratitude_Intent"}
    
#     if intent in no_rag_intents:
#         # Skip RAG entirely for these intents
#         response_result = await generate_response_optimized(
#             intent, "", user_info, query_language, "", query, stream=stream
#         )
        
#         if stream:
#             audio_task = create_audio_task_background("", user_info)
#             return response_result, audio_task
#         else:
#             audio_task = create_audio_task_background(response_result, user_info)
#             # Fire-and-forget save
#             asyncio.create_task(save_conversation_background(session_id, mobile_number, query, response_result))
#             return response_result, audio_task

#     # Step 5: Context determination (optimized)
#     follow_up_intents = {
#         "Contextual_Follow_Up", "Specific_Scheme_Eligibility_Intent", 
#         "Specific_Scheme_Apply_Intent", "Confirmation_New_RAG"
#     }
#     follow_up = intent in follow_up_intents
    
#     context_pair = ""
#     if follow_up and session_data.messages:
#         # Quick context extraction (last 2 messages max)
#         for i, msg in enumerate(reversed(session_data.messages[-4:])):
#             if msg["role"] == "assistant" and "Welcome" not in msg.get("content", ""):
#                 if i > 0:
#                     prev_msg = list(reversed(session_data.messages[-4:]))[i-1]
#                     if prev_msg["role"] == "user":
#                         context_pair = f"User: {prev_msg['content'][:100]}\nAssistant: {msg['content'][:100]}"
#                 break
    
#     tracker.checkpoint("context")

#     # Step 6: RAG Processing (with aggressive caching)
#     scheme_intents = {
#         "Specific_Scheme_Know_Intent", "Specific_Scheme_Apply_Intent", 
#         "Specific_Scheme_Eligibility_Intent", "Schemes_Know_Intent", 
#         "Contextual_Follow_Up", "Confirmation_New_RAG"
#     }
#     dfl_intents = {"DFL_Intent", "Non_Scheme_Know_Intent"}
    
#     rag_response = None
    
#     # Check session cache first (fastest)
#     cache_key = query if not follow_up else f"{query}_{context_pair[:50]}"
    
#     if intent in scheme_intents:
#         if cache_key in session_data.rag_cache:
#             rag_response = session_data.rag_cache[cache_key]
#             logger.info("Session cache hit - scheme")
#         else:
#             rag_response = await get_scheme_response_optimized(
#                 query,
#                 get_scheme_vector_store(),
#                 state=user_info.state_id,
#                 gender=user_info.gender,
#                 business_category=user_info.business_category,
#                 include_mudra=(intent == "Schemes_Know_Intent"),
#                 intent=intent,
#             )
#             session_data.rag_cache[cache_key] = rag_response
            
#     elif intent in dfl_intents:
#         if cache_key in session_data.dfl_rag_cache:
#             rag_response = session_data.dfl_rag_cache[cache_key]
#             logger.info("Session cache hit - DFL")
#         else:
#             rag_response = await get_dfl_response_async(
#                 query,
#                 get_dfl_vector_store(),
#                 state=user_info.state_id,
#                 gender=user_info.gender,
#                 business_category=user_info.business_category,
#             )
#             session_data.dfl_rag_cache[cache_key] = rag_response
    
#     tracker.checkpoint("rag")

#     # Step 7: Response Generation
#     rag_text = rag_response.get("text") if isinstance(rag_response, dict) else rag_response
#     if intent == "DFL_Intent" and (rag_text is None or "No relevant" in str(rag_text)):
#         rag_text = ""
    
#     scheme_guid = None
#     if isinstance(rag_response, dict) and intent == "Specific_Scheme_Eligibility_Intent":
#         scheme_guid = extract_scheme_guid(rag_response.get("sources", []))
    
#     response_result = await generate_response_optimized(
#         intent, rag_text or "", user_info, query_language,
#         context_pair, query, scheme_guid=scheme_guid, stream=stream
#     )
    
#     tracker.checkpoint("response")

#     # Step 8: Background Tasks (fire-and-forget for non-streaming)
#     if stream:
#         def create_streaming_audio_task():
#             async def streaming_audio_task(final_text: str) -> str:
#                 # Start both tasks but only wait for audio
#                 audio_task = asyncio.create_task(
#                     generate_audio_script_background(final_text, user_info, rag_text or "")
#                 )
                
#                 # Fire-and-forget save
#                 asyncio.create_task(
#                     save_conversation_background(session_id, mobile_number, query, final_text)
#                 )
                
#                 return await audio_task
            
#             return streaming_audio_task
        
#         audio_task = create_streaming_audio_task()
#     else:
#         response_text = response_result
        
#         async def background_audio_task(final_text: str = None) -> str:
#             text_to_use = final_text or response_text
            
#             # Start both tasks in parallel
#             audio_task = asyncio.create_task(
#                 generate_audio_script_background(text_to_use, user_info, rag_text or "")
#             )
#             save_task = asyncio.create_task(
#                 save_conversation_background(session_id, mobile_number, query, text_to_use)
#             )
            
#             # Wait for audio, let save complete in background
#             try:
#                 return await audio_task
#             except Exception as e:
#                 logger.error(f"Audio task failed: {e}")
#                 return "ऑडियो उत्पन्न करने में समस्या हुई।"
        
#         audio_task = background_audio_task

#     tracker.checkpoint("background")
#     tracker.log_summary()
    
#     logger.info(f"Query processing completed: {query[:50]}...")
#     return response_result, audio_task

# Fix for the streaming error - replace the problematic sections

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
    stream: bool = False
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
        # Start background conversation check
        conversations_task = asyncio.create_task(
            data_manager.get_conversations_async(mobile_number)
        )
        
        # Don't wait for it - assume new user for speed
        response = welcome_user(user_info.state_name, user_info.name, query_language)
        
        # Fire-and-forget welcome save
        asyncio.create_task(save_conversation_background(session_id, mobile_number, "welcome", response))
        
        audio_task = create_audio_task_background(response, user_info)
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
        audio_task = create_audio_task_background("", user_info)
        
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

# Export the optimized function
process_query_ultra_optimized = process_query_ultra_optimized