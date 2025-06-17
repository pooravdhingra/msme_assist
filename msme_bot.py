import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from data_loader import load_rag_data, load_dfl_data
from utils import get_embeddings
import streamlit as st
from data import DataManager
import re
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize DataManager
data_manager = DataManager()

# Initialize cached resources
@st.cache_resource
def init_llm():
    logger.info("Initializing LLM client")
    start_time = time.time()
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY environment variable not set")
    llm = ChatOpenAI(
        model="grok-3-mini-fast",
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        temperature=0
    )
    logger.info(f"LLM initialized in {time.time() - start_time:.2f} seconds")
    return llm

@st.cache_resource
def init_vector_store():
    logger.info("Loading vector store")
    start_time = time.time()
    vector_store = load_rag_data(faiss_index_path="faiss_index", version_file="faiss_version.txt")
    logger.info(f"Vector store loaded in {time.time() - start_time:.2f} seconds with {vector_store.index.ntotal} documents")
    return vector_store

@st.cache_resource
def init_dfl_vector_store():
    logger.info("Loading DFL vector store")
    start_time = time.time()
    google_drive_file_id = os.getenv("DFL_GOOGLE_DOC_ID")
    if not google_drive_file_id:
        raise ValueError("DFL_GOOGLE_DOC_ID environment variable not set")
    vector_store = load_dfl_data(google_drive_file_id)
    logger.info(
        f"DFL vector store loaded in {time.time() - start_time:.2f} seconds with {vector_store.index.ntotal} documents"
    )
    return vector_store

llm = init_llm()
scheme_vector_store = init_vector_store()
dfl_vector_store = init_dfl_vector_store()

# Dataclass to hold user context information
@dataclass
class UserContext:
    name: str
    state_id: str
    state_name: str
    business_name: str
    business_category: str
    turnover: str
    preferred_application_mode: str

# Retrieve user information from Streamlit session state
def get_user_context(session_state):
    try:
        user = session_state.user
        return UserContext(
            name=user["fname"],
            state_id=user.get("state_id", "Unknown"),
            state_name=user.get("state_name", "Unknown"),
            business_name=user.get("business_name", "Unknown"),
            business_category=user.get("business_category", "Unknown"),
            turnover=user.get("turnover", "Not Provided"),
            preferred_application_mode=user.get("preferred_application_mode", "Not Provided"),
        )
    except AttributeError:
        logger.error("User data not found in session state")
        return None

# Helper function to detect language
def detect_language(query):
    # Check for Devanagari script (Hindi)
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    if devanagari_pattern.search(query):
        return "Hindi"
    
    # Common Hindi words in Roman script for Hinglish detection
    hindi_words = [
        "kya", "kaise", "ke", "mein", "hai", "kaun", "kahan", "kab",
        "batao", "sarkari", "yojana", "paise", "karobar", "dukaan", "nayi", "naye", "chahiye", "madad", "karo",
        "dikhao", "samjhao", "tarika", "aur", "arey", "bhi", "kya", "hai", "hoga", "hogi", "ho", "hoon", "magar", "lekin", "par", "toh", "ab", "phir", "kuch", "thoda", "zyada", "sab", "koi", "kuchh", "aap", "tum", "main",
        "hum", "unhe", "unko", "unse", "yeh", "woh", "aisa", "aisi", "aise"
    ]
    query_lower = query.lower()
    hindi_word_count = sum(1 for word in hindi_words if word in query_lower)
    total_words = len(query_lower.split())
    
    # If more than 30% of words are Hindi or mixed with English
    if total_words > 0 and hindi_word_count / total_words > 0.25:
        return "Hinglish"
    
    return "English"

# Build conversation history string from stored messages
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
    - The message should welcome the user, mention their state ({state_name}), and offer assistance with schemes and documents applicable to their state and all central government schemes.
    - Response must be ≤50 words.
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

# Step 1: Process user query with RAG
def get_rag_response(query, vector_store, business_category=None, turnover=None, preferred_application_mode=None):
    start_time = time.time()
    try:
        details = []
        if business_category:
            details.append(f"business category: {business_category}")
        if turnover:
            details.append(f"turnover: {turnover}")
        if preferred_application_mode:
            details.append(f"preferred application mode: {preferred_application_mode}")

        full_query = query
        if details:
            full_query = f"{query}. {' '.join(details)}"

        logger.debug(f"Processing query: {full_query}")
        embeddings = get_embeddings()
        embed_start = time.time()
        query_embedding = embeddings.embed_query(full_query)
        logger.debug(f"Query embedding generated in {time.time() - embed_start:.2f} seconds (first 10 values): {query_embedding[:10]}")
        retrieve_start = time.time()
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        result = qa_chain.invoke({"query": full_query})
        logger.debug(f"Retrieval and QA completed in {time.time() - retrieve_start:.2f} seconds")
        response = result["result"]
        sources = result["source_documents"]
        logger.info(f"RAG response generated in {time.time() - start_time:.2f} seconds: {response}")
        if not sources:
            logger.warning(
                f"No documents retrieved for query with profile details: {full_query}."
            )
            if full_query != query:
                logger.info("Retrying RAG search without profile details")
                result = qa_chain.invoke({"query": query})
                response = result["result"]
                sources = result["source_documents"]
        if not sources:
            logger.warning(f"No documents retrieved for query: {query}")
            return "No relevant scheme information found."
        logger.info(f"Retrieved {len(sources)} documents for query: {query}")
        for i, doc in enumerate(sources):
            logger.debug(f"Document {i+1}:")
            logger.debug(f"  Content: {doc.page_content}")
            logger.debug(f"  Metadata: {doc.metadata}")
        return response
    except Exception as e:
        logger.error(f"RAG retrieval failed in {time.time() - start_time:.2f} seconds: {str(e)}")
        return "Error retrieving scheme information."

# Check query similarity for context
def is_query_related(query, prev_response):
    prompt = f"""You are an assistant for Haqdarshak, helping small business owners in India with government schemes, digital/financial literacy, and business growth. Determine if the current query is a follow-up to the previous bot response.

    **Input**:
    - Current Query: {query}
    - Previous Bot Response: {prev_response}

    **Instructions**:
    - A query is a follow-up if it is ambiguous (lacks specific scheme/document/bucket names like 'FSSAI', 'Udyam', 'PMFME', 'GST', 'UPI') and contextually refers to the topic or intent of the previous bot response.
    - Examples of ambiguous queries: 'Tell me more', 'How to apply?', 'What next?', 'Can you help with it?', 'और बताएं', 'आगे क्या?'.
    - The query is NOT a follow-up if it mentions a specific scheme, document, or topic (e.g., 'What is FSSAI?', 'How to use UPI?', 'एफएसएसएआई क्या है?') or is unrelated (e.g., 'What’s the weather?', 'मौसम कैसा है?').
    - Focus only on the previous bot response for context, not the previous query or broader conversation history.
    - Return 'True' if the query is a follow-up, 'False' otherwise.
    - Do not consider rule-based checks like keyword matching or similarity scores.

    **Output**:
    - Return only 'True' or 'False'.
    """

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        result = response.content.strip()
        logger.debug(f"LLM determined query '{query}' is {'related' if result == 'True' else 'not related'} to previous response: {prev_response[:500]}...")
        return result == "True"
    except Exception as e:
        logger.error(f"Failed to determine query relation: {str(e)}")
        return False

# Classify the intent of the user's query
def classify_intent(query, prev_response, conversation_history):
    """Return one of the predefined intent labels."""
    prompt = f"""You are an assistant for Haqdarshak. Classify the user's intent.

    **Input**:
    - Query: {query}
    - Previous Assistant Response: {prev_response}
    - Conversation History: {conversation_history}

    **Instructions**:
    Return only one label from the following:
    Specific_Scheme_Know_Intent,
    Specific_Scheme_Apply_Intent,
    Schemes_Know_Intent,
    Non_Scheme_Know_Intent,
    DFL_Intent,
    Out_of_Scope,
    Contextual_Follow_Up.
    """
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Failed to classify intent: {str(e)}")
        return "Out_of_Scope"

# Generate final response based on intent and RAG output
def generate_response(intent, rag_response, user_info, language, context):
    if intent == "Out_of_Scope":
        if language == "Hindi":
            return "क्षमा करें, मैं केवल सरकारी योजनाओं, डिजिटल या वित्तीय साक्षरता और व्यावसायिक वृद्धि पर मदद कर सकता हूँ।"
        if language == "Hinglish":
            return "Maaf kijiye, main sirf sarkari yojanaon, digital ya financial literacy aur business growth mein madad kar sakta hoon."
        return "Sorry, I can only help with government schemes, digital/financial literacy or business growth."

    prompt = f"""You are a helpful assistant for Haqdarshak assisting small business owners in India.

    **Input**:
    - Intent: {intent}
    - RAG Response: {rag_response}
    - User Name: {user_info.name}
    - State: {user_info.state_name} ({user_info.state_id})
    - Conversation Context: {context}
    - Language: {language}

    **Instructions**:
    Respond in the specified language using a friendly tone and short sentences. Start with 'Hi {user_info.name}!' for English, 'Namaste {user_info.name}!' for Hinglish, or 'नमस्ते {user_info.name}!' for Hindi.
    Keep the answer under 120 words and use the RAG Response to answer based on the intent.
    """
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}")
        if language == "Hindi":
            return "क्षमा करें, मैं आपका प्रश्न संसाधित नहीं कर सका।"
        if language == "Hinglish":
            return "Sorry, main aapka query process nahi kar saka."
        return "Sorry, I couldn't process your query."

# Generate unique interaction ID
def generate_interaction_id(query, timestamp):
    return f"{query[:500]}_{timestamp.strftime('%Y%m%d%H%M%S')}"


# Main function to process query
def process_query(query, scheme_vector_store, dfl_vector_store, session_id, mobile_number, user_language=None):
    start_time = time.time()
    logger.info(f"Starting query processing for: {query}")

    # Retrieve user data from session state using helper
    user_info = get_user_context(st.session_state)
    if not user_info:
        return "Error: User not logged in."
    user_name = user_info.name
    state_id = user_info.state_id
    state_name = user_info.state_name
    business_name = user_info.business_name
    business_category = user_info.business_category
    turnover = user_info.turnover
    preferred_application_mode = user_info.preferred_application_mode

    # Use user_language for welcome message, otherwise detect query language
    query_language = user_language if query.lower() == "welcome" and user_language else detect_language(query)
    logger.info(f"Using query language: {query_language}")

    # Check user type
    conversations = data_manager.get_conversations(mobile_number)
    has_user_messages = False
    for conv in conversations:
        for msg in conv["messages"]:
            if msg["role"] == "user" or (msg["role"] == "assistant" and "Welcome" not in msg["content"]):
                has_user_messages = True
                break
        if has_user_messages:
            break
    user_type = "returning" if has_user_messages else "new"
    logger.info(f"User type: {user_type}")

    profile_complete = data_manager.is_profile_complete(mobile_number)
    missing_fields = data_manager.get_missing_optional_fields(mobile_number)

    # Handle welcome query
    if query.lower() == "welcome":
        if user_type == "new":
            response = welcome_user(state_name, user_name, query_language)
            try:
                interaction_id = generate_interaction_id(response, datetime.utcnow())
                recent_conversations = data_manager.get_conversations(mobile_number)
                if not any(
                    msg["role"] == "assistant" and msg["content"] == response
                    for conv in recent_conversations for msg in conv["messages"]
                ):
                    data_manager.save_conversation(
                        session_id,
                        mobile_number,
                        [
                            {"role": "assistant", "content": response, "timestamp": datetime.utcnow(), "interaction_id": interaction_id}
                        ]
                    )
                    logger.info(f"Saved welcome message for new user in session {session_id} (Interaction ID: {interaction_id})")
                else:
                    logger.debug(f"Skipped saving duplicate welcome message: {response}")
            except Exception as e:
                logger.error(f"Failed to save welcome message for new user in session {session_id}: {str(e)}")
            logger.info(f"Generated welcome response for new user in {time.time() - start_time:.2f} seconds: {response}")
            return response
        else:
            logger.info(f"No welcome message for returning user")
            return None

    # Check if vector stores are valid
    try:
        doc_count = scheme_vector_store.index.ntotal
        logger.info(f"Scheme vector store contains {doc_count} documents")
        if doc_count == 0:
            logger.error("Vector store is empty")
            if query_language == "Hindi":
                return "कोई योजना डेटा उपलब्ध नहीं है। कृपया डेटा स्रोत की जाँच करें।"
            return "No scheme data available. Please check the data source."
        dfl_count = dfl_vector_store.index.ntotal
        logger.info(f"DFL vector store contains {dfl_count} documents")
        if dfl_count == 0:
            logger.error("DFL vector store is empty")
            if query_language == "Hindi":
                return "कोई DFL डेटा उपलब्ध नहीं है। कृपया डेटा स्रोत की जाँच करें।"
            return "No DFL data available. Please check the data source."
    except Exception as e:
        logger.error(f"Vector store check failed: {str(e)}")
        if query_language == "Hindi":
            return "योजना डेटा तक पहुँचने में त्रुटि।"
        return "Error accessing scheme data."

    # Check if query is related to any previous query in the session
    scheme_rag = None
    dfl_rag = None
    related_prev_query = None
    session_cache = st.session_state.rag_cache.get(session_id, {})
    dfl_session_cache = st.session_state.dfl_rag_cache.get(session_id, {})

    # Get the most recent query-response pair from the current session
    recent_query = None
    recent_response = None
    if st.session_state.messages:
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant" and "Welcome" not in msg["content"]:
                recent_response = msg["content"]
                msg_index = st.session_state.messages.index(msg)
                if msg_index > 0 and st.session_state.messages[msg_index - 1]["role"] == "user":
                    recent_query = st.session_state.messages[msg_index - 1]["content"]
                break

    conversation_history = build_conversation_history(st.session_state.messages)
    intent = classify_intent(query, recent_response or "", conversation_history)
    logger.info(f"Classified intent: {intent}")

    scheme_intents = {"Specific_Scheme_Know_Intent", "Specific_Scheme_Apply_Intent", "Schemes_Know_Intent", "Contextual_Follow_Up"}
    dfl_intents = {"DFL_Intent", "Non_Scheme_Know_Intent"}

    if recent_query and recent_response and is_query_related(query, recent_response):
        if intent in scheme_intents:
            scheme_rag = session_cache.get(recent_query, None)
        elif intent in dfl_intents:
            dfl_rag = dfl_session_cache.get(recent_query, None)
        related_prev_query = recent_query

    if scheme_rag is None and intent in scheme_intents:
        scheme_rag = get_rag_response(
            query,
            scheme_vector_store,
            business_category=business_category,
            turnover=turnover,
            preferred_application_mode=preferred_application_mode,
        )
        if session_id not in st.session_state.rag_cache:
            st.session_state.rag_cache[session_id] = {}
        cache_key = f"{query}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        st.session_state.rag_cache[session_id][cache_key] = scheme_rag

    if dfl_rag is None and intent in dfl_intents:
        dfl_rag = get_rag_response(
            query,
            dfl_vector_store,
            business_category=business_category,
            turnover=turnover,
            preferred_application_mode=preferred_application_mode,
        )
        if session_id not in st.session_state.dfl_rag_cache:
            st.session_state.dfl_rag_cache[session_id] = {}
        dfl_cache_key = f"{query}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        st.session_state.dfl_rag_cache[session_id][dfl_cache_key] = dfl_rag

    rag_response = scheme_rag if intent in scheme_intents else dfl_rag
    generated_response = generate_response(intent, rag_response or "", user_info, query_language, conversation_history)

    # Save conversation to MongoDB
    try:
        interaction_id = generate_interaction_id(query, datetime.utcnow())
        recent_conversations = data_manager.get_conversations(mobile_number)
        messages_to_save = [
            {"role": "user", "content": query, "timestamp": datetime.utcnow(), "interaction_id": interaction_id},
            {"role": "assistant", "content": generated_response, "timestamp": datetime.utcnow(), "interaction_id": interaction_id},
        ]
        if not any(
            any(msg.get("interaction_id") == interaction_id for msg in conv["messages"])
            for conv in recent_conversations
        ):
            data_manager.save_conversation(session_id, mobile_number, messages_to_save)
            logger.info(
                f"Saved conversation for session {session_id}: {query} -> {generated_response} (Interaction ID: {interaction_id})"
            )
        else:
            logger.debug(
                f"Skipped saving duplicate conversation for query: {query} (Interaction ID: {interaction_id})"
            )
    except Exception as e:
        logger.error(f"Failed to save conversation for session {session_id}: {str(e)}")

    return generated_response
