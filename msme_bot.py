import logging
import time
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
from fastapi import BackgroundTasks 

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
logger = logging.getLogger(__name__)
logging.getLogger("pymongo").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Initialize DataManager
data_manager = DataManager()

# Initialize cached resources
from functools import lru_cache

@lru_cache(maxsize=1)
def init_llm():
    """Initialise the default LLM client for all tasks except intent classification."""
    logger.info("Initializing LLM client")
    start_time = time.time() 
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    llm = ChatOpenAI(
        model="gpt-4.1-mini-2025-04-14",
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        temperature=0
    )
    logger.info(f"LLM initialized in {time.time() - start_time:.2f} seconds")
    return llm


@lru_cache(maxsize=1)
def init_intent_llm():
    """Initialise a dedicated LLM client for intent classification."""
    logger.info("Initializing Intent LLM client")
    start_time = time.time()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    intent_llm = ChatOpenAI(
        model="gpt-3.5",
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        temperature=0
    )
    logger.info(
        f"Intent LLM initialized in {time.time() - start_time:.2f} seconds"
    )
    return intent_llm

@lru_cache(maxsize=1)
def init_vector_store():
    logger.info("Loading vector store")
    start_time = time.time()
    index_host = os.getenv("PINECONE_SCHEME_HOST")
    if not index_host:
        raise ValueError("PINECONE_SCHEME_HOST environment variable not set")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    try:
        vector_store = load_rag_data(host=index_host, index_name=index_name, version_file="faiss_version.txt")
    except Exception as e:
        logger.error(f"Failed to load scheme index: {e}")
        raise
    logger.info(f"Vector store loaded in {time.time() - start_time:.2f} seconds")
    return vector_store

@lru_cache(maxsize=1)
def init_dfl_vector_store():
    logger.info("Loading DFL vector store")
    start_time = time.time()
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
    logger.info(
        f"DFL vector store loaded in {time.time() - start_time:.2f} seconds"
    )
    return vector_store

llm = init_llm()
intent_llm = init_intent_llm()
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
    gender: str


class SessionData:
    """Simple container for per-session information."""
    def __init__(self, user=None):
        self.user = user
        self.messages = []
        self.rag_cache = {}
        self.dfl_rag_cache = {}

# Retrieve user information from the session data
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
    
    # If more than 30% of words are Hindi or mixed with English
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

# Step 1: Process user query with RAG
def get_rag_response(query, vector_store, state="ALL_STATES", gender=None, business_category=None):
    start_time = time.time()
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
        retriever = PineconeRecordRetriever(
            index=vector_store, state=state, gender=gender, k=5
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        result = qa_chain.invoke({"query": full_query})
        response = result["result"]
        sources = result["source_documents"]
        logger.info(
            f"RAG response generated in {time.time() - start_time:.2f} seconds: {response}"
        )
        if not sources:
            logger.warning(f"No documents retrieved for query: {full_query}")
        else:
            logger.info(f"Retrieved {len(sources)} documents for query: {full_query}")
            # for i, doc in enumerate(sources):
                # logger.debug(f"Document {i+1}:")
                # logger.debug(f"  Content: {doc.page_content}")
                # logger.debug(f"  Metadata: {doc.metadata}")
        return {"text": response, "sources": sources}
    except Exception as e:
        logger.error(
            f"RAG retrieval failed in {time.time() - start_time:.2f} seconds: {str(e)}"
        )
        return {"text": "Error retrieving scheme information.", "sources": []}


def get_scheme_response(
    query,
    vector_store,
    state="ALL_STATES",
    gender=None,
    business_category=None,
    include_mudra=False,
    intent=None,
):
    """Retrieve scheme info, using keyword lookup for popular schemes."""
    logger.info("Querying scheme dataset")

    guid = None
    if intent == "Specific_Scheme_Know_Intent":
        guid = find_scheme_guid_by_query(query)

    if guid:
        logger.info(f"Directly fetching scheme details for GUID {guid}")
        docs = fetch_scheme_docs_by_guid(guid, vector_store)
        if docs:
            retriever = DocumentListRetriever(docs)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
            result = qa_chain.invoke({"query": query})
            rag = {"text": result["result"], "sources": result["source_documents"]}
        else:
            logger.warning("No documents fetched by GUID; falling back to search")
            guid = None

    if not guid:
        rag = get_rag_response(
            query,
            vector_store,
            state=state,
            gender=gender,
            business_category=business_category,
        )

    if include_mudra:
        logger.info("Fetching Pradhan Mantri Mudra Yojana details")
        mudra_guid = find_scheme_guid_by_query("pradhan mantri mudra yojana") or "SH0008BK"
        mudra_docs = fetch_scheme_docs_by_guid(mudra_guid, vector_store)

        if mudra_docs:
            retriever = DocumentListRetriever(mudra_docs)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
            result = qa_chain.invoke({"query": "Pradhan Mantri Mudra Yojana"})
            mudra_rag = {"text": result["result"], "sources": result["source_documents"]}
        else:
            logger.warning("Mudra documents not found; skipping")
            mudra_rag = {"text": "", "sources": []}

        if not isinstance(rag, dict):
            rag = {"text": str(rag), "sources": []}

        rag["text"] = f"{rag.get('text', '')}\n{mudra_rag.get('text', '')}"
        rag["sources"] = rag.get("sources", []) + mudra_rag.get("sources", [])

    return rag


def get_dfl_response(query, vector_store, state=None, gender=None, business_category=None):
    """Wrapper for DFL dataset retrieval with clearer logging."""
    logger.info("Querying DFL dataset")
    return get_rag_response(
        query,
        vector_store,
        state=None,
        gender=gender,
        business_category=business_category,
    )

# Classify the intent of the user's query
def classify_intent(query, conversation_history=""):
    """Return one of the predefined intent labels."""

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
        response = intent_llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Failed to classify intent: {str(e)}")
        return "Out_of_Scope"

# Generate final response based on intent and RAG output
def generate_response(intent, rag_response, user_info, language, context, query, scheme_guid=None, stream: bool = False):
    if intent == "Out_of_Scope":
        if language == "Hindi":
            return "क्षमा करें, मैं केवल सरकारी योजनाओं, डिजिटल या वित्तीय साक्षरता और व्यावसायिक वृद्धि पर मदद कर सकता हूँ।"
        if language == "Hinglish":
            return "Maaf kijiye, main sirf sarkari yojanaon, digital ya financial literacy aur business growth mein madad kar sakta hoon."
        return "Sorry, I can only help with government schemes, digital/financial literacy or business growth."

    if intent == "Gratitude_Intent":
        gratitude_prompt = f"""You are a friendly assistant for Haqdarshak. The user {user_info.name} has thanked you.

        **Instructions**:
        - Respond briefly in the same language ({language}) acknowledging the thanks and offering further help.
        - Use Devanagari script for Hindi and a natural mix of Hindi and English words in Roman script for Hinglish.
        - Keep the message under 30 words.

        **Output**:
        - Only the acknowledgement message in the user's language."""
        try:
            response = llm.invoke([{"role": "user", "content": gratitude_prompt}])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate gratitude response: {str(e)}")
            if language == "Hindi":
                return "धन्यवाद! क्या मैं और मदद कर सकता हूँ?"
            if language == "Hinglish":
                return "Thanks! Kya main aur madad kar sakta hoon?"
            return "You're welcome! Let me know if you need anything else."

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
            def token_generator():
                buffer = ""
                try:
                    for chunk in llm.stream([{"role": "user", "content": prompt}]):
                        token = chunk.content or ""
                        buffer += token
                        if token:
                            yield token
                except Exception as e:
                    logger.error(f"Failed to stream response: {str(e)}")
                    return

                if intent == "Specific_Scheme_Eligibility_Intent" and scheme_guid:
                    screening_link = f"https://customer.haqdarshak.com/check-eligibility/{scheme_guid}"
                    if screening_link not in buffer:
                        for char in "\n" + screening_link:
                            yield char

            return token_generator()
        else:
            response = llm.invoke([{"role": "user", "content": prompt}])
            final_text = response.content.strip()
            if intent == "Specific_Scheme_Eligibility_Intent" and scheme_guid:
                screening_link = f"https://customer.haqdarshak.com/check-eligibility/{scheme_guid}"
                if screening_link not in final_text:
                    final_text += f"\n{screening_link}"
            return final_text
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
        # logger.info(f"Generated Hindi audio script: {hindi_script}")
        return hindi_script
    except Exception as e:
        logger.error(f"Failed to generate Hindi audio script: {str(e)}")
        # Fallback: Attempt to translate the original response to Hindi as a last resort
        try:
            translation_prompt = f"Translate the following text into simple Hindi (Devanagari script), removing all punctuation and hyphens for a smooth audio output: {original_response}"
            translation_response = llm.invoke([{"role": "user", "content": translation_prompt}])
            hindi_script = translation_response.content.strip()
            # logger.warning(f"Falling back to direct translation for Hindi audio script: {hindi_script}")
            return hindi_script
        except Exception as inner_e:
            logger.error(f"Failed to fall back to direct translation: {str(inner_e)}")
            return "ऑडियो स्क्रिप्ट उत्पन्न करने में त्रुटि हुई है।" # Error generating audio script.

# Main function to process query
# def process_query(query, scheme_vector_store, dfl_vector_store, session_id, mobile_number, session_data: SessionData, user_language=None, stream: bool = False):
#     start_time = time.time()
#     timings = {}

#     def record(step_name, start):
#         timings[step_name] = time.time() - start

#     def log_timings():
#         total = time.time() - start_time
#         summary = "\n".join(f"{k}: {v:.2f}s" for k, v in timings.items())
#         summary += f"\nTotal: {total:.2f}s"
#         logger.info("Query processing timings:\n" + summary)

#     logger.info(f"Starting query processing for: {query}")

#     # Retrieve user data from session state using helper
#     step = time.time()
#     user_info = get_user_context(session_data)
#     record("user_context", step)
#     logger.info(f"User context in process_query: {user_info}")
#     if not user_info:
#         log_timings()
#         return "Error: User not logged in.", None  # Return tuple
#     user_name = user_info.name
#     state_id = user_info.state_id
#     state_name = user_info.state_name
#     business_category = user_info.business_category
#     gender = user_info.gender


#     # Use user_language for welcome message, otherwise detect query language
#     step = time.time()
#     query_language = user_language if query.lower() == "welcome" and user_language else detect_language(query)
#     record("language_detection", step)
#     logger.info(f"Using query language: {query_language}")

#     # Check user type and fetch recent conversations once
#     step = time.time()
#     conversations = data_manager.get_conversations(mobile_number)
#     user_type = "returning" if conversations else "new"
#     record("fetch_conversations", step)
#     logger.info(f"User type: {user_type}")



#     # Handle welcome query
#     if query.lower() == "welcome":
#         if user_type == "new":
#             response = welcome_user(state_name, user_name, query_language)

#             def audio_task(final_text=None):
#                 step_local = time.time()
#                 hindi_audio_script = generate_hindi_audio_script(
#                     response,
#                     user_info,
#                     "",
#                 )
#                 record("audio_script", step_local)
#                 try:
#                     interaction_id = generate_interaction_id(response, datetime.utcnow())
#                     if not any(
#                         msg["role"] == "assistant" and msg["content"] == response
#                         for conv in conversations for msg in conv["messages"]
#                     ):
#                         data_manager.save_conversation(
#                             session_id,
#                             mobile_number,
#                             [
#                                 {"role": "assistant", "content": response, "timestamp": datetime.utcnow(), "interaction_id": interaction_id, "audio_script": hindi_audio_script}
#                             ]
#                         )
#                         logger.info(f"Saved welcome message for new user in session {session_id} (Interaction ID: {interaction_id})")
#                     else:
#                         logger.debug(f"Skipped saving duplicate welcome message: {response}")
#                 except Exception as e:
#                     logger.error(f"Failed to save welcome message for new user in session {session_id}: {str(e)}")
#                 log_timings()
#                 return hindi_audio_script

#             logger.info(f"Generated welcome response for new user in {time.time() - start_time:.2f} seconds: {response}")
#             if stream:
#                 def gen():
#                     for ch in response:
#                         yield ch
#                 return gen(), audio_task
#             return response, audio_task
#         else:
#             logger.info(f"No welcome message for returning user")
#             log_timings()
#             return None, None

#     # Check if query is related to any previous query in the session
#     scheme_rag = None
#     dfl_rag = None
#     related_prev_query = None
#     session_cache = session_data.rag_cache
#     dfl_session_cache = session_data.dfl_rag_cache

#     # Get the most recent query-response pair from the current session
#     recent_query = None
#     recent_response = None
#     if session_data.messages:
#         for msg in reversed(session_data.messages):
#             if msg["role"] == "assistant" and "Welcome" not in msg["content"]:
#                 recent_response = msg["content"]
#                 msg_index = session_data.messages.index(msg)
#                 if msg_index > 0 and session_data.messages[msg_index - 1]["role"] == "user":
#                     recent_query = session_data.messages[msg_index - 1]["content"]
#                 break



#     # Build conversation history for intent classification
#     conversation_history = build_conversation_history(session_data.messages)
#     step = time.time()
#     intent = classify_intent(query, conversation_history)
#     record("intent_classification", step)

#     # Determine if the query is a follow-up based on intent
#     follow_up_intents = {
#         "Contextual_Follow_Up",
#         "Specific_Scheme_Eligibility_Intent",
#         "Specific_Scheme_Apply_Intent",
#         "Confirmation_New_RAG",
#     }
#     follow_up = intent in follow_up_intents

#     # Use conversation context only when the query is a follow-up
#     context_pair = f"User: {recent_query}\nAssistant: {recent_response}" if follow_up else ""

#     augmented_query = query
#     logger.info(f"Using conversation context: {context_pair}")
#     logger.info(f"Classified intent: {intent}")


#     # Append previous interaction for context when required
#     if intent in {
#         "Contextual_Follow_Up",
#         "Specific_Scheme_Eligibility_Intent",
#         "Specific_Scheme_Apply_Intent",
#         "Confirmation_New_RAG",
#     }:
#         if recent_query and recent_response:
#             augmented_query = (
#                 f"{augmented_query}. Previous User Query: {recent_query}. Previous Assistant Response: {recent_response}"
#             )


#     scheme_intents = {"Specific_Scheme_Know_Intent", "Specific_Scheme_Apply_Intent", "Specific_Scheme_Eligibility_Intent", "Schemes_Know_Intent", "Contextual_Follow_Up", "Confirmation_New_RAG"}
#     dfl_intents = {"DFL_Intent", "Non_Scheme_Know_Intent"}

#     if intent in scheme_intents:
#         if intent == "Specific_Scheme_Know_Intent":
#             step = time.time()
#             scheme_rag = get_scheme_response(
#                 query,
#                 scheme_vector_store,
#                 state=None,
#                 gender=None,
#                 business_category=None,
#                 include_mudra=False,
#                 intent=intent,
#             )
#             record("rag_retrieval", step)
#         else:
#             scheme_rag = session_cache.get(query)
#     elif intent in dfl_intents:
#         dfl_rag = dfl_session_cache.get(query)

#     if follow_up and recent_query and recent_response:
#         if intent in scheme_intents:
#             scheme_rag = session_cache.get(recent_query, scheme_rag)
#         elif intent in dfl_intents:
#             dfl_rag = dfl_session_cache.get(recent_query, dfl_rag)
#         related_prev_query = recent_query

#     if scheme_rag is None and intent in scheme_intents:
#         step = time.time()
#         scheme_rag = get_scheme_response(
#             augmented_query,
#             scheme_vector_store,
#             state=state_id,
#             gender=gender,
#             business_category=business_category,
#             include_mudra=intent == "Schemes_Know_Intent",
#             intent=intent,
#         )
#         record("rag_retrieval", step)
#         session_data.rag_cache[query] = scheme_rag

#     if dfl_rag is None and intent in dfl_intents:
#         step = time.time()
#         dfl_rag = get_dfl_response(
#             query,
#             dfl_vector_store,
#             state=state_id,
#             gender=gender,
#             business_category=business_category,
#         )
#         record("rag_retrieval", step)
#         session_data.dfl_rag_cache[query] = dfl_rag

#     rag_response = scheme_rag if intent in scheme_intents else dfl_rag
#     rag_text = rag_response.get("text") if isinstance(rag_response, dict) else rag_response
#     if intent == "DFL_Intent" and (
#         rag_text is None or "No relevant" in rag_text
#     ):
#         rag_text = ""
#     scheme_guid = None
#     if isinstance(rag_response, dict) and intent == "Specific_Scheme_Eligibility_Intent":
#         scheme_guid = extract_scheme_guid(rag_response.get("sources", []))
#     step = time.time()
#     gen_result = generate_response(
#         intent,
#         rag_text or "",
#         user_info,
#         query_language,
#         context_pair,
#         query,
#         scheme_guid=scheme_guid,
#         stream=stream,
#     )
#     record("generate_response", step)

#     if not stream:
#         generated_response = gen_result

#     def audio_task(final_text=None):
#         text_for_use = final_text if stream else generated_response
#         step_local = time.time()
#         hindi_audio_script = generate_hindi_audio_script(
#             text_for_use,
#             user_info,
#             rag_text or "",
#         )
#         record("audio_script", step_local)


#         try:
#             interaction_id = generate_interaction_id(query, datetime.utcnow())
#             messages_to_save = [
#                 {"role": "user", "content": query, "timestamp": datetime.utcnow(), "interaction_id": interaction_id},
#                 {"role": "assistant", "content": text_for_use, "timestamp": datetime.utcnow(), "interaction_id": interaction_id, "audio_script": hindi_audio_script},
#             ]
#             if not any(
#                 any(msg.get("interaction_id") == interaction_id for msg in conv["messages"])
#                 for conv in conversations
#             ):
#                 step_db = time.time()
#                 data_manager.save_conversation(session_id, mobile_number, messages_to_save)
#                 record("save_conversation", step_db)
#                 logger.info(
#                     f"Saved conversation for session {session_id}: {query} -> {text_for_use} (Interaction ID: {interaction_id})"
#                 )
#             else:
#                 logger.debug(
#                     f"Skipped saving duplicate conversation for query: {query} (Interaction ID: {interaction_id})"
#                 )
#         except Exception as e:
#             logger.error(f"Failed to save conversation for session {session_id}: {str(e)}")

#         log_timings()
#         return hindi_audio_script

#     if stream:
#         return gen_result, audio_task
#     return generated_response, audio_task

# 

def process_query(query, scheme_vector_store, dfl_vector_store, session_id, mobile_number, session_data: SessionData, user_language=None, stream: bool = False, background_tasks: BackgroundTasks = None):
    import time
    from datetime import datetime
    
    start_time = time.time()
    timings = {}

    def record(step_name, start):
        timings[step_name] = time.time() - start

    def log_timings():
        total = time.time() - start_time
        summary = "\n".join(f"{k}: {v:.2f}s" for k, v in timings.items())
        summary += f"\nTotal: {total:.2f}s"
        logger.info("Query processing timings:\n" + summary)

    # logger.info(f"Starting query processing for: {query}")

    step = time.time()
    user_info = get_user_context(session_data)
    record("user_context", step)

    if not user_info:
        log_timings()
        return "Error: User not logged in.", None

    user_name = user_info.name
    state_id = user_info.state_id
    state_name = user_info.state_name
    business_category = user_info.business_category
    gender = user_info.gender

    step = time.time()
    query_language = user_language if query.lower() == "welcome" and user_language else detect_language(query)
    record("language_detection", step)

    step = time.time()
    CONVERSATION_CACHE={}
    conversations = CONVERSATION_CACHE.get(mobile_number)
    if not conversations:
        conversations = data_manager.get_conversations(mobile_number)
        CONVERSATION_CACHE[mobile_number] = conversations
    record("fetch_conversations", step)

    step = time.time()
    intent = classify_intent(query, build_conversation_history(session_data.messages))
    record("intent_classification", step)

    follow_up_intents = {"Contextual_Follow_Up", "Specific_Scheme_Eligibility_Intent", "Specific_Scheme_Apply_Intent", "Confirmation_New_RAG"}
    recent_query, recent_response = None, None
    if session_data.messages:
        for msg in reversed(session_data.messages):
            if msg["role"] == "assistant" and "Welcome" not in msg["content"]:
                recent_response = msg["content"]
                idx = session_data.messages.index(msg)
                if idx > 0 and session_data.messages[idx - 1]["role"] == "user":
                    recent_query = session_data.messages[idx - 1]["content"]
                break

    context_pair = f"User: {recent_query}\nAssistant: {recent_response}" if intent in follow_up_intents else ""
    augmented_query = f"{query}. Previous User Query: {recent_query}. Previous Assistant Response: {recent_response}" if intent in follow_up_intents and recent_query else query

    scheme_rag, dfl_rag = None, None

    if intent in {"Specific_Scheme_Know_Intent", "Specific_Scheme_Apply_Intent", "Specific_Scheme_Eligibility_Intent", "Schemes_Know_Intent", "Contextual_Follow_Up", "Confirmation_New_RAG"}:
        step = time.time()
        scheme_rag = get_scheme_response(
            augmented_query,
            scheme_vector_store,
            state=state_id,
            gender=gender,
            business_category=business_category,
            include_mudra=intent == "Schemes_Know_Intent",
            intent=intent,
        )
        record("rag_retrieval", step)
        session_data.rag_cache[query] = scheme_rag
    elif intent in {"DFL_Intent", "Non_Scheme_Know_Intent"}:
        step = time.time()
        dfl_rag = get_dfl_response(
            query,
            dfl_vector_store,
            state=state_id,
            gender=gender,
            business_category=business_category,
        )
        record("rag_retrieval", step)
        session_data.dfl_rag_cache[query] = dfl_rag

    rag_text = (scheme_rag or dfl_rag or {}).get("text") if isinstance((scheme_rag or dfl_rag), dict) else (scheme_rag or dfl_rag)

    scheme_guid = extract_scheme_guid(scheme_rag.get("sources", [])) if intent == "Specific_Scheme_Eligibility_Intent" and isinstance(scheme_rag, dict) else None

    step = time.time()
    gen_result = generate_response(
        intent,
        rag_text or "",
        user_info,
        query_language,
        context_pair,
        query,
        scheme_guid=scheme_guid,
        stream=stream,
    )
    record("generate_response", step)

    def audio_task(final_text=None):
        step_local = time.time()
        text_for_use = final_text if stream else gen_result
        hindi_audio_script = generate_hindi_audio_script(text_for_use, user_info, rag_text or "")
        record("audio_script", step_local)

        if background_tasks:
            background_tasks.add_task(data_manager.save_conversation, session_id, mobile_number, [
                {"role": "user", "content": query, "timestamp": datetime.utcnow()},
                {"role": "assistant", "content": text_for_use, "timestamp": datetime.utcnow(), "audio_script": hindi_audio_script},
            ])
        log_timings()
        return hindi_audio_script

    if stream:
        return gen_result, audio_task
    return gen_result, audio_task