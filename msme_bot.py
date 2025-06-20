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
embeddings = get_embeddings()

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
    gender: str

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

def get_system_prompt(language, user_name="User"):

    """Return tone and style instructions."""

    system_rules = f"""1. **Language Handling**:
       - The query language is provided as {language} (English, Hindi, or Hinglish).
       - For Hindi queries, respond in Devanagari script using simple, clear words suitable for micro business owners with low Hindi proficiency.
       - For Hinglish queries, use a natural mix of simple English and Hindi words in Roman script.
       - For English queries, respond in simple English.
       
       2. **Response Guidelines**:
       - Scope: Only respond to queries about government schemes, digital/financial literacy, or business growth.
       - Tone and Style: Use simple, clear words, short sentences, friendly tone, relatable examples.
       - Response must be ≤120 words.
       - Never mention agent fees unless specified in RAG Response for scheme queries.
       - For returning users, use conversation history to maintain context.
       - Start the response with 'Hi {user_name}!' (English), 'Namaste {user_name}!' (Hinglish), or 'नमस्ते {user_name}!' (Hindi) unless Out_of_Scope."""

    system_prompt = system_rules.format(language=language, user_name=user_name)
    return system_prompt

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

# Summarize recent conversation for contextual RAG
def summarize_conversation(messages, current_query: str | None = None, max_pairs: int = 3) -> str:
    """Summarize recent conversation, prioritizing the most recent pairs. Do not include information from older pairs if context has switched (user now talking about different scheme or switched from schemes intent to dfl or vice versa).

    The current query guides what context from previous responses to include.
    """
    history_pairs = []
    pair = []
    for msg in reversed(messages):
        if msg["role"] == "assistant" and "Welcome" in msg["content"]:
            continue
        pair.append(msg)
        if len(pair) == 2:
            if pair[0]["role"] == "assistant" and pair[1]["role"] == "user":
                history_pairs.append((pair[1]["content"], pair[0]["content"]))
            elif pair[0]["role"] == "user" and pair[1]["role"] == "assistant":
                history_pairs.append((pair[0]["content"], pair[1]["content"]))
            pair = []
            if len(history_pairs) >= max_pairs:
                break

    convo_text = "".join(
        f"User: {u}\nAssistant: {a}\n" for u, a in history_pairs
    )
    if not convo_text:
        logger.debug("No complete message pairs to summarize")
        return ""
    logger.debug(f"Summarizing {len(history_pairs)} conversation pairs with query: {current_query}")

    base_prompt = (
        f"Summarize the last {max_pairs} query-response pairs below. "
        "Include only the most recent scheme or DFL topic mentioned and any details already provided. "
        "If multiple schemes or DFL topics appear, keep only the most recent."
    )
    if current_query:
        base_prompt += f" Only include context relevant to the current query: {current_query}."
    base_prompt += " Keep it under 100 words."

    prompt = f"{base_prompt}\n\n{convo_text}\n\nSummary:"

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        summary = response.content.strip()
        logger.debug(f"Conversation summary: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Failed to summarize conversation: {str(e)}")
        return convo_text[:500]

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
def get_rag_response(query, vector_store, conversation_summary=None, state="ALL_STATES", gender=None, business_category=None, turnover=None, preferred_application_mode=None):
    start_time = time.time()
    try:
        details = []
        if state:
            details.append(f"state: {state}")
        if gender:
            details.append(f"gender: {gender}")
        if business_category:
            details.append(f"business category: {business_category}")
        if turnover:
            details.append(f"turnover: {turnover}")
        if preferred_application_mode:
            details.append(f"preferred application mode: {preferred_application_mode}")

        full_query = query
        if conversation_summary:
            full_query = f"{full_query}. Context: {conversation_summary}"
        if details:
            full_query = f"{full_query}. {' '.join(details)}"

        logger.debug(f"Processing query: {full_query}")
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
        return {"text": response, "sources": sources}
    except Exception as e:
        logger.error(f"RAG retrieval failed in {time.time() - start_time:.2f} seconds: {str(e)}")
        return {"text": "Error retrieving scheme information.", "sources": []}


def get_scheme_response(query, vector_store, conversation_summary=None, state="ALL_STATES", gender=None, business_category=None, turnover=None, preferred_application_mode=None):
    """Wrapper for scheme dataset retrieval with clearer logging."""
    logger.info("Querying scheme dataset")
    return get_rag_response(
        query,
        vector_store,
        conversation_summary,
        state=state,
        gender=gender,
        business_category=business_category,
        turnover=turnover,
        preferred_application_mode=preferred_application_mode,
    )


def get_dfl_response(query, vector_store, conversation_summary=None, state="ALL_STATES", gender=None, business_category=None, turnover=None, preferred_application_mode=None):
    """Wrapper for DFL dataset retrieval with clearer logging."""
    logger.info("Querying DFL dataset")
    return get_rag_response(
        query,
        vector_store,
        conversation_summary,
        state=state,
        gender=gender,
        business_category=business_category,
        turnover=turnover,
        preferred_application_mode=preferred_application_mode,
    )

# Check query similarity for context
def is_query_related(query, prev_response):
    prompt = f"""You are an assistant for Haqdarshak, helping small business owners in India with government schemes, digital/financial literacy, and business growth. Determine if the current query is a follow-up to the previous bot response.

    **Input**:
    - Current Query: {query}
    - Previous Bot Response: {prev_response}

    **Instructions**:
    - A query is a related follow-up if it is ambiguous (lacks specific scheme/document/bucket names like 'FSSAI', 'Udyam', 'PMFME', 'GST', 'UPI') and contextually refers to the topic or intent of the previous bot response.
    - Examples of ambiguous queries: 'Tell me more', 'How to apply?', 'What next?', 'Can you help with it?', 'और बताएं', 'आगे क्या?'.
    - The query is not a follow-up if it mentions a specific scheme, document, or topic (e.g., 'What is FSSAI?', 'How to use UPI?', 'एफएसएसएआई क्या है?') or is unrelated (e.g., 'What’s the weather?', 'मौसम कैसा है?').
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
       - Specific_Scheme_Know_Intent (e.g., 'What is FSSAI?', 'PMFME ke baare mein batao', 'एफएसएसएआई क्या है?')
       - Specific_Scheme_Apply_Intent (e.g., 'Apply kaise karna hai', 'How to apply for FSSAI?', 'FSSAI kaise apply karu?', 'एफएसएसएआई के लिए आवेदन कैसे करें?')
       - Specific_Scheme_Eligibility_Intent (e.g., 'Eligibility batao', 'Am I eligible for FSSAI?', 'FSSAI eligibility?', 'एफएसएसएआई की पात्रता क्या है?')
       - Schemes_Know_Intent (e.g., 'Schemes for credit?', 'MSME ke liye schemes kya hain?', 'क्रेडिट के लिए योजनाएं?')
       - Non_Scheme_Know_Intent (e.g., 'How to use UPI?', 'GST kya hai?', 'यूपीआई का उपयोग कैसे करें?')
       - DFL_Intent (digital/financial literacy queries, e.g., 'How to use UPI?', 'UPI kaise use karein?', 'डिजिटल भुगतान कैसे करें?', 'Opening Bank Account', 'Why get Insurance', 'Why take loans', 'Online Safety', 'How can going digital help grow business', etc.)
       - Out_of_Scope (e.g., 'What's the weather?', 'Namaste', 'मौसम कैसा है?')
       - Contextual_Follow_Up (e.g., 'Tell me more', 'Aur batao', 'और बताएं')
       - Confirmation_New_RAG (Only to be chosen when user query is confirmation for initating another RAG search ("Yes", "Haan batao", "Haan dikhao", "Yes search again") AND previous assistant response says that the bot needs to fetch more details about some scheme. ('I need to fetch more details about [scheme name]. Please confirm if this is the scheme you meant.')

    **Tips**:
       - Use rule-based checks for Out_of_Scope (keywords: 'hello', 'hi', 'hey', 'weather', 'time', 'namaste', 'mausam', 'samay'). 
       - For Contextual_Follow_Up, prioritize the Previous Assistant Response for context. If the query refers to a specific part (e.g., 'the first scheme'), identify the referenced scheme or topic.
       - To distinguish between Specific_Scheme_Know_Intent and Scheme_Know_Intent, check for whether query is asking for information about specific scheme or general information about schemes. You can also refer to conversation history to see if the scheme being asked about has already been mentioned by the bot to the user first, in which case the intent is certainly Specific_Scheme_Know_Intent.
    """
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Failed to classify intent: {str(e)}")
        return "Out_of_Scope"

# Generate final response based on intent and RAG output
def generate_response(intent, rag_response, user_info, language, context, scheme_guid=None):
    if intent == "Out_of_Scope":
        if language == "Hindi":
            return "क्षमा करें, मैं केवल सरकारी योजनाओं, डिजिटल या वित्तीय साक्षरता और व्यावसायिक वृद्धि पर मदद कर सकता हूँ।"
        if language == "Hinglish":
            return "Maaf kijiye, main sirf sarkari yojanaon, digital ya financial literacy aur business growth mein madad kar sakta hoon."
        return "Sorry, I can only help with government schemes, digital/financial literacy or business growth."

    tone_prompt = get_system_prompt(language, user_info.name)

    special_schemes = ["Udyam", "FSSAI", "Shop Act"]
    link = "https://haqdarshak.com/contact"

    prompt = f"""You are a helpful assistant for Haqdarshak, supporting small business owners in India with government schemes, digital/financial literacy, and business growth.

    **Input**:
    - Intent: {intent}
    - RAG Response: {rag_response}
    - User Name: {user_info.name}
    - State: {user_info.state_name} ({user_info.state_id})
    - Gender: {user_info.gender}
    - Business Name: {user_info.business_name}
    - Business Category: {user_info.business_category}
    - Turnover: {user_info.turnover}
    - Preferred Application Mode: {user_info.preferred_application_mode}
    - Conversation Context: {context}
    - Language: {language}"""
    if scheme_guid:
        prompt += f"\n    - Scheme GUID: {scheme_guid}"

    prompt += f"""

    **Language Handling and Tone Instructions**:
    {tone_prompt}

    **Task**:
    **Generate Response Based on Intent**:
       - **Specific_Scheme_Know_Intent**: Share scheme name, purpose, benefits from **RAG Response** (≤120 words). Filter for schemes where 'applicability' includes state_id or 'ALL_STATES' or 'scheme type' is 'Centrally Sponsored Scheme' (CSS). List CSS schemes first, then state-specific. Ask: 'Want details on eligibility or how to apply?' (English), 'Eligibility ya apply karne ke baare mein jaanna chahte hain?' (Hinglish), or 'पात्रता या आवेदन करने के बारे में जानना चाहते हैं?' (Hindi).
       - **Specific_Scheme_Apply_Intent**: Share application process from **RAG Response** (≤120 words). Filter for schemes where 'applicability' includes state_id or 'ALL_STATES' or 'scheme type' is 'Centrally Sponsored Scheme' (CSS). For Udyam, FSSAI, or Shop Act, add: 'Haqdarshak can help you get this document for Only ₹99. Click: {link}' (English), 'Haqdarshak aapko yeh document sirf ₹99 mein dilane mein madad kar sakta hai. Click: {link}' (Hinglish), or 'हकदर्शक आपको यह दस्तावेज़ केवल ₹99 में दिलाने में मदद कर सकता है। क्लिक करें: {link}' (Hindi).
       - **Specific_Scheme_Eligibility_Intent**: Summarize eligibility rules from **RAG Response** (≤120 words) and provide a link to check eligibility: https://customer.haqdarshak.com/check-eligibility/{scheme_guid}. Ask the user to verify their eligibility there.
       - **Schemes_Know_Intent**: List schemes from **RAG Response** (2-3 lines each, ≤120 words). Filter for schemes where 'applicability' includes state_id or 'ALL_STATES' or 'scheme type' is 'Centrally Sponsored Scheme' (CSS). Ask: 'Want more details on any scheme?' (English), 'Kisi yojana ke baare mein aur jaanna chahte hain?' (Hinglish), or 'किसी योजना के बारे में और जानना चाहते हैं?' (Hindi).
       - **Non_Scheme_Know_Intent**: Answer using **RAG Response** in simple language (≤120 words). Use examples (e.g., 'Use UPI like PhonePe' or 'UPI ka istemal PhonePe jaise karo' or 'यूपीआई का उपयोग फोनपे की तरह करें'). Use verified external info if needed.
       - **DFL_Intent**: Respond using **RAG Response** in simple language (≤120 words) with relevant examples.
       - **Contextual_Follow_Up**: Use the Previous Assistant Response and Conversation Context to identify the topic. If the RAG Response does not match the referenced scheme, indicate a new RAG search is needed. Provide a relevant follow-up response (≤120 words) using the RAG Response, filtering for schemes where 'applicability' includes state_id or 'scheme type' is 'Centrally Sponsored Scheme' (CSS). If unclear, ask for clarification (e.g., 'Could you specify which scheme?' or 'Kaunsi scheme ke baare mein?' or 'कौन सी योजना के बारे में?').
       - **Confirmation_New_RAG**: If the user confirms to initiate a new RAG search, respond with the details of the scheme they are interested in, refer to conversation context for details.
    **Output**:
       - Return only the final response in the query's language (no intent label or intermediate steps). If a new RAG search is needed, indicate with: 'I need to fetch more details about [scheme name]. Please confirm if this is the scheme you meant.' (English), 'Mujhe [scheme name] ke baare mein aur jaankari leni hogi. Kya aap isi scheme ki baat kar rahe hain?' (Hinglish), or 'मुझे [scheme name] के बारे में और जानकारी लेनी होगी। क्या आप इसी योजना की बात कर रहे हैं?' (Hindi).
       - If RAG Response is empty or 'No relevant scheme information found,' and the query is a Contextual_Follow_Up referring to a specific scheme, indicate a new RAG search is needed. Otherwise, say: 'I don't have information on this right now.' (English), 'Mujhe iske baare mein abhi jaankari nahi hai.' (Hinglish), or 'मुझे इसके बारे में अभी जानकारी नहीं है।' (Hindi).
       - Do not mention any other scheme when a specific scheme is being talked about.
       - When intent is Schemes_Know, do not mention other schemes from past conversation, only the current relevant ones.
       - No need to mention user profile details in every response, only include where contextually relevant.
       - Scheme answers must come only from scheme data, and DFL answers must come from the DFL document. All answers must be given from provided internal data sources.
    """

    try:
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
    gender = user_info.gender

    # Use user_language for welcome message, otherwise detect query language
    query_language = user_language if query.lower() == "welcome" and user_language else detect_language(query)
    logger.info(f"Using query language: {query_language}")

    # Check user type and fetch recent conversations once
    conversations = data_manager.get_conversations(mobile_number, limit=10)
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
                if not any(
                    msg["role"] == "assistant" and msg["content"] == response
                    for conv in conversations for msg in conv["messages"]
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
    conversation_summary = st.session_state.get("conversation_summary", "")
    logger.info(f"Using conversation summary: {conversation_summary}")
    logger.info(f"Classified intent: {intent}")

    scheme_intents = {"Specific_Scheme_Know_Intent", "Specific_Scheme_Apply_Intent", "Specific_Scheme_Eligibility_Intent", "Schemes_Know_Intent", "Contextual_Follow_Up", "Confirmation_New_RAG"}
    dfl_intents = {"DFL_Intent", "Non_Scheme_Know_Intent"}

    if intent in scheme_intents:
        scheme_rag = session_cache.get(query)
    elif intent in dfl_intents:
        dfl_rag = dfl_session_cache.get(query)

    if recent_query and recent_response and is_query_related(query, recent_response):
        if intent in scheme_intents:
            scheme_rag = session_cache.get(recent_query, scheme_rag)
        elif intent in dfl_intents:
            dfl_rag = dfl_session_cache.get(recent_query, dfl_rag)
        related_prev_query = recent_query

    if scheme_rag is None and intent in scheme_intents:
        scheme_rag = get_scheme_response(
            query,
            scheme_vector_store,
            conversation_summary,
            state=state_id,
            gender=gender,
            business_category=business_category,
            turnover=turnover,
            preferred_application_mode=preferred_application_mode,
        )
        if session_id not in st.session_state.rag_cache:
            st.session_state.rag_cache[session_id] = {}
        st.session_state.rag_cache[session_id][query] = scheme_rag

    if dfl_rag is None and intent in dfl_intents:
        dfl_rag = get_dfl_response(
            query,
            dfl_vector_store,
            conversation_summary,
            state=state_id,
            gender=gender,
            business_category=business_category,
            turnover=turnover,
            preferred_application_mode=preferred_application_mode,
        )
        if session_id not in st.session_state.dfl_rag_cache:
            st.session_state.dfl_rag_cache[session_id] = {}
        st.session_state.dfl_rag_cache[session_id][query] = dfl_rag

    rag_response = scheme_rag if intent in scheme_intents else dfl_rag
    rag_text = rag_response.get("text") if isinstance(rag_response, dict) else rag_response
    scheme_guid = None
    if intent == "Specific_Scheme_Eligibility_Intent" and isinstance(rag_response, dict):
        for doc in rag_response.get("sources", []):
            guid = doc.metadata.get("guid") if doc and doc.metadata else None
            if guid:
                scheme_guid = guid
                break
    generated_response = generate_response(
        intent,
        rag_text or "",
        user_info,
        query_language,
        conversation_history,
        scheme_guid=scheme_guid,
    )

    # Save conversation to MongoDB
    try:
        interaction_id = generate_interaction_id(query, datetime.utcnow())
        messages_to_save = [
            {"role": "user", "content": query, "timestamp": datetime.utcnow(), "interaction_id": interaction_id},
            {"role": "assistant", "content": generated_response, "timestamp": datetime.utcnow(), "interaction_id": interaction_id},
        ]
        if not any(
            any(msg.get("interaction_id") == interaction_id for msg in conv["messages"])
            for conv in conversations
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
