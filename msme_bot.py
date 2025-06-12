import logging
import time
from datetime import datetime, timedelta
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

# Generate unique interaction ID
def generate_interaction_id(query, timestamp):
    return f"{query[:500]}_{timestamp.strftime('%Y%m%d%H%M%S')}"


# Main function to process query
def process_query(query, scheme_vector_store, dfl_vector_store, session_id, mobile_number, user_language=None):
    start_time = time.time()
    logger.info(f"Starting query processing for: {query}")
    
    # Retrieve user data from session state
    try:
        user = st.session_state.user
        user_name = user["fname"]
        state_id = user.get("state_id", "Unknown")
        state_name = user.get("state_name", "Unknown")
        business_name = user.get("business_name", "Unknown")
        business_category = user.get("business_category", "Unknown")
        turnover = user.get("turnover", "Not Provided")
        preferred_application_mode = user.get("preferred_application_mode", "Not Provided")
    except AttributeError:
        logger.error("User data not found in session state")
        return "Error: User not logged in."

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

    # Check if the current query is a follow-up to the most recent response
    if recent_query and recent_response and is_query_related(query, recent_response):
        scheme_rag = session_cache.get(recent_query, None)
        dfl_rag = dfl_session_cache.get(recent_query, None)
        related_prev_query = recent_query
        logger.info(f"Using cached RAG response from recent query: {recent_query}")

    # If no cached response is found or query is not a follow-up, perform RAG search
    if scheme_rag is None:
        scheme_rag = get_rag_response(
            query,
            scheme_vector_store,
            business_category=business_category,
            turnover=turnover,
            preferred_application_mode=preferred_application_mode
        )
        if session_id not in st.session_state.rag_cache:
            st.session_state.rag_cache[session_id] = {}
        cache_key = f"{query}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        st.session_state.rag_cache[session_id][cache_key] = scheme_rag
        logger.info(f"Stored new RAG response for query: {query} with key: {cache_key}")

    if dfl_rag is None:
        dfl_rag = get_rag_response(
            query,
            dfl_vector_store,
            business_category=business_category,
            turnover=turnover,
            preferred_application_mode=preferred_application_mode
        )
        if session_id not in st.session_state.dfl_rag_cache:
            st.session_state.dfl_rag_cache[session_id] = {}
        dfl_cache_key = f"{query}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        st.session_state.dfl_rag_cache[session_id][dfl_cache_key] = dfl_rag
        logger.info(f"Stored new DFL RAG response for query: {query} with key: {dfl_cache_key}")

    # Process query and RAG response with a single LLM call
    special_schemes = ["Udyam", "FSSAI", "Shop Act"]
    link = "https://haqdarshak.com/contact"
    
    # Prepare conversation history for prompt
    conversation_history = ""
    session_messages = []
    for msg in st.session_state.messages[-10:]:
        if msg["role"] == "assistant" and "Welcome" in msg["content"]:
            continue
        session_messages.append((msg["role"], msg["content"], msg["timestamp"]))
    session_messages = sorted(session_messages, key=lambda x: x[2], reverse=True)[:5]
    for role, content, _ in session_messages:
        conversation_history += f"{role.capitalize()}: {content}\n"

    prompt = f"""You are a helpful assistant for Haqdarshak, supporting small business owners in India with government schemes, digital/financial literacy, and business growth. The user is a {user_type} user named {user_name} from {state_name} (state_id: {state_id}). The user's business is called {business_name} in the {business_category} category. Annual turnover: {turnover}. Preferred application mode: {preferred_application_mode}.

    **Input**:
    - Query: {query}
    - Query Language: {query_language}
    - Business Name: {business_name}
    - Business Category: {business_category}
    - Turnover: {turnover}
    - Preferred Application Mode: {preferred_application_mode}
    - Scheme RAG Response: {scheme_rag}
    - DFL RAG Response: {dfl_rag}
    - Related Previous Query (if any): {related_prev_query if related_prev_query else 'None'}
    - Most Recent Assistant Response: {recent_response if 'recent_response' in locals() else 'None'}
    - Conversation History (last 5 query-response pairs, excluding welcome messages): {conversation_history}
    - Cached Scheme RAG Responses for Session: {st.session_state.rag_cache.get(session_id, {})}
    - Cached DFL RAG Responses for Session: {st.session_state.dfl_rag_cache.get(session_id, {})}

    **Instructions**:
    1. **Language Handling**:
       - The query language is provided as {query_language} (English, Hindi, or Hinglish).
       - For Hindi queries, respond in Devanagari script using simple, clear words suitable for micro business owners with low Hindi proficiency.
       - For Hinglish queries, use a natural mix of simple English and Hindi words in Roman script.
       - For English queries, respond in simple English.
       - Ensure responses use short sentences, a friendly tone, and relatable examples.

    2. **Classify the Query Intent**:
       - Specific_Scheme_Know_Intent (e.g., 'What is FSSAI?', 'PMFME ke baare mein batao', 'एफएसएसएआई क्या है?')
       - Specific_Scheme_Apply_Intent (e.g., 'How to apply for FSSAI?', 'FSSAI kaise apply karu?', 'एफएसएसएआई के लिए आवेदन कैसे करें?')
       - Schemes_Know_Intent (e.g., 'Schemes for credit?', 'MSME ke liye schemes kya hain?', 'क्रेडिट के लिए योजनाएं?')
       - Non_Scheme_Know_Intent (e.g., 'How to use UPI?', 'GST kya hai?', 'यूपीआई का उपयोग कैसे करें?')
       - DFL_Intent (digital/financial literacy topics)
       - Out_of_Scope (e.g., 'What’s the weather?', 'Namaste', 'मौसम कैसा है?')
       - Contextual_Follow_Up (e.g., 'Tell me more', 'Aur batao', 'और बताएं')
       - Use rule-based checks for Out_of_Scope (keywords: 'hello', 'hi', 'hey', 'weather', 'time', 'namaste', 'mausam', 'samay'). For Contextual_Follow_Up, prioritize the Most Recent Assistant Response for context. If the query refers to a specific part (e.g., 'the first scheme'), identify the referenced scheme or topic.

    3. **Generate Response Based on Intent**:
       - **Out_of_Scope**: Return: 'Sorry, I can only help with government schemes, digital/financial literacy, or business growth. Please ask about these topics.' (English), 'Maaf kijiye, main sirf sarkari yojanaon, digital/financial literacy, ya business growth ke baare mein madad kar sakta hoon. In vishayon ke baare mein poochhein.' (Hinglish), or 'क्षमा करें, मैं केवल सरकारी योजनाओं, डिजिटल/वित्तीय साक्षरता, या व्यवसाय वृद्धि के बारे में मदद कर सकता हूँ। कृपया इन विषयों के बारे में पूछें।' (Hindi).
       - **Specific_Scheme_Know_Intent**: Share scheme name, purpose, benefits from **Scheme RAG Response** (≤100 words). Filter for schemes where 'applicability' includes {state_id} or 'scheme type' is 'Centrally Sponsored Scheme' (CSS). List CSS schemes first, then state-specific. Ask: 'Want details on eligibility or how to apply?' (English), 'Eligibility ya apply karne ke baare mein jaanna chahte hain?' (Hinglish), or 'पात्रता या आवेदन करने के बारे में जानना चाहते हैं?' (Hindi).
       - **Specific_Scheme_Apply_Intent**: Share application process from **Scheme RAG Response** (≤100 words). Filter for schemes where 'applicability' includes {state_id} or 'scheme type' is 'Centrally Sponsored Scheme' (CSS). For Udyam, FSSAI, or Shop Act, add: 'Haqdarshak can help you get this document for Only ₹99. Click: {link}' (English), 'Haqdarshak aapko yeh document sirf ₹99 mein dilane mein madad kar sakta hai. Click: {link}' (Hinglish), or 'हकदर्शक आपको यह दस्तावेज़ केवल ₹99 में दिलाने में मदद कर सकता है। क्लिक करें: {link}' (Hindi).
       - **Schemes_Know_Intent**: List schemes from **Scheme RAG Response** (2–3 lines each, ≤100 words). Filter for schemes where 'applicability' includes {state_id} or 'scheme type' is 'Centrally Sponsored Scheme' (CSS). Ask: 'Want more details on any scheme?' (English), 'Kisi yojana ke baare mein aur jaanna chahte hain?' (Hinglish), or 'किसी योजना के बारे में और जानना चाहते हैं?' (Hindi).
       - **Non_Scheme_Know_Intent**: Answer using **DFL RAG Response** in simple language (≤100 words). Use examples (e.g., 'Use UPI like PhonePe' or 'UPI ka istemal PhonePe jaise karo' or 'यूपीआई का उपयोग फोनपे की तरह करें'). Use verified external info if needed.
       - **DFL_Intent**: Respond using **DFL RAG Response** in simple language (≤100 words) with relevant examples.
       - **Contextual_Follow_Up**: Use the Most Recent Assistant Response to identify the topic. If the RAG Response does not match the referenced scheme, indicate a new RAG search is needed. Provide a relevant follow-up response (≤100 words) using the RAG Response, filtering for schemes where 'applicability' includes {state_id} or 'scheme type' is 'Centrally Sponsored Scheme' (CSS). If unclear, ask for clarification (e.g., 'Could you specify which scheme?' or 'Kaunsi scheme ke baare mein?' or 'कौन सी योजना के बारे में?').
       - If RAG Response is empty or 'No relevant scheme information found,' and the query is a Contextual_Follow_Up referring to a specific scheme, indicate a new RAG search is needed. Otherwise, say: 'I don’t have information on this right now.' (English), 'Mujhe iske baare mein abhi jaankari nahi hai.' (Hinglish), or 'मुझे इसके बारे में अभी जानकारी नहीं है।' (Hindi).

    **Response Guidelines**:
    - Scope: Only respond to queries about government schemes, digital/financial literacy, or business growth.
    - Tone and Style: Use simple, clear words, short sentences, friendly tone, relatable examples.
    - Core Rules:
       - For scheme queries, ONLY use **Scheme RAG Response** or cached scheme responses.
       - For DFL queries, ONLY use **DFL RAG Response**.
       - Ensure scheme-related responses only include schemes where 'applicability' includes {state_id} or 'scheme type' is 'Centrally Sponsored Scheme' (CSS).
       - List CSS schemes first, followed by state-specific schemes.
       - Response must be ≤120 words.
       - Never mention agent fees unless specified in RAG Response.
       - For returning users, use conversation history to maintain context.
       - Start the response with 'Hi {user_name}!' (English), 'Namaste {user_name}!' (Hinglish), or 'नमस्ते {user_name}!' (Hindi) unless Out_of_Scope.

    **Output**:
    - Return only the final response in the query's language (no intent label or intermediate steps). If a new RAG search is needed, indicate with: 'I need to fetch more details about [scheme name]. Please confirm if this is the scheme you meant.' (English), 'Mujhe [scheme name] ke baare mein aur jaankari leni hogi. Kya aap isi scheme ki baat kar rahe hain?' (Hinglish), or 'मुझे [scheme name] के बारे में और जानकारी लेनी होगी। क्या आप इसी योजना की बात कर रहे हैं?' (Hindi).
    - Scheme answers must come only from scheme data, and DFL answers must come from the DFL document.
    """

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        generated_response = response.content.strip()
        logger.debug(f"LLM prompt: {prompt}")
        logger.debug(f"LLM response: {generated_response}")
        logger.info(f"Generated response in {time.time() - start_time:.2f} seconds: {generated_response}")

        # Handle new RAG search if needed
        if "I need to fetch more details about" in generated_response:
            match = re.search(r"I need to fetch more details about (.+?)\. Please confirm", generated_response)
            if match:
                scheme_name = match.group(1)
                logger.info(f"LLM indicated new RAG search needed for scheme: {scheme_name}")
                scheme_rag = get_rag_response(
                    scheme_name,
                    scheme_vector_store,
                    business_category=business_category,
                    turnover=turnover,
                    preferred_application_mode=preferred_application_mode
                )
                cache_key = f"{scheme_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
                st.session_state.rag_cache[session_id][cache_key] = scheme_rag
                logger.info(f"Stored new RAG response for scheme: {scheme_name} with key: {cache_key}")
                prompt = prompt.replace(f"Scheme RAG Response: {scheme_rag}", f"Scheme RAG Response: {scheme_rag}")
                response = llm.invoke([{"role": "user", "content": prompt}])
                generated_response = response.content.strip()
                logger.info(f"Generated response after new RAG search: {generated_response}")

        # Save conversation to MongoDB
        try:
            interaction_id = generate_interaction_id(query, datetime.utcnow())
            recent_conversations = data_manager.get_conversations(mobile_number)
            messages_to_save = [
                {"role": "user", "content": query, "timestamp": datetime.utcnow(), "interaction_id": interaction_id},
                {"role": "assistant", "content": generated_response, "timestamp": datetime.utcnow(), "interaction_id": interaction_id}
            ]
            if not any(
                any(msg["interaction_id"] == interaction_id for msg in conv["messages"] if "interaction_id" in msg)
                for conv in recent_conversations
            ):
                data_manager.save_conversation(
                    session_id,
                    mobile_number,
                    messages_to_save
                )
                logger.info(f"Saved conversation for session {session_id}: {query} -> {generated_response} (Interaction ID: {interaction_id})")
            else:
                logger.debug(f"Skipped saving duplicate conversation for query: {query} (Interaction ID: {interaction_id})")
        except Exception as e:
            logger.error(f"Failed to save conversation for session {session_id}: {str(e)}")

        return generated_response
    except Exception as e:
        logger.error(f"Response generation failed in {time.time() - start_time:.2f} seconds: {str(e)}")
        if query_language == "Hindi":
            return "क्षमा करें, मैं आपका प्रश्न संसाधित नहीं कर सका। कृपया पुनः प्रयास करें।"
        elif query_language == "Hinglish":
            return "Sorry, main aapka query process nahi kar saka. Please dobara try karein."
        return "Sorry, I couldn’t process your query. Please try again."