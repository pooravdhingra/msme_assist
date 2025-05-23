import logging
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from data_loader import load_rag_data
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
        model="grok-3",
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
    vector_store = load_rag_data()
    logger.info(f"Vector store loaded in {time.time() - start_time:.2f} seconds with {vector_store.index.ntotal} documents")
    return vector_store

llm = init_llm()
vector_store = init_vector_store()

# Welcome user
def welcome_user(state_name):
    return f"Welcome to Haqdarshak MSME Chatbot! Since you're from {state_name}, I'll help with schemes and documents applicable to your state and all central government schemes. How can I assist you today?"

def welcome_returning_user(user_name, state_name, conversation_summary=None):
    if not conversation_summary:
        return f"Welcome back, {user_name}! Since you're from {state_name}, I'll help with schemes and documents for your state and central schemes. How can I assist you today?"

    prompt = f"""You are a friendly assistant for Haqdarshak, helping small business owners in India. Create a natural, concise welcome message for a returning user named {user_name} from {state_name}. 

    **Instructions**:
    - Start with "Welcome back, {user_name}!"
    - Mention that since the user is from {state_name}, you will provide schemes and documents applicable to their state and central government schemes.
    - Summarize the prior conversation in 1-2 sentences (max 50 words) based on this summary: {conversation_summary}
    - Use conversational phrases like "Last time, we talked about...", "We discussed...", or similar, avoiding repetitive wording.
    - Mention that you provided helpful details for their business.
    - End with a question like "How can I help you today?" or a natural variant.
    - Keep the tone warm, simple, and relatable for micro business owners with low English proficiency.

    **Output**:
    - Return only the final welcome message.
    """

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate welcome message: {str(e)}")
        return f"Welcome back, {user_name}! Since you're from {state_name}, I'll help with schemes and documents for your state and central schemes. Last time, we discussed {conversation_summary.lower()}. I shared tips for your business. How can I help you today?"

# Step 1: Process user query with RAG
def get_rag_response(query, vector_store):
    start_time = time.time()
    try:
        logger.debug(f"Processing query: {query}")
        embeddings = get_embeddings()
        embed_start = time.time()
        query_embedding = embeddings.embed_query(query)
        logger.debug(f"Query embedding generated in {time.time() - embed_start:.2f} seconds (first 10 values): {query_embedding[:10]}")
        retrieve_start = time.time()
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        result = qa_chain.invoke({"query": query})
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
    - Examples of ambiguous queries: 'Tell me more', 'How to apply?', 'What next?', 'Can you help with it?'.
    - The query is NOT a follow-up if it mentions a specific scheme, document, or topic (e.g., 'What is FSSAI?', 'How to use UPI?') or is unrelated (e.g., 'What’s the weather?').
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
def process_query(query, vector_store, session_id, mobile_number):
    start_time = time.time()
    logger.info(f"Starting query processing for: {query}")
    
    # Retrieve user data from session state
    try:
        user = st.session_state.user
        user_name = user["fname"]
        state_id = user.get("state_id", "Unknown")
        state_name = user.get("state_name", "Unknown")
    except AttributeError:
        logger.error("User data not found in session state")
        return "Error: User not logged in."

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

    # Handle welcome query
    if query.lower() == "welcome":
        if user_type == "new":
            response = welcome_user(state_name)
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
        else:  # returning user
            if not has_user_messages:
                logger.info(f"No new welcome message for returning user with only initial welcome message")
                return None
            conversation_summary = ""
            conversation_history = ""
            all_messages = []
            for conv in conversations:
                for msg in conv["messages"]:
                    if "content" not in msg or "role" not in msg or "timestamp" not in msg:
                        logger.warning(f"Skipping malformed message in MongoDB: {msg}")
                        continue
                    if msg["role"] == "assistant" and "Welcome" in msg["content"]:
                        continue
                    all_messages.append((msg["role"], msg["content"], msg["timestamp"]))
            all_messages = sorted(all_messages, key=lambda x: x[2], reverse=True)[:10]
            user_count = 0
            for role, content, _ in all_messages:
                if role == "user":
                    user_count += 1
                    conversation_history += f"User: {content}\n"
                elif role == "assistant" and user_count > 0:
                    conversation_history += f"Assistant: {content}\n"
                if user_count >= 5:
                    break
            
            if conversation_history:
                summary_prompt = f"Summarize the following conversation history in 1-2 sentences (max 50 words):\n{conversation_history}"
                try:
                    summary_response = llm.invoke([{"role": "user", "content": summary_prompt}])
                    conversation_summary = summary_response.content.strip()
                    logger.info(f"Conversation summary generated: {conversation_summary}")
                except Exception as e:
                    logger.error(f"Failed to generate conversation summary: {str(e)}")
                    conversation_summary = ""

            response = welcome_returning_user(user_name, state_name, conversation_summary)
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
                    logger.info(f"Saved welcome message for returning user in session {session_id} (Interaction ID: {interaction_id})")
                else:
                    logger.debug(f"Skipped saving duplicate welcome message: {response}")
            except Exception as e:
                logger.error(f"Failed to save welcome message for returning user in session {session_id}: {str(e)}")
            logger.info(f"Generated welcome response in {time.time() - start_time:.2f} seconds: {response}")
            return response

    # Check if vector store is valid
    try:
        doc_count = vector_store.index.ntotal
        logger.info(f"Vector store contains {doc_count} documents")
        if doc_count == 0:
            logger.error("Vector store is empty")
            return "No scheme data available. Please check the data source."
    except Exception as e:
        logger.error(f"Vector store check failed: {str(e)}")
        return "Error accessing scheme data."

    # Check if query is related to any previous query in the session
    rag_response = None
    related_prev_query = None
    session_cache = st.session_state.rag_cache.get(session_id, {})

    # Get the most recent query-response pair from the current session
    recent_query = None
    recent_response = None
    if st.session_state.messages:
        # Look for the most recent assistant response (excluding welcome messages)
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant" and "Welcome" not in msg["content"]:
                recent_response = msg["content"]
                # Find the corresponding user query (the one just before this response)
                msg_index = st.session_state.messages.index(msg)
                if msg_index > 0 and st.session_state.messages[msg_index - 1]["role"] == "user":
                    recent_query = st.session_state.messages[msg_index - 1]["content"]
                break

    # Check if the current query is a follow-up to the most recent response
    if recent_query and recent_response and is_query_related(query, recent_response):
        # Use the cached RAG response for the most recent query, if available
        rag_response = session_cache.get(recent_query, None)
        related_prev_query = recent_query
        logger.info(f"Using cached RAG response from recent query: {recent_query}")

    # If no cached response is found or query is not a follow-up, perform RAG search
    if rag_response is None:
        rag_response = get_rag_response(query, vector_store)
        if session_id not in st.session_state.rag_cache:
            st.session_state.rag_cache[session_id] = {}
        # Store with a key combining query and timestamp to ensure uniqueness
        cache_key = f"{query}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        st.session_state.rag_cache[session_id][cache_key] = rag_response
        logger.info(f"Stored new RAG response for query: {query} with key: {cache_key}")

    # Process query and RAG response with a single LLM call
    special_schemes = ["Udyam", "FSSAI", "Shop Act"]
    link = "https://haqdarshak.com/contact"
    
    # Prepare conversation history for prompt, including session messages
    conversation_history = ""
    session_messages = []
    for msg in st.session_state.messages[-10:]:  # Last 10 messages for recency
        if msg["role"] == "assistant" and "Welcome" in msg["content"]:
            continue
        session_messages.append((msg["role"], msg["content"], msg["timestamp"]))
    session_messages = sorted(session_messages, key=lambda x: x[2], reverse=True)[:5]
    for role, content, _ in session_messages:
        conversation_history += f"{role.capitalize()}: {content}\n"

    prompt = f"""You are a helpful assistant for Haqdarshak, supporting small business owners in India with government schemes, digital/financial literacy, and business growth. The user is a {user_type} user named {user_name} from {state_name} (state_id: {state_id}).

    **Input**:
    - Query: {query}
    - RAG Response: {rag_response}
    - Related Previous Query (if any): {related_prev_query if related_prev_query else 'None'}
    - Most Recent Assistant Response: {recent_response if 'recent_response' in locals() else 'None'}
    - Conversation History (last 5 query-response pairs, excluding welcome messages): {conversation_history}
    - Cached RAG Responses for Session: {st.session_state.rag_cache.get(session_id, {})}

    **Instructions**:
    1. **Identify Query Language**:
       - Analyze the query to determine its language: English, Hindi, or Hinglish.
       - Use linguistic patterns: English queries use primarily English words; Hindi queries use Devanagari script or transliterated Hindi words; Hinglish queries mix English and Hindi words or use Romanized Hindi.
       - Do not make additional LLM calls for language detection; infer the language directly from the query text.

    2. **Classify the Query Intent**:
       - Specific_Scheme_Know_Intent (e.g., "What is FSSAI?", "PMFME ke baare mein batao")
       - Specific_Scheme_Apply_Intent (e.g., "How to apply for FSSAI?", "FSSAI kaise apply karu?")
       - Schemes_Know_Intent (e.g., "Schemes for credit?", "MSME ke liye schemes kya hain?")
       - Non_Scheme_Know_Intent (e.g., "How to use UPI?", "GST kya hai?")
       - Out_of_Scope (e.g., "What’s the weather?", "Namaste")
       - Contextual_Follow_Up (e.g., "Tell me more", "Aur batao", "Help me with it", "Tell me more about the first scheme")
       - Use rule-based checks for Out_of_Scope (keywords: "hello", "hi", "hey", "weather", "time", "namaste"). For Contextual_Follow_Up, prioritize the Most Recent Assistant Response for context. If the query refers to a specific part of the Most Recent Assistant Response (e.g., 'the first scheme', 'that scheme', 'the one you mentioned'), identify the referenced scheme or topic from the Most Recent Assistant Response.

    3. **Generate Response Based on Intent**:
       - **Out_of_Scope**: Return: "Sorry, I can only help with government schemes, digital/financial literacy, or business growth. Please ask about these topics." (in the query's language).
       - **Specific_Scheme_Know_Intent**: Share only scheme name, purpose, benefits from RAG Response (≤500 words). Filter for schemes where 'applicability' includes {state_id} or 'scheme type' is 'Centrally Sponsored Scheme' (CSS). List CSS schemes first, then state-specific schemes. Ask: “Want details on eligibility or how to apply?” (in the query's language).
       - **Specific_Scheme_Apply_Intent**: Share only application process from RAG Response (≤500 words). Filter for schemes where 'applicability' includes {state_id} or 'scheme type' is 'Centrally Sponsored Scheme' (CSS). List CSS schemes first, then state-specific schemes. For Udyam, FSSAI, or Shop Act, add: “Haqdarshak can help you get this document for Only ₹99. Click: {link}” (in the query's language).
       - **Schemes_Know_Intent**: List schemes from RAG Response (2–3 lines each, ≤500 words total). Filter for schemes where 'applicability' includes {state_id} or 'scheme type' is 'Centrally Sponsored Scheme' (CSS). List CSS schemes first, then state-specific schemes. Ask: “Want more details on any scheme?” (in the query's language).
       - **Non_Scheme_Know_Intent**: Answer using simple language (≤500 words). Use examples (e.g., “Use UPI like PhonePe” or “UPI ka istemal PhonePe jaise karo”). Use verified external info if needed (in the query's language).
       - **Contextual_Follow_Up**: Use the Most Recent Assistant Response to identify the topic. If the query refers to a specific part of the Most Recent Assistant Response (e.g., 'the first scheme', 'that scheme'), extract the referenced scheme or topic from the Most Recent Assistant Response. If the RAG Response does not match the referenced scheme, indicate that a new RAG search is needed for the identified scheme (but do not perform the search; use the provided RAG Response if it aligns). Provide a relevant follow-up response (≤500 words) using the RAG Response, filtering for schemes where 'applicability' includes {state_id} or 'scheme type' is 'Centrally Sponsored Scheme' (CSS). List CSS schemes first, then state-specific schemes. If the referenced scheme is unclear, ask for clarification (e.g., “Could you specify which scheme?” or “Kaunsi scheme ke baare mein?”) (in the query's language).
       - If RAG Response is empty or "No relevant scheme information found," and the query is a Contextual_Follow_Up referring to a specific scheme in the Most Recent Assistant Response, indicate that a new RAG search is needed for the identified scheme. Otherwise, say: “I don’t have information on this right now.” (in the query's language).

    **Response Guidelines**:
    - Scope: Only respond to queries about government schemes, digital/financial literacy, or business growth.
    - Language: Generate the response in the same language as the query (English, Hindi, or Hinglish). For Hindi, use simple transliterated Hindi (Roman script). For Hinglish, mix English and simple Hindi words naturally.
    - Tone and Style: Use simple, clear words, short sentences, friendly tone, relatable examples. Target micro business owners with low proficiency in the query's language.
    - Core Rules:
    - For scheme queries, ONLY use RAG Response or cached RAG responses for Contextual_Follow_Up unless a new RAG search is indicated for a referenced scheme.
    - Ensure all scheme-related responses only include schemes where 'applicability' includes {state_id} or 'scheme type' is 'Centrally Sponsored Scheme' (CSS).
    - List CSS schemes first, followed by state-specific schemes in the response.
    - Response must be ≤500 words.
    - Never mention agent fees unless specified in RAG Response.
    - Do not extrapolate beyond RAG Response or cached responses.
    - For returning users, use conversation history to maintain context, prioritizing the Most Recent Assistant Response for vague queries or references to specific response content.
    - Always start the response with "Hi {user_name}!" (or "Namaste {user_name}!" for Hindi/Hinglish queries) unless it's an Out_of_Scope response.

    **Output**:
    - Return only the final response to the user in the query's language (no intent label, language label, or intermediate steps). If a new RAG search is needed for a referenced scheme, indicate this in the response with: "I need to fetch more details about [scheme name]. Please confirm if this is the scheme you meant."
    """

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        generated_response = response.content.strip()
        logger.debug(f"LLM prompt: {prompt}")
        logger.debug(f"LLM response: {generated_response}")
        logger.info(f"Generated response in {time.time() - start_time:.2f} seconds: {generated_response}")

        # Check if the response indicates a new RAG search is needed
        if "I need to fetch more details about" in generated_response:
            # Extract the scheme name from the response
            match = re.search(r"I need to fetch more details about (.+?)\. Please confirm", generated_response)
            if match:
                scheme_name = match.group(1)
                logger.info(f"LLM indicated new RAG search needed for scheme: {scheme_name}")
                # Perform a new RAG search for the scheme
                rag_response = get_rag_response(scheme_name, vector_store)
                # Update the cache
                cache_key = f"{scheme_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
                st.session_state.rag_cache[session_id][cache_key] = rag_response
                logger.info(f"Stored new RAG response for scheme: {scheme_name} with key: {cache_key}")
                # Re-run the LLM with the new RAG response
                prompt = prompt.replace(f"RAG Response: {rag_response}", f"RAG Response: {rag_response}")
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
        return "Sorry, I couldn’t process your query. Please try again."