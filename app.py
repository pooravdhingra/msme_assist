import streamlit as st
import random
import string
from datetime import datetime, timedelta
from msme_bot import load_rag_data, process_query, welcome_user
from data import DataManager, STATE_MAPPING
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize DataManager with error handling
try:
    data_manager = DataManager()
except Exception as e:
    logger.error(f"Failed to initialize DataManager: {str(e)}")
    st.error("Database connection failed. Please try again later.")
    st.stop()

# Initialize Streamlit session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_rag_data()

if "page" not in st.session_state:
    st.session_state.page = "login"

if "user" not in st.session_state:
    st.session_state.user = None

if "otp" not in st.session_state:
    st.session_state.otp = None

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "otp_generated" not in st.session_state:
    st.session_state.otp_generated = False

if "welcome_message_sent" not in st.session_state:
    st.session_state.welcome_message_sent = False

if "last_query_id" not in st.session_state:
    st.session_state.last_query_id = None

if "rag_cache" not in st.session_state:
    st.session_state.rag_cache = {}

# Check for session ID in query parameters to restore session
query_params = st.query_params
if "session_id" in query_params and st.session_state.user is None:
    session_id = query_params["session_id"]
    try:
        session = data_manager.find_active_session(session_id)
        if session:
            st.session_state.user = session["user_data"]
            st.session_state.session_id = session_id
            st.session_state.page = "chat"
            st.session_state.messages = []
            logger.info(f"Restored session {session_id} from query parameters")
        else:
            logger.warning(f"No active session found for session_id {session_id}")
            st.query_params.clear()
            st.error("Session expired or invalid. Please log in again.")
            st.session_state.page = "login"
            st.rerun()
    except Exception as e:
        logger.error(f"Failed to restore session {session_id}: {str(e)}")
        st.error("Unable to restore session. Please log in again.")
        st.query_params.clear()
        st.session_state.page = "login"
        st.rerun()

# Generate session ID
def generate_session_id():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))

# Generate unique query ID
def generate_query_id(query, timestamp):
    return f"{query[:50]}_{timestamp.strftime('%Y%m%d%H%M%S')}"

# Registration page
def registration_page():
    st.title("Register")
    st.markdown("Please provide your personal and business details to register.")

    with st.form("registration_form"):
        st.subheader("Personal Details")
        fname = st.text_input("First Name")
        lname = st.text_input("Last Name")
        mobile_number = st.text_input("Mobile Number (10 digits)")
        state = st.selectbox("State", list(STATE_MAPPING.values()))

        st.subheader("Business Details")
        business_name = st.text_input("Business Name")
        business_category = st.selectbox("Business Category", ["Manufacturing", "Services", "Retail"])

        submit_button = st.form_submit_button("Register")

        if submit_button:
            if not fname or not lname:
                st.error("First name and last name are required.")
            elif not mobile_number.isdigit() or len(mobile_number) != 10:
                st.error("Mobile number must be 10 digits.")
            elif not business_name:
                st.error("Business name is required.")
            elif not state:
                st.error("State is required.")
            else:
                try:
                    success, message = data_manager.register_user(
                        fname, lname, mobile_number, state, business_name, business_category
                    )
                    if success:
                        st.success(message)
                        st.session_state.page = "login"
                        st.rerun()
                    else:
                        st.error(message)
                except Exception as e:
                    logger.error(f"Registration failed: {str(e)}")
                    st.error("Registration failed due to database error. Please try again.")

# Login page
def login_page():
    st.title("Login")
    st.markdown("Enter your mobile number to log in.")

    mobile_number = st.text_input("Mobile Number (10 digits)")
    if st.button("Generate OTP"):
        if not mobile_number.isdigit() or len(mobile_number) != 10:
            st.error("Mobile number must be 10 digits.")
        else:
            try:
                user = data_manager.find_user(mobile_number)
                if not user:
                    st.error("Mobile number not registered. Please register first.")
                else:
                    st.session_state.otp = str(random.randint(100000, 999999))
                    st.session_state.temp_mobile = mobile_number
                    st.session_state.otp_generated = True
                    st.info(f"Simulated OTP sent: {st.session_state.otp}")
            except Exception as e:
                logger.error(f"Failed to find user during login: {str(e)}")
                st.error("Error accessing database. Please try again.")

    if st.session_state.otp_generated:
        otp_input = st.text_input("Enter OTP", type="password")
        if st.button("Verify OTP"):
            if otp_input == st.session_state.otp:
                try:
                    user = data_manager.find_user(st.session_state.temp_mobile)
                    user_data = {
                        "fname": user.get("fname"),
                        "lname": user.get("lname"),
                        "mobile_number": user.get("mobile_number"),
                        "state_id": user.get("state_id", "Unknown"),
                        "state_name": user.get("state_name", "Unknown"),
                        "business_name": user.get("business_name"),
                        "business_category": user.get("business_category")
                    }
                    st.session_state.user = user_data
                    st.session_state.session_id = generate_session_id()
                    st.session_state.messages = []
                    data_manager.start_session(st.session_state.temp_mobile, st.session_state.session_id, user_data)
                    st.session_state.page = "chat"
                    st.session_state.otp_generated = False
                    st.session_state.otp = None
                    st.session_state.last_query_id = None
                    st.query_params["session_id"] = st.session_state.session_id
                    st.success("Login successful!")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Login failed: {str(e)}")
                    st.error("Login failed due to database error. Please try again.")
            else:
                st.error("Invalid OTP. Please try again.")

# Chat page
def chat_page():
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"Hi, {st.session_state.user['fname']}")
    with col2:
        if st.button("Logout", key="logout_button"):
            try:
                data_manager.end_session(st.session_state.session_id)
            except Exception as e:
                logger.error(f"Failed to end session: {str(e)}")
            st.session_state.page = "login"
            st.session_state.user = None
            st.session_state.otp = None
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state.otp_generated = False
            st.session_state.welcome_message_sent = False
            st.session_state.last_query_id = None
            st.query_params.clear()
            st.rerun()

    if not st.session_state.welcome_message_sent:
        try:
            conversations = data_manager.get_conversations(st.session_state.user["mobile_number"])
            has_user_messages = False
            for conv in conversations:
                for msg in conv["messages"]:
                    if msg["role"] == "user" or (msg["role"] == "assistant" and "Welcome" not in msg["content"]):
                        has_user_messages = True
                        break
                if has_user_messages:
                    break
            user_type = "returning" if has_user_messages else "new"

            if user_type == "new" or (user_type == "returning" and has_user_messages):
                welcome_response = process_query(
                    "welcome",
                    st.session_state.vector_store,
                    st.session_state.session_id,
                    st.session_state.user["mobile_number"]
                )
                if welcome_response:
                    if not any(msg["role"] == "assistant" and msg["content"] == welcome_response for msg in st.session_state.messages):
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": welcome_response,
                            "timestamp": datetime.utcnow()
                        })
                        logger.debug(f"Appended welcome message to session state: {welcome_response}")
                st.session_state.welcome_message_sent = True
        except Exception as e:
            logger.error(f"Failed to process welcome message: {str(e)}")
            st.error("Error loading conversation history. Please try again.")

    st.subheader("Conversation History")
    all_messages = []

    try:
        conversations = data_manager.get_conversations(st.session_state.user["mobile_number"])
        for conv in conversations:
            for msg in conv["messages"]:
                if "content" not in msg or "role" not in msg or "timestamp" not in msg:
                    logger.warning(f"Skipping malformed message in MongoDB: {msg}")
                    continue
                all_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"]
                })
    except Exception as e:
        logger.error(f"Failed to retrieve conversations: {str(e)}")
        st.error("Error loading conversation history. Please try again.")

    for msg in st.session_state.messages:
        if not any(m["role"] == msg["role"] and m["content"] == msg["content"] for m in all_messages):
            all_messages.append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"]
            })
            logger.debug(f"Added session message to all_messages: {msg['role']} - {msg['content']} ({msg['timestamp']})")

    all_messages.sort(key=lambda x: x["timestamp"])

    for msg in all_messages:
        with st.chat_message(msg["role"], avatar="logo.jpeg" if msg["role"] == "assistant" else None):
            st.markdown(f"{msg['content']} *({msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})*")

    query = st.chat_input("Type your query here...")
    if query:
        query_timestamp = datetime.utcnow()
        query_id = generate_query_id(query, query_timestamp)
        if query_id != st.session_state.last_query_id:
            st.session_state.last_query_id = query_id
            if not any(msg["role"] == "user" and msg["content"] == query for msg in st.session_state.messages):
                st.session_state.messages.append({
                    "role": "user",
                    "content": query,
                    "timestamp": query_timestamp
                })
                with st.chat_message("user"):
                    st.markdown(f"{query} *({query_timestamp.strftime('%Y-%m-%d %H:%M:%S')})*")
                logger.debug(f"Appended user query to session state: {query} (ID: {query_id})")

            try:
                response = process_query(
                    query,
                    st.session_state.vector_store,
                    st.session_state.session_id,
                    st.session_state.user["mobile_number"]
                )
                if not any(msg["role"] == "assistant" and msg["content"] == response for msg in st.session_state.messages):
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.utcnow()
                    })
                    with st.chat_message("assistant", avatar="logo.jpeg"):
                        st.markdown(f"{response} *({datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')})*")
                    logger.debug(f"Appended bot response to session state: {response} (Query ID: {query_id})")
            except Exception as e:
                logger.error(f"Failed to process query: {str(e)}")
                st.error("Error processing your query. Please try again.")

# Main app logic
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "register":
    registration_page()
elif st.session_state.page == "chat":
    chat_page()

# Link to registration
if st.session_state.page == "login":
    if st.button("New User? Register Here"):
        st.session_state.page = "register"
        st.rerun()

# Update existing users to include state_id and state_name
try:
    data_manager.update_existing_users_state()
except Exception as e:
    logger.error(f"Failed to update existing users' state: {str(e)}")