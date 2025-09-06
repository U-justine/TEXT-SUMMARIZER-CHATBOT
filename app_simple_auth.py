import streamlit as st
import PyPDF2
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import io
import re
import torch
from typing import Optional, List, Dict
import logging
import hashlib
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="PDF AI Assistant | Sign In",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple file-based user storage (no additional packages needed)
USERS_FILE = "users.json"
SESSIONS_FILE = "user_sessions.json"

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    """Save users to JSON file"""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=2)
        return True
    except:
        return False

def load_sessions():
    """Load user sessions from JSON file"""
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_sessions(sessions):
    """Save user sessions to JSON file"""
    try:
        with open(SESSIONS_FILE, 'w') as f:
            json.dump(sessions, f, indent=2)
        return True
    except:
        return False

def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username: str, email: str, password: str) -> bool:
    """Create new user"""
    users = load_users()

    # Check if username or email already exists
    for user_id, user_data in users.items():
        if user_data['username'] == username or user_data['email'] == email:
            return False

    # Create new user
    user_id = str(len(users) + 1)
    users[user_id] = {
        'username': username,
        'email': email,
        'password_hash': hash_password(password),
        'created_at': datetime.now().isoformat(),
        'last_login': None
    }

    return save_users(users)

def verify_user(username: str, password: str) -> Optional[Dict]:
    """Verify user credentials"""
    users = load_users()
    password_hash = hash_password(password)

    for user_id, user_data in users.items():
        if user_data['username'] == username and user_data['password_hash'] == password_hash:
            # Update last login
            user_data['last_login'] = datetime.now().isoformat()
            users[user_id] = user_data
            save_users(users)

            return {
                'id': user_id,
                'username': user_data['username'],
                'email': user_data['email']
            }

    return None

def save_user_session(user_id: str, session_name: str, pdf_name: str, extracted_text: str, summary: str = "") -> str:
    """Save user session"""
    sessions = load_sessions()

    if user_id not in sessions:
        sessions[user_id] = {}

    session_id = str(len(sessions[user_id]) + 1)
    sessions[user_id][session_id] = {
        'session_name': session_name,
        'pdf_name': pdf_name,
        'extracted_text': extracted_text,
        'summary': summary,
        'qa_history': [],
        'created_at': datetime.now().isoformat(),
        'word_count': len(extracted_text.split()) if extracted_text else 0
    }

    save_sessions(sessions)
    return session_id

def update_session_summary(user_id: str, session_id: str, summary: str):
    """Update session summary"""
    sessions = load_sessions()
    if user_id in sessions and session_id in sessions[user_id]:
        sessions[user_id][session_id]['summary'] = summary
        save_sessions(sessions)

def add_qa_to_session(user_id: str, session_id: str, question: str, answer: str, confidence: float = 0.0):
    """Add Q&A to session"""
    sessions = load_sessions()
    if user_id in sessions and session_id in sessions[user_id]:
        qa_entry = {
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        sessions[user_id][session_id]['qa_history'].append(qa_entry)
        save_sessions(sessions)

def get_user_sessions(user_id: str) -> List[Dict]:
    """Get user sessions"""
    sessions = load_sessions()
    if user_id not in sessions:
        return []

    user_sessions = []
    for session_id, session_data in sessions[user_id].items():
        user_sessions.append({
            'id': session_id,
            'session_name': session_data['session_name'],
            'pdf_name': session_data['pdf_name'],
            'created_at': session_data['created_at'],
            'word_count': session_data.get('word_count', 0)
        })

    # Sort by creation date (newest first)
    user_sessions.sort(key=lambda x: x['created_at'], reverse=True)
    return user_sessions

def get_session_data(user_id: str, session_id: str) -> Optional[Dict]:
    """Get session data"""
    sessions = load_sessions()
    if user_id in sessions and session_id in sessions[user_id]:
        return sessions[user_id][session_id]
    return None

# Enhanced Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}

    .main { font-family: 'Inter', sans-serif; }

    /* Authentication Forms */
    .auth-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(102,126,234,0.3);
    }

    .auth-form {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* User Welcome */
    .user-welcome {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(16,185,129,0.3);
    }

    .user-avatar {
        width: 60px;
        height: 60px;
        background: rgba(255,255,255,0.2);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin: 0 auto 1rem;
        font-weight: bold;
    }

    /* Sidebar History */
    .history-item {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .history-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.2);
    }

    .history-date {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.5rem;
    }

    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
    }

    .welcome-message {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #78350f;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        border-left: 4px solid #f59e0b;
        font-size: 1.1rem;
        font-weight: 500;
    }

    /* Status Cards */
    .success-message {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
        font-weight: 500;
    }

    .error-message {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #7f1d1d;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ef4444;
        margin: 1rem 0;
        font-weight: 500;
    }

    .warning-message {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #78350f;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
        font-weight: 500;
    }

    /* File Upload */
    .uploaded-file-info {
        background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #0ea5e9;
        box-shadow: 0 4px 12px rgba(14,165,233,0.2);
    }

    /* Summary Box */
    .summary-box {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-left: 6px solid #10b981;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        position: relative;
    }

    .summary-box::before {
        content: '✨';
        position: absolute;
        top: -15px;
        left: 20px;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 4px 12px rgba(16,185,129,0.3);
    }

    /* Q&A Response */
    .qa-response {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border-left: 6px solid #f59e0b;
        box-shadow: 0 6px 20px rgba(245,158,11,0.2);
        position: relative;
    }

    .qa-response::before {
        content: '🤖';
        position: absolute;
        top: -15px;
        left: 20px;
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 4px 12px rgba(245,158,11,0.3);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.6);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }

    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(102,126,234,0.3);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in { animation: fadeIn 0.6s ease-out; }

    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        position: relative;
        padding-left: 1rem;
    }

    .section-header::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading functions
@st.cache_resource
def load_summarization_model():
    """Load and cache the summarization model"""
    try:
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
        return summarizer
    except Exception as e:
        logger.error(f"Error loading summarization model: {e}")
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def load_qa_model():
    """Load and cache the Q&A model"""
    try:
        model_name = "deepset/roberta-base-squad2"
        return pipeline("question-answering", model=model_name, tokenizer=model_name)
    except Exception as e:
        logger.error(f"Error loading Q&A model: {e}")
        return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

        # Clean the extracted text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        text = text.strip()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def summarize_text(text: str, summarizer) -> str:
    """Generate summary of the text"""
    try:
        if len(text.split()) < 50:
            return "Text is too short to summarize effectively."

        # Split text into chunks if needed
        max_length = 1024
        words = text.split()
        if len(words) <= max_length:
            summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        else:
            # Process in chunks
            chunks = [' '.join(words[i:i+max_length]) for i in range(0, len(words), max_length)]
            summaries = []

            for chunk in chunks:
                if len(chunk.split()) > 30:
                    summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                    summaries.append(summary[0]['summary_text'])

            return ' '.join(summaries)

    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        return f"Error generating summary: {str(e)}"

def answer_question(question: str, context: str, qa_model) -> dict:
    """Answer question based on the document context"""
    try:
        if not context or not question:
            return {"answer": "Please provide both a question and document context.", "confidence": 0}

        max_context_length = 2000
        if len(context) > max_context_length:
            context = context[:max_context_length]

        result = qa_model(question=question, context=context)

        confidence_threshold = 0.1
        if result['score'] < confidence_threshold:
            return {
                "answer": "I'm not confident enough to answer this question based on the provided document.",
                "confidence": result['score']
            }

        return {"answer": result['answer'], "confidence": result['score']}

    except Exception as e:
        logger.error(f"Error in question answering: {e}")
        return {"answer": f"Error processing question: {str(e)}", "confidence": 0}

def show_auth_page():
    """Show authentication page"""
    st.markdown('<h1 class="main-header">🤖 PDF AI Assistant</h1>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Sign Up"])

    with tab1:
        st.markdown("""
        <div class="auth-container">
            <div class="auth-form">
                <h2>🔑 Welcome Back!</h2>
                <p>Sign in to access your document history and personalized AI assistant</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("signin_form"):
            username = st.text_input("👤 Username")
            password = st.text_input("🔒 Password", type="password")
            submitted = st.form_submit_button("🚀 Sign In", use_container_width=True)

            if submitted:
                if username and password:
                    user = verify_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.session_state.authenticated = True
                        st.success(f"Welcome back, {user['username']}! 🎉")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.warning("⚠️ Please fill in all fields")

    with tab2:
        st.markdown("""
        <div class="auth-container">
            <div class="auth-form">
                <h2>📝 Join Us!</h2>
                <p>Create an account to save your document sessions and chat history</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("signup_form"):
            new_username = st.text_input("👤 Choose Username")
            new_email = st.text_input("📧 Email Address")
            new_password = st.text_input("🔒 Create Password", type="password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password")
            submitted = st.form_submit_button("✨ Create Account", use_container_width=True)

            if submitted:
                if new_username and new_email and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("❌ Passwords don't match")
                    elif len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters")
                    else:
                        if create_user(new_username, new_email, new_password):
                            st.success("🎉 Account created successfully! Please sign in.")
                        else:
                            st.error("❌ Username or email already exists")
                else:
                    st.warning("⚠️ Please fill in all fields")

def show_sidebar():
    """Show enhanced sidebar with user info and history"""
    with st.sidebar:
        if st.session_state.get('authenticated'):
            user = st.session_state.user
            username = user['username']

            st.markdown(f"""
            <div class="user-welcome">
                <div class="user-avatar">
                    {username[0].upper()}
                </div>
                <h3>Welcome, {username}! 👋</h3>
                <p style="margin: 0; opacity: 0.9;">Ready to analyze documents?</p>
            </div>
            """, unsafe_allow_html=True)

            # Logout button
            if st.button("🚪 Logout", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

            st.markdown("---")

            # User History Section
            st.markdown("### 📚 Your Document History")

            user_sessions = get_user_sessions(user['id'])

            if user_sessions:
                for session in user_sessions[:10]:  # Show last 10 sessions
                    session_date = session['created_at'][:10]  # Extract date part

                    if st.button(
                        f"📄 {session['pdf_name'][:20]}...",
                        key=f"session_{session['id']}",
                        help=f"Session: {session['session_name']}\nCreated: {session_date}\nWords: {session['word_count']:,}"
                    ):
                        # Load session data
                        session_data = get_session_data(user['id'], session['id'])
                        if session_data:
                            st.session_state.current_session_id = session['id']
                            st.session_state.extracted_text = session_data['extracted_text']
                            st.session_state.summary = session_data['summary']
                            st.session_state.qa_history = session_data.get('qa_history', [])
                            st.success(f"📂 Loaded: {session['session_name']}")
                            st.rerun()

                    st.markdown(f'<div class="history-date">{session_date} • {session["word_count"]:,} words</div>', unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.info("📝 No document history yet. Upload your first PDF to get started!")

            # Quick Stats
            st.markdown("### 📊 Your Stats")
            total_sessions = len(user_sessions)
            total_words = sum(session.get('word_count', 0) for session in user_sessions)

            st.metric("📄 Documents Processed", total_sessions)
            st.metric("📝 Total Words Processed", f"{total_words:,}")

            # Total questions across all sessions
            sessions = load_sessions()
            total_questions = 0
            if user['id'] in sessions:
                for session_data in sessions[user['id']].values():
                    total_questions += len(session_data.get('qa_history', []))

            st.metric("❓ Questions Asked", total_questions)

        else:
            st.markdown("### 🔐 Authentication Required")
            st.info("Please sign in to access your document history and personalized features.")

def display_metrics(text: str):
    """Display enhanced metrics for the text"""
    words = len(text.split())
    chars = len(text)
    sentences = len([s for s in text.split('.') if s.strip()])
    reading_time = max(1, words // 200)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{words:,}</div>
            <div class="metric-label">Words</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{chars:,}</div>
            <div class="metric-label">Characters</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{sentences:,}</div>
            <div class="metric-label">Sentences</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{reading_time}</div>
            <div class="metric-label">Min Read</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None

    # Show authentication page if not logged in
    if not st.session_state.authenticated:
        show_auth_page()
        return

    # Show sidebar with user info and history
    show_sidebar()

    # Main App Content
    user = st.session_state.user
    st.markdown('<h1 class="main-header">🤖 PDF AI Assistant</h1>', unsafe_allow_html=True)

    # Personalized welcome message
    st.markdown(f'''
    <div class="welcome-message fade-in">
        🌟 Hello <strong>{user["username"]}</strong>! Ready to unlock insights from your documents with AI?
    </div>
    ''', unsafe_allow_html=True)

    # Main application tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📋 AI Summary", "🤖 Q&A Assistant"])

    with tab1:
        st.markdown('<h2 class="section-header">📤 Upload PDF Document</h2>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to extract text and generate summaries"
        )

        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)

            st.markdown(f'''
            <div class="uploaded-file-info fade-in">
                <h4 style="margin-bottom: 1rem; color: #0ea5e9;">📄 File Information</h4>
                <p><strong>Name:</strong> {uploaded_file.name}</p>
                <p><strong>Size:</strong> {file_size_mb:.2f} MB</p>
                <p><strong>Type:</strong> {uploaded_file.type}</p>
            </div>
            ''', unsafe_allow_html=True)

            # Session name input
            session_name = st.text_input(
                "📝 Session Name (optional)",
                value=f"{uploaded_file.name.replace('.pdf', '')} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                help="Give this session a memorable name"
            )

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔍 Extract & Process Text", key="extract_btn", use_container_width=True):
                    with st.spinner("🔄 Extracting text from PDF..."):
                        extracted_text = extract_text_from_pdf(uploaded_file)

                    if extracted_text:
                        st.session_state.extracted_text = extracted_text

                        # Save session
                        session_id = save_user_session(
                            user['id'],
                            session_name,
                            uploaded_file.name,
                            extracted_text
                        )
                        st.session_state.current_session_id = session_id

                        st.markdown('<div class="success-message fade-in">✅ Text extracted successfully! Document is ready for analysis.</div>', unsafe_allow_html=True)
                        display_metrics(extracted_text)
                    else:
                        st.markdown('<div class="error-message">❌ Failed to extract text from the PDF. Please ensure it\'s not password-protected.</div>', unsafe_allow_html=True)

        # Display extracted text if available
        if st.session_state.extracted_text:
            st.markdown('<h2 class="section-header">📄 Extracted Content</h2>', unsafe_allow_html=True)
            with st.expander("📖 View Extracted Text", expanded=False):
                st.text_area(
                    "Document Content",
                    st.session_state.extracted_text,
                    height=300,
                    disabled=True
                )

    with tab2:
        st.markdown('<h2 class="section-header">📋 AI-Powered Summary</h2>', unsafe_allow_html=True)

        if not st.session_state.extracted_text:
            st.markdown('<div class="warning-message">📄 Please upload and process a PDF document first to generate a summary.</div>', unsafe_allow_html=True)
        else:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("✨ Generate AI Summary", key="summary_btn", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing your document..."):
                        summarizer = load_summarization_model()
                        summary = summarize_text(st.session_state.extracted_text, summarizer)
                        st.session_state.summary = summary

                        # Update session with summary if we have a session
                        if st.session_state.current_session_id:
                            update_session_summary(user['id'], st.session_state.current_session_id, summary)

            if st.session_state.summary:
                st.markdown(f'''
                <div class="summary-box fade-in">
                    <h4 style="margin-top: 1rem; margin-bottom: 1rem; color: #059669;">📝 Document Summary</h4>
                    <p style="font-size: 1.1rem; line-height: 1.6; color: #374151;">
                        {st.session_state.summary}
                    </p>
                </div>
                ''', unsafe_allow_html=True)

                # Summary statistics
                original_words = len(st.session_state.extracted_text.split())
                summary_words = len(st.session_state.summary.split())
                compression_ratio = (summary_words / original_words) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original", f"{original_words:,} words")
                with col2:
                    st.metric("Summary", f"{summary_words:,} words")
                with col3:
                    st.metric("Compression", f"{compression_ratio:.1f}%")

    with tab3:
        st.markdown('<h2 class="section-header">🤖 AI Question & Answer</h2>', unsafe_allow_html=True)

        if not st.session_state.extracted_text:
            st.markdown('<div class="warning-message">📄 Please upload and process a PDF document first to ask questions.</div>', unsafe_allow_html=True)
        else:
            question = st.text_input(
                f"Ask a question about your document, {user['username']}:",
                placeholder="e.g., What is the main topic of this document?",
                key="question_input"
            )

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Ask AI Assistant", key="ask_btn", use_container_width=True):
                    if question:
                        with st.spinner("🤔 AI is thinking..."):
                            qa_model = load_qa_model()
                            result = answer_question(question, st.session_state.extracted_text, qa_model)

                        # Store in session state
                        qa_entry = {
                            "question": question,
                            "answer": result["answer"],
                            "confidence": result["confidence"]
                        }
                        st.session_state.qa_history.append(qa_entry)

                        # Save to file if we have a session
                        if st.session_state.current_session_id:
                            add_qa_to_session(
                                user['id'],
                                st.session_state.current_session_id,
                                question,
                                result["answer"],
                                result["confidence"]
                            )

                        # Display answer
                        confidence_color = "#10b981" if result["confidence"] > 0.7 else "#f59e0b" if result["confidence"] > 0.3 else "#ef4444"
                        confidence_text = "High" if result["confidence"] > 0.7 else "Medium" if result["confidence"] > 0.3 else "Low"

                        st.markdown(f'''
                        <div class="qa-response fade-in">
                            <h4 style="color: #d97706; margin-bottom: 1rem;">❓ Question</h4>
                            <p style="font-style: italic; margin-bottom: 1.5rem; font-size: 1.1rem;">"{question}"</p>

                            <h4 style="color: #d97706; margin-bottom: 1rem;">💡 Answer</h4>
                            <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1rem;">{result["answer"]}</p>

                            <div style="background: {confidence_color}; color: white; padding: 0.3rem 1rem; border-radius: 20px; display: inline-block; font-size: 0.9rem; font-weight: 600;">
                                🎯 {confidence_text} Confidence ({result["confidence"]:.1%})
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-message">❓ Please enter a question about your document.</div>', unsafe_allow_html=True)

            # Display current session Q&A history
            if st.session_state.qa_history:
                st.markdown("### 💬 Current Session Conversations")
                for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):  # Show last 5
                    with st.expander(f"💭 {qa['question'][:60]}...", expanded=(i == 0)):
                        confidence_color = "#10b981" if qa["confidence"] > 0.7 else "#f59e0b" if qa["confidence"] > 0.3 else "#ef4444"
                        confidence_text = "High" if qa["confidence"] > 0.7 else "Medium" if qa["confidence"] > 0.3 else "Low"

                        st.markdown(f'''
                        <div style="background: #f8fafc; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                            <p><strong>❓ Question:</strong> {qa['question']}</p>
                            <p><strong>💡 Answer:</strong> {qa['answer']}</p>
                            <div style="background: {confidence_color}; color: white; padding: 0.3rem 1rem; border-radius: 15px; display: inline-block; font-size: 0.8rem; margin-top: 0.5rem;">
                                {confidence_text} Confidence ({qa['confidence']:.1%})
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)

    # Enhanced Footer
    st.markdown("---")
    st.markdown(f'''
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 15px; margin-top: 2rem;">
        <h3 style="color: #1e293b; margin-bottom: 1rem;">🚀 PDF AI Assistant</h3>
        <p style="color: #64748b; margin-bottom: 1rem;">
            Powered by state-of-the-art AI models • Personalized for <strong>{user["username"]}</strong>
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <span style="background: #667eea; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                📄 Smart PDF Processing
            </span>
            <span style="background: #10b981; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                🧠 AI Summarization
            </span>
            <span style="background: #f59e0b; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                💬 Intelligent Q&A
            </span>
        </div>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 1rem;">
            Built with ❤️ using Streamlit • Your documents, your AI assistant
        </p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
