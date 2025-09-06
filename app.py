import streamlit as st
import PyPDF2
from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from typing import Optional, List
import logging
import socket
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_internet_connection():
    """Check if internet connection is available"""
    try:
        # Try to connect to a reliable host
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        try:
            # Try Hugging Face specifically
            response = requests.get("https://huggingface.co", timeout=5)
            return response.status_code == 200
        except:
            return False

def simple_text_summarizer(text: str, max_sentences: int = 3) -> str:
    """Simple rule-based text summarizer for offline mode"""
    try:
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Simple scoring based on length and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = len(sentence.split())  # Word count
            if i < len(sentences) * 0.1:  # Boost first 10%
                score *= 1.2
            if i > len(sentences) * 0.8:  # Boost last 20%
                score *= 1.1
            scored_sentences.append((score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True)
        top_sentences = [s[1] for s in scored_sentences[:max_sentences]]
        
        # Sort back to original order
        result_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                result_sentences.append(sentence)
        
        return '. '.join(result_sentences) + '.'
    except Exception as e:
        logger.error(f"Error in simple summarizer: {e}")
        return text[:500] + "..." if len(text) > 500 else text

def simple_qa_search(question: str, context: str) -> str:
    """Simple keyword-based Q&A for offline mode"""
    try:
        question_lower = question.lower()
        context_sentences = [s.strip() for s in context.split('.') if s.strip()]
        
        # Extract keywords from question
        question_words = [w for w in question_lower.split() if len(w) > 3]
        
        # Score sentences based on keyword matches
        scored_sentences = []
        for sentence in context_sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for word in question_words if word in sentence_lower)
            if score > 0:
                scored_sentences.append((score, sentence))
        
        if scored_sentences:
            # Return the best matching sentence
            scored_sentences.sort(reverse=True)
            return scored_sentences[0][1]
        else:
            return "I couldn't find relevant information in the document to answer your question."
    except Exception as e:
        logger.error(f"Error in simple Q&A: {e}")
        return "I encountered an error while processing your question."

# Set page config
st.set_page_config(
    page_title="PDF Text Summarizer & ChatBot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Modern CSS for Professional PDF AI Assistant
st.markdown("""
<style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    .stApp > header {visibility: hidden;}

    /* Global Reset & Base */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        min-height: 100vh;
    }

    /* Premium Hero Header */
    .main-header {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e293b 0%, #475569 25%, #6366f1 50%, #8b5cf6 75%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 2rem 0 1rem 0;
        letter-spacing: -0.02em;
        line-height: 1.1;
        position: relative;
    }

    .main-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 4px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 2px;
    }

    /* Premium Section Headers */
    .section-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 3rem 0 2rem 0;
        position: relative;
        padding: 1.5rem 0 1rem 0;
        text-align: center;
        letter-spacing: -0.01em;
    }

    .section-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 4px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        border-radius: 2px;
    }

    .section-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 200px;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, #e2e8f0 50%, transparent 100%);
    }

    /* Premium File Upload Info */
    .uploaded-file-info {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.08), 0 8px 16px rgba(0,0,0,0.04);
        position: relative;
        overflow: hidden;
    }

    .uploaded-file-info::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    }

    /* Premium Summary Box */
    .summary-box {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 3rem;
        border-radius: 24px;
        margin: 3rem 0;
        box-shadow: 0 25px 50px rgba(0,0,0,0.1), 0 10px 20px rgba(0,0,0,0.05);
        position: relative;
        border: 1px solid #f1f5f9;
        overflow: hidden;
    }

    .summary-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%);
    }

    .summary-box::after {
        content: '✨';
        position: absolute;
        top: 2rem;
        right: 2rem;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 8px 20px rgba(16,185,129,0.4);
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    /* Premium Q&A Response */
    .qa-response {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 3rem;
        border-radius: 24px;
        margin: 3rem 0;
        box-shadow: 0 25px 50px rgba(0,0,0,0.1), 0 10px 20px rgba(0,0,0,0.05);
        position: relative;
        border: 1px solid #f1f5f9;
        overflow: hidden;
    }

    .qa-response::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 50%, #b45309 100%);
    }

    .qa-response::after {
        content: '🤖';
        position: absolute;
        top: 2rem;
        right: 2rem;
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 8px 20px rgba(245,158,11,0.4);
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    /* Premium Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 1rem 2.5rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 25px rgba(99,102,241,0.3), 0 4px 10px rgba(0,0,0,0.1);
        text-transform: none;
        letter-spacing: 0.02em;
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 20px 40px rgba(99,102,241,0.4), 0 8px 20px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #5b21b6 0%, #7c3aed 50%, #db2777 100%);
    }

    .stButton > button:active {
        transform: translateY(-1px) scale(0.98);
    }

    /* Premium Metrics */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }

    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        color: #0f172a;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0,0,0,0.08), 0 8px 16px rgba(0,0,0,0.04);
        border: 1px solid #f1f5f9;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 60px rgba(0,0,0,0.12), 0 12px 24px rgba(0,0,0,0.08);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        font-size: 1rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Premium Status Messages */
    .success-message {
        background: linear-gradient(145deg, #ffffff 0%, #f0fdf4 100%);
        color: #065f46;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #bbf7d0;
        margin: 2rem 0;
        font-weight: 500;
        box-shadow: 0 10px 25px rgba(16,185,129,0.1);
        position: relative;
        overflow: hidden;
    }

    .success-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }

    .error-message {
        background: linear-gradient(145deg, #ffffff 0%, #fef2f2 100%);
        color: #7f1d1d;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #fecaca;
        margin: 2rem 0;
        font-weight: 500;
        box-shadow: 0 10px 25px rgba(239,68,68,0.1);
        position: relative;
        overflow: hidden;
    }

    .error-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }

    .warning-message {
        background: linear-gradient(145deg, #ffffff 0%, #fffbeb 100%);
        color: #78350f;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #fed7aa;
        margin: 2rem 0;
        font-weight: 500;
        box-shadow: 0 10px 25px rgba(245,158,11,0.1);
        position: relative;
        overflow: hidden;
    }

    .warning-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }

    /* Premium Animations */
    @keyframes fadeIn {
        from { 
            opacity: 0; 
            transform: translateY(30px) scale(0.95); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0) scale(1); 
        }
    }

    @keyframes slideInLeft {
        from { 
            opacity: 0; 
            transform: translateX(-50px); 
        }
        to { 
            opacity: 1; 
            transform: translateX(0); 
        }
    }

    @keyframes slideInRight {
        from { 
            opacity: 0; 
            transform: translateX(50px); 
        }
        to { 
            opacity: 1; 
            transform: translateX(0); 
        }
    }

    .fade-in {
        animation: fadeIn 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .slide-in-left {
        animation: slideInLeft 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .slide-in-right {
        animation: slideInRight 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Premium Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid #e2e8f0;
    }

    .css-1d391kg .stMarkdown h3 {
        color: #0f172a;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Enhanced Responsive Design */
    @media (max-width: 768px) {
        .main-header { 
            font-size: 2.8rem; 
            margin: 1rem 0;
        }
        .section-header { 
            font-size: 1.8rem; 
            margin: 2rem 0 1rem 0;
        }
        .metric-container {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }
        .summary-box, .qa-response {
            padding: 2rem;
            margin: 2rem 0;
        }
    }

    @media (max-width: 480px) {
        .main-header { 
            font-size: 2.2rem; 
        }
        .section-header { 
            font-size: 1.5rem; 
        }
        .metric-container {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# Dynamic theme overrides
def _get_theme_override_css(mode: str) -> str:
    """Return CSS overrides for the selected theme mode ('Light' or 'Dark')."""
    if mode == "Dark":
        return """
        <style>
            .main { color: #e2e8f0; }
            .stApp { background-color: #0f172a; }
            .section-header { color: #e5e7eb; }
            .uploaded-file-info { background: linear-gradient(135deg, #0b1220 0%, #111827 100%); border-color: #38bdf8; }
            .summary-box { background: linear-gradient(145deg, #0b1220 0%, #0f172a 100%); border-left-color: #34d399; }
            .qa-response { background: linear-gradient(135deg, #3f2d0c 0%, #4d3a10 100%); border-left-color: #f59e0b; }
            .metric-card { background: linear-gradient(135deg, #3730a3 0%, #6d28d9 100%); }
            .stButton > button { box-shadow: 0 4px 15px rgba(88, 28, 135, 0.5); }
            .stButton > button:hover { box-shadow: 0 8px 25px rgba(88, 28, 135, 0.7); }
            .stTextArea textarea { background-color: #111827; color: #e5e7eb; }
            .stTextInput input { background-color: #111827; color: #e5e7eb; }
            .block-container { padding-top: 2rem; }
        </style>
        """
    # Light (default) keeps existing styles but slightly softens backgrounds
    return """
    <style>
        .stApp { background-color: #f8fafc; }
        .block-container { padding-top: 2rem; }
    </style>
    """

@st.cache_resource
def load_summarization_model():
    """Load and cache the summarization model with network-aware error handling"""
    # Check internet connection first
    if not check_internet_connection():
        logger.warning("🌐 No internet connection detected - using offline mode")
        return "offline"
    
    model_options = [
        "sshleifer/distilbart-cnn-12-6",
        "facebook/bart-large-cnn",
        "t5-small"
    ]
    
    for i, model_name in enumerate(model_options):
        try:
            logger.info(f"Attempting to load summarization model {i+1}/{len(model_options)}: {model_name}")
            
            # Try with minimal settings for stability
            summarizer = pipeline(
                "summarization", 
                model=model_name,
                device=-1,  # Force CPU for stability
                batch_size=1
            )
            
            logger.info(f"✅ Summarization model loaded successfully: {model_name}")
            return summarizer
            
        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
                logger.warning(f"🌐 Network error loading {model_name}: {e}")
                return "offline"
            else:
                logger.warning(f"❌ Failed to load {model_name}: {e}")
                continue
    
    logger.error("❌ All summarization models failed to load")
    return None

@st.cache_resource
def load_qa_model():
    """Load and cache the Q&A model with network-aware error handling"""
    # Check internet connection first
    if not check_internet_connection():
        logger.warning("🌐 No internet connection detected - using offline mode")
        return "offline"
    
    model_options = [
        "distilbert-base-uncased-distilled-squad",
        "deepset/roberta-base-squad2", 
        "bert-base-uncased"
    ]
    
    for i, model_name in enumerate(model_options):
        try:
            logger.info(f"Attempting to load Q&A model {i+1}/{len(model_options)}: {model_name}")
            
            # Try with minimal settings first
            qa_pipeline = pipeline(
                "question-answering", 
                model=model_name,
                device=-1,  # Force CPU for stability
                batch_size=1,
                return_all_scores=False
            )
            
            logger.info(f"✅ Q&A model loaded successfully: {model_name}")
            return qa_pipeline
            
        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
                logger.warning(f"🌐 Network error loading {model_name}: {e}")
                return "offline"
            else:
                logger.warning(f"❌ Failed to load {model_name}: {e}")
                continue
    
    logger.error("❌ All Q&A models failed to load")
    return None

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text

        # Clean the extracted text
        text = clean_text(text)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def clean_text(text: str) -> str:
    """Clean and preprocess extracted text"""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
    # Remove extra spaces
    text = text.strip()
    return text

def chunk_text(text: str, max_length: int = None) -> List[str]:
    """Split text into chunks for processing with fast mode optimization"""
    if max_length is None:
        # Use fast mode setting if available
        fast_mode = st.session_state.get('fast_mode', True)
        max_length = 512 if fast_mode else 1024
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > max_length:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        current_chunk.append(word)
        current_length += word_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def summarize_text(text: str, summarizer) -> str:
    """Generate summary of the text with offline mode support"""
    try:
        if summarizer == "offline":
            # Use offline summarizer
            logger.info("Using offline text summarizer")
            return simple_text_summarizer(text)
        elif not summarizer:
            return "⚠️ Summarization model not available. Please refresh the page."
            
        if len(text.split()) < 30:
            return "Text is too short to summarize effectively."

        # Optimize chunk size for faster processing
        fast_mode = st.session_state.get('fast_mode', True)
        chunk_size = 512 if fast_mode else 1024
        chunks = chunk_text(text, max_length=chunk_size)
        summaries = []

        # Process fewer chunks in fast mode
        max_chunks = min(2 if fast_mode else 3, len(chunks))
        
        for i, chunk in enumerate(chunks[:max_chunks]):
            if len(chunk.split()) > 20:  # Lower threshold for faster processing
                try:
                    # Adjust summary length based on fast mode
                    max_summary_length = 80 if fast_mode else 100
                    min_summary_length = 15 if fast_mode else 20
                    
                    summary = summarizer(
                        chunk, 
                        max_length=max_summary_length,
                        min_length=min_summary_length, 
                        do_sample=False,
                        truncation=True
                    )
                    summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    logger.warning(f"Error processing chunk {i}: {e}")
                    continue

        if not summaries:
            return "Unable to generate summary from the provided text."

        # Combine summaries
        combined_summary = ' '.join(summaries)

        # Only create final summary if we have multiple chunks and it's not too long
        if len(summaries) > 1 and len(combined_summary.split()) > 80:
            try:
                final_summary = summarizer(
                    combined_summary, 
                    max_length=150, 
                    min_length=40, 
                    do_sample=False,
                    truncation=True
                )
                return final_summary[0]['summary_text']
            except Exception as e:
                logger.warning(f"Error in final summarization: {e}")
                return combined_summary

        return combined_summary

    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        return f"⚠️ Error generating summary: {str(e)}"

def check_model_status():
    """Check if models are loaded and working"""
    status = {
        'summarization': False,
        'qa': False,
        'summarization_model': None,
        'qa_model': None,
        'offline_mode': False
    }
    
    try:
        # Test summarization model
        summarizer = load_summarization_model()
        if summarizer == "offline":
            status['summarization'] = True
            status['summarization_model'] = summarizer
            status['offline_mode'] = True
        elif summarizer:
            status['summarization'] = True
            status['summarization_model'] = summarizer
    except Exception as e:
        logger.warning(f"Summarization model check failed: {e}")
    
    try:
        # Test Q&A model
        qa_model = load_qa_model()
        if qa_model == "offline":
            status['qa'] = True
            status['qa_model'] = qa_model
            status['offline_mode'] = True
        elif qa_model:
            status['qa'] = True
            status['qa_model'] = qa_model
    except Exception as e:
        logger.warning(f"Q&A model check failed: {e}")
    
    return status

def answer_question(question: str, context: str, qa_model=None) -> str:
    """Answer question based on the document context with offline mode support"""
    try:
        # Try to load model if not provided
        if not qa_model:
            qa_model = load_qa_model()
            
        if qa_model == "offline":
            # Use offline Q&A
            logger.info("Using offline Q&A search")
            answer = simple_qa_search(question, context)
            return f"""
            <div style="background: #f0f9ff; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #0ea5e9;">
                <h4 style="color: #0ea5e9; margin-bottom: 1rem;">💡 Answer (Offline Mode)</h4>
                <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1rem;">{answer}</p>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="background: #0ea5e9; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                        🌐 Offline Mode
                    </div>
                </div>
            </div>
            """
        elif not qa_model:
            return """
            <div style="background: #fef2f2; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #ef4444;">
                <h4 style="color: #dc2626; margin-bottom: 1rem;">⚠️ Q&A Model Unavailable</h4>
                <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1rem;">
                    The Q&A model failed to load. This could be due to:
                </p>
                <ul style="margin-left: 1.5rem; margin-bottom: 1rem;">
                    <li>Network connectivity issues</li>
                    <li>Insufficient system resources</li>
                    <li>Model download problems</li>
                </ul>
                <p style="font-size: 0.9rem; color: #64748b;">
                    Please try refreshing the page or check your internet connection.
                </p>
            </div>
            """
            
        if not context or not question:
            return "Please provide both a question and document context."

        # Optimize context length for faster processing
        max_context_length = 1000
        if len(context) > max_context_length:
            context = context[:max_context_length]

        result = qa_model(question=question, context=context)

        confidence_threshold = 0.05
        if result['score'] < confidence_threshold:
            return """
            <div style="background: #fffbeb; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #f59e0b;">
                <h4 style="color: #d97706; margin-bottom: 1rem;">🤔 Low Confidence Answer</h4>
                <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1rem;">
                    I'm not confident enough to answer this question based on the provided document. 
                    Please try rephrasing your question or check if the information is available in the document.
                </p>
            </div>
            """

        # Format response with confidence indicator
        confidence_percent = result['score'] * 100
        confidence_color = "#10b981" if confidence_percent > 70 else "#f59e0b" if confidence_percent > 30 else "#ef4444"
        
        return f"""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px; border-left: 4px solid {confidence_color};">
            <h4 style="color: #0ea5e9; margin-bottom: 1rem;">💡 Answer</h4>
            <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1rem;">{result['answer']}</p>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="background: {confidence_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                    Confidence: {confidence_percent:.1f}%
                </div>
            </div>
        </div>
        """

    except Exception as e:
        logger.error(f"Error in question answering: {e}")
        return f"""
        <div style="background: #fef2f2; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #ef4444;">
            <h4 style="color: #dc2626; margin-bottom: 1rem;">⚠️ Processing Error</h4>
            <p style="font-size: 1.1rem; line-height: 1.6;">
                Error processing question: {str(e)}
            </p>
        </div>
        """

def display_metrics(text: str):
    """Display enhanced metrics for the text"""
    words = len(text.split())
    chars = len(text)
    sentences = len([s for s in text.split('.') if s.strip()])
    reading_time = max(1, words // 200)  # Approximate reading time

    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-value">{words:,}</div>
            <div class="metric-label">Words</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{chars:,}</div>
            <div class="metric-label">Characters</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{sentences:,}</div>
            <div class="metric-label">Sentences</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{reading_time}</div>
            <div class="metric-label">Min Read</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Premium Hero Header
    st.markdown('''
    <div class="fade-in">
        <h1 class="main-header">🤖 PDF AI Assistant</h1>
        <div style="text-align: center; margin-bottom: 4rem;">
            <p style="font-size: 1.4rem; color: #475569; font-weight: 400; line-height: 1.6; max-width: 800px; margin: 0 auto;">
                Transform your documents with cutting-edge AI technology. Extract insights, generate summaries, and get intelligent answers from your PDFs in seconds.
            </p>
            <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                <span style="background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 0.5rem 1.5rem; border-radius: 25px; font-size: 0.9rem; font-weight: 500;">
                    🚀 Lightning Fast
                </span>
                <span style="background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 0.5rem 1.5rem; border-radius: 25px; font-size: 0.9rem; font-weight: 500;">
                    🧠 AI Powered
                </span>
                <span style="background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 0.5rem 1.5rem; border-radius: 25px; font-size: 0.9rem; font-weight: 500;">
                    📄 PDF Ready
                </span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Initialize session state
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False

    # Premium Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 1rem; border-radius: 16px; margin-bottom: 1rem;">
                <h3 style="margin: 0; font-size: 1.2rem; font-weight: 700;">🎯 Quick Start Guide</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #6366f1; margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="background: #6366f1; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 600; margin-right: 0.75rem;">1</span>
                <strong>📤 Upload</strong> your PDF document
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="background: #6366f1; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 600; margin-right: 0.75rem;">2</span>
                <strong>🔍 Extract</strong> text from the PDF
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="background: #6366f1; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 600; margin-right: 0.75rem;">3</span>
                <strong>✨ Generate</strong> AI-powered summary
            </div>
            <div style="display: flex; align-items: center;">
                <span style="background: #6366f1; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 600; margin-right: 0.75rem;">4</span>
                <strong>💬 Ask</strong> intelligent questions
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Theme toggle
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #10b981; margin-bottom: 2rem;">
            <h4 style="margin: 0 0 1rem 0; color: #0f172a; font-weight: 600;">🎨 Appearance</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if 'theme' not in st.session_state:
            st.session_state.theme = "Light"
        st.session_state.theme = st.radio("Theme", options=["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1, horizontal=True)

        # UI style toggle (Classic tabs vs DocuSum layout)
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #f59e0b; margin-bottom: 2rem;">
            <h4 style="margin: 0 0 1rem 0; color: #0f172a; font-weight: 600;">🧩 Layout Style</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if 'ui_style' not in st.session_state:
            st.session_state.ui_style = "Classic"
        st.session_state.ui_style = st.radio("Layout", options=["Classic", "DocuSum"], index=0 if st.session_state.ui_style == "Classic" else 1, horizontal=True)

        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #8b5cf6; margin-bottom: 2rem;">
            <h4 style="margin: 0 0 1rem 0; color: #0f172a; font-weight: 600;">🤖 AI Models (Optimized)</h4>
            <div style="margin-bottom: 1rem;">
                <strong style="color: #6366f1;">Summarization</strong><br>
                <span style="color: #10b981; font-size: 0.9rem;">⚡ DistilBART (Fast)</span><br>
                <span style="color: #64748b; font-size: 0.9rem;">🔹 BART Large (fallback)</span>
            </div>
            <div>
                <strong style="color: #6366f1;">Question Answering</strong><br>
                <span style="color: #10b981; font-size: 0.9rem;">⚡ DistilBERT (Fast)</span><br>
                <span style="color: #64748b; font-size: 0.9rem;">🔹 RoBERTa (fallback)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Performance Mode Toggle
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #10b981; margin-bottom: 2rem;">
            <h4 style="margin: 0 0 1rem 0; color: #0f172a; font-weight: 600;">⚡ Performance Mode</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if 'fast_mode' not in st.session_state:
            st.session_state.fast_mode = True
        st.session_state.fast_mode = st.checkbox("Enable Fast Mode", value=st.session_state.fast_mode, help="Faster processing with optimized models and smaller chunks")

        # Model Status Check
        if 'model_status_checked' not in st.session_state:
            with st.spinner("🔍 Checking model status..."):
                st.session_state.model_status = check_model_status()
                st.session_state.model_status_checked = True
        
        # Performance Status
        fast_mode = st.session_state.get('fast_mode', True)
        performance_status = "⚡ Fast Mode" if fast_mode else "🎯 Quality Mode"
        status_color = "#10b981" if fast_mode else "#6366f1"
        
        st.markdown(f"""
        <div style="background: {status_color}; color: white; padding: 1rem; border-radius: 12px; text-align: center; margin-bottom: 1rem;">
            <strong>{performance_status}</strong><br>
            <small>{'Optimized for speed' if fast_mode else 'Optimized for quality'}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Status Indicators
        model_status = st.session_state.get('model_status', {})
        offline_mode = model_status.get('offline_mode', False)
        
        if offline_mode:
            summarization_status = "🌐 Offline"
            qa_status = "🌐 Offline"
            status_color = "#0ea5e9"
            status_text = "Offline Mode"
        else:
            summarization_status = "✅ Ready" if model_status.get('summarization', False) else "❌ Failed"
            qa_status = "✅ Ready" if model_status.get('qa', False) else "❌ Failed"
            status_color = "#6366f1"
            status_text = "Model Status"
        
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 1rem; border-radius: 12px; border-left: 4px solid {status_color}; margin-bottom: 1rem;">
            <h4 style="margin: 0 0 0.5rem 0; color: #0f172a; font-weight: 600; font-size: 0.9rem;">🤖 {status_text}</h4>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                <span>📝 Summarization: {summarization_status}</span>
                <span>💬 Q&A: {qa_status}</span>
            </div>
            {f'<div style="margin-top: 0.5rem; font-size: 0.7rem; color: #64748b;">🌐 Working offline due to network issues</div>' if offline_mode else ''}
        </div>
        """, unsafe_allow_html=True)
        
        # Retry button for failed models
        if not model_status.get('summarization', False) or not model_status.get('qa', False):
            if st.button("🔄 Retry Model Loading", help="Try to reload failed models"):
                st.session_state.model_status_checked = False
                st.rerun()

        if st.session_state.extracted_text:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 1.5rem; border-radius: 12px;">
                <h4 style="margin: 0 0 1rem 0; font-weight: 600;">📊 Document Stats</h4>
            </div>
            """, unsafe_allow_html=True)
            words = len(st.session_state.extracted_text.split())
            st.metric("Words", f"{words:,}")
            st.metric("Characters", f"{len(st.session_state.extracted_text):,}")
            
            # Estimated processing time
            estimated_time = "2-5 seconds" if fast_mode else "5-10 seconds"
            st.metric("Est. Processing Time", estimated_time)

    # Apply theme CSS overrides after sidebar selection so they take precedence
    st.markdown(_get_theme_override_css(st.session_state.get('theme', 'Light')), unsafe_allow_html=True)

    # DocuSum layout (two-panel) or Classic tabs
    if st.session_state.ui_style == "DocuSum":
        # Inject DocuSum-flavored CSS
        st.markdown("""
        <style>
            .docu-hero { margin-top: 0.5rem; padding: 1.5rem 1rem 0.5rem; text-align: center; }
            .docu-title { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; background: linear-gradient(135deg,#6366f1,#8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .docu-sub { color: #64748b; margin-top: .25rem; }
            .docu-grid { display: grid; grid-template-columns: 380px 1fr; gap: 1.5rem; }
            @media (max-width: 900px) { .docu-grid { grid-template-columns: 1fr; } }
            .docu-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.06); overflow: hidden; }
            .docu-card .head { background: linear-gradient(135deg,#6366f1,#8b5cf6); color: #fff; padding: 14px 16px; font-weight: 600; display:flex; align-items:center; gap:.5rem; }
            .docu-card .body { padding: 16px; }
            .docu-upload { border: 2px dashed #cbd5e1; border-radius: 12px; padding: 18px; background: #f8fafc; }
            .docu-stats { display:grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap:.5rem; margin-top: .75rem; }
            .docu-pill { background: #f1f5f9; border:1px solid #e2e8f0; border-radius: 999px; padding: .4rem .75rem; font-size:.8rem; color:#334155; text-align:center; }
            .docu-chat { height: 520px; display:flex; flex-direction:column; }
            .docu-chat-box { flex:1; background: #f8fafc; border:1px solid #e5e7eb; border-radius: 12px; padding: 12px; overflow:auto; }
            .docu-input { display:flex; gap:.5rem; margin-top:.75rem; }
            .stButton>button { background: linear-gradient(135deg,#6366f1,#8b5cf6); color:#fff; border:none; border-radius: 999px; padding:.7rem 1.4rem; font-weight:700; box-shadow: 0 8px 20px rgba(99,102,241,.35); }
            .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 12px 24px rgba(99,102,241,.45); }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="docu-hero"><div class="docu-title">DocuSum</div><div class="docu-sub">Upload, Summarize and Chat with your PDFs</div></div>', unsafe_allow_html=True)

        # Grid
        st.markdown('<div class="docu-grid">', unsafe_allow_html=True)
        col_left, col_right = st.columns([1, 2], gap="large")

        with col_left:
            st.markdown('<div class="docu-card"><div class="head">📄 Upload Document</div><div class="body">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed", key="docu_file_uploader")
            if uploaded_file is not None:
                st.session_state.pdf_uploaded = True
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.markdown(f"<div class='docu-upload'><b>Name:</b> {uploaded_file.name}<br/><b>Size:</b> {file_size_mb:.2f} MB</div>", unsafe_allow_html=True)

                if st.button("🔍 Extract & Process Text", use_container_width=True, key="docu_extract_btn"):
                    with st.spinner("Extracting text from PDF..."):
                        extracted_text = extract_text_from_pdf(uploaded_file)
                        st.session_state.extracted_text = extracted_text
                if st.session_state.extracted_text:
                    # Stats pills
                    words = len(st.session_state.extracted_text.split())
                    sentences = len([s for s in st.session_state.extracted_text.split('.') if s.strip()])
                    st.markdown(f"<div class='docu-stats'><div class='docu-pill'>Words: {words:,}</div><div class='docu-pill'>Sentences: {sentences:,}</div><div class='docu-pill'>Ready ✅</div></div>", unsafe_allow_html=True)

                st.markdown('</div></div>', unsafe_allow_html=True)

            # Summary card
            st.markdown('<div class="docu-card" style="margin-top:1rem;"><div class="head">📝 AI Summary</div><div class="body">', unsafe_allow_html=True)
            if not st.session_state.extracted_text:
                st.info("Upload and extract a PDF first to generate a summary.")
            else:
                if st.button("✨ Generate AI Summary", use_container_width=True, key="docu_summary_btn"):
                    with st.spinner("AI is analyzing your document..."):
                        summarizer = load_summarization_model()
                        summary = summarize_text(st.session_state.extracted_text, summarizer)
                        st.session_state.summary = summary
                if st.session_state.summary:
                    st.markdown(f"<div style='background:#f8fafc;border:1px solid #e5e7eb;border-radius:12px;padding:12px;'>{st.session_state.summary}</div>", unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="docu-card"><div class="head">🤖 AI Assistant</div><div class="body docu-chat">', unsafe_allow_html=True)
            # Chat history preview
            chat_box_html = ""
            if st.session_state.get('qa_history'):
                for qa in st.session_state.qa_history[-6:]:
                    chat_box_html += f"<div style='margin-bottom:.5rem'><div style='font-weight:600;color:#111827'>You</div><div style='background:#e2e8f0;padding:.5rem .75rem;border-radius:10px'>{qa['question']}</div></div>"
                    chat_box_html += f"<div style='margin:6px 0 10px'><div style='font-weight:600;color:#111827'>Assistant</div><div style='background:#eef2ff;border:1px solid #e0e7ff;padding:.5rem .75rem;border-radius:10px'>{qa['answer']}</div></div>"
            st.markdown(f"<div class='docu-chat-box'>{chat_box_html}</div>", unsafe_allow_html=True)

            question = st.text_input("Ask a question about your document:", placeholder="e.g., What is the main topic?", label_visibility="collapsed", key="docu_question_input")
            ask_col1, ask_col2 = st.columns([1,3])
            with ask_col2:
                if st.button("🚀 Ask AI Assistant", use_container_width=True, key="docu_ask_btn"):
                    if question:
                        with st.spinner("AI is thinking..."):
                            qa_model = load_qa_model()
                            answer = answer_question(question, st.session_state.extracted_text, qa_model)
                        st.session_state.qa_history.append({"question": question, "answer": answer})
            st.markdown('</div></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Main content with tabs (Classic)
        tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "📋 AI Summary", "🤖 Q&A Assistant"])

        with tab1:
            st.markdown('<h2 class="section-header">📤 Upload PDF Document</h2>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type="pdf",
                help="Upload a PDF document to extract text and generate summaries",
                key="classic_file_uploader"
            )
            if uploaded_file is not None:
                st.session_state.pdf_uploaded = True
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.markdown(f'''
                <div class="uploaded-file-info fade-in">
                    <h4 style="margin-bottom: 1rem; color: #0ea5e9;">📄 File Information</h4>
                    <p><strong>Name:</strong> {uploaded_file.name}</p>
                    <p><strong>Size:</strong> {file_size_mb:.2f} MB</p>
                    <p><strong>Type:</strong> {uploaded_file.type}</p>
                </div>
                ''', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔍 Extract & Process Text", key="classic_extract_btn", use_container_width=True):
                        with st.spinner("🔄 Extracting text from PDF..."):
                            extracted_text = extract_text_from_pdf(uploaded_file)
                            st.session_state.extracted_text = extracted_text
                        if st.session_state.extracted_text:
                            st.markdown('<div class="success-message fade-in">✅ Text extracted successfully! Document is ready for analysis.</div>', unsafe_allow_html=True)
                            display_metrics(st.session_state.extracted_text)
                        else:
                            st.markdown('<div class="error-message">❌ Failed to extract text from the PDF. Please ensure it\'s not password-protected.</div>', unsafe_allow_html=True)
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
                if st.button("✨ Generate AI Summary", key="classic_summary_btn", use_container_width=True):
                    # Show loading progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        status_text.text("🔄 Loading AI model...")
                        progress_bar.progress(20)
                        
                        summarizer = load_summarization_model()
                        if not summarizer:
                            st.error("⚠️ Failed to load summarization model. Please refresh the page.")
                            return
                            
                        status_text.text("🧠 AI is analyzing your document...")
                        progress_bar.progress(60)
                        
                        summary = summarize_text(st.session_state.extracted_text, summarizer)
                        progress_bar.progress(90)
                        
                        status_text.text("✨ Generating summary...")
                        progress_bar.progress(100)
                        
                        st.session_state.summary = summary
                        status_text.text("✅ Summary generated successfully!")
                        
                        # Clear progress indicators after a moment
                        import time
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()
                        
                    except Exception as e:
                        st.error(f"⚠️ Error generating summary: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()
                if st.session_state.summary:
                    st.markdown(f'''
                    <div class="summary-box fade-in">
                        <h4 style="margin-top: 1rem; margin-bottom: 1rem; color: #059669;">📝 Document Summary</h4>
                        <p style="font-size: 1.1rem; line-height: 1.6; color: #374151;">
                            {st.session_state.summary}
                        </p>
                    </div>
                    ''', unsafe_allow_html=True)
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
                    "Ask a question about your document:",
                    placeholder="e.g., What is the main topic of this document?",
                    key="classic_question_input"
                )
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Ask AI Assistant", key="classic_ask_btn", use_container_width=True):
                        if question:
                            with st.spinner("🤔 AI is thinking..."):
                                answer = answer_question(question, st.session_state.extracted_text)
                                st.session_state.qa_history.append({
                                    "question": question,
                                    "answer": answer
                                })
                            # Display the formatted answer (already includes styling)
                            st.markdown(answer, unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="warning-message">❓ Please enter a question about your document.</div>', unsafe_allow_html=True)
                if st.session_state.qa_history:
                    st.markdown("### 💬 Recent Conversations")
                    for i, qa in enumerate(reversed(st.session_state.qa_history[-3:])):
                        with st.expander(f"💭 {qa['question'][:60]}...", expanded=(i == 0)):
                            st.markdown(f"**❓ Question:** {qa['question']}")
                            st.markdown(f"**💡 Answer:** {qa['answer']}")


    # Premium Footer
    st.markdown("---")
    st.markdown('''
    <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 24px; margin-top: 4rem; position: relative; overflow: hidden;">
        <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);"></div>
        
        <div style="margin-bottom: 2rem;">
            <h3 style="color: white; margin-bottom: 1rem; font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">🚀 PDF AI Assistant</h3>
            <p style="color: #cbd5e1; margin-bottom: 2rem; font-size: 1.1rem; max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.6;">
                Powered by cutting-edge AI models from Hugging Face Transformers. Transform your documents with intelligent automation.
            </p>
        </div>
        
        <div style="display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 2rem;">
            <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 1rem 2rem; border-radius: 16px; font-size: 0.9rem; font-weight: 600; box-shadow: 0 8px 20px rgba(99,102,241,0.3);">
                📄 Smart PDF Processing
            </div>
            <div style="background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 1rem 2rem; border-radius: 16px; font-size: 0.9rem; font-weight: 600; box-shadow: 0 8px 20px rgba(16,185,129,0.3);">
                🧠 AI Summarization
            </div>
            <div style="background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 1rem 2rem; border-radius: 16px; font-size: 0.9rem; font-weight: 600; box-shadow: 0 8px 20px rgba(245,158,11,0.3);">
                💬 Intelligent Q&A
            </div>
        </div>
        
        <div style="border-top: 1px solid #334155; padding-top: 2rem;">
            <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
                Built with ❤️ using Streamlit & Hugging Face • Transform your documents with AI
            </p>
            <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 1rem;">
                <span style="color: #64748b; font-size: 0.8rem;">⚡ Lightning Fast</span>
                <span style="color: #64748b; font-size: 0.8rem;">🔒 Secure</span>
                <span style="color: #64748b; font-size: 0.8rem;">🌐 Cloud Ready</span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

