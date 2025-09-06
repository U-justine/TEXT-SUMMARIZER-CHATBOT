import streamlit as st
import PyPDF2
import io
import re
import torch
import gc
import logging
import time
from typing import Optional, List, Dict, Any
import os
import nltk
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data silently
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Set page config
st.set_page_config(
    page_title="📝 Text Summarizer ChatBot - Cloud Edition",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Platform detection
def detect_platform():
    """Detect hosting platform to optimize accordingly"""
    hostname = os.environ.get("HOSTNAME", "")
    if "streamlit" in hostname.lower():
        return "streamlit_cloud"
    elif os.environ.get("SPACE_ID"):
        return "huggingface"
    elif os.environ.get("RAILWAY_ENVIRONMENT"):
        return "railway"
    else:
        return "local"

PLATFORM = detect_platform()

# Platform-specific configurations
PLATFORM_CONFIGS = {
    "streamlit_cloud": {
        "max_text_length": 2000,
        "summarizer_model": "t5-small",
        "qa_model": "distilbert-base-uncased-distilled-squad",
        "timeout": 30,
        "use_gpu": False
    },
    "huggingface": {
        "max_text_length": 5000,
        "summarizer_model": "sshleifer/distilbart-cnn-12-6",
        "qa_model": "distilbert-base-cased-distilled-squad",
        "timeout": 60,
        "use_gpu": True
    },
    "railway": {
        "max_text_length": 4000,
        "summarizer_model": "sshleifer/distilbart-cnn-12-6",
        "qa_model": "distilbert-base-cased-distilled-squad",
        "timeout": 45,
        "use_gpu": False
    },
    "local": {
        "max_text_length": 10000,
        "summarizer_model": "facebook/bart-large-cnn",
        "qa_model": "deepset/roberta-base-squad2",
        "timeout": 120,
        "use_gpu": True
    }
}

CONFIG = PLATFORM_CONFIGS[PLATFORM]

# Enhanced CSS with loading animations
st.markdown(f"""
<style>
    .main {{
        padding-top: 1rem;
    }}

    .platform-badge {{
        position: fixed;
        top: 10px;
        right: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        z-index: 1000;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }}

    .hero-section {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }}

    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }}

    .feature-card {{
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
    }}

    .feature-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}

    .summary-box {{
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }}

    .metric-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }}

    .warning-card {{
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }}

    .success-card {{
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }}

    .loading-spinner {{
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }}

    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}

    .progress-bar {{
        width: 100%;
        height: 6px;
        background-color: #e0e0e0;
        border-radius: 3px;
        overflow: hidden;
    }}

    .progress-fill {{
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 0.3s ease;
    }}
</style>
""", unsafe_allow_html=True)

# Platform badge
st.markdown(f"""
<div class="platform-badge">
    🌐 Running on {PLATFORM.replace('_', ' ').title()}
</div>
""", unsafe_allow_html=True)

class TimeoutError(Exception):
    pass

def with_timeout(func, timeout_seconds=30):
    """Execute function with timeout"""
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

    if hasattr(signal, 'SIGALRM'):  # Unix systems
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            result = func()
            return result
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:  # Windows or systems without SIGALRM
        return func()

def cleanup_memory():
    """Clean up memory after processing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@st.cache_resource
def load_ai_models():
    """Load AI models with platform optimization"""
    models = {}

    try:
        # Show loading progress
        progress_container = st.container()
        with progress_container:
            st.info("🔄 Loading AI models... This may take a few minutes on first run.")
            progress_bar = st.progress(0)
            status_text = st.empty()

        # Determine device
        device = 0 if (CONFIG["use_gpu"] and torch.cuda.is_available()) else -1

        # Load summarization model
        status_text.text(f"Loading summarization model: {CONFIG['summarizer_model']}")
        progress_bar.progress(25)

        try:
            from transformers import pipeline

            models['summarizer'] = pipeline(
                "summarization",
                model=CONFIG["summarizer_model"],
                device=device,
                trust_remote_code=True
            )

            # Test the model
            test_result = models['summarizer'](
                "This is a test sentence for the summarization model.",
                max_length=20,
                min_length=5,
                do_sample=False
            )

            progress_bar.progress(50)
            status_text.text("✅ Summarization model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            models['summarizer'] = None

        # Load QA model
        status_text.text(f"Loading Q&A model: {CONFIG['qa_model']}")
        progress_bar.progress(75)

        try:
            models['qa'] = pipeline(
                "question-answering",
                model=CONFIG["qa_model"],
                device=device,
                trust_remote_code=True
            )

            # Test the model
            test_result = models['qa'](
                question="What is this?",
                context="This is a test sentence for the question answering model."
            )

            progress_bar.progress(100)
            status_text.text("✅ All models loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load QA model: {e}")
            models['qa'] = None

        # Clear loading UI
        progress_container.empty()

        return models

    except Exception as e:
        logger.error(f"Critical error loading models: {e}")
        return {"summarizer": None, "qa": None}

def simple_extractive_summary(text: str, num_sentences: int = 3) -> str:
    """Fallback extractive summarization using NLTK"""
    try:
        if len(text.strip()) < 100:
            return "⚠️ Text is too short for meaningful summarization."

        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text

        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]

        word_freq = Counter(words)

        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sentence_words = word_tokenize(sentence.lower())
            sentence_words = [word for word in sentence_words if word.isalnum() and word not in stop_words]

            if len(sentence_words) == 0:
                continue

            score = sum(word_freq[word] for word in sentence_words) / len(sentence_words)
            sentence_scores[i] = score

        import heapq
        top_sentences = heapq.nlargest(num_sentences, sentence_scores.items(), key=lambda x: x[1])
        top_sentences.sort(key=lambda x: x[0])

        summary = ' '.join([sentences[i] for i, score in top_sentences])
        return summary

    except Exception as e:
        return f"Error generating summary: {str(e)}"

def smart_summarize(text: str, models: Dict[str, Any], max_length: int = 130) -> str:
    """Smart summarization with AI model + fallback"""

    # Truncate text if too long
    if len(text) > CONFIG["max_text_length"]:
        text = text[:CONFIG["max_text_length"]]
        st.warning(f"⚠️ Text truncated to {CONFIG['max_text_length']} characters for optimal performance on {PLATFORM}")

    if models.get("summarizer"):
        try:
            def summarize():
                return models["summarizer"](
                    text,
                    max_length=max_length,
                    min_length=max(10, max_length // 4),
                    do_sample=False,
                    truncation=True
                )

            with st.spinner("🤖 AI model analyzing your text..."):
                result = with_timeout(summarize, CONFIG["timeout"])
                cleanup_memory()
                return result[0]['summary_text']

        except TimeoutError:
            st.warning(f"⏱️ AI processing timed out after {CONFIG['timeout']}s. Using fallback method.")
        except Exception as e:
            st.warning(f"⚠️ AI model error: {str(e)}. Using fallback method.")

    # Fallback to extractive summarization
    with st.spinner("📝 Using backup summarization method..."):
        return simple_extractive_summary(text, num_sentences=max(3, max_length // 30))

def simple_qa(question: str, context: str) -> Dict[str, Any]:
    """Fallback QA using keyword matching"""
    try:
        question_words = set(word_tokenize(question.lower()))
        stop_words = set(stopwords.words('english'))
        question_words = {word for word in question_words if word.isalnum() and word not in stop_words}

        sentences = sent_tokenize(context)
        sentence_scores = {}

        for i, sentence in enumerate(sentences):
            sentence_words = set(word_tokenize(sentence.lower()))
            sentence_words = {word for word in sentence_words if word.isalnum() and word not in stop_words}

            overlap = len(question_words.intersection(sentence_words))
            if len(question_words) > 0:
                score = overlap / len(question_words)
                sentence_scores[i] = score

        if sentence_scores:
            best_idx = max(sentence_scores.items(), key=lambda x: x[1])[0]
            return {
                "answer": sentences[best_idx],
                "score": sentence_scores[best_idx]
            }
        else:
            return {
                "answer": "No relevant information found in the text.",
                "score": 0.0
            }

    except Exception as e:
        return {
            "answer": f"Error processing question: {str(e)}",
            "score": 0.0
        }

def smart_qa(question: str, context: str, models: Dict[str, Any]) -> Dict[str, Any]:
    """Smart QA with AI model + fallback"""

    if models.get("qa"):
        try:
            def ask_question():
                return models["qa"](question=question, context=context[:CONFIG["max_text_length"]])

            with st.spinner("🤖 AI analyzing your question..."):
                result = with_timeout(ask_question, CONFIG["timeout"])
                cleanup_memory()
                return result

        except TimeoutError:
            st.warning(f"⏱️ AI processing timed out after {CONFIG['timeout']}s. Using fallback method.")
        except Exception as e:
            st.warning(f"⚠️ AI model error: {str(e)}. Using fallback method.")

    # Fallback to simple QA
    with st.spinner("🔍 Using backup question answering..."):
        return simple_qa(question, context)

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF with progress tracking"""
    try:
        start_time = time.time()

        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)

        # Limit pages for performance
        max_pages = min(20 if PLATFORM == "streamlit_cloud" else 50, total_pages)

        text = ""
        progress_bar = st.progress(0)
        status_text = st.empty()

        for page_num in range(max_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"

            progress = (page_num + 1) / max_pages
            progress_bar.progress(progress)
            status_text.text(f"Processing page {page_num + 1} of {max_pages}")

        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()

        # Limit text length
        if len(text) > CONFIG["max_text_length"]:
            text = text[:CONFIG["max_text_length"]]
            st.info(f"📄 Text truncated to {CONFIG['max_text_length']} characters for optimal performance")

        end_time = time.time()

        progress_bar.empty()
        status_text.empty()

        if max_pages < total_pages:
            st.info(f"📄 Processed {max_pages} of {total_pages} pages in {end_time - start_time:.1f}s")

        return text

    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'extracted_text': '',
        'summary': '',
        'user_input_text': '',
        'text_summary': '',
        'qa_history': [],
        'models_loaded': False,
        'sample_text': ''
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def show_platform_info():
    """Display platform-specific information"""
    st.markdown(f"""
    <div class="feature-card">
        <h4>🌐 Platform: {PLATFORM.replace('_', ' ').title()}</h4>
        <p><strong>Max Text Length:</strong> {CONFIG['max_text_length']:,} characters</p>
        <p><strong>Models:</strong> {CONFIG['summarizer_model']} + {CONFIG['qa_model']}</p>
        <p><strong>Processing Timeout:</strong> {CONFIG['timeout']} seconds</p>
        <p><strong>GPU Support:</strong> {'✅ Enabled' if CONFIG['use_gpu'] else '❌ CPU Only'}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    init_session_state()

    # Hero section
    st.markdown("""
    <div class="hero-section">
        <h1>📝 Text Summarizer ChatBot</h1>
        <h3>Cloud-Optimized AI-Powered Text Analysis</h3>
        <p>✨ Intelligent summarization • 🤖 Smart Q&A • ⚡ Optimized for cloud hosting</p>
    </div>
    """, unsafe_allow_html=True)

    # Platform info
    with st.expander("🌐 Platform Information", expanded=False):
        show_platform_info()

    # Load models
    if not st.session_state.models_loaded:
        models = load_ai_models()
        st.session_state.models = models
        st.session_state.models_loaded = True
    else:
        models = st.session_state.models

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 PDF Upload", "📝 Text Input", "✨ Summary", "❓ Q&A"])

    with tab1:
        st.header("📄 PDF Document Upload")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help=f"Upload a PDF document (processing optimized for {PLATFORM})"
        )

        if uploaded_file is not None:
            file_size = uploaded_file.size / (1024 * 1024)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.info(f"📄 **File:** {uploaded_file.name}")
                st.info(f"📊 **Size:** {file_size:.2f} MB")

            with col2:
                if st.button("🔍 Extract Text", key="pdf_extract"):
                    extracted_text = extract_text_from_pdf(uploaded_file)

                    if extracted_text:
                        st.session_state.extracted_text = extracted_text

                        # Show statistics
                        word_count = len(extracted_text.split())
                        char_count = len(extracted_text)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{word_count:,}</h3>
                                <p>Words Extracted</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{char_count:,}</h3>
                                <p>Characters</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{word_count//200 + 1}</h3>
                                <p>Est. Minutes</p>
                            </div>
                            """, unsafe_allow_html=True)

        if st.session_state.extracted_text:
            with st.expander("📖 View Extracted Text", expanded=False):
                st.text_area(
                    "Extracted content:",
                    st.session_state.extracted_text,
                    height=300,
                    disabled=True
                )

    with tab2:
        st.header("📝 Direct Text Input")

        # Sample texts optimized for cloud
        st.subheader("💡 Try Sample Texts")

        sample_texts = {
            "Tech News": """
            Artificial intelligence is revolutionizing healthcare with new diagnostic tools that can detect diseases earlier than ever before. Machine learning algorithms analyze medical images with 95% accuracy, helping doctors make faster decisions. The technology processes thousands of scans in minutes, identifying patterns invisible to human eyes. Major hospitals report 30% improvement in early detection rates. However, concerns about data privacy and algorithm bias remain. Researchers emphasize the need for diverse training data and transparent AI systems. The next decade promises even more advanced AI tools that could transform patient care globally.
            """.strip(),

            "Science Brief": """
            Climate scientists have discovered that ocean temperatures are rising faster than previously thought. New data from deep-sea sensors shows a 2-degree increase in average temperatures over the past decade. This warming affects marine ecosystems and weather patterns worldwide. Fish populations are migrating toward cooler waters, disrupting local fishing industries. The research team recommends immediate action to reduce carbon emissions. They warn that continued warming could lead to irreversible changes in ocean currents. International cooperation is essential to address this global challenge.
            """.strip(),

            "Business Update": """
            The global e-commerce market reached $4.2 trillion in 2023, driven by mobile shopping and improved logistics. Online retail now represents 20% of total retail sales worldwide. Small businesses benefit from digital platforms that provide access to global markets. Social commerce through platforms like Instagram and TikTok shows 40% growth year-over-year. Challenges include supply chain disruptions and increasing customer acquisition costs. Companies invest heavily in AI-powered recommendation systems and personalized shopping experiences. The trend toward sustainable packaging and carbon-neutral delivery options continues to grow.
            """.strip()
        }

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔬 Tech News", use_container_width=True):
                st.session_state.sample_text = sample_texts["Tech News"]
        with col2:
            if st.button("🌊 Science Brief", use_container_width=True):
                st.session_state.sample_text = sample_texts["Science Brief"]
        with col3:
            if st.button("💼 Business Update", use_container_width=True):
                st.session_state.sample_text = sample_texts["Business Update"]

        # Text input area
        user_text = st.text_area(
            f"Enter your text here (max {CONFIG['max_text_length']:,} characters):",
            value=st.session_state.get('sample_text', ''),
            height=300,
            max_chars=CONFIG['max_text_length'],
            help=f"Optimized for {PLATFORM} - shorter texts process faster"
        )

        if user_text:
            st.session_state.extracted_text = user_text
            st.session_state.user_input_text = user_text

            # Clear sample text
            if 'sample_text' in st.session_state and st.session_state.sample_text:
                st.session_state.sample_text = ''

            # Show text statistics
            word_count = len(user_text.split())
            char_count = len(user_text)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{word_count:,}</h3>
                    <p>Words</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{char_count:,}</h3>
                    <p>Characters</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                usage_percent = (char_count / CONFIG['max_text_length']) * 100
                color = "#28a745" if usage_percent < 50 else "#ffc107" if usage_percent < 80 else "#dc3545"
                st.markdown(f"""
                <div class="metric-card" style="background: {color}">
                    <h3>{usage_percent:.1f}%</h3>
                    <p>Usage</p>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        st.header("✨ AI-Powered Summarization")

        if not st.session_state.extracted_text:
            st.markdown("""
            <div class="warning-card">
                <h4>📄 No Text Available</h4>
                <p>Please upload a PDF or enter text in the previous tabs first.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Summary options
            col1, col2 = st.columns(2)
            with col1:
                summary_length = st.selectbox(
                    "Summary length:",
                    ["Short (50 words)", "Medium (100 words)", "Long (150 words)"],
                    help="Longer summaries may take more time on some platforms"
                )

            with col2:
                include_stats = st.checkbox(
                    "Show detailed statistics",
                    value=True,
                    help="Display compression ratio and other metrics"
                )

            # Generate summary
            if st.button("✨ Generate AI Summary", type="primary", use_container_width=True):
                max_lengths = {
                    "Short (50 words)": 60,
                    "Medium (100 words)": 120,
                    "Long (150 words)": 180
                }

                max_len = max_lengths[summary_length]

                start_time = time.time()
                summary = smart_summarize(st.session_state.extracted_text, models, max_len)
                end_time = time.time()

                st.session_state.summary = summary
                st.session_state.processing_time = end_time - start_time

            # Display summary
            if st.session_state.summary:
                st.markdown(f"""
                <div class="summary-box">
                    <h4>📝 Generated Summary</h4>
                    <p style="font-size: 1.1rem; line-height: 1.6;">
                        {st.session_state.summary}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if include_stats:
                    original_words = len(st.session_state.extracted_text.split())
                    summary_words = len(st.session_state.summary.split())
                    compression = (summary_words / original_words) * 100 if original_words > 0 else 0
                    processing_time = getattr(st.session_state, 'processing_time', 0)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{original_words:,}</h3>
                            <p>Original Words</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{summary_words:,}</h3>
                            <p>Summary Words</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{compression:.1f}%</h3>
                            <p>Compression</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{processing_time:.1f}s</h3>
                            <p>Process Time</p>
                        </div>
                        """, unsafe_allow_html=True)

    with tab4:
        st.header("❓ Intelligent Q&A System")

        if not st.session_state.extracted_text:
            st.markdown("""
            <div class="warning-card">
                <h4>📄 No Text Context</h4>
                <p>Please upload a PDF or enter text first to ask questions.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            question = st.text_input(
                "Ask a question about your text:",
                placeholder="e.g., What are the main points? Who is mentioned?",
                help="Ask specific questions for better results"
            )

            if question and st.button("🚀 Get AI Answer", type="primary", use_container_width=True):
                start_time = time.time()
                result = smart_qa(question, st.session_state.extracted_text, models)
                end_time = time.time()

                # Add to history
                st.session_state.qa_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "confidence": result.get("score", 0),
                    "timestamp": time.time(),
                    "processing_time": end_time - start_time
                })

            # Display current answer
            if st.session_state.qa_history:
                latest = st.session_state.qa_history[-1]

                st.markdown(f"""
                <div class="summary-box">
                    <h4>❓ Question:</h4>
                    <p style="font-style: italic; margin-bottom: 1rem;">{latest['question']}</p>

                    <h4>💡 Answer:</h4>
                    <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1rem;">{latest['answer']}</p>

                    <div style="display: flex; gap: 1rem; align-items: center;">
                        <span><strong>Confidence:</strong> {latest['confidence']:.1%}</span>
                        <span><strong>Time:</strong> {latest.get('processing_time', 0):.1f}s</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Q&A History
            if len(st.session_state.qa_history) > 1:
                with st.expander(f"📚 Previous Questions ({len(st.session_state.qa_history) - 1})", expanded=False):
                    for i, qa in enumerate(reversed(st.session_state.qa_history[:-1])):
                        st.markdown(f"""
                        **Q{len(st.session_state.qa_history) - i - 1}:** {qa['question']}
                        **A:** {qa['answer'][:100]}{'...' if len(qa['answer']) > 100 else ''}
                        *Confidence: {qa['confidence']:.1%}*

                        ---
                        """)

            # Question suggestions
            st.subheader("💡 Question Suggestions")
            suggestions = [
                "What is the main topic?",
                "What are the key findings?",
                "Who are the important people mentioned?",
                "What are the main challenges discussed?",
                "What solutions are proposed?",
                "What is the conclusion?"
            ]

            cols = st.columns(3)
            for i, suggestion in enumerate(suggestions):
                with cols[i % 3]:
                    if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                        result = smart_qa(suggestion, st.session_state.extracted_text, models)
                        st.session_state.qa_history.append({
                            "question": suggestion,
                            "answer": result["answer"],
                            "confidence": result.get("score", 0),
                            "timestamp": time.time(),
                            "processing_time": 0
                        })
                        st.experimental_rerun()

    # Footer with performance info
    st.markdown("---")

    # Performance dashboard
    with st.expander("📊 Performance Dashboard", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🌐 Platform Status")
            st.write(f"**Environment:** {PLATFORM.title()}")
            st.write(f"**Max Text:** {CONFIG['max_text_length']:,} chars")
            st.write(f"**Timeout:** {CONFIG['timeout']}s")

        with col2:
            st.subheader("🤖 Model Status")
            summarizer_status = "✅ Loaded" if models.get("summarizer") else "❌ Failed"
            qa_status = "✅ Loaded" if models.get("qa") else "❌ Failed"
            st.write(f"**Summarizer:** {summarizer_status}")
            st.write(f"**Q&A System:** {qa_status}")

        with col3:
            st.subheader("📈 Usage Stats")
            st.write(f"**Questions Asked:** {len(st.session_state.qa_history)}")
            current_text_len = len(st.session_state.extracted_text)
            usage_percent = (current_text_len / CONFIG['max_text_length']) * 100 if current_text_len else 0
            st.write(f"**Text Usage:** {usage_percent:.1f}%")

    # Help section
    with st.expander("ℹ️ Platform Help & Tips", expanded=False):
        st.markdown(f"""
        ### 🎯 Optimized for {PLATFORM.replace('_', ' ').title()}

        **Performance Tips:**
        - Keep text under {CONFIG['max_text_length']:,} characters for best performance
        - Shorter texts process faster (under 1000 words ideal)
        - Use simple, clear questions for better Q&A results
        - Models are cached after first load (subsequent runs are faster)

        **Troubleshooting:**
        - If processing times out, try shorter text
        - Refresh the page if models fail to load
        - Check your internet connection for model downloads
        - Use the fallback methods if AI models are unavailable

        **Feature Status:**
        - ✅ PDF text extraction
        - ✅ Direct text input
        - ✅ AI summarization with fallback
        - ✅ Intelligent Q&A with fallback
        - ✅ Performance optimization
        - ✅ Cloud hosting ready

        **Models Used:**
        - **Summarization:** {CONFIG['summarizer_model']}
        - **Question Answering:** {CONFIG['qa_model']}
        - **Fallback:** NLTK-based extractive methods
        """)

    st.markdown("""
    <div class="success-card">
        <h4>🚀 Cloud-Optimized Text Summarizer</h4>
        <p>Built for reliable performance across different hosting platforms with intelligent fallbacks and resource management.</p>
        <p><strong>Current Platform:</strong> {platform} • <strong>Status:</strong> Optimized ✅</p>
    </div>
    """.format(platform=PLATFORM.replace('_', ' ').title()), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
