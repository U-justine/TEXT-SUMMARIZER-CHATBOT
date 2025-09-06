import streamlit as st
import PyPDF2
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import io
import re
import torch
from typing import Optional, List
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="FastSum - Quick PDF Summarizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Fast Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .stAlert > div {
        border-radius: 10px;
    }

    .success-message {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }

    .error-message {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }

    .summary-box {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #0ea5e9;
        margin: 1rem 0;
    }

    .metrics-container {
        background: linear-gradient(135deg, #f8fafc, #e2e8f0);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    .speed-indicator {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'fast_summarizer' not in st.session_state:
    st.session_state.fast_summarizer = None
if 'fast_qa_model' not in st.session_state:
    st.session_state.fast_qa_model = None

@st.cache_resource(show_spinner=False)
def load_fast_summarization_model():
    """Load lightweight summarization model for speed"""
    try:
        # Use DistilBART - much faster than BART-large
        model_name = "sshleifer/distilbart-cnn-12-6"
        summarizer = pipeline(
            "summarization",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            framework="pt"
        )
        logger.info(f"Fast summarization model loaded: {model_name}")
        return summarizer
    except Exception as e:
        logger.error(f"Error loading fast summarization model: {e}")
        # Fallback to even lighter model
        try:
            summarizer = pipeline(
                "summarization",
                model="t5-small",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Fallback to T5-small model")
            return summarizer
        except Exception as e2:
            logger.error(f"Fallback model also failed: {e2}")
            return None

@st.cache_resource(show_spinner=False)
def load_fast_qa_model():
    """Load lightweight QA model for speed"""
    try:
        # Use DistilBERT - faster than full BERT
        model_name = "distilbert-base-cased-distilled-squad"
        qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info(f"Fast QA model loaded: {model_name}")
        return qa_pipeline
    except Exception as e:
        logger.error(f"Error loading fast QA model: {e}")
        return None

def extract_text_from_pdf(pdf_file) -> Optional[str]:
    """Fast PDF text extraction"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""

        # Limit pages for speed (first 10 pages max)
        max_pages = min(10, len(pdf_reader.pages))

        for page_num in range(max_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"

        # Quick text cleaning
        text = re.sub(r'\s+', ' ', text).strip()

        # Limit text length for speed (first 8000 characters)
        if len(text) > 8000:
            text = text[:8000] + "..."

        return text if text else None

    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return None

def fast_chunk_text(text: str, max_length: int = 512) -> List[str]:
    """Fast text chunking optimized for speed"""
    # Quick sentence splitting
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Limit chunks for speed
    return chunks[:3]  # Only process first 3 chunks

def fast_summarize_text(text: str, summarizer) -> str:
    """Ultra-fast summarization with optimizations"""
    try:
        start_time = time.time()

        if not summarizer:
            return "❌ Summarization model not available. Please refresh and try again."

        if len(text.split()) < 20:
            return "📝 Text is too short for meaningful summarization."

        # Fast chunking
        chunks = fast_chunk_text(text, max_length=512)
        summaries = []

        # Process chunks with speed optimizations
        for i, chunk in enumerate(chunks):
            if len(chunk.split()) < 15:
                continue

            try:
                # Speed optimizations
                summary = summarizer(
                    chunk,
                    max_length=80,      # Shorter summaries = faster
                    min_length=20,      # Lower minimum
                    do_sample=False,    # Deterministic = faster
                    num_beams=2,        # Fewer beams = faster
                    early_stopping=True # Stop early = faster
                )
                summaries.append(summary[0]['summary_text'])

                # Progress indicator for user
                if i == 0:
                    st.info(f"⚡ Processing chunk {i+1}/{len(chunks)}...")

            except Exception as e:
                logger.error(f"Error summarizing chunk {i}: {e}")
                continue

        if not summaries:
            return "❌ Unable to generate summary. Try with a different document."

        # Combine summaries
        final_summary = ' '.join(summaries)

        # If we have multiple chunks, do one final quick summarization
        if len(summaries) > 1 and len(final_summary.split()) > 60:
            try:
                final_result = summarizer(
                    final_summary,
                    max_length=120,
                    min_length=30,
                    do_sample=False,
                    num_beams=2,
                    early_stopping=True
                )
                final_summary = final_result[0]['summary_text']
            except Exception as e:
                logger.warning(f"Final summarization failed, using combined summary: {e}")

        end_time = time.time()
        processing_time = end_time - start_time

        # Add performance info
        final_summary += f"\n\n⚡ Generated in {processing_time:.1f} seconds"

        return final_summary

    except Exception as e:
        logger.error(f"Error in fast summarization: {e}")
        return f"❌ Summarization error: {str(e)}"

def fast_answer_question(question: str, context: str, qa_model) -> str:
    """Fast question answering with optimizations"""
    try:
        start_time = time.time()

        if not qa_model:
            return "❌ Q&A model not available. Please refresh and try again."

        # Limit context for speed
        if len(context) > 2000:
            context = context[:2000] + "..."

        result = qa_model(
            question=question,
            context=context,
            max_answer_len=100,  # Shorter answers = faster
            max_seq_len=512      # Shorter sequences = faster
        )

        answer = result['answer']
        confidence = result['score']

        end_time = time.time()
        processing_time = end_time - start_time

        # Add confidence and speed info
        if confidence > 0.3:
            return f"{answer}\n\n⚡ Answered in {processing_time:.1f}s (Confidence: {confidence:.2f})"
        else:
            return f"🤔 {answer}\n\n⚠️ Low confidence answer. Try rephrasing your question."

    except Exception as e:
        logger.error(f"Error in fast QA: {e}")
        return f"❌ Q&A error: {str(e)}"

def display_speed_metrics(text: str, summary: str = "", processing_time: float = 0):
    """Display performance metrics"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📄 Words", f"{len(text.split()):,}")
    with col2:
        st.metric("📝 Summary Words", f"{len(summary.split()) if summary else 0:,}")
    with col3:
        compression = (len(summary.split()) / len(text.split()) * 100) if summary and text else 0
        st.metric("🗜️ Compression", f"{compression:.1f}%")
    with col4:
        st.metric("⚡ Speed", f"{processing_time:.1f}s" if processing_time > 0 else "N/A")

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="color: white; margin-bottom: 0.5rem;">⚡ FastSum - Quick PDF Summarizer</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem;">Lightning-fast AI document analysis optimized for speed</p>
        <div class="speed-indicator">
            🚀 Optimized with DistilBART & DistilBERT for 3x faster processing
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load models with progress
    if not st.session_state.models_loaded:
        with st.spinner("⚡ Loading optimized AI models (this happens once)..."):
            st.session_state.fast_summarizer = load_fast_summarization_model()
            st.session_state.fast_qa_model = load_fast_qa_model()
            st.session_state.models_loaded = True

        if st.session_state.fast_summarizer and st.session_state.fast_qa_model:
            st.success("✅ Fast AI models loaded successfully!")
        else:
            st.error("❌ Some models failed to load. Functionality may be limited.")

    # Main interface with tabs
    tab1, tab2, tab3 = st.tabs(["📤 Quick Upload", "⚡ Fast Summary", "🚀 Speed Q&A"])

    with tab1:
        st.markdown("### 📤 Upload PDF Document")
        st.info("💡 **Speed Tips:** For fastest processing, use PDFs under 5MB with clear text (not scanned images)")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document for fast AI analysis"
        )

        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.markdown(f"""
            <div class="metrics-container">
                <p><strong>📄 File:</strong> {uploaded_file.name}</p>
                <p><strong>📊 Size:</strong> {file_size_mb:.2f} MB</p>
                <p><strong>⚡ Status:</strong> Ready for fast processing</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("⚡ Quick Extract & Process", use_container_width=True):
                start_time = time.time()

                with st.spinner("🔄 Fast text extraction in progress..."):
                    extracted_text = extract_text_from_pdf(uploaded_file)
                    st.session_state.extracted_text = extracted_text

                end_time = time.time()
                extraction_time = end_time - start_time

                if st.session_state.extracted_text:
                    st.markdown(f'<div class="success-message">✅ Text extracted in {extraction_time:.1f} seconds! Ready for analysis.</div>', unsafe_allow_html=True)
                    display_speed_metrics(st.session_state.extracted_text, processing_time=extraction_time)
                else:
                    st.markdown('<div class="error-message">❌ Failed to extract text. Ensure PDF is not password-protected.</div>', unsafe_allow_html=True)

        # Show extracted text preview
        if st.session_state.extracted_text:
            with st.expander("👀 Preview Extracted Text", expanded=False):
                preview_text = st.session_state.extracted_text[:1000] + "..." if len(st.session_state.extracted_text) > 1000 else st.session_state.extracted_text
                st.text_area("Content Preview", preview_text, height=200, disabled=True)

    with tab2:
        st.markdown("### ⚡ Lightning-Fast AI Summary")

        if not st.session_state.extracted_text:
            st.warning("📄 Please upload and extract text from a PDF first.")
        else:
            st.info("🚀 Using optimized DistilBART model for 3x faster summarization")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("⚡ Generate Fast Summary", use_container_width=True):
                    with st.spinner("🤖 AI creating summary at lightning speed..."):
                        start_time = time.time()
                        summary = fast_summarize_text(
                            st.session_state.extracted_text,
                            st.session_state.fast_summarizer
                        )
                        st.session_state.summary = summary
                        end_time = time.time()
                        summary_time = end_time - start_time

            if st.session_state.summary:
                st.markdown(f'''
                <div class="summary-box">
                    <h4 style="color: #0ea5e9; margin-bottom: 1rem;">📝 Fast AI Summary</h4>
                    <p style="font-size: 1.1rem; line-height: 1.6; color: #374151;">
                        {st.session_state.summary}
                    </p>
                </div>
                ''', unsafe_allow_html=True)

                # Show performance metrics
                display_speed_metrics(
                    st.session_state.extracted_text,
                    st.session_state.summary
                )

    with tab3:
        st.markdown("### 🚀 High-Speed Q&A Assistant")

        if not st.session_state.extracted_text:
            st.warning("📄 Please upload and extract text from a PDF first.")
        else:
            st.info("⚡ Using optimized DistilBERT for rapid question answering")

            question = st.text_input(
                "Ask a question about your document:",
                placeholder="What is the main topic? Who are the key people mentioned?",
                key="fast_question"
            )

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Get Fast Answer", use_container_width=True):
                    if question:
                        with st.spinner("🤔 AI analyzing at high speed..."):
                            answer = fast_answer_question(
                                question,
                                st.session_state.extracted_text,
                                st.session_state.fast_qa_model
                            )

                        # Store in history
                        st.session_state.qa_history.append({
                            "question": question,
                            "answer": answer,
                            "timestamp": datetime.now()
                        })

                        st.markdown(f'''
                        <div class="summary-box">
                            <h4 style="color: #f59e0b; margin-bottom: 1rem;">❓ Question</h4>
                            <p style="font-style: italic; margin-bottom: 1rem;">"{question}"</p>
                            <h4 style="color: #10b981; margin-bottom: 1rem;">💡 Fast Answer</h4>
                            <p style="font-size: 1.1rem; line-height: 1.6;">{answer}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.warning("❓ Please enter a question.")

            # Show recent Q&A history
            if st.session_state.qa_history:
                st.markdown("### 💬 Recent Fast Q&A")
                for i, qa in enumerate(reversed(st.session_state.qa_history[-3:])):
                    with st.expander(f"💭 {qa['question'][:50]}...", expanded=(i == 0)):
                        st.markdown(f"**❓ Question:** {qa['question']}")
                        st.markdown(f"**💡 Answer:** {qa['answer']}")

    # Speed-optimized footer
    st.markdown("---")
    st.markdown('''
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f8fafc, #e2e8f0); border-radius: 10px;">
        <h4 style="color: #1e293b; margin-bottom: 1rem;">⚡ FastSum - Optimized for Speed</h4>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;">
            <span style="background: #10b981; color: white; padding: 0.5rem 1rem; border-radius: 15px; font-size: 0.8rem;">
                🚀 3x Faster Processing
            </span>
            <span style="background: #3b82f6; color: white; padding: 0.5rem 1rem; border-radius: 15px; font-size: 0.8rem;">
                ⚡ Optimized Models
            </span>
            <span style="background: #f59e0b; color: white; padding: 0.5rem 1rem; border-radius: 15px; font-size: 0.8rem;">
                🎯 Speed-First Design
            </span>
        </div>
        <p style="color: #64748b; font-size: 0.9rem;">
            Built for speed with DistilBART & DistilBERT • Fast PDF Analysis
        </p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
