import streamlit as st
import logging
from typing import Optional
import torch

# Import local modules
from config import (
    APP_CONFIG, MODEL_CONFIG, TEXT_CONFIG, PDF_CONFIG, UI_CONFIG,
    PERFORMANCE_CONFIG, ERROR_MESSAGES, SUCCESS_MESSAGES, CUSTOM_CSS
)
from utils import TextProcessor, ModelManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, 'INFO'),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon=APP_CONFIG["page_icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state=APP_CONFIG["initial_sidebar_state"]
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize processors and managers
@st.cache_resource
def get_text_processor():
    """Initialize and cache text processor"""
    return TextProcessor()

@st.cache_resource
def get_model_manager():
    """Initialize and cache model manager"""
    return ModelManager()

def initialize_session_state():
    """Initialize session state variables"""
    default_states = {
        'extracted_text': "",
        'summary': "",
        'pdf_uploaded': False,
        'file_info': {},
        'text_stats': {},
        'qa_history': [],
        'processing_status': {
            'text_extracted': False,
            'summary_generated': False,
            'models_loaded': False
        }
    }

    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def display_file_info(uploaded_file):
    """Display information about the uploaded file"""
    file_size_mb = uploaded_file.size / (1024 * 1024)

    st.markdown('<div class="uploaded-file-info">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**📄 File Name:** {uploaded_file.name}")
        st.write(f"**📏 File Size:** {file_size_mb:.2f} MB")

    with col2:
        st.write(f"**📅 File Type:** {uploaded_file.type}")
        status = "✅ Ready" if file_size_mb <= PDF_CONFIG["max_file_size_mb"] else "⚠️ Too Large"
        st.write(f"**📊 Status:** {status}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Store file info in session state
    st.session_state.file_info = {
        'name': uploaded_file.name,
        'size_mb': file_size_mb,
        'type': uploaded_file.type
    }

    return file_size_mb <= PDF_CONFIG["max_file_size_mb"]

def display_text_statistics(text: str):
    """Display text statistics"""
    text_processor = get_text_processor()
    stats = text_processor.get_text_statistics(text)

    st.session_state.text_stats = stats

    st.markdown('<div class="statistics-box">', unsafe_allow_html=True)
    st.subheader("📊 Document Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Words", f"{stats['word_count']:,}")
    with col2:
        st.metric("Characters", f"{stats['character_count']:,}")
    with col3:
        st.metric("Sentences", f"{stats['sentence_count']:,}")
    with col4:
        st.metric("Avg Words/Sentence", f"{stats['avg_words_per_sentence']}")

    st.markdown('</div>', unsafe_allow_html=True)

def extract_and_process_text(uploaded_file):
    """Extract and process text from uploaded PDF"""
    text_processor = get_text_processor()

    try:
        with st.spinner("🔍 Extracting text from PDF..."):
            extracted_text = text_processor.extract_text_from_pdf(uploaded_file)

        if not extracted_text:
            st.error(ERROR_MESSAGES["text_extraction_failed"])
            return None

        # Validate text
        is_valid, message = text_processor.validate_text_for_processing(extracted_text)
        if not is_valid:
            st.error(f"❌ {message}")
            return None

        st.session_state.extracted_text = extracted_text
        st.session_state.processing_status['text_extracted'] = True

        st.success("✅ " + SUCCESS_MESSAGES["text_extracted"])
        return extracted_text

    except Exception as e:
        logger.error(f"Text extraction error: {e}")
        st.error(f"❌ {ERROR_MESSAGES['text_extraction_failed']}: {str(e)}")
        return None

def generate_summary():
    """Generate summary of extracted text"""
    if not st.session_state.extracted_text:
        st.error("No text available for summarization")
        return

    model_manager = get_model_manager()

    try:
        with st.spinner("🤖 Loading summarization model..."):
            summarizer = model_manager.load_summarization_model()

        with st.spinner("✨ Generating summary..."):
            result = model_manager.summarize_text(
                st.session_state.extracted_text,
                summarizer,
                max_length=MODEL_CONFIG["summarization"]["max_length"],
                min_length=MODEL_CONFIG["summarization"]["min_length"]
            )

        if result['success']:
            st.session_state.summary = result['summary']
            st.session_state.processing_status['summary_generated'] = True
            st.success("✅ " + SUCCESS_MESSAGES["summary_generated"])

            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Length", f"{result['original_length']} words")
            with col2:
                st.metric("Summary Length", f"{result['summary_length']} words")
            with col3:
                st.metric("Compression Ratio", f"{result['compression_ratio']:.1%}")

        else:
            st.error(f"❌ {result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Summarization error: {e}")
        st.error(f"❌ {ERROR_MESSAGES['summarization_failed']}: {str(e)}")

def process_question(question: str):
    """Process user question and generate answer"""
    if not question.strip():
        st.warning(ERROR_MESSAGES["no_question_provided"])
        return

    if not st.session_state.extracted_text:
        st.error(ERROR_MESSAGES["no_context_available"])
        return

    model_manager = get_model_manager()

    try:
        with st.spinner("🤖 Loading Q&A model..."):
            qa_model = model_manager.load_qa_model()

        with st.spinner("🔍 Processing your question..."):
            result = model_manager.answer_question(
                question,
                st.session_state.extracted_text,
                qa_model,
                confidence_threshold=MODEL_CONFIG["question_answering"]["confidence_threshold"]
            )

        # Store Q&A pair in history
        qa_pair = {
            'question': question,
            'answer': result['answer'],
            'success': result['success'],
            'confidence': result.get('confidence', 0)
        }
        st.session_state.qa_history.append(qa_pair)

        # Display answer
        st.markdown('<div class="qa-response">', unsafe_allow_html=True)
        st.markdown(f"**❓ Question:** {question}")

        if result['success']:
            st.markdown(f"**💡 Answer:** {result['answer']}")
            if 'confidence' in result:
                confidence_color = "green" if result['confidence'] > 0.5 else "orange" if result['confidence'] > 0.2 else "red"
                st.markdown(f"**🎯 Confidence:** <span style='color: {confidence_color}'>{result['confidence']:.1%}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**⚠️ Response:** {result['answer']}")

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Q&A processing error: {e}")
        st.error(f"❌ {ERROR_MESSAGES['qa_failed']}: {str(e)}")

def display_sidebar():
    """Display sidebar with instructions and information"""
    with st.sidebar:
        st.markdown("## 📋 How to Use")
        st.markdown("""
        1. **📤 Upload** a PDF document
        2. **🔍 Extract** text from the PDF
        3. **✨ Generate** a summary (optional)
        4. **❓ Ask** questions about the content
        """)

        st.markdown("---")

        # Model Information
        st.markdown("## 🤖 AI Models")
        with st.expander("Summarization", expanded=False):
            st.write(f"**Primary:** {MODEL_CONFIG['summarization']['primary_model']}")
            st.write(f"**Fallback:** {MODEL_CONFIG['summarization']['fallback_model']}")
            st.write(f"**Max Length:** {MODEL_CONFIG['summarization']['max_length']} tokens")

        with st.expander("Question Answering", expanded=False):
            st.write(f"**Primary:** {MODEL_CONFIG['question_answering']['primary_model']}")
            st.write(f"**Fallback:** {MODEL_CONFIG['question_answering']['fallback_model']}")
            st.write(f"**Confidence Threshold:** {MODEL_CONFIG['question_answering']['confidence_threshold']}")

        # System Information
        if st.checkbox("Show System Info"):
            model_manager = get_model_manager()
            device_info = model_manager.get_device_info()
            st.json(device_info)

        st.markdown("---")

        # Processing Status
        if any(st.session_state.processing_status.values()):
            st.markdown("## 📈 Processing Status")
            for status, value in st.session_state.processing_status.items():
                icon = "✅" if value else "⭕"
                st.write(f"{icon} {status.replace('_', ' ').title()}")

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()

    # Display header
    st.markdown(f'<h1 class="main-header">{APP_CONFIG["page_icon"]} {APP_CONFIG["title"]}</h1>', unsafe_allow_html=True)

    # Display sidebar
    display_sidebar()

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "📋 Summary", "🤖 Q&A Chat"])

    with tab1:
        st.markdown('<h2 class="section-header">📤 Upload PDF Document</h2>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help=f"Upload a PDF document (max {PDF_CONFIG['max_file_size_mb']}MB)",
            key="pdf_uploader"
        )

        if uploaded_file is not None:
            st.session_state.pdf_uploaded = True

            # Display file information
            is_valid_size = display_file_info(uploaded_file)

            if is_valid_size:
                col1, col2 = st.columns([1, 3])

                with col1:
                    extract_button = st.button(
                        "🔍 Extract Text",
                        key="extract_btn",
                        disabled=st.session_state.processing_status.get('text_extracted', False)
                    )

                if extract_button:
                    extracted_text = extract_and_process_text(uploaded_file)
                    if extracted_text:
                        st.rerun()
            else:
                st.error(f"❌ File size exceeds {PDF_CONFIG['max_file_size_mb']}MB limit")

        # Display extracted text and statistics
        if st.session_state.extracted_text:
            st.markdown('<h2 class="section-header">📄 Extracted Content</h2>', unsafe_allow_html=True)

            # Text statistics
            display_text_statistics(st.session_state.extracted_text)

            # Extracted text viewer
            with st.expander("📖 View Extracted Text", expanded=UI_CONFIG["default_expanded_sections"]["extracted_text"]):
                st.text_area(
                    "Document Content",
                    st.session_state.extracted_text,
                    height=UI_CONFIG["max_text_area_height"],
                    disabled=True,
                    key="extracted_text_display"
                )

    with tab2:
        st.markdown('<h2 class="section-header">📋 Text Summary</h2>', unsafe_allow_html=True)

        if not st.session_state.extracted_text:
            st.info("👆 Please upload and extract text from a PDF first")
        else:
            col1, col2 = st.columns([1, 3])

            with col1:
                summary_button = st.button(
                    "✨ Generate Summary",
                    key="summary_btn",
                    disabled=st.session_state.processing_status.get('summary_generated', False)
                )

            if summary_button:
                generate_summary()

            # Display summary
            if st.session_state.summary:
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.markdown("### 📝 Generated Summary")
                st.write(st.session_state.summary)
                st.markdown('</div>', unsafe_allow_html=True)

                # Copy summary button
                if st.button("📋 Copy Summary"):
                    st.code(st.session_state.summary, language=None)

    with tab3:
        st.markdown('<h2 class="section-header">🤖 Q&A ChatBot</h2>', unsafe_allow_html=True)

        if not st.session_state.extracted_text:
            st.info("👆 Please upload and extract text from a PDF first")
        else:
            # Question input
            col1, col2 = st.columns([3, 1])

            with col1:
                question = st.text_input(
                    "Ask a question about the document:",
                    placeholder="e.g., What is the main topic of this document?",
                    key="question_input"
                )

            with col2:
                st.write("")  # Spacing
                ask_button = st.button("🚀 Ask", key="ask_btn")

            if ask_button and question:
                process_question(question)

            # Display Q&A history
            if st.session_state.qa_history:
                st.markdown("### 💬 Conversation History")

                for i, qa_pair in enumerate(reversed(st.session_state.qa_history[-5:])):  # Show last 5 Q&As
                    with st.expander(f"Q{len(st.session_state.qa_history)-i}: {qa_pair['question'][:50]}...", expanded=(i==0)):
                        st.markdown(f"**❓ Question:** {qa_pair['question']}")
                        st.markdown(f"**💡 Answer:** {qa_pair['answer']}")
                        if 'confidence' in qa_pair:
                            st.write(f"**🎯 Confidence:** {qa_pair['confidence']:.1%}")
                        st.write(f"**✅ Success:** {'Yes' if qa_pair['success'] else 'No'}")

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p><strong>🚀 {APP_CONFIG['title']} v{APP_CONFIG['version']}</strong></p>
        <p>Built with ❤️ using Streamlit • Powered by Hugging Face Transformers</p>
        <p>Upload PDF documents to extract text, generate summaries, and ask questions!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")
