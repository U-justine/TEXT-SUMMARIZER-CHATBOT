import streamlit as st
import re
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import heapq
import PyPDF2
import io

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set page config
st.set_page_config(
    page_title="📝 Simple Text Summarizer - Fallback Mode",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }

    .summary-box {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }

    .metric-box {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }

    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }

    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""

        # Limit to first 20 pages for performance
        max_pages = min(20, len(pdf_reader.pages))

        for page_num in range(max_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"

        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()

        if max_pages < len(pdf_reader.pages):
            st.info(f"📄 Processed first {max_pages} pages of {len(pdf_reader.pages)} total pages")

        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def simple_extractive_summary(text, num_sentences=3):
    """
    Create a simple extractive summary using basic NLP techniques
    This doesn't require heavy AI models and works offline
    """
    try:
        if len(text.strip()) < 100:
            return "Text is too short for meaningful summarization."

        # Tokenize into sentences
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text

        # Tokenize into words and remove stopwords
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]

        # Calculate word frequency
        word_freq = Counter(words)

        # Score sentences based on word frequency
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sentence_words = word_tokenize(sentence.lower())
            sentence_words = [word for word in sentence_words if word.isalnum() and word not in stop_words]

            if len(sentence_words) == 0:
                continue

            score = sum(word_freq[word] for word in sentence_words) / len(sentence_words)
            sentence_scores[i] = score

        # Get top sentences
        top_sentences = heapq.nlargest(num_sentences, sentence_scores.items(), key=lambda x: x[1])
        top_sentences.sort(key=lambda x: x[0])  # Sort by original order

        summary = ' '.join([sentences[i] for i, score in top_sentences])
        return summary

    except Exception as e:
        return f"Error generating summary: {str(e)}"

def keyword_extraction(text, num_keywords=10):
    """Extract key words from text"""
    try:
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and len(word) > 3 and word not in stop_words]

        word_freq = Counter(words)
        return [word for word, freq in word_freq.most_common(num_keywords)]
    except:
        return []

def simple_question_answering(question, text):
    """
    Basic question answering using keyword matching and sentence ranking
    This is a simple fallback that doesn't require AI models
    """
    try:
        question_words = set(word_tokenize(question.lower()))
        stop_words = set(stopwords.words('english'))
        question_words = {word for word in question_words if word.isalnum() and word not in stop_words}

        sentences = sent_tokenize(text)
        sentence_scores = {}

        for i, sentence in enumerate(sentences):
            sentence_words = set(word_tokenize(sentence.lower()))
            sentence_words = {word for word in sentence_words if word.isalnum() and word not in stop_words}

            # Calculate overlap
            overlap = len(question_words.intersection(sentence_words))
            if len(question_words) > 0:
                score = overlap / len(question_words)
                sentence_scores[i] = score

        if sentence_scores:
            best_sentence_idx = max(sentence_scores.items(), key=lambda x: x[1])[0]
            return {
                "answer": sentences[best_sentence_idx],
                "confidence": sentence_scores[best_sentence_idx]
            }
        else:
            return {
                "answer": "No relevant information found in the text.",
                "confidence": 0.0
            }
    except Exception as e:
        return {
            "answer": f"Error processing question: {str(e)}",
            "confidence": 0.0
        }

def main():
    """Main application"""

    st.title("📝 Simple Text Summarizer - Fallback Mode")

    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Fallback Mode Active</h4>
        <p>This is a simplified version that works without AI models. It uses basic NLP techniques for:</p>
        <ul>
            <li>Extractive summarization (selects important sentences)</li>
            <li>Keyword extraction</li>
            <li>Simple question answering</li>
        </ul>
        <p><strong>Note:</strong> Results may be less sophisticated than AI-powered versions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    if 'summary' not in st.session_state:
        st.session_state.summary = ""

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 PDF Upload", "📝 Text Input", "✨ Summary", "❓ Q&A"])

    with tab1:
        st.header("📄 Upload PDF Document")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to extract and analyze text"
        )

        if uploaded_file is not None:
            file_size = uploaded_file.size / (1024 * 1024)  # MB
            st.info(f"📄 File: {uploaded_file.name} ({file_size:.2f} MB)")

            if st.button("🔍 Extract Text from PDF", key="pdf_extract"):
                with st.spinner("Extracting text from PDF..."):
                    extracted_text = extract_text_from_pdf(uploaded_file)

                if extracted_text:
                    st.session_state.extracted_text = extracted_text
                    st.markdown("""
                    <div class="success-box">
                        <h4>✅ Text Extraction Successful!</h4>
                        <p>Your PDF has been processed and text extracted.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Show statistics
                    word_count = len(extracted_text.split())
                    char_count = len(extracted_text)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h3>{word_count:,}</h3>
                            <p>Words</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h3>{char_count:,}</h3>
                            <p>Characters</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h3>{word_count//200 + 1}</h3>
                            <p>Est. Minutes</p>
                        </div>
                        """, unsafe_allow_html=True)

        # Show extracted text if available
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

        # Sample texts
        st.subheader("💡 Try Sample Texts")
        col1, col2, col3 = st.columns(3)

        sample_news = """
        Scientists at Stanford University have developed a revolutionary new battery technology that could charge electric vehicles in just five minutes. The breakthrough involves using silicon nanowires that can store ten times more energy than traditional lithium-ion batteries. The research team, led by Dr. Sarah Chen, has been working on this technology for over five years. Initial tests show that the new batteries maintain 90% of their capacity after 1000 charge cycles, significantly outperforming current technology. The innovation could transform the electric vehicle industry by eliminating range anxiety and reducing charging time. Major automotive manufacturers have already expressed interest in licensing the technology. Commercial production is expected to begin within three years, pending regulatory approval and scaling up manufacturing processes.
        """

        sample_science = """
        A new study published in Nature reveals that climate change is causing rapid shifts in ocean currents, with potentially catastrophic consequences for global weather patterns. Researchers analyzed temperature and salinity data from over 3,000 monitoring stations worldwide over the past 30 years. The findings show that the Atlantic Meridional Overturning Circulation (AMOC), which helps regulate temperatures in Europe and North America, has weakened by 15% since 1990. Lead author Dr. Michael Rodriguez warns that continued weakening could lead to more extreme weather events, including severe winters in Europe and increased hurricane activity in the Atlantic. The study recommends immediate action to reduce greenhouse gas emissions to prevent further deterioration of ocean circulation patterns.
        """

        sample_business = """
        Global e-commerce sales reached a record $5.8 trillion in 2023, representing a 12% increase from the previous year. Mobile commerce now accounts for 58% of all online transactions, highlighting the shift toward smartphone-based shopping. Amazon maintained its market leadership with 38% market share, followed by Alibaba at 22% and eBay at 8%. The growth was driven by improved logistics infrastructure, enhanced payment security, and changing consumer behavior following recent global events. Small and medium enterprises showed remarkable growth, with 67% reporting increased online sales. However, challenges remain in supply chain optimization and sustainable packaging. Industry experts predict continued growth of 10-12% annually through 2025, supported by emerging technologies like augmented reality shopping and voice commerce.
        """

        with col1:
            if st.button("📰 Tech News", use_container_width=True):
                st.session_state.input_text = sample_news

        with col2:
            if st.button("🔬 Science Research", use_container_width=True):
                st.session_state.input_text = sample_science

        with col3:
            if st.button("📊 Business Report", use_container_width=True):
                st.session_state.input_text = sample_business

        # Initialize input text
        if 'input_text' not in st.session_state:
            st.session_state.input_text = ""

        # Text input area
        user_text = st.text_area(
            "Enter or paste your text here:",
            value=st.session_state.input_text,
            height=300,
            max_chars=20000,
            help="You can paste up to 20,000 characters"
        )

        if user_text and user_text != st.session_state.input_text:
            st.session_state.input_text = user_text

        if user_text:
            st.session_state.extracted_text = user_text

            # Show text statistics
            word_count = len(user_text.split())
            char_count = len(user_text)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>{word_count:,}</h3>
                    <p>Words</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>{char_count:,}</h3>
                    <p>Characters</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>{word_count//200 + 1}</h3>
                    <p>Est. Minutes</p>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        st.header("✨ Text Summarization")

        if not st.session_state.extracted_text:
            st.markdown("""
            <div class="warning-box">
                <h4>📄 No Text Available</h4>
                <p>Please upload a PDF or enter text first to generate a summary.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Summary options
            col1, col2 = st.columns(2)
            with col1:
                num_sentences = st.selectbox(
                    "Summary length:",
                    [3, 5, 7, 10],
                    help="Number of sentences in the summary"
                )

            with col2:
                extract_keywords = st.checkbox(
                    "Include keywords",
                    value=True,
                    help="Extract key terms from the text"
                )

            if st.button("✨ Generate Summary", key="generate_summary"):
                with st.spinner("Analyzing text and generating summary..."):
                    # Generate summary
                    summary = simple_extractive_summary(
                        st.session_state.extracted_text,
                        num_sentences=num_sentences
                    )
                    st.session_state.summary = summary

                    # Extract keywords
                    keywords = []
                    if extract_keywords:
                        keywords = keyword_extraction(st.session_state.extracted_text)

            # Display summary
            if st.session_state.summary:
                st.markdown(f"""
                <div class="summary-box">
                    <h4>📝 Summary</h4>
                    <p>{st.session_state.summary}</p>
                </div>
                """, unsafe_allow_html=True)

                # Show keywords if extracted
                if extract_keywords and 'keywords' in locals() and keywords:
                    st.subheader("🔑 Key Terms")
                    keyword_tags = " • ".join([f"`{kw}`" for kw in keywords[:10]])
                    st.markdown(keyword_tags)

                # Summary statistics
                if st.session_state.extracted_text:
                    original_words = len(st.session_state.extracted_text.split())
                    summary_words = len(st.session_state.summary.split())
                    compression = (summary_words / original_words) * 100 if original_words > 0 else 0

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h3>{original_words:,}</h3>
                            <p>Original Words</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h3>{summary_words:,}</h3>
                            <p>Summary Words</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h3>{compression:.1f}%</h3>
                            <p>Compression</p>
                        </div>
                        """, unsafe_allow_html=True)

    with tab4:
        st.header("❓ Simple Question Answering")

        if not st.session_state.extracted_text:
            st.markdown("""
            <div class="warning-box">
                <h4>📄 No Text Context</h4>
                <p>Please upload a PDF or enter text first to ask questions.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            question = st.text_input(
                "Ask a question about your text:",
                placeholder="e.g., What is the main topic? Who are the key people mentioned?",
                help="Ask specific questions for better results"
            )

            if question and st.button("🔍 Find Answer", key="find_answer"):
                with st.spinner("Searching for answer in text..."):
                    result = simple_question_answering(question, st.session_state.extracted_text)

                st.markdown(f"""
                <div class="summary-box">
                    <h4>❓ Question:</h4>
                    <p><em>{question}</em></p>

                    <h4>💡 Answer:</h4>
                    <p>{result['answer']}</p>

                    <h4>📊 Confidence:</h4>
                    <p>{result['confidence']:.1%} (based on keyword matching)</p>
                </div>
                """, unsafe_allow_html=True)

            # Question suggestions
            st.subheader("💡 Question Suggestions")
            suggestions = [
                "What is the main topic?",
                "What are the key findings?",
                "Who are the important people mentioned?",
                "What are the main challenges?",
                "What solutions are proposed?",
                "What are the conclusions?"
            ]

            cols = st.columns(3)
            for i, suggestion in enumerate(suggestions):
                with cols[i % 3]:
                    if st.button(suggestion, key=f"suggest_{i}"):
                        st.session_state.suggested_question = suggestion
                        st.experimental_rerun()

    # Footer
    st.markdown("---")
    st.markdown("### 📝 Simple Text Summarizer - Fallback Mode")
    st.info("""
    **This fallback version uses basic NLP techniques:**
    - ✅ Works offline (no internet required for processing)
    - ✅ No AI model downloads needed
    - ✅ Fast processing
    - ⚠️ Less sophisticated than AI-powered versions
    - ⚠️ Extractive summaries only (selects existing sentences)
    """)

    st.caption("Built with Python, NLTK, and Streamlit")

if __name__ == "__main__":
    main()
