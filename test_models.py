#!/usr/bin/env python3
"""
Model Testing and Diagnostic Script
Tests different AI models to identify loading issues
"""

import streamlit as st
import logging
import sys
import traceback
from transformers import pipeline
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_environment():
    """Test the Python environment and dependencies"""
    st.title("🔧 Model Diagnostic Tool")
    st.markdown("This tool helps diagnose model loading issues.")

    st.subheader("📋 Environment Check")

    # Python version
    python_version = sys.version
    st.info(f"**Python Version:** {python_version}")

    # Check dependencies
    dependencies = {
        'torch': None,
        'transformers': None,
        'streamlit': None,
        'tokenizers': None
    }

    for dep in dependencies:
        try:
            module = __import__(dep)
            if hasattr(module, '__version__'):
                dependencies[dep] = module.__version__
            else:
                dependencies[dep] = "Available (no version info)"
        except ImportError:
            dependencies[dep] = "❌ NOT INSTALLED"

    # Display dependency status
    for dep, version in dependencies.items():
        if "❌" in str(version):
            st.error(f"**{dep}:** {version}")
        else:
            st.success(f"**{dep}:** {version}")

    # Check PyTorch CUDA
    if torch.cuda.is_available():
        st.success(f"**CUDA Available:** Yes ({torch.cuda.get_device_name(0)})")
    else:
        st.warning("**CUDA Available:** No (using CPU)")

    return dependencies

def test_model_loading():
    """Test loading different models"""
    st.subheader("🤖 Model Loading Tests")

    # Summarization models to test
    summarization_models = [
        "sshleifer/distilbart-cnn-12-6",
        "facebook/bart-large-cnn",
        "t5-small",
        "google/pegasus-xsum"
    ]

    # Q&A models to test
    qa_models = [
        "distilbert-base-cased-distilled-squad",
        "distilbert-base-uncased-distilled-squad",
        "deepset/roberta-base-squad2"
    ]

    st.markdown("### 📝 Summarization Models")

    for model_name in summarization_models:
        with st.expander(f"Testing {model_name}", expanded=False):
            try:
                st.info(f"Loading {model_name}...")

                # Load model with CPU only
                summarizer = pipeline(
                    "summarization",
                    model=model_name,
                    device=-1,  # CPU only
                    trust_remote_code=True
                )

                # Test with simple text
                test_text = "Artificial intelligence is rapidly transforming many industries. Machine learning algorithms are being used to automate tasks, improve efficiency, and create new opportunities for businesses. Companies are investing heavily in AI research and development to stay competitive in the digital age."

                result = summarizer(
                    test_text,
                    max_length=50,
                    min_length=20,
                    do_sample=False,
                    truncation=True
                )

                summary = result[0]['summary_text']
                st.success(f"✅ **Model loaded successfully!**")
                st.write(f"**Test Summary:** {summary}")

                # Performance test
                import time
                start_time = time.time()
                for _ in range(3):
                    summarizer(test_text[:200], max_length=30, min_length=10)
                avg_time = (time.time() - start_time) / 3
                st.info(f"**Average processing time:** {avg_time:.2f} seconds")

            except Exception as e:
                st.error(f"❌ **Failed to load {model_name}**")
                st.code(f"Error: {str(e)}")
                st.code(f"Traceback:\n{traceback.format_exc()}")

    st.markdown("### 🤔 Question Answering Models")

    for model_name in qa_models:
        with st.expander(f"Testing {model_name}", expanded=False):
            try:
                st.info(f"Loading {model_name}...")

                qa_pipeline = pipeline(
                    "question-answering",
                    model=model_name,
                    device=-1,  # CPU only
                    trust_remote_code=True
                )

                # Test with simple Q&A
                test_context = "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans. AI research has been highly successful in developing effective techniques for solving a wide range of problems."
                test_question = "What is artificial intelligence?"

                result = qa_pipeline(
                    question=test_question,
                    context=test_context
                )

                st.success(f"✅ **Model loaded successfully!**")
                st.write(f"**Question:** {test_question}")
                st.write(f"**Answer:** {result['answer']}")
                st.write(f"**Confidence:** {result['score']:.2%}")

                # Performance test
                import time
                start_time = time.time()
                for _ in range(3):
                    qa_pipeline(question="What is AI?", context=test_context)
                avg_time = (time.time() - start_time) / 3
                st.info(f"**Average processing time:** {avg_time:.2f} seconds")

            except Exception as e:
                st.error(f"❌ **Failed to load {model_name}**")
                st.code(f"Error: {str(e)}")
                st.code(f"Traceback:\n{traceback.format_exc()}")

def test_internet_connection():
    """Test internet connection to Hugging Face"""
    st.subheader("🌐 Internet Connection Test")

    try:
        import requests

        # Test Hugging Face API
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code == 200:
            st.success("✅ **Hugging Face accessible**")
        else:
            st.warning(f"⚠️ **Hugging Face returned status code:** {response.status_code}")
    except Exception as e:
        st.error(f"❌ **Internet connection issue:** {str(e)}")
        st.info("💡 **Suggestion:** Check your internet connection or firewall settings")

def show_troubleshooting_tips():
    """Show troubleshooting suggestions"""
    st.subheader("💡 Troubleshooting Tips")

    st.markdown("""
    ### Common Solutions:

    **1. Internet Connection Issues**
    - Ensure stable internet connection
    - Check if firewall is blocking Hugging Face domains
    - Try using a VPN if in a restricted network

    **2. Memory Issues**
    - Close other applications to free up RAM
    - Restart your browser
    - Try smaller models first (t5-small, distilbert)

    **3. Dependency Issues**
    - Update transformers: `pip install transformers --upgrade`
    - Update torch: `pip install torch --upgrade`
    - Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

    **4. Cache Issues**
    - Clear Hugging Face cache: `rm -rf ~/.cache/huggingface/`
    - Clear Streamlit cache by refreshing the page
    - Restart the Streamlit app

    **5. CUDA Issues**
    - If GPU errors occur, the app will fallback to CPU
    - This is normal and expected on most systems
    - CPU processing is slower but should still work

    **6. Model Download Issues**
    - Models are downloaded on first use (can take time)
    - Ensure sufficient disk space (2-3GB per model)
    - Wait patiently during first load
    """)

def show_recommended_action():
    """Show recommended next steps"""
    st.subheader("🎯 Recommended Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Test Simple Model", use_container_width=True):
            try:
                st.info("Testing t5-small model (lightest option)...")
                summarizer = pipeline("summarization", model="t5-small", device=-1)
                test_result = summarizer("This is a test sentence for the summarization model.", max_length=20, min_length=5)
                st.success("✅ Basic model works! The issue might be with specific models.")
                st.write(f"Test result: {test_result[0]['summary_text']}")
            except Exception as e:
                st.error(f"❌ Even basic model failed: {str(e)}")
                st.info("💡 This suggests a fundamental dependency or connection issue.")

    with col2:
        if st.button("🌐 Test Connection", use_container_width=True):
            test_internet_connection()

def main():
    """Main diagnostic function"""

    # Test environment
    deps = test_environment()

    # Check critical dependencies
    critical_missing = [dep for dep, version in deps.items() if "❌" in str(version)]

    if critical_missing:
        st.error(f"**Critical dependencies missing:** {', '.join(critical_missing)}")
        st.markdown("### 🔧 Installation Commands")
        st.code("""
pip install transformers torch streamlit tokenizers
# or
pip install -r requirements.txt
        """)
        st.stop()

    # Test internet connection
    test_internet_connection()

    # Test model loading
    if st.button("🚀 Start Model Tests", type="primary"):
        test_model_loading()

    # Show troubleshooting
    show_troubleshooting_tips()

    # Recommended actions
    show_recommended_action()

    # Debug info
    with st.expander("🔍 Debug Information", expanded=False):
        st.code(f"""
Python Version: {sys.version}
Platform: {sys.platform}
Streamlit Version: {st.__version__ if hasattr(st, '__version__') else 'Unknown'}
PyTorch Version: {torch.__version__ if torch else 'Not available'}
CUDA Available: {torch.cuda.is_available() if torch else 'Unknown'}
        """)

if __name__ == "__main__":
    main()
