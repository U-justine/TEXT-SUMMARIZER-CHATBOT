import streamlit as st

def render_footer():
    """
    Render a footer using pure Streamlit components as an alternative to HTML
    """
    # Add some space
    st.markdown("---")

    # Main title
    st.markdown("### 🚀 PDF AI Assistant")
    st.markdown("*Powered by state-of-the-art AI models from Hugging Face*")

    # Model badges using columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🤖 **BART Large CNN**\nText Summarization")

    with col2:
        st.success("💬 **RoBERTa Squad2**\nQuestion Answering")

    with col3:
        st.warning("📄 **PyPDF2 Engine**\nDocument Processing")

    # Bottom section
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit • Transform your documents with AI")
    st.caption("Upload • Extract • Summarize • Ask Questions")

def render_footer_compact():
    """
    Render a more compact footer version
    """
    st.markdown("---")

    # Center-aligned content using columns
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 🚀 PDF AI Assistant")
        st.markdown("*Powered by Hugging Face AI models*")

        # Model tags
        st.markdown("🤖 BART Large CNN  •  💬 RoBERTa Squad2  •  📄 PyPDF2 Engine")

        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit")
        st.caption("Upload • Extract • Summarize • Ask Questions")

def render_footer_with_metrics():
    """
    Render footer with metrics-style display
    """
    st.markdown("---")
    st.markdown("### 🚀 PDF AI Assistant")
    st.markdown("*Powered by state-of-the-art AI models*")

    # Use metrics for a clean look
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Summarization", "BART Large CNN", "🤖")

    with col2:
        st.metric("Q&A System", "RoBERTa Squad2", "💬")

    with col3:
        st.metric("PDF Engine", "PyPDF2", "📄")

    st.markdown("---")
    st.markdown("**Built with ❤️ using Streamlit** • Transform your documents with AI")
    st.caption("Upload • Extract • Summarize • Ask Questions")

def render_footer_expander():
    """
    Render footer inside an expander for a cleaner look
    """
    with st.expander("ℹ️ About This Application", expanded=False):
        st.markdown("### 🚀 PDF AI Assistant")
        st.markdown("This application uses state-of-the-art AI models from Hugging Face to provide:")

        st.markdown("**🤖 Text Summarization**")
        st.markdown("- Primary: BART Large CNN (`facebook/bart-large-cnn`)")
        st.markdown("- Fallback: DistilBART CNN for faster processing")

        st.markdown("**💬 Question Answering**")
        st.markdown("- Primary: RoBERTa Squad2 (`deepset/roberta-base-squad2`)")
        st.markdown("- Fallback: DistilBERT for faster processing")

        st.markdown("**📄 Document Processing**")
        st.markdown("- PyPDF2 for reliable PDF text extraction")
        st.markdown("- Smart text chunking for large documents")

        st.markdown("---")
        st.markdown("**Built with ❤️ using Streamlit** • Transform your documents with AI")

# Example usage functions that can be imported
def add_footer(style="default"):
    """
    Add footer to the Streamlit app with different style options

    Args:
        style (str): Footer style - "default", "compact", "metrics", or "expander"
    """
    if style == "compact":
        render_footer_compact()
    elif style == "metrics":
        render_footer_with_metrics()
    elif style == "expander":
        render_footer_expander()
    else:
        render_footer()

# Test function
def main():
    """Test all footer styles"""
    st.title("Footer Style Test")

    style = st.selectbox(
        "Choose footer style:",
        ["default", "compact", "metrics", "expander"]
    )

    st.markdown("### Sample Content Above Footer")
    st.write("This is where your main app content would go...")

    # Add the selected footer
    add_footer(style)

if __name__ == "__main__":
    main()
