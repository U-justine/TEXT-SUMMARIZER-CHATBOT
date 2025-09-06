import streamlit as st

def test_html_rendering():
    """Test script to debug HTML rendering issues in Streamlit"""

    st.title("HTML Rendering Test")

    st.subheader("Test 1: Simple HTML with unsafe_allow_html=True")
    st.markdown("""
    <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
        <h3 style="color: #333;">This should be styled HTML</h3>
        <p style="color: #666;">If you see this with styling, HTML is working!</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Test 2: HTML without unsafe_allow_html (should show as text)")
    st.markdown("""
    <div style="background-color: #f0f0f0; padding: 20px;">
        <h3>This should show as raw HTML text</h3>
    </div>
    """)

    st.subheader("Test 3: The problematic footer HTML")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background-color: rgba(102, 126, 234, 0.1); border-radius: 15px; margin-top: 2rem;">
        <h3 style="color: #1e293b; margin-bottom: 1rem;">🚀 PDF AI Assistant</h3>
        <p style="color: #64748b; margin-bottom: 1.5rem;">
            Powered by state-of-the-art AI models from Hugging Face
        </p>
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <span style="background-color: #667eea; color: white; padding: 8px 16px; border-radius: 20px; margin: 4px; display: inline-block;">
                🤖 BART Large CNN
            </span>
            <span style="background-color: #10b981; color: white; padding: 8px 16px; border-radius: 20px; margin: 4px; display: inline-block;">
                💬 RoBERTa Squad2
            </span>
            <span style="background-color: #f59e0b; color: white; padding: 8px 16px; border-radius: 20px; margin: 4px; display: inline-block;">
                📄 PyPDF2 Engine
            </span>
        </div>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 1rem 0;">
        <p style="color: #94a3b8; margin-bottom: 0.5rem;">
            Built with ❤️ using Streamlit • Transform your documents with AI
        </p>
        <p style="color: #cbd5e1; font-style: italic;">
            Upload • Extract • Summarize • Ask Questions
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Test 4: Native Streamlit Alternative")
    st.info("🚀 **PDF AI Assistant** - Powered by state-of-the-art AI models")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🤖 **BART Large CNN**")
    with col2:
        st.markdown("💬 **RoBERTa Squad2**")
    with col3:
        st.markdown("📄 **PyPDF2 Engine**")

    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit • Transform your documents with AI")
    st.caption("Upload • Extract • Summarize • Ask Questions")

    st.subheader("Debugging Info")
    st.write(f"Streamlit version: {st.__version__}")

    # Show what the user is experiencing
    st.subheader("What you might be seeing:")
    st.code('''
<div style="display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1.5rem;">
    <span style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 0.75rem 1.5rem; border-radius: 25px; font-size: 0.9rem; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);">
        🤖 BART Large CNN
    </span>
    <span style="background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 0.75rem 1.5rem; border-radius: 25px; font-size: 0.9rem; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);">
        💬 RoBERTa Squad2
    </span>
    <span style="background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 0.75rem 1.5rem; border-radius: 25px; font-size: 0.9rem; box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);">
        📄 PyPDF2 Engine
    </span>
</div>

<div style="border-top: 1px solid rgba(255,255,255,0.3); padding-top: 1rem;">
    <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem;">
        Built with ❤️ using Streamlit • Transform your documents with AI
    </p>
    <p style="color: #cbd5e1; font-size: 0.8rem; font-style: italic;">
        Upload • Extract • Summarize • Ask Questions
    </p>
</div>
    ''')

if __name__ == "__main__":
    test_html_rendering()
