import streamlit as st

def main():
    """Demo of the text input feature"""
    st.title("📝 Text Summarization Demo")

    st.markdown("""
    This demo shows the new text input feature that allows users to:
    - Paste any text directly without needing PDFs
    - Get AI-powered summaries with customizable length
    - Ask questions about their text
    - See detailed metrics and statistics
    """)

    # Sample texts for demonstration
    sample_texts = {
        "News Article": """
        Scientists at MIT have developed a new artificial intelligence system that can generate highly realistic images from text descriptions. The breakthrough technology, called DALL-E 3, represents a significant advancement in the field of generative AI. The system can create detailed artwork, photographs, and illustrations based on natural language prompts provided by users. Researchers believe this technology could revolutionize creative industries, from advertising and marketing to art and design. The AI model was trained on millions of image-text pairs, allowing it to understand complex relationships between visual elements and linguistic descriptions. Early tests show that the system can generate images with unprecedented accuracy and creativity, often producing results that are indistinguishable from human-created content. However, experts also warn about potential misuse of such technology, particularly in creating deepfakes or misleading visual content. The research team is working on implementing safety measures and ethical guidelines to ensure responsible deployment of this powerful technology.
        """,

        "Scientific Paper": """
        Background: Climate change poses significant challenges to global food security, particularly affecting crop yields in developing regions. This study examines the impact of rising temperatures and changing precipitation patterns on wheat production in South Asia. Methods: We analyzed climate data from 1990 to 2020 across five countries and correlated temperature and rainfall variations with wheat yield statistics. Machine learning models were employed to predict future yield scenarios under different climate change projections. Results: Our analysis reveals a strong negative correlation between temperature increases above 2°C and wheat productivity. Regions experiencing irregular rainfall patterns showed up to 25% reduction in yields compared to historical averages. The predictive models suggest that without adaptation strategies, wheat production could decline by 40% by 2050. Conclusions: Urgent implementation of climate-resilient agricultural practices is essential to maintain food security in the region. These findings highlight the need for international cooperation in developing sustainable farming techniques and climate adaptation technologies.
        """,

        "Business Report": """
        Executive Summary: The global e-commerce market experienced unprecedented growth during 2023, with total sales reaching $6.2 trillion, representing a 15% increase from the previous year. Mobile commerce accounted for 60% of all online transactions, highlighting the continued shift toward smartphone-based shopping experiences. Key market drivers include improved internet infrastructure, enhanced payment security, and changing consumer behavior patterns accelerated by recent global events. North America and Asia-Pacific regions dominated the market, contributing 70% of total e-commerce revenue. Small and medium enterprises (SMEs) showed remarkable adaptability, with 78% expanding their online presence during the reporting period. Challenges remain in logistics optimization, customer data privacy, and sustainable packaging solutions. Looking ahead, artificial intelligence integration, voice commerce, and augmented reality shopping experiences are expected to drive the next wave of e-commerce innovation. Industry experts project continued growth of 12-14% annually through 2025, contingent upon technological advancement and regulatory stability.
        """
    }

    st.markdown("## 📝 New Features in Text Input Tab")

    # Feature highlights
    col1, col2 = st.columns(2)

    with col1:
        st.info("**✨ Key Features:**")
        st.write("• Direct text input (no PDF required)")
        st.write("• Customizable summary length")
        st.write("• Real-time text statistics")
        st.write("• Question & Answer capability")
        st.write("• Sample text examples")
        st.write("• Support for up to 10,000 characters")

    with col2:
        st.success("**🎯 Perfect for:**")
        st.write("• News articles and blog posts")
        st.write("• Research papers and reports")
        st.write("• Essays and academic content")
        st.write("• Business documents")
        st.write("• Web content and reviews")
        st.write("• Social media posts")

    st.markdown("## 📊 Sample Texts")
    st.markdown("Choose a sample text to see how the feature works:")

    # Display sample texts
    for title, text in sample_texts.items():
        with st.expander(f"📄 {title} Sample", expanded=False):
            st.text_area(f"{title} content:", text, height=200, disabled=True)

            # Show statistics for each sample
            words = len(text.split())
            chars = len(text)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Words", f"{words:,}")
            with col2:
                st.metric("Characters", f"{chars:,}")
            with col3:
                st.metric("Est. Reading Time", f"{words//200 + 1} min")

    st.markdown("## 🚀 How to Use")

    st.markdown("""
    ### Step 1: Navigate to the "Text Input" Tab
    - Open the main application
    - Click on the **📝 Text Input** tab

    ### Step 2: Enter Your Text
    - Paste your text in the large text area
    - Or click one of the sample buttons to try examples
    - Text statistics will appear automatically

    ### Step 3: Choose Summary Options
    - **Summary Length:** Short (25-50 words), Medium (50-100 words), Long (100-200 words)
    - **Summary Style:** Extractive or Abstractive

    ### Step 4: Generate Summary
    - Click "✨ Generate Summary from Text"
    - AI will analyze your text and create a summary
    - View compression metrics and statistics

    ### Step 5: Ask Questions (Optional)
    - Use the Q&A section to ask questions about your text
    - Get AI-powered answers with context understanding
    """)

    st.markdown("## 💡 Tips for Best Results")

    st.info("""
    **For Better Summaries:**
    - Use text with at least 50 characters
    - Well-structured content works best
    - Remove excessive formatting or special characters

    **For Better Q&A:**
    - Ask specific, clear questions
    - Questions about main topics work better than details
    - Try different phrasings if needed
    """)

    st.markdown("## 🔧 Technical Details")

    with st.expander("Technical Information", expanded=False):
        st.markdown("""
        **Models Used:**
        - **Summarization:** DistilBART CNN (sshleifer/distilbart-cnn-12-6)
        - **Question Answering:** DistilBERT Squad (distilbert-base-cased-distilled-squad)

        **Limits:**
        - Maximum text input: 10,000 characters
        - Processing time: 2-10 seconds depending on text length
        - Supported languages: Primarily English

        **Performance Optimizations:**
        - Text chunking for long content
        - GPU acceleration when available
        - Model caching for faster subsequent requests
        """)

    st.markdown("---")
    st.markdown("**Ready to try it out?** Go to the main app and click the **📝 Text Input** tab!")

if __name__ == "__main__":
    main()
