# 📄 PDF Text Summarizer & ChatBot

A powerful web application that enables users to upload PDF documents, extract and summarize their text content, and interact with an intelligent Q&A chatbot to get answers related to the document content.

## ✨ Features

- **📤 PDF Upload**: Upload PDF files for processing with drag-and-drop interface
- **🔍 Text Extraction**: Extract and clean text from uploaded PDF documents using PyPDF2
- **📋 Text Summarization**: Generate concise, meaningful summaries using state-of-the-art transformer models
- **🤖 Q&A Chatbot**: Ask questions about the document content and receive relevant, contextual answers
- **📊 Text Analytics**: View detailed statistics about your document (word count, sentences, etc.)
- **🎨 User-Friendly Interface**: Clean, responsive UI with modern design and intuitive navigation
- **⚡ Performance Optimized**: Efficient text processing with chunking and caching for large documents
- **🔧 Model Selection**: Multiple AI models available for different use cases

## 🚀 Technologies Used

- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[Hugging Face Transformers](https://huggingface.co/transformers/)** - Pre-trained AI models for NLP tasks
- **[PyPDF2](https://pypdf2.readthedocs.io/)** - PDF text extraction
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[BART](https://huggingface.co/facebook/bart-large-cnn)** - Text summarization model
- **[RoBERTa](https://huggingface.co/deepset/roberta-base-squad2)** - Question answering model

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- 4GB+ RAM recommended for optimal performance
- GPU support optional but recommended for faster processing

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Text Summarizer ChatBot"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download models (first run will automatically download):**
   The application will automatically download the required models on first use. This may take several minutes.

## 🏃‍♂️ Running the Application

1. **Start the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:8501
   ```

3. **Upload a PDF file** and start exploring!

## 💡 Usage Guide

### 1. Upload PDF Document
- Click on "Choose a PDF file" or drag and drop your PDF
- Supported formats: PDF files only
- File size limit: Recommended under 50MB for optimal performance

### 2. Extract Text
- Click "🔍 Extract Text" to process the uploaded PDF
- View extracted text statistics (word count, character count, etc.)
- Review the extracted content in the expandable text area

### 3. Generate Summary
- Click "✨ Generate Summary" to create an AI-powered summary
- The system automatically handles long documents by chunking
- View the generated summary in the dedicated section

### 4. Ask Questions
- Type your question in the Q&A input field
- Click "🚀 Ask Question" to get AI-powered answers
- Questions should be related to the document content
- The system provides confidence scores for answers

## 🎯 Model Information

### Summarization Models
- **Primary**: BART Large CNN (`facebook/bart-large-cnn`)
  - High-quality summaries
  - Optimized for news articles and documents
  - Fallback: DistilBART CNN for faster processing

### Question Answering Models
- **Primary**: RoBERTa Base Squad2 (`deepset/roberta-base-squad2`)
  - High accuracy question answering
  - Handles both answerable and unanswerable questions
  - Fallback: DistilBERT Squad for faster processing

## 📁 Project Structure

```
Text Summarizer ChatBot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── utils/
│   ├── __init__.py       # Package initialization
│   ├── text_processor.py # Text processing utilities
│   └── model_utils.py    # ML model management
└── README.md             # Project documentation
```

## 🔧 Configuration

### Model Settings
The application uses sensible defaults but can be customized:

- **Summarization**:
  - Max summary length: 150 tokens
  - Min summary length: 30 tokens
  - Chunk size for long documents: 1024 tokens

- **Question Answering**:
  - Confidence threshold: 0.1
  - Max context length: 2000 characters

### Performance Optimization
- Models are cached using Streamlit's `@st.cache_resource`
- Text is chunked for efficient processing of large documents
- GPU acceleration automatically used if available

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory Errors**
- Try using smaller PDF files (< 20MB)
- Restart the application to clear memory cache
- Consider using a machine with more RAM

**2. Model Download Issues**
- Ensure stable internet connection
- Check if you have sufficient disk space (2-3GB for models)
- Try running the application multiple times

**3. PDF Text Extraction Problems**
- Ensure PDF contains text (not just images)
- Try converting scanned PDFs to text-based PDFs first
- Some encrypted PDFs may not be supported

**4. Slow Performance**
- First run is slower due to model loading
- Consider using GPU if available
- Close other memory-intensive applications

### Error Messages

- **"Text is too short to summarize"**: Upload a document with more content
- **"I'm not confident enough to answer"**: Try rephrasing your question or ask about different content
- **"Error extracting text from PDF"**: Check if PDF is corrupted or encrypted

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for providing excellent pre-trained models
- **Streamlit** for the amazing web app framework
- **PyPDF2** developers for PDF processing capabilities
- **Facebook AI Research** for the BART model
- **Deepset** for the RoBERTa QA model

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information
4. Include error messages and system information

## 🔮 Future Enhancements

- [ ] Support for multiple file formats (DOCX, TXT, etc.)
- [ ] Multi-language support
- [ ] Document comparison features
- [ ] Export summaries to different formats
- [ ] Integration with cloud storage services
- [ ] Advanced analytics and insights
- [ ] User authentication and document history
- [ ] API endpoints for programmatic access

---

**Made with ❤️ using Streamlit and Hugging Face Transformers**

*For more information about the underlying technologies, visit [Streamlit Documentation](https://docs.streamlit.io/) and [Hugging Face Documentation](https://huggingface.co/docs).*