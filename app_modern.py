from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import PyPDF2
import io
import logging
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import torch
from werkzeug.utils import secure_filename
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
summarizer = None
qa_pipeline = None
qa_tokenizer = None
qa_model = None

# Initialize models
def load_models():
    global summarizer, qa_pipeline, qa_tokenizer, qa_model

    try:
        logger.info("Loading summarization model...")
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )

        logger.info("Loading Q&A model...")
        model_name = "distilbert-base-cased-distilled-squad"
        qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa_pipeline = pipeline(
            "question-answering",
            model=qa_model,
            tokenizer=qa_tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

        logger.info("Models loaded successfully!")

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        # Fallback to smaller models if the main ones fail
        try:
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            logger.info("Fallback models loaded successfully!")
        except Exception as fallback_error:
            logger.error(f"Fallback model loading also failed: {str(fallback_error)}")

# Load models on startup
load_models()

# Utility functions
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"

        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None

def chunk_text(text, max_chunk_length=1024):
    """Split text into smaller chunks for processing"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def summarize_text(text):
    """Generate summary of the text"""
    try:
        if not summarizer:
            return "Summarization model not available. Please try again later."

        # Chunk text if it's too long
        max_length = 1024
        if len(text) > max_length:
            chunks = chunk_text(text, max_length)
            summaries = []

            for chunk in chunks[:3]:  # Limit to first 3 chunks for performance
                if len(chunk.strip()) > 50:  # Only summarize substantial chunks
                    try:
                        summary = summarizer(
                            chunk,
                            max_length=150,
                            min_length=50,
                            do_sample=False
                        )
                        summaries.append(summary[0]['summary_text'])
                    except Exception as chunk_error:
                        logger.error(f"Error summarizing chunk: {str(chunk_error)}")
                        continue

            return " ".join(summaries) if summaries else "Unable to generate summary."

        else:
            summary = summarizer(
                text,
                max_length=200,
                min_length=50,
                do_sample=False
            )
            return summary[0]['summary_text']

    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        return "I apologize, but I encountered an error while generating the summary. Please try again."

def answer_question(question, context):
    """Answer a question based on the document context"""
    try:
        if not qa_pipeline:
            return "Question answering model not available. Please try again later."

        # Limit context length for performance
        max_context_length = 2000
        if len(context) > max_context_length:
            # Try to find the most relevant part of the context
            context = context[:max_context_length] + "..."

        result = qa_pipeline(question=question, context=context)

        # Format the response
        answer = result['answer']
        confidence = result['score']

        if confidence > 0.3:
            return f"{answer}"
        else:
            return "I'm not confident about this answer based on the document content. Could you rephrase your question or ask about a different topic?"

    except Exception as e:
        logger.error(f"Error in question answering: {str(e)}")
        return "I apologize, but I encountered an error while processing your question. Please try again."

# Routes
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and file.filename.lower().endswith('.pdf'):
            # Secure the filename
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename

            # Save file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract text
            with open(filepath, 'rb') as pdf_file:
                text = extract_text_from_pdf(pdf_file)

            if text:
                # Store text in session or database (for demo, we'll return it)
                return jsonify({
                    'success': True,
                    'filename': file.filename,
                    'text_length': len(text),
                    'preview': text[:200] + "..." if len(text) > 200 else text,
                    'file_id': filename  # Use this to reference the file later
                })
            else:
                return jsonify({'error': 'Could not extract text from PDF'}), 400

        else:
            return jsonify({'error': 'Only PDF files are allowed'}), 400

    except Exception as e:
        logger.error(f"Error in file upload: {str(e)}")
        return jsonify({'error': 'An error occurred during upload'}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    """Generate summary of uploaded document"""
    try:
        data = request.get_json()
        file_id = data.get('file_id')

        if not file_id:
            return jsonify({'error': 'No file ID provided'}), 400

        # Read the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        # Extract text and generate summary
        with open(filepath, 'rb') as pdf_file:
            text = extract_text_from_pdf(pdf_file)

        if text:
            summary = summarize_text(text)
            return jsonify({
                'success': True,
                'summary': summary,
                'original_length': len(text),
                'summary_length': len(summary)
            })
        else:
            return jsonify({'error': 'Could not extract text from file'}), 400

    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        return jsonify({'error': 'An error occurred during summarization'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer questions about the uploaded document"""
    try:
        data = request.get_json()
        question = data.get('question')
        file_id = data.get('file_id')

        if not question or not file_id:
            return jsonify({'error': 'Question and file ID are required'}), 400

        # Read the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        # Extract text and answer question
        with open(filepath, 'rb') as pdf_file:
            text = extract_text_from_pdf(pdf_file)

        if text:
            answer = answer_question(question, text)
            return jsonify({
                'success': True,
                'question': question,
                'answer': answer
            })
        else:
            return jsonify({'error': 'Could not extract text from file'}), 400

    except Exception as e:
        logger.error(f"Error in question answering: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your question'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle general chat messages"""
    try:
        data = request.get_json()
        message = data.get('message')
        file_id = data.get('file_id')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Check if it's a question about the document
        if file_id:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as pdf_file:
                    text = extract_text_from_pdf(pdf_file)

                if text:
                    # Determine if it's a summarization request or question
                    if any(word in message.lower() for word in ['summary', 'summarize', 'sum up', 'overview']):
                        response = summarize_text(text)
                    else:
                        response = answer_question(message, text)

                    return jsonify({
                        'success': True,
                        'response': response,
                        'type': 'document_based'
                    })

        # General chat response
        general_responses = {
            'hello': "Hello! I'm your AI document assistant. Upload a PDF to get started!",
            'hi': "Hi there! How can I help you with your document today?",
            'help': "I can help you summarize documents and answer questions about them. Just upload a PDF file to begin!",
            'thank': "You're welcome! Is there anything else I can help you with?",
        }

        message_lower = message.lower()
        for key, response in general_responses.items():
            if key in message_lower:
                return jsonify({
                    'success': True,
                    'response': response,
                    'type': 'general'
                })

        # Default response
        return jsonify({
            'success': True,
            'response': "I'm here to help you with document analysis. Please upload a PDF file so I can assist you with summaries and answer questions about its content.",
            'type': 'general'
        })

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your message'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'summarizer': summarizer is not None,
            'qa_pipeline': qa_pipeline is not None
        },
        'timestamp': datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413

# Static files
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Run the app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
