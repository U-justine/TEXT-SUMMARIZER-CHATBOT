"""
Configuration settings for PDF Text Summarizer & ChatBot application
"""

import os
from typing import Dict, Any

# Application Settings
APP_CONFIG = {
    "title": "PDF Text Summarizer & ChatBot",
    "page_icon": "📄",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "version": "1.0.0"
}

# Model Configuration
MODEL_CONFIG = {
    "summarization": {
        "primary_model": "facebook/bart-large-cnn",
        "fallback_model": "sshleifer/distilbart-cnn-12-6",
        "max_length": 150,
        "min_length": 30,
        "do_sample": False,
        "chunk_size": 1024,
        "max_input_length": 1024
    },
    "question_answering": {
        "primary_model": "deepset/roberta-base-squad2",
        "fallback_model": "distilbert-base-uncased-distilled-squad",
        "confidence_threshold": 0.1,
        "max_context_length": 2000,
        "return_all_scores": True
    }
}

# Text Processing Settings
TEXT_CONFIG = {
    "min_text_length": 20,  # Minimum words for processing
    "max_text_length": 50000,  # Maximum words allowed
    "chunk_overlap": 50,  # Overlap between chunks in words
    "sentence_min_length": 5,  # Minimum sentence length in characters
    "meaningful_content_ratio": 0.5  # Minimum ratio of meaningful characters
}

# PDF Processing Settings
PDF_CONFIG = {
    "max_file_size_mb": 50,  # Maximum file size in MB
    "allowed_extensions": [".pdf"],
    "encoding": "utf-8"
}

# UI Configuration
UI_CONFIG = {
    "sidebar_width": 300,
    "max_text_area_height": 300,
    "default_expanded_sections": {
        "extracted_text": False,
        "summary": True,
        "qa": True
    },
    "show_statistics": True,
    "show_model_info": True
}

# Performance Settings
PERFORMANCE_CONFIG = {
    "enable_gpu": True,
    "torch_dtype": "float16",  # Use float16 for GPU, float32 for CPU
    "low_cpu_mem_usage": True,
    "cache_models": True,
    "max_workers": 4  # For multiprocessing if needed
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console"],  # Options: console, file
    "log_file": "app.log"
}

# Error Messages
ERROR_MESSAGES = {
    "pdf_upload_failed": "Failed to upload PDF file. Please try again.",
    "text_extraction_failed": "Could not extract text from PDF. The file might be corrupted or password-protected.",
    "text_too_short": "The extracted text is too short for meaningful processing. Please upload a document with more content.",
    "text_too_long": "The document is too large to process. Please try a smaller file.",
    "summarization_failed": "Failed to generate summary. Please try again or check your internet connection.",
    "qa_failed": "Could not process your question. Please try rephrasing it.",
    "model_load_failed": "Failed to load AI models. Please check your internet connection and try again.",
    "insufficient_confidence": "I'm not confident enough to answer this question based on the provided document.",
    "no_question_provided": "Please enter a question before asking.",
    "no_context_available": "No document context available. Please upload and process a PDF first."
}

# Success Messages
SUCCESS_MESSAGES = {
    "pdf_uploaded": "PDF file uploaded successfully!",
    "text_extracted": "Text extracted successfully from PDF!",
    "summary_generated": "Summary generated successfully!",
    "question_answered": "Question answered successfully!",
    "model_loaded": "AI models loaded successfully!"
}

# Default Settings
DEFAULTS = {
    "summary_max_length": 150,
    "summary_min_length": 30,
    "qa_confidence_threshold": 0.1,
    "text_chunk_size": 1024,
    "context_max_length": 2000
}

# Theme Configuration
THEME_CONFIG = {
    "primary_color": "#1f77b4",
    "background_color": "#ffffff",
    "secondary_background_color": "#f0f2f6",
    "text_color": "#262730",
    "font": "sans serif"
}

# Custom CSS Styles
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin: 1rem 0;
        font-weight: 600;
    }
    .uploaded-file-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .summary-box {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .qa-response {
        background-color: #e6f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2e8b57;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .statistics-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .error-message {
        background-color: #ffe6e6;
        color: #d63384;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #d63384;
    }
    .success-message {
        background-color: #d1f2eb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1565c0;
        transform: translateY(-1px);
    }
    .model-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
"""

# API Configuration (if needed for external services)
API_CONFIG = {
    "timeout": 30,
    "max_retries": 3,
    "retry_delay": 1
}

# Environment Variables
def get_env_var(var_name: str, default_value: Any = None) -> Any:
    """Get environment variable with default fallback"""
    return os.getenv(var_name, default_value)

# Dynamic Configuration Updates
def update_config(section: str, key: str, value: Any) -> bool:
    """Update configuration dynamically"""
    try:
        config_map = {
            "app": APP_CONFIG,
            "model": MODEL_CONFIG,
            "text": TEXT_CONFIG,
            "pdf": PDF_CONFIG,
            "ui": UI_CONFIG,
            "performance": PERFORMANCE_CONFIG
        }

        if section in config_map and key in config_map[section]:
            config_map[section][key] = value
            return True
        return False
    except Exception:
        return False

# Configuration Validation
def validate_config() -> Dict[str, bool]:
    """Validate configuration settings"""
    validation_results = {}

    # Validate model configuration
    validation_results["model_config"] = (
        isinstance(MODEL_CONFIG["summarization"]["max_length"], int) and
        MODEL_CONFIG["summarization"]["max_length"] > 0
    )

    # Validate text configuration
    validation_results["text_config"] = (
        TEXT_CONFIG["min_text_length"] > 0 and
        TEXT_CONFIG["max_text_length"] > TEXT_CONFIG["min_text_length"]
    )

    # Validate PDF configuration
    validation_results["pdf_config"] = (
        PDF_CONFIG["max_file_size_mb"] > 0 and
        len(PDF_CONFIG["allowed_extensions"]) > 0
    )

    return validation_results
