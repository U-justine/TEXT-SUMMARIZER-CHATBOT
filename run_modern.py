#!/usr/bin/env python3
"""
DocuSum - Modern Document Summarizer & ChatBot
Startup script for the Flask application with modern UI
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask',
        'PyPDF2',
        'transformers',
        'torch'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Please install them using: pip install -r requirements_modern.txt")
        return False

    return True

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'static/css', 'static/js', 'templates']

    for directory in directories:
        dir_path = current_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def main():
    """Main function to start the DocuSum application"""
    logger.info("Starting DocuSum - Modern Document Summarizer & ChatBot")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Create necessary directories
    create_directories()

    # Set environment variables
    os.environ.setdefault('FLASK_APP', 'app_modern.py')
    os.environ.setdefault('FLASK_ENV', 'development')

    try:
        # Import and run the Flask app
        from app_modern import app

        logger.info("=" * 60)
        logger.info("DocuSum Application Started Successfully!")
        logger.info("Open your browser and go to: http://localhost:5000")
        logger.info("=" * 60)

        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )

    except ImportError as e:
        logger.error(f"Failed to import Flask app: {e}")
        logger.info("Make sure app_modern.py exists in the current directory")
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Application stopped by user")

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
