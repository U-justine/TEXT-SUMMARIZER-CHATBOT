#!/usr/bin/env python3
"""
Launch script for PDF Text Summarizer & ChatBot application
"""

import os
import sys
import subprocess
import argparse
import webbrowser
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'transformers', 'torch', 'PyPDF2',
        'tokenizers', 'numpy', 'pandas'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def install_requirements():
    """Install requirements from requirements.txt"""
    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        print("❌ requirements.txt file not found")
        return False

    print("📦 Installing requirements...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True, capture_output=True, text=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def launch_streamlit_app(app_file="app.py", port=8501, open_browser=True):
    """Launch the Streamlit application"""
    app_path = Path(__file__).parent / app_file

    if not app_path.exists():
        print(f"❌ Application file {app_file} not found")
        # Try alternative app files
        alternative_files = ["app_enhanced.py", "main.py"]
        for alt_file in alternative_files:
            alt_path = Path(__file__).parent / alt_file
            if alt_path.exists():
                print(f"🔄 Using {alt_file} instead")
                app_path = alt_path
                app_file = alt_file
                break
        else:
            print("❌ No valid application file found")
            return False

    print(f"🚀 Starting {app_file}...")
    print(f"🌐 Application will be available at: http://localhost:{port}")

    # Prepare streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.headless", "true" if not open_browser else "false"
    ]

    try:
        # Start the application
        process = subprocess.Popen(cmd)

        if open_browser:
            # Wait a moment for server to start, then open browser
            print("⏳ Waiting for server to start...")
            time.sleep(3)
            webbrowser.open(f"http://localhost:{port}")

        print("✅ Application started successfully!")
        print("📝 Instructions:")
        print("   1. Upload a PDF document")
        print("   2. Extract text from the PDF")
        print("   3. Generate summary (optional)")
        print("   4. Ask questions about the content")
        print("\n🛑 Press Ctrl+C to stop the application")

        # Wait for the process to finish
        process.wait()

    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        try:
            process.terminate()
        except:
            pass
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False

    return True

def show_help():
    """Show help information"""
    help_text = """
📄 PDF Text Summarizer & ChatBot - Launch Script

USAGE:
    python run_app.py [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -p, --port PORT         Port number (default: 8501)
    --no-browser           Don't open browser automatically
    --install              Install requirements first
    --check                Only check requirements
    --app APP_FILE         Specify app file (default: app.py)

EXAMPLES:
    python run_app.py                    # Launch with default settings
    python run_app.py -p 8080           # Launch on port 8080
    python run_app.py --install         # Install requirements first
    python run_app.py --app app_enhanced.py  # Use enhanced app

FEATURES:
    ✨ Text Summarization using BART/T5 models
    🤖 Q&A ChatBot using RoBERTa/DistilBERT
    📄 PDF text extraction with PyPDF2
    🎨 Modern Streamlit interface
    📊 Text analytics and statistics
    """
    print(help_text)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Launch PDF Text Summarizer & ChatBot",
        add_help=False
    )

    parser.add_argument('-h', '--help', action='store_true',
                       help='Show help message')
    parser.add_argument('-p', '--port', type=int, default=8501,
                       help='Port number (default: 8501)')
    parser.add_argument('--no-browser', action='store_true',
                       help="Don't open browser automatically")
    parser.add_argument('--install', action='store_true',
                       help='Install requirements first')
    parser.add_argument('--check', action='store_true',
                       help='Only check requirements')
    parser.add_argument('--app', default='app.py',
                       help='Specify app file (default: app.py)')

    args = parser.parse_args()

    if args.help:
        show_help()
        return

    print("=" * 60)
    print("🚀 PDF Text Summarizer & ChatBot Launcher")
    print("=" * 60)

    # Check Python version
    check_python_version()

    # Install requirements if requested
    if args.install:
        if not install_requirements():
            sys.exit(1)

    # Check requirements
    if not check_requirements():
        if not args.check:
            print("\n💡 Run with --install to install missing packages")
        sys.exit(1)

    # If only checking, exit here
    if args.check:
        print("✅ All requirements satisfied!")
        return

    # Launch the application
    success = launch_streamlit_app(
        app_file=args.app,
        port=args.port,
        open_browser=not args.no_browser
    )

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
