#!/usr/bin/env python3
"""
Enhanced Text Summarizer ChatBot Launch Script
Launches the app with text input capabilities
"""

import os
import sys
import subprocess
import streamlit.web.cli as stcli
from pathlib import Path

def main():
    """Launch the enhanced Text Summarizer ChatBot"""

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    app_path = script_dir / "app_attractive.py"

    # Check if the app file exists
    if not app_path.exists():
        print(f"❌ Error: {app_path} not found!")
        print("Please make sure you're running this script from the correct directory.")
        sys.exit(1)

    print("🚀 Starting Enhanced Text Summarizer ChatBot...")
    print("📝 Now with Direct Text Input Feature!")
    print("=" * 60)
    print("Features Available:")
    print("  📤 PDF Upload & Processing")
    print("  📝 Direct Text Input & Summarization")
    print("  ⚡ Fast AI Summarization")
    print("  🚀 Speed Q&A System")
    print("  📊 Text Statistics & Metrics")
    print("  💡 Sample Text Examples")
    print("=" * 60)
    print("🌐 Opening browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)

    # Set up Streamlit arguments
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8501",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
        "--theme.base=light"
    ]

    # Launch Streamlit
    try:
        stcli.main()
    except KeyboardInterrupt:
        print("\n👋 Thanks for using Text Summarizer ChatBot!")
        print("   Application stopped by user.")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("💡 Try running manually with: streamlit run app_attractive.py")

if __name__ == "__main__":
    main()
