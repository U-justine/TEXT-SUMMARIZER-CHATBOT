#!/usr/bin/env python3
"""
Setup script for PDF Text Summarizer & ChatBot
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🚀 PDF Text Summarizer & ChatBot Setup")
    print("=" * 60)

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ Python 3.8+ required. Current: {python_version.major}.{python_version.minor}")
        return False
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

    # Check platform
    system = platform.system()
    print(f"✅ Platform: {system} {platform.release()}")

    # Check available memory (if possible)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"✅ Available RAM: {memory_gb:.1f} GB")
        if memory_gb < 4:
            print("⚠️  Warning: 4GB+ RAM recommended for optimal performance")
    except ImportError:
        print("ℹ️  Could not check memory (psutil not installed)")

    return True

def create_virtual_environment():
    """Create virtual environment"""
    venv_path = Path("venv")

    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True

    print("📦 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_pip_command():
    """Get the appropriate pip command for the platform"""
    if platform.system() == "Windows":
        return ["venv\\Scripts\\python.exe", "-m", "pip"]
    else:
        return ["venv/bin/python", "-m", "pip"]

def install_requirements():
    """Install required packages"""
    requirements_file = Path("requirements.txt")

    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False

    print("📦 Installing requirements...")
    pip_cmd = get_pip_command()

    try:
        # Upgrade pip first
        subprocess.run(pip_cmd + ["install", "--upgrade", "pip"], check=True)

        # Install requirements
        subprocess.run(pip_cmd + ["install", "-r", "requirements.txt"], check=True)

        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def download_models():
    """Download and cache ML models"""
    print("🤖 Downloading ML models (this may take a few minutes)...")

    # Create a simple script to download models
    download_script = '''
import os
os.environ['TRANSFORMERS_CACHE'] = './models_cache'
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

print("📥 Downloading summarization model...")
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    print("✅ Summarization model downloaded")
except Exception as e:
    print(f"⚠️ Could not download summarization model: {e}")

print("📥 Downloading Q&A model...")
try:
    qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    print("✅ Q&A model downloaded")
except Exception as e:
    print(f"⚠️ Could not download Q&A model: {e}")

print("🎉 Model download completed!")
'''

    # Write and execute the download script
    script_path = Path("download_models.py")
    with open(script_path, "w") as f:
        f.write(download_script)

    try:
        python_cmd = get_pip_command()[0]  # Get python executable path
        subprocess.run([python_cmd, str(script_path)], check=True)
        script_path.unlink()  # Remove the temporary script
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Could not download models: {e}")
        print("📝 Models will be downloaded on first use")
        if script_path.exists():
            script_path.unlink()
        return False

def create_launch_scripts():
    """Create platform-specific launch scripts"""
    print("📝 Creating launch scripts...")

    # Windows batch script
    windows_script = '''@echo off
echo Starting PDF Text Summarizer & ChatBot...
call venv\\Scripts\\activate
python run_app.py %*
pause
'''

    # Unix shell script
    unix_script = '''#!/bin/bash
echo "Starting PDF Text Summarizer & ChatBot..."
source venv/bin/activate
python run_app.py "$@"
'''

    try:
        # Create Windows script
        with open("start_app.bat", "w") as f:
            f.write(windows_script)

        # Create Unix script
        with open("start_app.sh", "w") as f:
            f.write(unix_script)

        # Make Unix script executable
        if platform.system() != "Windows":
            os.chmod("start_app.sh", 0o755)

        print("✅ Launch scripts created")
        return True
    except Exception as e:
        print(f"⚠️ Could not create launch scripts: {e}")
        return False

def create_desktop_shortcut():
    """Create desktop shortcut (Windows only)"""
    if platform.system() != "Windows":
        return True

    try:
        import winshell
        from win32com.client import Dispatch

        desktop = winshell.desktop()
        shortcut_path = os.path.join(desktop, "PDF Summarizer ChatBot.lnk")

        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = os.path.join(os.getcwd(), "start_app.bat")
        shortcut.WorkingDirectory = os.getcwd()
        shortcut.IconLocation = os.path.join(os.getcwd(), "start_app.bat")
        shortcut.save()

        print("✅ Desktop shortcut created")
        return True
    except ImportError:
        print("ℹ️  Desktop shortcut creation requires pywin32 (optional)")
        return True
    except Exception as e:
        print(f"⚠️ Could not create desktop shortcut: {e}")
        return True

def print_completion_message():
    """Print setup completion message"""
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)

    print("\n📋 Next steps:")
    if platform.system() == "Windows":
        print("   • Double-click 'start_app.bat' to launch the application")
        print("   • Or run: start_app.bat")
    else:
        print("   • Run: ./start_app.sh")
        print("   • Or run: bash start_app.sh")

    print("   • Or manually activate venv and run: python run_app.py")

    print("\n🌐 The application will open in your web browser")
    print("📄 Upload a PDF file to get started!")

    print("\n💡 Tips:")
    print("   • Recommended: PDFs with text content (not scanned images)")
    print("   • File size limit: 50MB for optimal performance")
    print("   • First run may be slower due to model loading")

    print("\n🆘 Need help?")
    print("   • Check README.md for detailed instructions")
    print("   • Run with --help for command line options")

def main():
    """Main setup function"""
    print_header()

    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ Please run this script from the project root directory")
        print("   (where requirements.txt is located)")
        sys.exit(1)

    steps = [
        ("Checking system requirements", check_system_requirements),
        ("Creating virtual environment", create_virtual_environment),
        ("Installing requirements", install_requirements),
        ("Creating launch scripts", create_launch_scripts),
    ]

    # Execute setup steps
    for step_name, step_func in steps:
        print(f"\n🔧 {step_name}...")
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            sys.exit(1)

    # Optional steps
    print(f"\n🔧 Downloading ML models (optional)...")
    download_models()  # Don't fail if this doesn't work

    if platform.system() == "Windows":
        print(f"\n🔧 Creating desktop shortcut (optional)...")
        create_desktop_shortcut()  # Don't fail if this doesn't work

    print_completion_message()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1)
