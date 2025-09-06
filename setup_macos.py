#!/usr/bin/env python3
"""
macOS-specific setup script for PDF Text Summarizer & ChatBot
Handles common macOS Python/pip installation issues
"""

import os
import sys
import subprocess
import platform

def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🍎 PDF Text Summarizer & ChatBot - macOS Setup")
    print("=" * 60)

def check_macos():
    """Verify we're running on macOS"""
    if platform.system() != "Darwin":
        print("❌ This script is designed for macOS only")
        return False

    macos_version = platform.mac_ver()[0]
    print(f"✅ macOS {macos_version} detected")
    return True

def check_xcode_tools():
    """Check if Xcode command line tools are installed"""
    print("🔍 Checking Xcode command line tools...")

    try:
        result = subprocess.run(['xcode-select', '-p'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Xcode command line tools are installed")
            return True
        else:
            print("❌ Xcode command line tools not found")
            return False
    except FileNotFoundError:
        print("❌ xcode-select not found")
        return False

def install_xcode_tools():
    """Install Xcode command line tools"""
    print("📦 Installing Xcode command line tools...")
    print("⏳ A dialog will appear - please click 'Install'")

    try:
        subprocess.run(['xcode-select', '--install'], check=True)
        print("✅ Xcode command line tools installation initiated")
        print("⚠️  Please wait for installation to complete, then run this script again")
        return False  # Need to restart after installation
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Xcode tools: {e}")
        return False

def check_python():
    """Check Python installation"""
    print("🐍 Checking Python installation...")

    # Check python3
    try:
        result = subprocess.run([sys.executable, '--version'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ {version}")

            # Check if it's the right version
            if sys.version_info >= (3, 8):
                return True
            else:
                print(f"❌ Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
                return False
        else:
            print("❌ Python not found")
            return False
    except Exception as e:
        print(f"❌ Error checking Python: {e}")
        return False

def check_pip():
    """Check if pip is available"""
    print("📦 Checking pip...")

    # Try different pip commands
    pip_commands = [
        [sys.executable, '-m', 'pip', '--version'],
        ['pip3', '--version'],
        ['pip', '--version']
    ]

    for cmd in pip_commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ pip found: {result.stdout.strip()}")
                return cmd[:-1]  # Return command without --version
        except FileNotFoundError:
            continue

    print("❌ pip not found")
    return None

def install_pip():
    """Install pip using get-pip.py"""
    print("📦 Installing pip...")

    try:
        # Download get-pip.py
        import urllib.request

        pip_url = "https://bootstrap.pypa.io/get-pip.py"
        print(f"📥 Downloading get-pip.py from {pip_url}")

        with urllib.request.urlopen(pip_url) as response:
            with open('get-pip.py', 'wb') as f:
                f.write(response.read())

        # Install pip
        result = subprocess.run([sys.executable, 'get-pip.py'], check=True)

        # Clean up
        os.remove('get-pip.py')

        print("✅ pip installed successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to install pip: {e}")
        print("💡 Try installing Python from python.org or using Homebrew")
        return False

def install_requirements(pip_cmd):
    """Install requirements using the available pip command"""
    print("📦 Installing requirements...")

    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False

    try:
        # Upgrade pip first
        print("⬆️ Upgrading pip...")
        subprocess.run(pip_cmd + ['install', '--upgrade', 'pip'],
                      check=True)

        # Install requirements
        print("📦 Installing packages...")
        subprocess.run(pip_cmd + ['install', '-r', 'requirements.txt'],
                      check=True)

        print("✅ Requirements installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   • Try: python3 -m pip install --user -r requirements.txt")
        print("   • Or install Homebrew and use: brew install python")
        print("   • Check your internet connection")
        return False

def suggest_alternatives():
    """Suggest alternative installation methods"""
    print("\n🔄 Alternative Installation Methods:")
    print("\n1️⃣ Using Homebrew (Recommended):")
    print("   • Install Homebrew:")
    print("     /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
    print("   • Install Python: brew install python")
    print("   • Then run: pip3 install -r requirements.txt")

    print("\n2️⃣ Using Python.org installer:")
    print("   • Download Python from: https://www.python.org/downloads/")
    print("   • Install the .pkg file")
    print("   • Then run: python3 -m pip install -r requirements.txt")

    print("\n3️⃣ Using Anaconda:")
    print("   • Download Anaconda from: https://www.anaconda.com/")
    print("   • Install and use: conda install --file requirements.txt")

def create_launch_script():
    """Create macOS-specific launch script"""
    print("📝 Creating launch script...")

    script_content = '''#!/bin/bash
# macOS Launch script for PDF Text Summarizer & ChatBot

echo "🚀 Starting PDF Text Summarizer & ChatBot..."

# Find Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found. Please install Python first."
    exit 1
fi

# Check if streamlit is installed
if ! $PYTHON_CMD -c "import streamlit" &> /dev/null; then
    echo "❌ Streamlit not installed. Please run setup first."
    exit 1
fi

# Launch the app
echo "🌐 Opening http://localhost:8501 in your browser..."
$PYTHON_CMD -m streamlit run app.py --server.port 8501

echo "👋 Application stopped."
'''

    try:
        with open('launch_macos.sh', 'w') as f:
            f.write(script_content)

        # Make executable
        os.chmod('launch_macos.sh', 0o755)

        print("✅ Launch script created: launch_macos.sh")
        return True
    except Exception as e:
        print(f"⚠️ Could not create launch script: {e}")
        return False

def main():
    """Main setup function"""
    print_header()

    # Check if we're on macOS
    if not check_macos():
        return False

    # Check Xcode tools
    if not check_xcode_tools():
        install_xcode_tools()
        return False

    # Check Python
    if not check_python():
        print("\n💡 Please install Python 3.8+ from:")
        print("   • https://www.python.org/downloads/")
        print("   • Or use Homebrew: brew install python")
        return False

    # Check pip
    pip_cmd = check_pip()
    if not pip_cmd:
        if not install_pip():
            suggest_alternatives()
            return False
        pip_cmd = check_pip()
        if not pip_cmd:
            suggest_alternatives()
            return False

    # Install requirements
    if not install_requirements(pip_cmd):
        suggest_alternatives()
        return False

    # Create launch script
    create_launch_script()

    # Success message
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)

    print("\n🚀 To start the application:")
    print("   ./launch_macos.sh")
    print("   OR")
    print("   python3 -m streamlit run app.py")

    print("\n🌐 The application will open at: http://localhost:8501")
    print("📄 Upload a PDF file to get started!")

    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Setup incomplete. Please follow the suggestions above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
