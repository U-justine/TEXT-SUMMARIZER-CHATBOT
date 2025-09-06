#!/usr/bin/env python3
"""
Smart Cloud Deployment Script for Text Summarizer ChatBot
Automatically detects platform, fixes common issues, and optimizes for hosting
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
import time

class CloudDeployer:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.platform = self.detect_platform()
        self.config = self.get_platform_config()

    def detect_platform(self):
        """Detect hosting platform"""
        if os.environ.get("STREAMLIT_SHARING"):
            return "streamlit_cloud"
        elif os.environ.get("SPACE_ID"):
            return "huggingface"
        elif os.environ.get("RAILWAY_ENVIRONMENT"):
            return "railway"
        elif os.environ.get("VERCEL"):
            return "vercel"
        elif os.environ.get("NETLIFY"):
            return "netlify"
        else:
            return "local"

    def get_platform_config(self):
        """Get platform-specific configuration"""
        configs = {
            "streamlit_cloud": {
                "max_file_size": "100MB",
                "memory_limit": "1GB",
                "timeout": 30,
                "python_version": "3.9",
                "recommended_models": ["t5-small", "distilbert-base-uncased-distilled-squad"],
                "requirements_file": "requirements_cloud.txt"
            },
            "huggingface": {
                "max_file_size": "500MB",
                "memory_limit": "16GB",
                "timeout": 60,
                "python_version": "3.9",
                "recommended_models": ["sshleifer/distilbart-cnn-12-6", "distilbert-base-cased-distilled-squad"],
                "requirements_file": "requirements.txt"
            },
            "railway": {
                "max_file_size": "200MB",
                "memory_limit": "8GB",
                "timeout": 45,
                "python_version": "3.9",
                "recommended_models": ["sshleifer/distilbart-cnn-12-6", "distilbert-base-cased-distilled-squad"],
                "requirements_file": "requirements.txt"
            },
            "local": {
                "max_file_size": "unlimited",
                "memory_limit": "system",
                "timeout": 120,
                "python_version": "3.9+",
                "recommended_models": ["facebook/bart-large-cnn", "deepset/roberta-base-squad2"],
                "requirements_file": "requirements.txt"
            }
        }
        return configs.get(self.platform, configs["local"])

    def check_dependencies(self):
        """Check if required dependencies are installed"""
        print("🔍 Checking dependencies...")

        required_packages = [
            "streamlit", "transformers", "torch", "PyPDF2", "nltk"
        ]

        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package}")
                missing_packages.append(package)

        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            return False

        print("✅ All dependencies satisfied!")
        return True

    def install_dependencies(self):
        """Install missing dependencies"""
        print("📦 Installing dependencies...")

        requirements_file = self.project_root / self.config["requirements_file"]

        if requirements_file.exists():
            print(f"  Using {requirements_file.name}")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install",
                    "-r", str(requirements_file),
                    "--upgrade", "--quiet"
                ], check=True)
                print("  ✅ Dependencies installed successfully!")
                return True
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to install dependencies: {e}")
                return False
        else:
            print(f"  ❌ Requirements file not found: {requirements_file}")
            return False

    def optimize_for_platform(self):
        """Optimize app configuration for specific platform"""
        print(f"⚙️  Optimizing for {self.platform}...")

        # Create platform-specific app file
        optimized_app = self.create_optimized_app()

        if optimized_app:
            print("  ✅ Created platform-optimized app")
        else:
            print("  ⚠️  Using default app configuration")

        # Create platform-specific configuration files
        self.create_config_files()

        print(f"✅ Optimization complete for {self.platform}!")

    def create_optimized_app(self):
        """Create a platform-optimized version of the app"""
        try:
            # Use the cloud-optimized version if available
            cloud_app = self.project_root / "app_cloud_optimized.py"
            if cloud_app.exists():
                # Copy to main app file for deployment
                main_app = self.project_root / "app.py"
                shutil.copy2(cloud_app, main_app)
                return True

            # Otherwise, modify existing app
            return self.modify_existing_app()

        except Exception as e:
            print(f"  ❌ Error creating optimized app: {e}")
            return False

    def modify_existing_app(self):
        """Modify existing app for platform compatibility"""
        app_files = ["app_attractive.py", "app.py", "main.py"]

        for app_file in app_files:
            app_path = self.project_root / app_file
            if app_path.exists():
                try:
                    # Read original app
                    with open(app_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Apply platform-specific modifications
                    modified_content = self.apply_platform_modifications(content)

                    # Write optimized version
                    optimized_path = self.project_root / f"app_optimized_{self.platform}.py"
                    with open(optimized_path, 'w', encoding='utf-8') as f:
                        f.write(modified_content)

                    print(f"  ✅ Created optimized version: {optimized_path.name}")
                    return True

                except Exception as e:
                    print(f"  ❌ Error modifying {app_file}: {e}")
                    continue

        return False

    def apply_platform_modifications(self, content):
        """Apply platform-specific code modifications"""
        modifications = []

        if self.platform == "streamlit_cloud":
            modifications = [
                # Force CPU usage
                ('device=0 if torch.cuda.is_available() else -1', 'device=-1  # Force CPU for Streamlit Cloud'),
                # Reduce timeout
                ('timeout=60', 'timeout=30'),
                # Use smaller models
                ('facebook/bart-large-cnn', 't5-small'),
                ('deepset/roberta-base-squad2', 'distilbert-base-uncased-distilled-squad'),
                # Limit text length
                ('max_chars=10000', 'max_chars=2000'),
            ]

        elif self.platform == "huggingface":
            modifications = [
                # Add GPU support
                ('device=-1', 'device=0 if torch.cuda.is_available() else -1'),
                # Increase timeout
                ('timeout=30', 'timeout=60'),
                # Use medium models
                ('t5-small', 'sshleifer/distilbart-cnn-12-6'),
            ]

        # Apply modifications
        modified_content = content
        for old, new in modifications:
            modified_content = modified_content.replace(old, new)

        # Add platform detection at the top
        platform_header = f'''
# Platform: {self.platform}
# Auto-generated optimized version
import os
os.environ["PLATFORM"] = "{self.platform}"

'''

        modified_content = platform_header + modified_content
        return modified_content

    def create_config_files(self):
        """Create platform-specific configuration files"""

        if self.platform == "streamlit_cloud":
            self.create_streamlit_config()

        elif self.platform == "huggingface":
            self.create_huggingface_config()

        elif self.platform == "railway":
            self.create_railway_config()

        # Create .gitignore
        self.create_gitignore()

        # Create README for deployment
        self.create_deployment_readme()

    def create_streamlit_config(self):
        """Create Streamlit Cloud configuration"""
        config_dir = self.project_root / ".streamlit"
        config_dir.mkdir(exist_ok=True)

        config_content = """
[theme]
base = "light"
primaryColor = "#667eea"

[server]
headless = true
port = $PORT
enableCORS = false

[browser]
gatherUsageStats = false
        """

        with open(config_dir / "config.toml", 'w') as f:
            f.write(config_content.strip())

        print("  ✅ Created Streamlit config")

    def create_huggingface_config(self):
        """Create Hugging Face Spaces configuration"""
        config_content = {
            "title": "Text Summarizer ChatBot",
            "emoji": "📝",
            "colorFrom": "blue",
            "colorTo": "green",
            "sdk": "streamlit",
            "sdk_version": "1.28.0",
            "app_file": "app.py",
            "pinned": False,
            "license": "mit"
        }

        with open(self.project_root / "README.md", 'w') as f:
            f.write("---\n")
            for key, value in config_content.items():
                f.write(f"{key}: {value}\n")
            f.write("---\n\n")
            f.write("# Text Summarizer ChatBot\n\n")
            f.write("AI-powered text summarization and question answering system.\n")

        print("  ✅ Created Hugging Face config")

    def create_railway_config(self):
        """Create Railway configuration"""
        config_content = {
            "build": {
                "buildCommand": "pip install -r requirements.txt"
            },
            "deploy": {
                "startCommand": "streamlit run app.py --server.port $PORT --server.address 0.0.0.0",
                "healthcheckPath": "/healthz"
            }
        }

        with open(self.project_root / "railway.json", 'w') as f:
            json.dump(config_content, f, indent=2)

        # Create Procfile as backup
        with open(self.project_root / "Procfile", 'w') as f:
            f.write("web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0\n")

        print("  ✅ Created Railway config")

    def create_gitignore(self):
        """Create .gitignore file"""
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Streamlit
.streamlit/secrets.toml

# Models cache
.cache/
models/
*.pkl
*.model

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
        """

        gitignore_path = self.project_root / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, 'w') as f:
                f.write(gitignore_content.strip())
            print("  ✅ Created .gitignore")

    def create_deployment_readme(self):
        """Create deployment-specific README"""
        readme_content = f"""
# 🚀 Deployment Guide - Text Summarizer ChatBot

## Platform: {self.platform.replace('_', ' ').title()}

### Configuration
- **Memory Limit:** {self.config['memory_limit']}
- **Timeout:** {self.config['timeout']} seconds
- **Python Version:** {self.config['python_version']}
- **Recommended Models:** {', '.join(self.config['recommended_models'])}

### Quick Deploy

#### For {self.platform.replace('_', ' ').title()}:
"""

        if self.platform == "streamlit_cloud":
            readme_content += """
1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click!

Your app will be live at: `https://your-app-name.streamlit.app`
"""

        elif self.platform == "huggingface":
            readme_content += """
1. Create account at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space (Streamlit SDK)
3. Upload your files or connect GitHub
4. Your app will be live automatically!

Access at: `https://huggingface.co/spaces/your-username/your-space`
"""

        elif self.platform == "railway":
            readme_content += """
1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login`
3. Initialize: `railway init`
4. Deploy: `railway up`

Your app will be live with a Railway URL.
"""

        readme_content += f"""

### Files Created:
- `app_optimized_{self.platform}.py` - Platform-optimized version
- Configuration files for {self.platform}
- `.gitignore` - Git ignore rules
- This deployment guide

### Performance Tips:
- Keep text under {self.config.get('max_file_size', '2MB')} for best performance
- First model load may take 1-2 minutes
- Subsequent requests are much faster due to caching

### Troubleshooting:
- If models fail to load, check internet connection
- For timeout errors, try shorter text
- Refresh page if app becomes unresponsive

### Support:
- Platform: {self.platform}
- Optimization Level: Production Ready ✅
- Fallback Methods: Enabled ✅
"""

        with open(self.project_root / f"DEPLOY_{self.platform.upper()}.md", 'w') as f:
            f.write(readme_content)

        print(f"  ✅ Created deployment guide")

    def test_deployment(self):
        """Test the deployment locally"""
        print("🧪 Testing deployment...")

        # Test import of main modules
        test_results = {}

        try:
            import streamlit
            test_results['streamlit'] = f"✅ {streamlit.__version__}"
        except Exception as e:
            test_results['streamlit'] = f"❌ {e}"

        try:
            import transformers
            test_results['transformers'] = f"✅ {transformers.__version__}"
        except Exception as e:
            test_results['transformers'] = f"❌ {e}"

        try:
            import torch
            test_results['torch'] = f"✅ {torch.__version__}"
        except Exception as e:
            test_results['torch'] = f"❌ {e}"

        print("\n📊 Test Results:")
        for package, result in test_results.items():
            print(f"  {package}: {result}")

        # Test basic functionality
        try:
            from transformers import pipeline
            summarizer = pipeline("summarization", model="t5-small", device=-1)
            test_text = "This is a test sentence for deployment verification."
            result = summarizer(test_text, max_length=20, min_length=5)
            print("  ✅ Basic model functionality works!")
            return True
        except Exception as e:
            print(f"  ❌ Model test failed: {e}")
            return False

    def deploy(self):
        """Main deployment workflow"""
        print(f"🚀 Starting deployment for {self.platform.upper()}...")
        print("=" * 60)

        # Step 1: Check dependencies
        if not self.check_dependencies():
            print("\n📦 Installing missing dependencies...")
            if not self.install_dependencies():
                print("❌ Failed to install dependencies. Please install manually.")
                return False

        # Step 2: Optimize for platform
        self.optimize_for_platform()

        # Step 3: Test deployment
        if not self.test_deployment():
            print("\n⚠️  Deployment test failed, but continuing...")

        # Step 4: Final instructions
        print("\n🎉 Deployment preparation complete!")
        print("=" * 60)

        print(f"\n📋 Next Steps for {self.platform.replace('_', ' ').title()}:")

        if self.platform == "streamlit_cloud":
            print("1. Push code to GitHub repository")
            print("2. Visit https://share.streamlit.io")
            print("3. Connect your GitHub repo")
            print("4. Click 'Deploy'!")

        elif self.platform == "huggingface":
            print("1. Visit https://huggingface.co/spaces")
            print("2. Create new Space (Streamlit)")
            print("3. Upload files or connect GitHub")
            print("4. Your app will deploy automatically!")

        elif self.platform == "railway":
            print("1. Install Railway CLI: npm install -g @railway/cli")
            print("2. Run: railway login")
            print("3. Run: railway init")
            print("4. Run: railway up")

        else:
            print("1. Run locally: streamlit run app.py")
            print("2. Or deploy to your preferred cloud platform")

        print(f"\n📁 Files created:")
        print(f"  - app_optimized_{self.platform}.py (optimized app)")
        print(f"  - DEPLOY_{self.platform.upper()}.md (deployment guide)")
        print(f"  - Platform-specific config files")

        print("\n✨ Your app is ready for deployment!")
        return True

def main():
    """Main function"""
    print("🌐 Smart Cloud Deployment Tool")
    print("Text Summarizer ChatBot - Deployment Optimizer")
    print("=" * 60)

    deployer = CloudDeployer()

    print(f"📍 Detected Platform: {deployer.platform.replace('_', ' ').title()}")
    print(f"💾 Memory Limit: {deployer.config['memory_limit']}")
    print(f"⏱️  Timeout: {deployer.config['timeout']}s")
    print(f"🤖 Recommended Models: {', '.join(deployer.config['recommended_models'])}")

    # Ask user to confirm
    response = input("\n🚀 Proceed with deployment optimization? (y/n): ").lower().strip()

    if response in ['y', 'yes', '']:
        success = deployer.deploy()
        if success:
            print("\n🎉 Ready to deploy! Check the generated files for next steps.")
        else:
            print("\n❌ Deployment preparation failed. Check errors above.")
    else:
        print("\n👋 Deployment cancelled by user.")

if __name__ == "__main__":
    main()
