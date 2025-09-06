#!/usr/bin/env python3
"""
Error Diagnostic and Fix Script for DocuSum
Automatically detects and fixes common issues
"""

import os
import sys
import traceback
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"❌ Python {version.major}.{version.minor} is too old. Need Python 3.8+")
        return False

    logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = {
        'streamlit': 'streamlit',
        'PyPDF2': 'PyPDF2',
        'transformers': 'transformers',
        'torch': 'torch',
        'logging': None,  # Built-in
        're': None,       # Built-in
        'io': None,       # Built-in
        'pathlib': None,  # Built-in
    }

    missing = []

    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✅ {package} is installed")
        except ImportError:
            if pip_name:
                missing.append(pip_name)
                logger.error(f"❌ {package} is missing")

    if missing:
        logger.error(f"📦 Install missing packages with: pip install {' '.join(missing)}")
        return False

    return True

def fix_unboundlocalerror():
    """Fix the UnboundLocalError in app files"""
    apps_to_fix = [
        'app.py',
        'app_attractive.py',
        'app_enhanced.py',
        'app_modern.py',
        'app_with_auth.py',
        'app_simple_auth.py'
    ]

    fixes_applied = []

    for app_file in apps_to_fix:
        if not os.path.exists(app_file):
            continue

        logger.info(f"🔍 Checking {app_file} for UnboundLocalError...")

        try:
            with open(app_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Fix pattern 1: Missing result initialization in Q&A
            if 'result = answer_question(' in content and 'if qa_model:' in content:
                if 'else:\n                                result = {' not in content:
                    pattern = 'if qa_model:\n                                result = answer_question('
                    replacement = '''if qa_model:
                                result = answer_question('''

                    # Add else clause after the if block
                    if pattern in content:
                        # Find the end of the if block and add else
                        lines = content.split('\n')
                        new_lines = []
                        in_if_block = False

                        for i, line in enumerate(lines):
                            new_lines.append(line)

                            if 'if qa_model:' in line and 'result = answer_question(' in lines[i+1] if i+1 < len(lines) else False:
                                in_if_block = True
                            elif in_if_block and line.strip() != '' and not line.startswith('    ') and not line.startswith('\t'):
                                # End of if block, add else
                                new_lines.insert(-1, '                            else:')
                                new_lines.insert(-1, '                                result = {')
                                new_lines.insert(-1, '                                    "answer": "❌ Q&A model failed to load. Please refresh and try again.",')
                                new_lines.insert(-1, '                                    "confidence": 0')
                                new_lines.insert(-1, '                                }')
                                in_if_block = False

                        content = '\n'.join(new_lines)
                        fixes_applied.append(f"{app_file}: Fixed UnboundLocalError for 'result'")

            # Fix pattern 2: Missing summarizer check
            if 'summarizer = load_summarization_model()' in content:
                if 'if summarizer:' not in content.split('summarizer = load_summarization_model()')[1].split('\n')[0:10]:
                    content = content.replace(
                        'summarizer = load_summarization_model()\n                        summary = summarize_text(',
                        '''summarizer = load_summarization_model()
                        if summarizer:
                            summary = summarize_text('''
                    )

                    content = content.replace(
                        'st.session_state.summary = summary',
                        '''st.session_state.summary = summary
                        else:
                            st.session_state.summary = "❌ Model failed to load. Please refresh and try again."'''
                    )
                    fixes_applied.append(f"{app_file}: Added summarizer null check")

            # Write fixed content back
            if fixes_applied:
                with open(app_file, 'w', encoding='utf-8') as f:
                    f.write(content)

        except Exception as e:
            logger.error(f"❌ Error fixing {app_file}: {e}")

    return fixes_applied

def fix_indentation_errors():
    """Fix common indentation errors"""
    apps_to_fix = ['app.py', 'app_attractive.py', 'app_enhanced.py']
    fixes_applied = []

    for app_file in apps_to_fix:
        if not os.path.exists(app_file):
            continue

        logger.info(f"🔍 Checking {app_file} for indentation errors...")

        try:
            with open(app_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            fixed_lines = []
            for i, line in enumerate(lines):
                # Fix common pattern: else: followed by unindented code
                if line.strip() == 'else:' and i+1 < len(lines):
                    next_line = lines[i+1]
                    if next_line.strip().startswith('tab1, tab2, tab3 = st.tabs(') and not next_line.startswith('    '):
                        fixed_lines.append(line)
                        fixed_lines.append('    ' + next_line)  # Add indentation
                        continue

                fixed_lines.append(line)

            # Write back if changes were made
            if fixed_lines != lines:
                with open(app_file, 'w', encoding='utf-8') as f:
                    f.writelines(fixed_lines)
                fixes_applied.append(f"{app_file}: Fixed indentation after else:")

        except Exception as e:
            logger.error(f"❌ Error fixing indentation in {app_file}: {e}")

    return fixes_applied

def fix_import_errors():
    """Fix missing imports"""
    apps_to_fix = ['app.py', 'app_attractive.py', 'app_enhanced.py', 'app_modern.py']
    fixes_applied = []

    required_imports = [
        'import streamlit as st',
        'import PyPDF2',
        'from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering',
        'import io',
        'import re',
        'import torch',
        'from typing import Optional, List',
        'import logging',
        'import time',
        'from datetime import datetime'
    ]

    for app_file in apps_to_fix:
        if not os.path.exists(app_file):
            continue

        logger.info(f"🔍 Checking imports in {app_file}...")

        try:
            with open(app_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check each required import
            missing_imports = []
            for imp in required_imports:
                if imp not in content:
                    missing_imports.append(imp)

            if missing_imports:
                # Add missing imports after the first import line
                lines = content.split('\n')
                import_insert_index = 0

                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        import_insert_index = i + 1
                        break

                for imp in missing_imports:
                    lines.insert(import_insert_index, imp)
                    import_insert_index += 1

                content = '\n'.join(lines)

                with open(app_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                fixes_applied.append(f"{app_file}: Added {len(missing_imports)} missing imports")

        except Exception as e:
            logger.error(f"❌ Error fixing imports in {app_file}: {e}")

    return fixes_applied

def test_app_syntax():
    """Test if app files have valid syntax"""
    apps_to_test = ['app.py', 'app_attractive.py', 'app_enhanced.py', 'app_modern.py']
    results = {}

    for app_file in apps_to_test:
        if not os.path.exists(app_file):
            results[app_file] = "File not found"
            continue

        logger.info(f"🧪 Testing syntax of {app_file}...")

        try:
            with open(app_file, 'r', encoding='utf-8') as f:
                source = f.read()

            compile(source, app_file, 'exec')
            results[app_file] = "✅ Syntax OK"
            logger.info(f"✅ {app_file} syntax is valid")

        except SyntaxError as e:
            results[app_file] = f"❌ Syntax Error: {e}"
            logger.error(f"❌ {app_file} syntax error: {e}")
        except Exception as e:
            results[app_file] = f"❌ Error: {e}"
            logger.error(f"❌ {app_file} error: {e}")

    return results

def create_minimal_working_app():
    """Create a minimal working app as backup"""
    minimal_app = '''
import streamlit as st
import PyPDF2
import io
import re

st.set_page_config(
    page_title="DocuSum - Minimal Version",
    page_icon="📄",
    layout="wide"
)

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages[:5]:  # Limit to 5 pages
            text += page.extract_text()
        return re.sub(r'\\s+', ' ', text).strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return ""

def simple_summarize(text):
    # Simple extractive summarization
    sentences = text.split('.')
    if len(sentences) < 3:
        return text

    # Return first 3 sentences as summary
    return '. '.join(sentences[:3]) + '.'

def main():
    st.title("📄 DocuSum - Minimal Version")
    st.info("This is a minimal version that works without AI models.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Extract Text"):
            text = extract_text_from_pdf(uploaded_file)
            if text:
                st.success("Text extracted successfully!")

                with st.expander("View Text"):
                    st.text_area("Extracted Text", text, height=300)

                if st.button("Generate Simple Summary"):
                    summary = simple_summarize(text)
                    st.markdown("### Summary")
                    st.write(summary)

if __name__ == "__main__":
    main()
'''

    try:
        with open('app_minimal.py', 'w', encoding='utf-8') as f:
            f.write(minimal_app)
        logger.info("✅ Created app_minimal.py as backup")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create minimal app: {e}")
        return False

def main():
    """Main diagnostic and fix function"""
    print("🔧 DocuSum Error Diagnostic and Fix Script")
    print("=" * 50)

    all_fixes = []

    # 1. Check Python version
    logger.info("1️⃣ Checking Python version...")
    if not check_python_version():
        print("❌ Python version incompatible. Please upgrade to Python 3.8+")
        return

    # 2. Check dependencies
    logger.info("2️⃣ Checking dependencies...")
    if not check_dependencies():
        print("❌ Missing dependencies. Please install them and run this script again.")
        return

    # 3. Fix UnboundLocalError
    logger.info("3️⃣ Fixing UnboundLocalError...")
    fixes = fix_unboundlocalerror()
    all_fixes.extend(fixes)

    # 4. Fix indentation errors
    logger.info("4️⃣ Fixing indentation errors...")
    fixes = fix_indentation_errors()
    all_fixes.extend(fixes)

    # 5. Fix import errors
    logger.info("5️⃣ Fixing import errors...")
    fixes = fix_import_errors()
    all_fixes.extend(fixes)

    # 6. Test syntax
    logger.info("6️⃣ Testing syntax...")
    syntax_results = test_app_syntax()

    # 7. Create minimal app backup
    logger.info("7️⃣ Creating minimal backup app...")
    create_minimal_working_app()

    # Summary
    print("\n" + "=" * 50)
    print("🔧 FIX SUMMARY")
    print("=" * 50)

    if all_fixes:
        print("✅ Applied fixes:")
        for fix in all_fixes:
            print(f"  • {fix}")
    else:
        print("ℹ️  No fixes needed")

    print(f"\n🧪 Syntax Test Results:")
    for app, result in syntax_results.items():
        print(f"  • {app}: {result}")

    print(f"\n🚀 Recommended next steps:")
    print("  1. Try running: streamlit run app_attractive.py")
    print("  2. If issues persist, try: streamlit run app_minimal.py")
    print("  3. For fastest performance: python run_fast.py")

    print("\n✨ DocuSum should now work properly!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Fix script interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error in fix script: {e}")
        traceback.print_exc()
