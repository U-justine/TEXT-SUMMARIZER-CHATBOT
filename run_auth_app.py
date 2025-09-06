#!/usr/bin/env python3
"""
Enhanced Launch Script for PDF AI Assistant with Authentication
Includes admin features, database management, and improved UI
"""

import os
import sys
import subprocess
import argparse
import webbrowser
import time
import sqlite3
from pathlib import Path
import streamlit as st

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
        return True

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'transformers', 'torch', 'PyPDF2',
        'tokenizers', 'numpy', 'pandas', 'sqlite3'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            if package == 'sqlite3':
                import sqlite3  # Built-in module
            else:
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

def init_database():
    """Initialize database with sample data"""
    print("🗄️ Initializing database...")

    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        # Check if database is empty
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]

        if table_count == 0:
            print("📦 Creating database tables...")

            # Users table
            cursor.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)

            # User sessions table
            cursor.execute("""
                CREATE TABLE user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_name TEXT NOT NULL,
                    pdf_name TEXT NOT NULL,
                    extracted_text TEXT,
                    summary TEXT,
                    word_count INTEGER DEFAULT 0,
                    character_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_favorite BOOLEAN DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            # Q&A history table
            cursor.execute("""
                CREATE TABLE qa_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES user_sessions (id)
                )
            """)

            # Activity log table
            cursor.execute("""
                CREATE TABLE activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            conn.commit()
            print("✅ Database tables created successfully")

        # Check if we need to create a demo user
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]

        if user_count == 0:
            create_demo_user(cursor)
            conn.commit()

        conn.close()
        print("✅ Database initialized successfully")
        return True

    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False

def create_demo_user(cursor):
    """Create demo user account"""
    import hashlib

    print("👤 Creating demo user account...")

    demo_password = "demo123"
    salt = "pdf_ai_assistant_salt_2024"
    password_hash = hashlib.sha256((demo_password + salt).encode()).hexdigest()

    cursor.execute("""
        INSERT INTO users (username, email, password_hash, full_name)
        VALUES (?, ?, ?, ?)
    """, ("demo", "demo@example.com", password_hash, "Demo User"))

    print("✅ Demo user created:")
    print("   Username: demo")
    print("   Password: demo123")

def get_database_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        # Get user count
        cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
        user_count = cursor.fetchone()[0]

        # Get session count
        cursor.execute("SELECT COUNT(*) FROM user_sessions")
        session_count = cursor.fetchone()[0]

        # Get Q&A count
        cursor.execute("SELECT COUNT(*) FROM qa_history")
        qa_count = cursor.fetchone()[0]

        # Get recent activity (last 7 days)
        cursor.execute("""
            SELECT COUNT(*) FROM activity_log
            WHERE created_at >= date('now', '-7 days')
        """)
        recent_activity = cursor.fetchone()[0]

        conn.close()

        return {
            "users": user_count,
            "sessions": session_count,
            "questions": qa_count,
            "recent_activity": recent_activity
        }
    except Exception as e:
        print(f"Error getting database stats: {e}")
        return {}

def show_database_info():
    """Display database information"""
    print("\n📊 Database Information:")
    print("-" * 40)

    stats = get_database_stats()
    if stats:
        print(f"👥 Active Users: {stats.get('users', 0)}")
        print(f"📄 Document Sessions: {stats.get('sessions', 0)}")
        print(f"❓ Questions Asked: {stats.get('questions', 0)}")
        print(f"🔄 Recent Activity (7 days): {stats.get('recent_activity', 0)}")
    else:
        print("❌ Could not retrieve database statistics")
    print("-" * 40)

def backup_database():
    """Create database backup"""
    try:
        import shutil
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"users_backup_{timestamp}.db"

        shutil.copy("users.db", backup_name)
        print(f"✅ Database backup created: {backup_name}")
        return True
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return False

def reset_database():
    """Reset database (delete all data)"""
    response = input("⚠️  Are you sure you want to reset the database? (yes/no): ")
    if response.lower() == 'yes':
        try:
            if os.path.exists("users.db"):
                os.remove("users.db")
            init_database()
            print("✅ Database reset successfully")
        except Exception as e:
            print(f"❌ Error resetting database: {e}")

def launch_streamlit_app(app_file="app_with_auth.py", port=8501, open_browser=True):
    """Launch the Streamlit application"""
    app_path = Path(__file__).parent / app_file

    if not app_path.exists():
        print(f"❌ Application file {app_file} not found")
        return False

    print(f"🚀 Starting {app_file}...")
    print(f"🌐 Application will be available at: http://localhost:{port}")

    # Prepare streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.headless", "true" if not open_browser else "false",
        "--theme.base", "light",
        "--theme.primaryColor", "#667eea",
        "--theme.backgroundColor", "#ffffff",
        "--theme.secondaryBackgroundColor", "#f0f2f6",
        "--theme.textColor", "#262730"
    ]

    try:
        # Start the application
        print("⏳ Initializing application...")
        process = subprocess.Popen(cmd)

        if open_browser:
            # Wait for server to start, then open browser
            print("⏳ Waiting for server to start...")
            time.sleep(5)
            webbrowser.open(f"http://localhost:{port}")

        print("✅ Application started successfully!")
        print("\n" + "=" * 60)
        print("🎉 PDF AI Assistant with Authentication is now running!")
        print("=" * 60)

        print("\n📋 Quick Start Guide:")
        print("   1. 🔑 Sign up for a new account or use demo account")
        print("   2. 📤 Upload a PDF document")
        print("   3. 🔍 Extract text from the PDF")
        print("   4. ✨ Generate AI-powered summary")
        print("   5. 🤖 Ask questions about your document")
        print("   6. 📚 Access your document history in the sidebar")

        print("\n👤 Demo Account:")
        print("   Username: demo")
        print("   Password: demo123")

        print("\n🛑 Press Ctrl+C to stop the application")

        # Wait for the process to finish
        process.wait()

    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False

    return True

def show_help():
    """Show help information"""
    help_text = """
🤖 PDF AI Assistant with Authentication - Launch Script

USAGE:
    python run_auth_app.py [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -p, --port PORT         Port number (default: 8501)
    --no-browser           Don't open browser automatically
    --init-db              Initialize database only
    --db-stats             Show database statistics
    --backup-db            Create database backup
    --reset-db             Reset database (WARNING: deletes all data)
    --demo-mode            Run with demo data
    --app APP_FILE         Specify app file (default: app_with_auth.py)

EXAMPLES:
    python run_auth_app.py                    # Launch with default settings
    python run_auth_app.py -p 8080           # Launch on port 8080
    python run_auth_app.py --init-db         # Initialize database only
    python run_auth_app.py --db-stats        # Show database statistics
    python run_auth_app.py --backup-db       # Create database backup

FEATURES:
    🔐 User Authentication (Sign up/Sign in)
    📚 Personal Document History
    🤖 AI-Powered Summarization
    💬 Intelligent Q&A ChatBot
    📊 User Statistics and Activity Tracking
    🌟 Favorite Documents
    🔄 Session Management
    📱 Responsive Design

DATABASE:
    The app uses SQLite for user data storage.
    Database file: users.db

DEMO ACCOUNT:
    Username: demo
    Password: demo123
    """
    print(help_text)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Launch PDF AI Assistant with Authentication",
        add_help=False
    )

    parser.add_argument('-h', '--help', action='store_true',
                       help='Show help message')
    parser.add_argument('-p', '--port', type=int, default=8501,
                       help='Port number (default: 8501)')
    parser.add_argument('--no-browser', action='store_true',
                       help="Don't open browser automatically")
    parser.add_argument('--init-db', action='store_true',
                       help='Initialize database only')
    parser.add_argument('--db-stats', action='store_true',
                       help='Show database statistics')
    parser.add_argument('--backup-db', action='store_true',
                       help='Create database backup')
    parser.add_argument('--reset-db', action='store_true',
                       help='Reset database (WARNING: deletes all data)')
    parser.add_argument('--demo-mode', action='store_true',
                       help='Run with demo data')
    parser.add_argument('--app', default='app_with_auth.py',
                       help='Specify app file (default: app_with_auth.py)')

    args = parser.parse_args()

    if args.help:
        show_help()
        return

    print("=" * 60)
    print("🤖 PDF AI Assistant with Authentication")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Database operations
    if args.init_db:
        init_database()
        return

    if args.db_stats:
        show_database_info()
        return

    if args.backup_db:
        backup_database()
        return

    if args.reset_db:
        reset_database()
        return

    # Check requirements
    if not check_requirements():
        print("\n💡 Install requirements with:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    # Initialize database
    if not init_database():
        sys.exit(1)

    # Show database info
    show_database_info()

    # Launch the application
    success = launch_streamlit_app(
        app_file=args.app,
        port=args.port,
        open_browser=not args.no_browser
    )

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Launcher interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
