"""
User Management Utility for PDF AI Assistant
Handles all user-related database operations and session management
"""

import sqlite3
import hashlib
import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class UserManager:
    """Manages user authentication, sessions, and history"""

    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                profile_image TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)

        # User sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_name TEXT NOT NULL,
                pdf_name TEXT NOT NULL,
                pdf_path TEXT,
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
            CREATE TABLE IF NOT EXISTS qa_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                processing_time REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES user_sessions (id)
            )
        """)

        # User preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                theme TEXT DEFAULT 'default',
                language TEXT DEFAULT 'en',
                notifications BOOLEAN DEFAULT 1,
                auto_save BOOLEAN DEFAULT 1,
                max_history_items INTEGER DEFAULT 50,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Activity log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        conn.commit()
        conn.close()

    def hash_password(self, password: str) -> str:
        """Hash password using SHA256 with salt"""
        salt = "pdf_ai_assistant_salt_2024"
        return hashlib.sha256((password + salt).encode()).hexdigest()

    def create_user(self, username: str, email: str, password: str, full_name: str = "") -> bool:
        """Create new user account"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            password_hash = self.hash_password(password)
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, full_name)
                VALUES (?, ?, ?, ?)
            """, (username, email, password_hash, full_name))

            user_id = cursor.lastrowid

            # Create default preferences
            cursor.execute("""
                INSERT INTO user_preferences (user_id)
                VALUES (?)
            """, (user_id,))

            conn.commit()
            conn.close()

            # Log activity
            self.log_activity(user_id, "user_created", f"New user account created: {username}")

            return True
        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to create user: {e}")
            conn.close()
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating user: {e}")
            conn.close()
            return False

    def verify_user(self, username: str, password: str) -> Optional[Dict]:
        """Verify user credentials and return user info"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            password_hash = self.hash_password(password)
            cursor.execute("""
                SELECT id, username, email, full_name, created_at, last_login
                FROM users
                WHERE username = ? AND password_hash = ? AND is_active = 1
            """, (username, password_hash))

            user = cursor.fetchone()

            if user:
                user_data = {
                    "id": user[0],
                    "username": user[1],
                    "email": user[2],
                    "full_name": user[3] or user[1],
                    "created_at": user[4],
                    "last_login": user[5]
                }

                # Update last login
                cursor.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (user[0],))

                conn.commit()
                conn.close()

                # Log activity
                self.log_activity(user[0], "user_login", "User logged in successfully")

                return user_data

            conn.close()
            return None

        except Exception as e:
            logger.error(f"Error verifying user: {e}")
            return None

    def update_user_profile(self, user_id: int, full_name: str = None, email: str = None) -> bool:
        """Update user profile information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            updates = []
            params = []

            if full_name is not None:
                updates.append("full_name = ?")
                params.append(full_name)

            if email is not None:
                updates.append("email = ?")
                params.append(email)

            if updates:
                params.append(user_id)
                query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
                cursor.execute(query, params)
                conn.commit()

            conn.close()
            self.log_activity(user_id, "profile_updated", "User profile updated")
            return True

        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return False

    def save_session(self, user_id: int, session_name: str, pdf_name: str,
                    extracted_text: str, summary: str = "", pdf_path: str = "") -> int:
        """Save user session to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            word_count = len(extracted_text.split()) if extracted_text else 0
            character_count = len(extracted_text) if extracted_text else 0

            cursor.execute("""
                INSERT INTO user_sessions
                (user_id, session_name, pdf_name, pdf_path, extracted_text, summary, word_count, character_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, session_name, pdf_name, pdf_path, extracted_text, summary, word_count, character_count))

            session_id = cursor.lastrowid
            conn.commit()
            conn.close()

            self.log_activity(user_id, "session_created", f"New session created: {session_name}")
            return session_id

        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return 0

    def update_session_summary(self, session_id: int, summary: str) -> bool:
        """Update session summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE user_sessions
                SET summary = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (summary, session_id))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error updating session summary: {e}")
            return False

    def save_qa(self, session_id: int, question: str, answer: str,
                confidence: float = 0.0, processing_time: float = 0.0) -> bool:
        """Save Q&A interaction to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO qa_history (session_id, question, answer, confidence, processing_time)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, question, answer, confidence, processing_time))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error saving Q&A: {e}")
            return False

    def get_user_sessions(self, user_id: int, limit: int = 20, include_favorites: bool = True) -> List[Dict]:
        """Get user sessions from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            order_clause = "ORDER BY is_favorite DESC, created_at DESC" if include_favorites else "ORDER BY created_at DESC"

            cursor.execute(f"""
                SELECT id, session_name, pdf_name, word_count, character_count,
                       created_at, updated_at, is_favorite
                FROM user_sessions
                WHERE user_id = ?
                {order_clause}
                LIMIT ?
            """, (user_id, limit))

            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    "id": row[0],
                    "session_name": row[1],
                    "pdf_name": row[2],
                    "word_count": row[3],
                    "character_count": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "is_favorite": bool(row[7])
                })

            conn.close()
            return sessions

        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []

    def get_session_data(self, session_id: int) -> Optional[Dict]:
        """Get complete session data by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT session_name, pdf_name, pdf_path, extracted_text, summary,
                       word_count, character_count, created_at, is_favorite
                FROM user_sessions
                WHERE id = ?
            """, (session_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    "session_name": row[0],
                    "pdf_name": row[1],
                    "pdf_path": row[2],
                    "extracted_text": row[3],
                    "summary": row[4],
                    "word_count": row[5],
                    "character_count": row[6],
                    "created_at": row[7],
                    "is_favorite": bool(row[8])
                }
            return None

        except Exception as e:
            logger.error(f"Error getting session data: {e}")
            return None

    def get_session_qa_history(self, session_id: int) -> List[Dict]:
        """Get Q&A history for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT question, answer, confidence, processing_time, created_at
                FROM qa_history
                WHERE session_id = ?
                ORDER BY created_at ASC
            """, (session_id,))

            history = []
            for row in cursor.fetchall():
                history.append({
                    "question": row[0],
                    "answer": row[1],
                    "confidence": row[2],
                    "processing_time": row[3],
                    "created_at": row[4]
                })

            conn.close()
            return history

        except Exception as e:
            logger.error(f"Error getting Q&A history: {e}")
            return []

    def toggle_session_favorite(self, session_id: int) -> bool:
        """Toggle favorite status of a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE user_sessions
                SET is_favorite = CASE WHEN is_favorite = 1 THEN 0 ELSE 1 END
                WHERE id = ?
            """, (session_id,))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error toggling favorite: {e}")
            return False

    def delete_session(self, session_id: int, user_id: int) -> bool:
        """Delete a user session and its Q&A history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete Q&A history first
            cursor.execute("DELETE FROM qa_history WHERE session_id = ?", (session_id,))

            # Delete session
            cursor.execute("DELETE FROM user_sessions WHERE id = ? AND user_id = ?", (session_id, user_id))

            conn.commit()
            conn.close()

            self.log_activity(user_id, "session_deleted", f"Session {session_id} deleted")
            return True

        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False

    def get_user_statistics(self, user_id: int) -> Dict:
        """Get comprehensive user statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Session statistics
            cursor.execute("""
                SELECT
                    COUNT(*) as total_sessions,
                    SUM(word_count) as total_words,
                    SUM(character_count) as total_characters,
                    COUNT(CASE WHEN is_favorite = 1 THEN 1 END) as favorite_sessions
                FROM user_sessions
                WHERE user_id = ?
            """, (user_id,))

            session_stats = cursor.fetchone()

            # Q&A statistics
            cursor.execute("""
                SELECT
                    COUNT(*) as total_questions,
                    AVG(confidence) as avg_confidence,
                    AVG(processing_time) as avg_processing_time
                FROM qa_history qh
                JOIN user_sessions us ON qh.session_id = us.id
                WHERE us.user_id = ?
            """, (user_id,))

            qa_stats = cursor.fetchone()

            # Recent activity (last 30 days)
            cursor.execute("""
                SELECT COUNT(*) as recent_sessions
                FROM user_sessions
                WHERE user_id = ? AND created_at >= date('now', '-30 days')
            """, (user_id,))

            recent_activity = cursor.fetchone()

            conn.close()

            return {
                "total_sessions": session_stats[0] or 0,
                "total_words_processed": session_stats[1] or 0,
                "total_characters_processed": session_stats[2] or 0,
                "favorite_sessions": session_stats[3] or 0,
                "total_questions_asked": qa_stats[0] or 0,
                "average_confidence": qa_stats[1] or 0,
                "average_processing_time": qa_stats[2] or 0,
                "recent_sessions_30d": recent_activity[0] or 0
            }

        except Exception as e:
            logger.error(f"Error getting user statistics: {e}")
            return {}

    def log_activity(self, user_id: int, action: str, details: str = "",
                    ip_address: str = "", user_agent: str = ""):
        """Log user activity"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO activity_log (user_id, action, details, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, action, details, ip_address, user_agent))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error logging activity: {e}")

    def get_user_activity(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Get user activity log"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT action, details, created_at
                FROM activity_log
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))

            activities = []
            for row in cursor.fetchall():
                activities.append({
                    "action": row[0],
                    "details": row[1],
                    "created_at": row[2]
                })

            conn.close()
            return activities

        except Exception as e:
            logger.error(f"Error getting user activity: {e}")
            return []

    def cleanup_old_data(self, days_old: int = 90):
        """Clean up old data (sessions and activities older than specified days)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Clean old activity logs
            cursor.execute("""
                DELETE FROM activity_log
                WHERE created_at < date('now', '-{} days')
            """.format(days_old))

            # Optionally clean old sessions (be careful with this)
            # cursor.execute("""
            #     DELETE FROM user_sessions
            #     WHERE created_at < date('now', '-{} days') AND is_favorite = 0
            # """.format(days_old * 2))  # Keep sessions longer than activity logs

            conn.commit()
            conn.close()

            logger.info(f"Cleaned up data older than {days_old} days")

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
