"""
Database module for JARVISv3
Implements persistence for production deployment
"""
import os
import asyncio
import json
from datetime import datetime, UTC
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

from .config import settings

# Use aiosqlite for now, will be replaced with asyncpg + SQLAlchemy for production
import aiosqlite

logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")

try:
    from sqlalchemy import text
except ImportError:
    pass

try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

class DatabaseManager:
    """Manages database operations for JARVISv3"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.logger = logging.getLogger(__name__)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the database and create tables if they don't exist"""
        if self._initialized:
            return
        
        # Create the database directory if it doesn't exist
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tables
        await self._create_tables()
        self._initialized = True
        self.logger.info(f"Database initialized at {self.db_path}")
        
        # Create default admin user if none exists
        await self._create_default_admin_user()
    
    async def _create_default_admin_user(self):
        """Create a default admin user if no users exist"""
        # Check if any users exist
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM users") as cursor:
                count = await cursor.fetchone()
                user_count = count[0] if count else 0
        
        if user_count == 0:
            # Create default admin user
            admin_user = {
                'user_id': 'admin_123',
                'username': 'admin',
                'email': 'admin@JARVISv3.local',
                'role': 'admin',
                'permissions': ['read', 'write', 'execute', 'admin'],
                'api_keys': [os.getenv("INITIAL_ADMIN_KEY", "JARVISv3_demo_key")]
            }
            await self.create_user(admin_user)
            self.logger.info("Default admin user created")
    
    async def _create_tables(self):
        """Create all required database tables"""
        async with aiosqlite.connect(self.db_path) as db:
            # Users table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    role TEXT DEFAULT 'user',
                    permissions TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    api_keys TEXT
                )
            ''')
            
            # Workflows table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    workflow_name TEXT NOT NULL,
                    workflow_type TEXT DEFAULT 'chat',
                    initiating_query TEXT NOT NULL,
                    status TEXT DEFAULT 'running',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    tokens_used INTEGER DEFAULT 0,
                    validation_passed BOOLEAN DEFAULT 1,
                    execution_time REAL DEFAULT 0.0,
                    context_snapshot TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Context objects table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS context_objects (
                    context_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    context_type TEXT NOT NULL,
                    context_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    size_bytes INTEGER DEFAULT 0,
                    FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
                )
            ''')
            
            # Budget tracking table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS budget_tracking (
                    budget_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    workflow_id TEXT,
                    monthly_limit_usd REAL DEFAULT 100.0,
                    daily_limit_usd REAL DEFAULT 10.0,
                    token_limit INTEGER DEFAULT 100000,
                    monthly_spent_usd REAL DEFAULT 0.0,
                    daily_spent_usd REAL DEFAULT 0.0,
                    tokens_consumed INTEGER DEFAULT 0,
                    period_start_date DATE,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
                )
            ''')
            
            # Observability logs table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS observability_logs (
                    log_id TEXT PRIMARY KEY,
                    workflow_id TEXT,
                    node_id TEXT,
                    log_level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context_snapshot TEXT,
                    FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
                )
            ''')

            # Conversations table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT DEFAULT '[]'
                )
            ''')

            # Messages table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tokens INTEGER DEFAULT 0,
                    mode TEXT DEFAULT 'chat',
                    tags TEXT DEFAULT '[]',
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            ''')

            # Workflow checkpoints table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    state_data TEXT NOT NULL,
                    context_data TEXT NOT NULL,
                    results_data TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
                )
            ''')
            
            await db.commit()
            self.logger.info("Database tables created successfully")
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                'SELECT user_id, username, email, role, permissions, created_at, last_login, is_active, api_keys FROM users WHERE user_id = ?',
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {
                        'user_id': row[0],
                        'username': row[1],
                        'email': row[2],
                        'role': row[3],
                        'permissions': json.loads(row[4]) if row[4] else [],
                        'created_at': row[5],
                        'last_login': row[6],
                        'is_active': bool(row[7]),
                        'api_keys': json.loads(row[8]) if row[8] else []
                    }
                return None
    
    async def create_user(self, user_data: Dict[str, Any]) -> bool:
        """Create a new user"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    '''INSERT INTO users 
                    (user_id, username, email, role, permissions, api_keys) 
                    VALUES (?, ?, ?, ?, ?, ?)''',
                    (
                        user_data['user_id'],
                        user_data['username'],
                        user_data['email'],
                        user_data.get('role', 'user'),
                        json.dumps(user_data.get('permissions', [])),
                        json.dumps(user_data.get('api_keys', []))
                    )
                )
                await db.commit()
                self.logger.info(f"User created: {user_data['username']}")
                return True
        except aiosqlite.IntegrityError:
            self.logger.warning(f"User already exists: {user_data['username']}")
            return False
    
    async def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                '''SELECT workflow_id, user_id, workflow_name, workflow_type, initiating_query, 
                          status, created_at, completed_at, tokens_used, validation_passed, 
                          execution_time, context_snapshot FROM workflows WHERE workflow_id = ?''',
                (workflow_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {
                        'workflow_id': row[0],
                        'user_id': row[1],
                        'workflow_name': row[2],
                        'workflow_type': row[3],
                        'initiating_query': row[4],
                        'status': row[5],
                        'created_at': row[6],
                        'completed_at': row[7],
                        'tokens_used': row[8],
                        'validation_passed': bool(row[9]),
                        'execution_time': row[10],
                        'context_snapshot': json.loads(row[11]) if row[11] else {}
                    }
                return None
    
    async def create_workflow(self, workflow_data: Dict[str, Any]) -> bool:
        """Create a new workflow record"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    '''INSERT INTO workflows 
                    (workflow_id, user_id, workflow_name, workflow_type, initiating_query, context_snapshot) 
                    VALUES (?, ?, ?, ?, ?, ?)''',
                    (
                        workflow_data['workflow_id'],
                        workflow_data['user_id'],
                        workflow_data['workflow_name'],
                        workflow_data.get('workflow_type', 'chat'),
                        workflow_data['initiating_query'],
                        json.dumps(workflow_data.get('context_snapshot', {}))
                    )
                )
                await db.commit()
                self.logger.info(f"Workflow created: {workflow_data['workflow_id']}")
                return True
        except Exception as e:
            self.logger.error(f"Error creating workflow: {e}")
            return False
    
    async def update_workflow_status(self, workflow_id: str, status: str, **kwargs) -> bool:
        """Update workflow status and other fields"""
        try:
            set_clause = []
            values = []
            
            set_clause.append("status = ?")
            values.append(status)
            
            for key, value in kwargs.items():
                if key in ['completed_at', 'tokens_used', 'validation_passed', 'execution_time', 'context_snapshot']:
                    set_clause.append(f"{key} = ?")
                    if key == 'context_snapshot' and isinstance(value, dict):
                        values.append(json.dumps(value))
                    else:
                        values.append(value)
            
            set_clause_str = ", ".join(set_clause)
            values.append(workflow_id)
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    f"UPDATE workflows SET {set_clause_str} WHERE workflow_id = ?",
                    values
                )
                await db.commit()
                self.logger.info(f"Workflow {workflow_id} updated to status: {status}")
                return True
        except Exception as e:
            self.logger.error(f"Error updating workflow: {e}")
            return False
    
    async def save_context_object(self, context_data: Dict[str, Any]) -> bool:
        """Save a context object to the database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    '''INSERT INTO context_objects 
                    (context_id, workflow_id, context_type, context_data, size_bytes) 
                    VALUES (?, ?, ?, ?, ?)''',
                    (
                        context_data['context_id'],
                        context_data['workflow_id'],
                        context_data['context_type'],
                        json.dumps(context_data['context_data']),
                        len(json.dumps(context_data['context_data']))
                    )
                )
                await db.commit()
                self.logger.info(f"Context object saved: {context_data['context_id']}")
                return True
        except Exception as e:
            self.logger.error(f"Error saving context object: {e}")
            return False
    
    async def get_context_objects(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get all context objects for a workflow"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                '''SELECT context_id, context_type, context_data, created_at, size_bytes 
                   FROM context_objects WHERE workflow_id = ?''',
                (workflow_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        'context_id': row[0],
                        'context_type': row[1],
                        'context_data': json.loads(row[2]) if row[2] else {},
                        'created_at': row[3],
                        'size_bytes': row[4]
                    }
                    for row in rows
                ]
    
    async def save_budget_record(self, budget_data: Dict[str, Any]) -> bool:
        """Save a budget tracking record"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    '''INSERT OR REPLACE INTO budget_tracking 
                    (budget_id, user_id, workflow_id, monthly_limit_usd, daily_limit_usd, 
                     token_limit, monthly_spent_usd, daily_spent_usd, tokens_consumed, period_start_date) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (
                        budget_data['budget_id'],
                        budget_data['user_id'],
                        budget_data.get('workflow_id'),
                        budget_data.get('monthly_limit_usd', 100.0),
                        budget_data.get('daily_limit_usd', 10.0),
                        budget_data.get('token_limit', 100000),
                        budget_data.get('monthly_spent_usd', 0.0),
                        budget_data.get('daily_spent_usd', 0.0),
                        budget_data.get('tokens_consumed', 0),
                        budget_data.get('period_start_date', datetime.now(UTC).date().isoformat())
                    )
                )
                await db.commit()
                self.logger.info(f"Budget record saved: {budget_data['budget_id']}")
                return True
        except Exception as e:
            self.logger.error(f"Error saving budget record: {e}")
            return False
    
    async def update_budget_usage(self, user_id: str, workflow_id: str, cost_usd: float = 0.0, tokens: int = 0) -> bool:
        """Update budget usage for a user and workflow"""
        try:
            # Get current budget record
            budget = await self.get_budget_record(user_id)
            if not budget:
                # Create a new budget record if it doesn't exist
                budget_id = f"budget_{user_id}"
                await self.save_budget_record({
                    'budget_id': budget_id,
                    'user_id': user_id,
                    'workflow_id': workflow_id,
                    'monthly_spent_usd': cost_usd,
                    'daily_spent_usd': cost_usd,
                    'tokens_consumed': tokens
                })
                return True
            
            # Update existing budget record
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    '''UPDATE budget_tracking 
                       SET monthly_spent_usd = monthly_spent_usd + ?, 
                           daily_spent_usd = daily_spent_usd + ?, 
                           tokens_consumed = tokens_consumed + ?,
                           workflow_id = ?
                       WHERE user_id = ?''',
                    (cost_usd, cost_usd, tokens, workflow_id, user_id)
                )
                await db.commit()
                self.logger.info(f"Budget updated for user {user_id}: +${cost_usd}, +{tokens} tokens")
                return True
        except Exception as e:
            self.logger.error(f"Error updating budget usage: {e}")
            return False
    
    async def get_budget_record(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get budget record for a user"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                '''SELECT budget_id, monthly_limit_usd, daily_limit_usd, token_limit,
                          monthly_spent_usd, daily_spent_usd, tokens_consumed, period_start_date
                   FROM budget_tracking WHERE user_id = ?''',
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {
                        'budget_id': row[0],
                        'monthly_limit_usd': row[1],
                        'daily_limit_usd': row[2],
                        'token_limit': row[3],
                        'monthly_spent_usd': row[4],
                        'daily_spent_usd': row[5],
                        'tokens_consumed': row[6],
                        'period_start_date': row[7]
                    }
                return None
    
    async def log_observability_event(self, log_data: Dict[str, Any]) -> bool:
        """Log an observability event"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    '''INSERT INTO observability_logs 
                    (log_id, workflow_id, node_id, log_level, message, context_snapshot) 
                    VALUES (?, ?, ?, ?, ?, ?)''',
                    (
                        log_data['log_id'],
                        log_data.get('workflow_id'),
                        log_data.get('node_id'),
                        log_data['log_level'],
                        log_data['message'],
                        json.dumps(log_data.get('context_snapshot', {}))
                    )
                )
                await db.commit()
                self.logger.info(f"Observability event logged: {log_data['log_id']}")
                return True
        except Exception as e:
            self.logger.error(f"Error logging observability event: {e}")
            return False
    
    async def get_observability_logs(self, workflow_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get observability logs, optionally filtered by workflow_id"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                if workflow_id:
                    async with db.execute(
                        '''SELECT log_id, workflow_id, node_id, log_level, message, timestamp, context_snapshot
                           FROM observability_logs 
                           WHERE workflow_id = ?
                           ORDER BY timestamp DESC
                           LIMIT ?''',
                        (workflow_id, limit)
                    ) as cursor:
                        rows = await cursor.fetchall()
                else:
                    async with db.execute(
                        '''SELECT log_id, workflow_id, node_id, log_level, message, timestamp, context_snapshot
                           FROM observability_logs 
                           ORDER BY timestamp DESC
                           LIMIT ?''',
                        (limit,)
                    ) as cursor:
                        rows = await cursor.fetchall()
                
                return [
                    {
                        'log_id': row[0],
                        'workflow_id': row[1],
                        'node_id': row[2],
                        'log_level': row[3],
                        'message': row[4],
                        'timestamp': row[5],
                        'context_snapshot': json.loads(row[6]) if row[6] else {}
                    }
                    for row in rows
                ]
        except Exception as e:
            self.logger.error(f"Error getting observability logs: {e}")
            return []

    async def create_conversation(self, title: str, conversation_id: Optional[str] = None) -> str:
        """Create a new conversation"""
        import uuid
        cid = conversation_id or str(uuid.uuid4())
            
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO conversations (conversation_id, title) VALUES (?, ?)",
                    (cid, title)
                )
                await db.commit()
                return cid
        except Exception as e:
            self.logger.error(f"Error creating conversation: {e}")
            return ""

    async def get_conversations(self) -> List[Dict[str, Any]]:
        """Get all conversations"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM conversations ORDER BY updated_at DESC") as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific conversation by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Delete messages first (foreign key constraint)
                await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
                # Delete conversation
                await db.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
                await db.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error deleting conversation {conversation_id}: {e}")
            return False

    async def add_message(self, conversation_id: str, role: str, content: str, tokens: int = 0, mode: str = "chat") -> str:
        """Add a message to a conversation"""
        import uuid
        message_id = str(uuid.uuid4())
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    '''INSERT INTO messages 
                    (message_id, conversation_id, role, content, tokens, mode) 
                    VALUES (?, ?, ?, ?, ?, ?)''',
                    (message_id, conversation_id, role, content, tokens, mode)
                )
                # Update conversation timestamp
                await db.execute(
                    "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
                    (conversation_id,)
                )
                await db.commit()
                return message_id
        except Exception as e:
            self.logger.error(f"Error adding message: {e}")
            return ""

    async def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get messages for a conversation"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
                (conversation_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def save_workflow_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Save a workflow checkpoint"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    '''INSERT OR REPLACE INTO workflow_checkpoints 
                    (checkpoint_id, workflow_id, node_id, state_data, context_data, results_data) 
                    VALUES (?, ?, ?, ?, ?, ?)''',
                    (
                        checkpoint_data['checkpoint_id'],
                        checkpoint_data['workflow_id'],
                        checkpoint_data['node_id'],
                        json.dumps(checkpoint_data['state_data']),
                        json.dumps(checkpoint_data['context_data']),
                        json.dumps(checkpoint_data['results_data'])
                    )
                )
                await db.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            return False

    async def get_latest_checkpoint(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint for a workflow"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                '''SELECT * FROM workflow_checkpoints 
                   WHERE workflow_id = ? 
                   ORDER BY timestamp DESC LIMIT 1''',
                (workflow_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    res = dict(row)
                    res['state_data'] = json.loads(res['state_data'])
                    res['context_data'] = json.loads(res['context_data'])
                    res['results_data'] = json.loads(res['results_data'])
                    return res
                return None

    # Tagging support
    async def set_conversation_tags(self, conversation_id: str, tags: List[str]) -> bool:
        """Set tags for a conversation"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "UPDATE conversations SET tags = ?, updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
                    (json.dumps(tags), conversation_id)
                )
                await db.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error setting conversation tags: {e}")
            return False

    async def set_message_tags(self, message_id: str, tags: List[str]) -> bool:
        """Set tags for a message"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "UPDATE messages SET tags = ? WHERE message_id = ?",
                    (json.dumps(tags), message_id)
                )
                await db.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error setting message tags: {e}")
            return False

    async def filter_conversations_by_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """Filter conversations by tags (exact match on any tag)"""
        # Note: SQLite JSON querying is limited. We'll do simple string matching or load-and-filter.
        # Load-and-filter is safer for correctness.
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM conversations") as cursor:
                rows = await cursor.fetchall()
                results = []
                for row in rows:
                    conv = dict(row)
                    try:
                        conv_tags = json.loads(conv['tags']) if conv['tags'] else []
                        # Check if any requested tag is in conversation tags
                        if any(t in conv_tags for t in tags):
                            results.append(conv)
                    except Exception:
                        continue
                return results

    # Utilities
    async def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """Get statistics for a conversation"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT COUNT(*), SUM(tokens) FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return {
                    "message_count": row[0] if row else 0,
                    "token_count": row[1] if row and row[1] else 0
                }

    async def export_all_data(self) -> Dict[str, Any]:
        """Export all data from the database"""
        export = {
            "version": "v3",
            "timestamp": datetime.now(UTC).isoformat(),
            "users": [],
            "conversations": [],
            "messages": [],
            "workflows": []
        }
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Export users
            async with db.execute("SELECT * FROM users") as cursor:
                rows = await cursor.fetchall()
                export["users"] = [dict(r) for r in rows]
                
            # Export conversations
            async with db.execute("SELECT * FROM conversations") as cursor:
                rows = await cursor.fetchall()
                export["conversations"] = [dict(r) for r in rows]
                
            # Export messages
            async with db.execute("SELECT * FROM messages") as cursor:
                rows = await cursor.fetchall()
                export["messages"] = [dict(r) for r in rows]
                
            # Export workflows (optional, maybe heavy)
            async with db.execute("SELECT * FROM workflows") as cursor:
                rows = await cursor.fetchall()
                export["workflows"] = [dict(r) for r in rows]
                
        return export

    async def import_data(self, data: Dict[str, Any], merge: bool = True) -> Dict[str, int]:
        """Import data into the database"""
        counts = {"users": 0, "conversations": 0, "messages": 0, "workflows": 0}
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Users
                for user in data.get("users", []):
                    try:
                        await db.execute(
                            "INSERT OR IGNORE INTO users (user_id, username, email, role, permissions, api_keys) VALUES (?, ?, ?, ?, ?, ?)",
                            (user['user_id'], user['username'], user['email'], user.get('role', 'user'), user.get('permissions'), user.get('api_keys'))
                        )
                        counts["users"] += 1
                    except Exception:
                        pass
                
                # Conversations
                for conv in data.get("conversations", []):
                    try:
                        await db.execute(
                            "INSERT OR IGNORE INTO conversations (conversation_id, title, tags) VALUES (?, ?, ?)",
                            (conv['conversation_id'], conv.get('title'), conv.get('tags', '[]'))
                        )
                        counts["conversations"] += 1
                    except Exception:
                        pass
                        
                # Messages
                for msg in data.get("messages", []):
                    try:
                        await db.execute(
                            "INSERT OR IGNORE INTO messages (message_id, conversation_id, role, content, tokens, mode, tags) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (msg['message_id'], msg['conversation_id'], msg['role'], msg['content'], msg.get('tokens', 0), msg.get('mode', 'chat'), msg.get('tags', '[]'))
                        )
                        counts["messages"] += 1
                    except Exception:
                        pass
                        
                await db.commit()
                return counts
        except Exception as e:
            self.logger.error(f"Error importing data: {e}")
            return counts


# Global instance
database_manager = DatabaseManager()
