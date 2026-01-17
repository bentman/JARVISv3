"""
Authentication and Authorization module for JARVISv3
Implements security measures for production deployment
"""
import os
import jwt
from jwt import InvalidTokenError, ExpiredSignatureError
import secrets
from datetime import datetime, timedelta, UTC
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import hashlib
from .config import settings

logger = logging.getLogger(__name__)

# In-memory user store (would be replaced with database in production)
users_db = {}
user_sessions = {}

class User:
    """User model for authentication"""
    
    def __init__(self, user_id: str, username: str, email: str, role: str = "user", permissions: Optional[List[str]] = None, password_hash: Optional[str] = None):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.role = role
        self.permissions = permissions if permissions is not None else []
        self.created_at = datetime.now(UTC)
        self.last_login = None
        self.is_active = True
        self.api_keys = []
        self.password_hash = password_hash  # Initialize password hash field
    
    def dict(self):
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'permissions': self.permissions,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
        }


class AuthManager:
    """Manages authentication and authorization for JARVISv3"""
    
    def __init__(self):
        self.security = HTTPBearer()
        self.logger = logging.getLogger(__name__)
        self._initialize_default_user()
    
    def _initialize_default_user(self):
        """Initialize a default admin user if no users exist"""
        if not users_db:
            # Create a default admin user for development
            admin_user = User(
                user_id="admin_123",
                username="admin",
                email="admin@JARVISv3.local",
                role="admin",
                permissions=["read", "write", "execute", "admin"]
            )
            # Set a default password hash for the admin user
            admin_user.password_hash = self._hash_password("admin123")  # Default password for dev
            users_db["admin_123"] = admin_user
            self.logger.info("Default admin user created for development")
    
    def create_access_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token for user"""
        if expires_delta:
            expire = datetime.now(UTC) + expires_delta
        else:
            expire = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(to_encode, settings.effective_secret_key, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token for user"""
        expire = datetime.now(UTC) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": "refresh"
        }
        
        encoded_jwt = jwt.encode(to_encode, settings.effective_secret_key, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify JWT token and return user_id if valid"""
        try:
            payload = jwt.decode(token, settings.effective_secret_key, algorithms=[settings.ALGORITHM])
            user_id: str = payload.get("sub")
            token_type: str = payload.get("type")
            
            if user_id is None or token_type != "access":
                return None
            
            # Check if token is expired (handled by jwt.decode)
            return user_id
            
        except ExpiredSignatureError:
            self.logger.warning("Expired token attempted")
            return None
        except InvalidTokenError:
            self.logger.warning("Invalid token attempted")
            return None
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        # In a real implementation, this would check against a database with hashed passwords
        # For now, we'll implement a simple check against our in-memory store
        
        # Find user by username
        user = None
        for u in users_db.values():
            if u.username == username:
                user = u
                break
        
        if user is None:
            return None
        
        # Verify password against stored hash
        if user.password_hash and self._verify_password(password, user.password_hash):
            user.last_login = datetime.now(UTC)
            # Update the user in the database
            for uid, u in users_db.items():
                if u.user_id == user.user_id:
                    u.last_login = user.last_login
                    break
            return user
        
        return None
    
    def _hash_password(self, password: str) -> str:
        """Hash a password for storing using passlib (bcrypt)"""
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.hash(password)
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify a stored password against plain text password using passlib"""
        if not stored_hash:
            return False
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.verify(password, stored_hash)
    
    async def get_current_user(self, request: Request) -> Optional[User]:
        """Get current user from request"""
        # First check for API key in header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            user_id = self._find_user_by_api_key(api_key)
            if user_id and user_id in users_db:
                return users_db[user_id]
        
        # Then check for JWT token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        user_id = self.verify_token(token)
        
        if user_id and user_id in users_db:
            return users_db[user_id]
        
        return None
    
    def _find_user_by_api_key(self, api_key: str) -> Optional[str]:
        """Find user ID by API key"""
        # In a real implementation, this would look up the API key in a database
        # For now, we'll just return a default user if it matches a pattern
        if api_key == "JARVISv3_demo_key":
            return "admin_123"
        return None
    
    async def create_user(self, username: str, email: str, password: str, role: str = "user") -> Optional[User]:
        """Create a new user"""
        # Check if user already exists
        for user in users_db.values():
            if user.username == username or user.email == email:
                return None  # User already exists
        
        # Hash the password
        password_hash = self._hash_password(password)
        
        # Create new user
        user_id = f"user_{secrets.token_urlsafe(16)}"
        new_user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            password_hash=password_hash  # Include the hashed password
        )
        
        # Set default permissions based on role
        if role == "admin":
            new_user.permissions = ["read", "write", "execute", "admin"]
        elif role == "user":
            new_user.permissions = ["read", "write"]
        else:
            new_user.permissions = ["read"]
        
        users_db[user_id] = new_user
        self.logger.info(f"New user created: {username}")
        
        return new_user
    
    async def generate_api_key(self, user_id: str) -> str:
        """Generate a new API key for a user"""
        if user_id not in users_db:
            raise ValueError("User not found")
        
        # Generate a secure API key
        api_key = secrets.token_urlsafe(32)
        users_db[user_id].api_keys.append(api_key)
        
        self.logger.info(f"New API key generated for user: {user_id}")
        return api_key
    
    async def check_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission"""
        if user.role == "admin":
            return True  # Admin has all permissions
        return permission in user.permissions if user.permissions else False
    
    async def require_permission(self, user: User, permission: str):
        """Raise HTTP 403 if user doesn't have permission"""
        if not await self.check_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )


# Global instance
auth_manager = AuthManager()
