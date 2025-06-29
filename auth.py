from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from database import User, LoginAttempt

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security scheme
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[str]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate a user with username and password"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get a user by username"""
    return db.query(User).filter(User.username == username).first()

def create_user(db: Session, username: str, password: str, email: Optional[str] = None) -> User:
    """Create a new user"""
    hashed_password = get_password_hash(password)
    db_user = User(username=username, password_hash=hashed_password, email=email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def log_login_attempt(
    db: Session, 
    username: str, 
    success: bool, 
    request: Request, 
    failure_reason: Optional[str] = None
):
    """Log a login attempt"""
    login_attempt = LoginAttempt(
        username=username,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        success=success,
        failure_reason=failure_reason
    )
    db.add(login_attempt)
    db.commit()

def get_current_user(token: str, db: Session) -> Optional[User]:
    """Get current user from token"""
    username = verify_token(token)
    if username is None:
        return None
    return get_user_by_username(db, username) 