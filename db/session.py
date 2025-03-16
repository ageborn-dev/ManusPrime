# db/session.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path

# Create database directory if it doesn't exist
Path("data").mkdir(exist_ok=True)

# Define database URL (SQLite for simplicity)
DATABASE_URL = "sqlite:///data/manusprime.db"

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Create sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import Base from models
from db.models import Base

# Function to create all tables
def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)

# Function to get a database session
def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()