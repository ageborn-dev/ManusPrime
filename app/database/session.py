from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path

from app.config import PROJECT_ROOT
from app.database.models import Base

# Define the SQLite database URL
DATABASE_URL = f"sqlite:///{PROJECT_ROOT}/manusprime.db"

# Create the SQLAlchemy engine
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to create all tables in the database
def create_tables():
    Base.metadata.create_all(bind=engine)

# Function to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
