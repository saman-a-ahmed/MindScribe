"""
Initialize database schema
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import init_db, engine
from app.models import Base


def main():
    """Initialize database tables"""
    print("Initializing database...")
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✓ Database initialized successfully!")
        print(f"Database URL: {engine.url}")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
