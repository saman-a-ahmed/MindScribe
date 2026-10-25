"""
Initialize the MindScribe database schema
"""

import sys
from pathlib import Path

# Add project root to path so `app` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import init_db, engine
import app.models  # noqa: F401  (registers tables on Base.metadata)


def main():
    """Initialize database tables"""
    print("Initializing database...")
    try:
        init_db()
        print("✓ Database initialized successfully!")
        print(f"Database URL: {engine.url}")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
