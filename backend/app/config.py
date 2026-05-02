import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/db.sqlite3")
APP_NAME = "Crime Detection API"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
