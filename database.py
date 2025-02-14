import os
from databases import Database
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import MetaData

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:arnavmodi@localhost/url")

database = Database(DATABASE_URL)

# Define metadata and bind it to the engine
metadata = MetaData()
async_engine = create_async_engine(DATABASE_URL, echo=True)
