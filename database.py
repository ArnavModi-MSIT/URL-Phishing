from databases import Database
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import MetaData

DATABASE_URL = "postgresql+asyncpg://postgres:arnavmodi@localhost/url"

database = Database(DATABASE_URL)

async_engine = create_async_engine(DATABASE_URL)
metadata = MetaData()
