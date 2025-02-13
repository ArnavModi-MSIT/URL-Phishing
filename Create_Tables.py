import asyncio
from database import async_engine, metadata

async def create_tables():
    async with async_engine.begin() as conn:  
        await conn.run_sync(metadata.create_all)

asyncio.run(create_tables())  # Run the function
