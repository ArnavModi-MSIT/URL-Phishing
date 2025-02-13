from sqlalchemy import Table, Column, Integer, String
from database import metadata

phishing_data = Table(
    "phishing_data",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("url", String, unique=True),
    Column("label", String),
)
