import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, DateTime, Text, LargeBinary, ForeignKey, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
DB_URL = os.getenv("DATABASE_URL", "postgresql://agentx:supersecret@postgres:5432/chat_history")
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(String, primary_key=True)
    report = Column(Text)
    author = Column(String)
    assigned_to = Column(String)
    status = Column(String, default="Abierto")
    created_at = Column(DateTime, default=datetime.utcnow)
    veredicto = Column(Text, nullable=True)
    planes_accion = Column(JSON, default=list)
    session_id = Column(String, nullable=True)

class TicketThread(Base):
    __tablename__ = "ticket_threads"
    id = Column(String, primary_key=True)
    ticket_id = Column(String, ForeignKey("tickets.id"))
    author = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Attachment(Base):
    __tablename__ = "attachments"
    id = Column(String, primary_key=True)
    ticket_id = Column(String, ForeignKey("tickets.id"))
    filename = Column(String)
    file_type = Column(String)
    file_data = Column(LargeBinary, nullable=True)

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True)
    title = Column(String, default="Nueva conversación")
    created_at = Column(DateTime, default=datetime.utcnow)

class Repository(Base):
    __tablename__ = "repositories"
    id = Column(String, primary_key=True)
    url = Column(String)
    last_updated = Column(DateTime)

def init_db():
    Base.metadata.create_all(engine)
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE tickets ADD COLUMN IF NOT EXISTS veredicto TEXT;"))
            conn.execute(text("ALTER TABLE tickets ADD COLUMN IF NOT EXISTS planes_accion JSON DEFAULT '[]'::json;"))
            conn.execute(text("CREATE TABLE IF NOT EXISTS repositories (id TEXT PRIMARY KEY, url TEXT, last_updated TIMESTAMP);"))
            conn.execute(text("ALTER TABLE attachments ADD COLUMN IF NOT EXISTS file_data BYTEA;"))
            conn.execute(text("ALTER TABLE tickets ADD COLUMN IF NOT EXISTS session_id TEXT;"))
            conn.execute(text("CREATE TABLE IF NOT EXISTS chat_sessions (id TEXT PRIMARY KEY, title TEXT DEFAULT 'Nueva conversación', created_at TIMESTAMP DEFAULT NOW());"))
    except Exception:
        pass
