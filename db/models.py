from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class TwinProfile(Base):
    __tablename__ = "twin_profiles"
    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    status = Column(Text, default="draft")  # draft / active / locked
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # новые поля расписания угасания
    current_state = Column(Text, default="SETUP")        # SETUP/ACTIVE/FADING/FAREWELL/LOCKED
    schedule_position = Column(Integer, default=0)       # s — позиция в расписании
    activated_at = Column(DateTime(timezone=True))       # когда перешёл в ACTIVE
    last_interaction_at = Column(DateTime(timezone=True)) # последнее сообщение пользователя

class TwinAttribute(Base):
    __tablename__ = "twin_attributes"
    id = Column(Integer, primary_key=True)
    twin_id = Column(Integer, ForeignKey("twin_profiles.id", ondelete="CASCADE"))
    block = Column(Text, nullable=False)          # style / character / worldview / fundamental
    attribute_key = Column(Text, nullable=False)  # speech_length, humor_type, ...
    raw_answer = Column(Text)
    normalized_value = Column(Text)               # JSON-строка для массивов
    prompt_fragment = Column(Text)                # готовый фрагмент для системного промпта

class TwinValue(Base):
    __tablename__ = "twin_values"
    id = Column(Integer, primary_key=True)
    twin_id = Column(Integer, ForeignKey("twin_profiles.id", ondelete="CASCADE"))
    kind = Column(Text, nullable=False)   # value / despised
    item = Column(Text, nullable=False)   # название ценности или качества
    rank = Column(Integer)                # порядок приоритета (для ценностей)

class TwinMemory(Base):
    __tablename__ = "twin_memories"
    id = Column(Integer, primary_key=True)
    twin_id = Column(Integer, ForeignKey("twin_profiles.id", ondelete="CASCADE"))
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1024))
    trigger_topics = Column(Text)         # JSON-строка, например '["кризис","выбор"]'
    
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    twin_id = Column(Integer, ForeignKey("twin_profiles.id", ondelete="CASCADE"))
    user_id = Column(Text, nullable=False)   # ID пользователя в мессенджере (VK/MAX)
    role = Column(Text, nullable=False)      # 'user' / 'assistant'
    content = Column(Text, nullable=False)
    grief_orientation = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ClientOnboarding(Base):
    __tablename__ = "client_onboarding"
    user_id = Column(Text, primary_key=True)
    channel = Column(Text, nullable=False)
    status = Column(Text, nullable=False, default="new")
    step_index = Column(Integer, nullable=False, default=0)
    answers_json = Column(Text, nullable=False, default="{}")
    conversation_text = Column("docx_content", Text)
    conversation_approved = Column("docx_approved", Integer, nullable=False, default=0)
    previous_status = Column(Text)
    consent_version = Column(Text)
    consent_signed_at = Column(DateTime(timezone=True))
    twin_id = Column(Integer, ForeignKey("twin_profiles.id", ondelete="SET NULL"))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())