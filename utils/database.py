from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config.config import Config
from loguru import logger

Base = declarative_base()

class RecognitionEvent(Base):
    """Model for recognition events."""
    __tablename__ = 'recognition_events'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<RecognitionEvent(name='{self.name}', confidence={self.confidence})>"

class RecognitionDatabase:
    """Database for logging recognition events."""
    
    def __init__(self):
        self.engine = create_engine(Config.DATABASE_URL, echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self. engine)
        self.session = Session()
        logger.info("Database initialized")
    
    def log_recognition(self, name: str, confidence: float):
        """Log a recognition event."""
        event = RecognitionEvent(name=name, confidence=confidence)
        self.session.add(event)
        self.session.commit()
    
    def get_recent_events(self, limit: int = 100):
        """Get recent recognition events."""
        return self.session.query(RecognitionEvent)\
            .order_by(RecognitionEvent.timestamp.desc())\
            .limit(limit)\
            .all()
    
    def get_statistics(self):
        """Get recognition statistics."""
        # Implementation for statistics
        pass