from sqlalchemy import (
    Column,
    String,
    Integer,
    )
from sqlalchemy.orm import relationship

from managers.DB_Models import Base


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key = True)
    
    username = Column(String, unique=True,nullable=False)
    password_hash = Column(String,nullable = False)

    app_settings = relationship("AppSetting", back_populates='user')
    devices = relationship('Device',back_populates='user')

