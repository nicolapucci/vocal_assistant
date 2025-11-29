from sqlalchemy import (
    Column,
    String,
    Integer,
    )
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash
from managers.DB_Models import Base


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key = True)
    
    username = Column(String, unique=True,nullable=False)
    password_hash = Column(String(256),nullable = False)

    app_settings = relationship("AppSetting", back_populates='user')
    devices = relationship('Device',back_populates='user')


    def set_password(self, clear_password):
        self.password_hash = generate_password_hash(clear_password)

    def check_password(self, clear_password):
        return check_password_hash(self.password_hash, clear_password)