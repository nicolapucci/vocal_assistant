from sqlalchemy import Column, String, Integer, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from managers.DB_Models import Base

class Device(Base):
    __tablename__ = 'devices'

    id = Column(Integer, primary_key=True)
    
    device_name = Column(String,nullable=False)
    
    device_token = Column(String, unique=True, nullable=False) 
    
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    
    device_type = Column(String, nullable=False, index=True)
    
    is_active = Column(Boolean, default=True) 
    
    user = relationship("User", back_populates="devices")


    
    __mapper_args__ = {
        'polymorphic_on': device_type,
        'polymorphic_identity': 'generic'
    }