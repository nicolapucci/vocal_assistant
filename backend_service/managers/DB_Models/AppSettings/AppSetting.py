from sqlalchemy import (
    Column,
    String,
    ForeignKey,
    Integer
    )
from sqlalchemy.orm import relationship, declared_attr

from managers.DB_Models import Base

class AppSetting(Base):

    __tablename__ = 'app_settings'

    id = Column(Integer, primary_key=True)

    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    app_name = Column(String, nullable=False)


    user = relationship('User', back_populates='app_settings')

    __mapper_args__ = {
        'polymorphic_on': app_name,
        'polymorphic_identity':'generic'
    }

    