from sqlalchemy import (
    Column,
    String,
    ForeignKey,
    Integer,
    )


from .AppSetting import AppSetting

class SpotifySetting(AppSetting):
    __tablename__ = 'spotify_settings'
    
    id = Column(Integer, ForeignKey('app_settings.id'), primary_key=True)
    
    default_spotify_device_id = Column(String, nullable=True)
    
    #i token li metto qua perchè non è detto che tutte le app usino lo stesso sistemta (access e refresh token)
    access_token = Column(String,nullable=True)
    refresh_token = Column(String,nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'spotify',
    }