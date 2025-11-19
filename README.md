# Vocal Assistant
***all models are in local, with the intention ofnusing cloud ones***

### Technologies
#### edge_device
- Picovoice for Wake Word Detection
- Pygame for audio

#### speech processor
- Whisper for Speech-to-Text
- openAi TTS for Text-to-Speech

#### trainers
- models based on roBERTa for NLU
- transformers for model training
- AmazonScience/massive dataset for model training
- huggingface datasets

#### backend_container
- flask for backend
- redis fir session db
- postgres db
- docker compose for redis, postgres and backend_service
