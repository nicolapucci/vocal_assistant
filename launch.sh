#!/bin/bash

NLU_DIR="processors/NLUProcessor"
SPEECH_DIR="processors/SpeechProcessor"
DEVICE_DIR="edge_device_simulator"


echo "▶️ 1/4 Avvio dei servizi Docker (in background)..."
docker-compose up -d

echo "▶️ 2/4 Avvio del servizio NLU su porta 5002 (in background)..."
(cd "$NLU_DIR" && source .venv/Scripts/activate &&  flask run --port 5002) &
NLU_PID=$!


echo "▶️ 3/4 Avvio del servizio Speech su porta 5001 (in background)..."
(cd "$SPEECH_DIR" && source .venv/Scripts/activate && flask run --port 5001) &
SPEECH_PID=$!

echo "▶️ 4/4 Avvio del Edge device..."
(cd "$DEVICE_DIR" && source .venv/Scripts/activate && python edge_mic_script.py)
DEVICE_PID=$!

# -----------------------------------------------------------
# --- MESSAGGIO FINALE E GESTIONE DELLO SPEGNIMENTO ---
# -----------------------------------------------------------

echo "✅ Tutti i 4 servizi sono stati avviati!"
echo ""
echo "Per visualizzare i log del device o dei servizi Flask, controlla i log di sistema o il terminale del servizio specifico."
echo ""
echo "Premi [Invio] per terminare tutti i processi Flask/Python e spegnere Docker."

# Attende che l'utente prema Invio
read -r

echo ""
echo "🛑 Terminazione dei servizi..."

# Uccidi i processi in background che abbiamo lanciato
kill $EDGE_PID $NLU_PID $SPEECH_PID 2>/dev/null

# Spegni i container Docker
echo "🛑 Spegnimento di Docker Compose..."
docker-compose down

echo "Adiós!"