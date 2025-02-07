# TelegramAPI

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment:
    - On Windows: `venv\Scripts\activate`
    - On Unix or MacOS: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and fill in your tokens.

## Running
1. Start the FastAPI app: `uvicorn telegrambot:app --host 0.0.0.0 --port 8000`
2. Run the Telegram bot: `python telegrambot.py`

## Using Docker
1. Build and start the services: `docker-compose up --build`
2. The FastAPI app will be available at `http://localhost:8000`
3. The Telegram bot will start automatically.
