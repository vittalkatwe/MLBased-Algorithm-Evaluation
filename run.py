
from waitress import serve
from app import app
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting the Waitress server...")
    serve(app, host='0.0.0.0', port=8000, threads=10)  # Explicitly set threads for concurrency
    logger.info("Server stopped.")