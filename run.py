

from waitress import serve
from app import app
import logging
import os  # Import os module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting the Waitress server...")
    port = int(os.environ.get("PORT", 8000))
    serve(app, host='0.0.0.0', port=port, threads=10)
    logger.info("Server stopped.")
