import logging
import os

class SystemLogger:
    def __init__(self, log_dir="logs", log_file="events.log"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.log_path = os.path.join(self.log_dir, log_file)
        
        # Use UTF-8 encoding for file handler to support all Unicode characters
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("AI_Face_Tracking")

    def info(self, message):
        # Sanitize message to ASCII-safe for console (Windows cp1252 fallback)
        safe = message.encode('ascii', errors='replace').decode('ascii')
        self.logger.info(safe)

    def error(self, message):
        safe = message.encode('ascii', errors='replace').decode('ascii')
        self.logger.error(safe)

    def warning(self, message):
        safe = message.encode('ascii', errors='replace').decode('ascii')
        self.logger.warning(safe)

    def debug(self, message):
        safe = message.encode('ascii', errors='replace').decode('ascii')
        self.logger.debug(safe)

    def clear_logs(self):
        """Empties the log file."""
        if os.path.exists(self.log_path):
            with open(self.log_path, 'w', encoding='utf-8'):
                pass
