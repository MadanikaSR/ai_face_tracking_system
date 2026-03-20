import logging
import os

class SystemLogger:
    def __init__(self, log_dir="logs", log_file="events.log"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.log_path = os.path.join(self.log_dir, log_file)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("AI_Face_Tracking")

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def debug(self, message):
        self.logger.debug(message)

    def clear_logs(self):
        """Empties the log file."""
        if os.path.exists(self.log_path):
            with open(self.log_path, 'w'):
                pass
