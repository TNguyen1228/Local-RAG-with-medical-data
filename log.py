import json
import logging

LOG_FILE = "./query_document_response_logs.json"

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    # Create a custom formatter that produces well-formatted JSON
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            if isinstance(record.msg, dict):
                return json.dumps(record.msg, ensure_ascii=False, indent=2)
            return json.dumps({"message": record.getMessage()}, ensure_ascii=False, indent=2)
    
    handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    handler.setFormatter(JsonFormatter())
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger
