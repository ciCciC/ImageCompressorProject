import os
from dotenv import dotenv_values

CODE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

config = {
    **dotenv_values(".env"),
    **dotenv_values(".env.test")
}

QDRANT_URL = config.get("QDRANT_URL", "http://localhost:6333/")
QDRANT_API_KEY = config.get("QDRANT_API_KEY", "")
COLLECTION_NAME = config.get("COLLECTION_NAME", "latent-images")
TEXT_FIELD_NAME = config.get("TEXT_FIELD_NAME", "document")