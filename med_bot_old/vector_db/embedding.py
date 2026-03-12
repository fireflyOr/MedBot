from langchain_openai import OpenAIEmbeddings
from med_bot.config import LLMOD_AI_API_KEY, LLMOD_AI_BASE_URL


# -------------------------
# File-local constants
# -------------------------

EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
EMBEDDING_DIM = 1536


# -------------------------
# Public API
# -------------------------

def get_embeddings():
    """
    Create and return the embeddings client.
    """
    return OpenAIEmbeddings(
        api_key=LLMOD_AI_API_KEY,
        base_url=LLMOD_AI_BASE_URL,
        model=EMBEDDING_MODEL,
    )
