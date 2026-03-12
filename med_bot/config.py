import os
from dotenv import load_dotenv

load_dotenv()

# LLMMod.ai
LLMOD_AI_API_KEY = os.environ["LLMOD_AI_API_KEY"]
LLMOD_AI_BASE_URL = os.environ["LLMOD_AI_BASE_URL"]

# Pinecone
PC_API_KEY = os.environ["PC_API_KEY"]
PC_ABSTRACTS_INDEX_NAME = os.environ["PC_ABSTRACTS_INDEX_NAME"]
PC_SYMPTOMS_INDEX_NAME = os.environ["PC_SYMPTOMS_INDEX_NAME"]

# Supabase
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
SUPABASE_TABLE_NAME=os.environ["SUPABASE_TABLE_NAME"]

# Weather API
WEATHER_API_KEY = os.environ["WEATHER_API_KEY"]