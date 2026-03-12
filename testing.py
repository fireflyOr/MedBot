import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
from pinecone import Pinecone
from openai import OpenAI