from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTING_CONFIG_PATH = PROJECT_ROOT / "config" / "routing.yaml"
