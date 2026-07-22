from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTING_CONFIG_PATH = PROJECT_ROOT / "config" / "routing.yaml"
DB_PATH = PROJECT_ROOT / "data" / "router.db"
CLASSIFIER_MODEL_PATH = PROJECT_ROOT / "data" / "classifier_model.joblib"

# "heuristic" (default) or "sklearn" - which Classifier implementation the API
# wires up at startup. See classifier/heuristic.py and classifier/sklearn_classifier.py.
CLASSIFIER = os.environ.get("CLASSIFIER", "heuristic")
