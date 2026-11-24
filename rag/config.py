"""Central settings, so the notebook and the evaluation always use the same values."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "data" / "diabetes_guidelines.pdf"
CHROMA_DIR = ROOT / "data" / "chroma_db"
COLLECTION = "idf_t2d_2025"

# Running title printed at the top of every even page of the PDF.
RUNNING_TITLE = "IDF Global Clinical Practice Recommendations for Managing Type 2 Diabetes – 2025"

# Chosen with eval/retrieval_eval.py (see README "Evaluation").
# bge models are trained for search and expect this instruction in front of each query.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # 768-dimensional embeddings
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 100  # characters shared between neighbouring chunks
TOP_K = 5  # chunks passed to Claude per question
CANDIDATES = 50  # how deep each search method looks before the results are fused
RRF_K = 60  # standard constant for reciprocal rank fusion

CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 700
