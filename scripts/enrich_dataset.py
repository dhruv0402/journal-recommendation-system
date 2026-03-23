import sys
from pathlib import Path

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from src.phase2.dataset_semantic_enricher import extract_dataset_semantics
print("Dataset enriched successfully.")