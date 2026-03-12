import re
import pandas as pd
from langchain_core.documents import Document

# -------------------------
# File-local constants
# -------------------------

ABSTRACTS_DATASET_PATH = "/data/RePORTER_PRJABS_C_FY2025.csv"

NEEDED_COLS = [
    "APPLICATION_ID",
    "ABSTRACT_TEXT",
]

TEXT_COLUMNS = [
    "ABSTRACT_TEXT",
]

# -------------------------
# Helpers
# -------------------------

def normalize_ws(text):
    """Collapse repeated whitespace and strip ends."""
    if pd.isna(text):
        return None
    return re.sub(r"\s+", " ", str(text)).strip()

def coerce_application_id(x):
    """Best-effort stable ID coercion to string (keeps big ints safe)."""
    if pd.isna(x):
        return None
    # Avoid turning large ints into scientific notation
    try:
        # If it's numeric but integral, represent as int string
        if isinstance(x, (int,)) or (isinstance(x, float) and x.is_integer()):
            return str(int(x))
    except Exception:
        pass
    return str(x).strip()

# -------------------------
# Public API
# -------------------------

def load_medical_abstracts(
    dataset_path: str = ABSTRACTS_DATASET_PATH,
    n_first_docs: int | None = None,
    drop_missing_abstracts: bool = True,
) -> list[Document]:
    """
    Load a CSV of medical abstracts and return a list of LangChain Documents.

    CSV columns required:
      - APPLICATION_ID
      - ABSTRACT_TEXT

    Each Document:
      - page_content: ABSTRACT_TEXT
      - metadata: {"application_id": APPLICATION_ID}

    Args:
        dataset_path: Path to CSV file.
        n_first_docs: Optional limit on number of rows (dev).
        drop_missing_abstracts: If True, drop rows with empty/NaN abstracts.

    Returns:
        list[Document]
    """
    # Read only needed columns; keep IDs as strings where possible
    df = pd.read_csv(
        dataset_path,
        usecols=NEEDED_COLS,
        dtype={"APPLICATION_ID": "string"},
        keep_default_na=True,
    ).copy()

    # Optionally limit dataset size (DEV ONLY)
    if n_first_docs is not None:
        df = df.iloc[:n_first_docs].copy()

    # Normalize whitespace in text columns
    for col in TEXT_COLUMNS:
        df[col] = df[col].apply(normalize_ws)

    # Clean/normalize application IDs
    # (If dtype coercion above fails for some files, this still stabilizes output)
    df["APPLICATION_ID"] = df["APPLICATION_ID"].apply(coerce_application_id)

    if drop_missing_abstracts:
        df = df[df["ABSTRACT_TEXT"].notna() & (df["ABSTRACT_TEXT"].str.len() > 0)]

    # Build LangChain Documents
    documents: list[Document] = []
    for _, row in df.iterrows():
        documents.append(
            Document(
                page_content=row["ABSTRACT_TEXT"],
                metadata={"application_id": row["APPLICATION_ID"]},
            )
        )

    return documents
