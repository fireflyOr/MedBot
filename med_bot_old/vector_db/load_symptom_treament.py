import re
import pandas as pd
from langchain_core.documents import Document


# -------------------------
# File-local constants
# -------------------------

SYMPTOMS_DATASET_PATH = "data/Diseases_Symptoms.csv"

SYMPTOM_NEEDED_COLS = [
    "Name",
    "Symptoms",
    "Treatments",
]

SYMPTOM_TEXT_COLUMNS = [
    "Symptoms",
    "Treatments",
]


# -------------------------
# Helpers
# -------------------------

def normalize_ws(text):
    """Collapse repeated whitespace and strip ends."""
    if pd.isna(text):
        return None
    return re.sub(r"\s+", " ", str(text)).strip()


# -------------------------
# Public API
# -------------------------

def load_symptom_treatments(
    dataset_path: str = SYMPTOMS_DATASET_PATH,
    n_first_docs: int | None = None,
    drop_missing_symptoms: bool = True,
) -> list[Document]:
    """
    Load a symptom-treatment CSV and return LangChain Documents.

    CSV columns required:
      - Name
      - Symptoms
      - Treatments

    Each Document:
      - page_content: Symptoms
      - metadata:
            {
                "name": Name,
                "treatments": Treatments
            }
    """

    df = pd.read_csv(
        dataset_path,
        usecols=SYMPTOM_NEEDED_COLS,
        keep_default_na=True,
    ).copy()

    if n_first_docs is not None:
        df = df.iloc[:n_first_docs].copy()

    # Normalize text
    for col in SYMPTOM_TEXT_COLUMNS:
        df[col] = df[col].apply(normalize_ws)

    if drop_missing_symptoms:
        df = df[df["Symptoms"].notna() & (df["Symptoms"].str.len() > 0)]

    documents: list[Document] = []

    for _, row in df.iterrows():
        documents.append(
            Document(
                page_content=row["Symptoms"],
                metadata={
                    "name": normalize_ws(row["Name"]),
                    "treatments": row["Treatments"],
                },
            )
        )

    return documents
