from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------
# File-local constants
# -------------------------

CHUNK_SIZE = 1024
OVERLAP_RATIO = 0.15


# -------------------------
# Public API
# -------------------------

def chunk_documents(documents):
    """
    Split documents into overlapping chunks using
    RecursiveCharacterTextSplitter.

    Args:
        documents (list[Document]): Input documents

    Returns:
        list[Document]: Chunked documents
    """

    chunk_overlap = int(CHUNK_SIZE * OVERLAP_RATIO)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    return splitter.split_documents(documents)
