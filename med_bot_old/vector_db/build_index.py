from med_bot.vector_db.load_abstracts import load_medical_abstracts
from med_bot.vector_db.load_symptom_treament import load_symptom_treatments
from med_bot.vector_db.chunking import chunk_documents
from med_bot.vector_db.embedding import get_embeddings
from med_bot.vector_db.vector_db import get_index, upsert_documents
from med_bot.config import PC_ABSTRACTS_INDEX_NAME, PC_SYMPTOMS_INDEX_NAME


def main(n_docs=10, abstracts=False):
    if abstracts:
        docs = load_medical_abstracts(n_first_docs=n_docs)
        index_name = PC_ABSTRACTS_INDEX_NAME
    else:
        docs = load_symptom_treatments(n_first_docs=n_docs)
        index_name = PC_SYMPTOMS_INDEX_NAME

    print(f"Loaded {len(docs)} abstracts")

    # 2) Chunk
    splits = chunk_documents(docs)
    print(f"Created {len(splits)} chunks")

    # 3) Embeddings + Pinecone index
    embeddings = get_embeddings()
    index = get_index(index_name)

    # 4) Upsert
    upsert_documents(index=index, splits=splits, embeddings=embeddings)


    print("Done.")


if __name__ == "__main__":
    main()
