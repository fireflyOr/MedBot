from med_bot.build_vector_dbs.load_abstracts import load_medical_abstracts
from med_bot.build_vector_dbs.load_symptom_treament import load_symptom_treatments
from med_bot.build_vector_dbs.chunking import chunk_documents
from med_bot.vector_db.embedding import get_embeddings
from med_bot.vector_db.index import get_index
from med_bot.build_vector_dbs.upsert import upsert_documents
from med_bot.config import PC_ABSTRACTS_INDEX_NAME, PC_SYMPTOMS_INDEX_NAME

def main(n_docs=None, abstracts=False, file_path=None, start_chunk=0):
    if abstracts:
        if file_path:
            docs = load_medical_abstracts(dataset_path=file_path, n_first_docs=n_docs)
        else:
            docs = load_medical_abstracts(n_first_docs=n_docs)
        index_name = PC_ABSTRACTS_INDEX_NAME
        id_col = "application_id"
    else:
        docs = load_symptom_treatments(n_first_docs=n_docs)
        index_name = PC_SYMPTOMS_INDEX_NAME
        id_col = "disease_code"

    print(f"Loaded {len(docs)} docs")

    # 2) Chunk
    splits = chunk_documents(docs)
    print(f"Created {len(splits)} chunks")

    # Skip the part that was already uploaded to the server to save time and money
    if start_chunk > 0:
        splits = splits[start_chunk:]
        print(f"Skipping first {start_chunk} chunks. Resuming from chunk {start_chunk}...")

    # 3) Embeddings + Pinecone index
    embeddings = get_embeddings()
    index = get_index(index_name)

    # 4) Upsert
    upsert_documents(index=index, splits=splits, embeddings=embeddings, id_col=id_col)
    print("Done with current batch.")


if __name__ == "__main__":
    # The absolute last file to upload, using the correct "_new" filename!
    abstract_files = [
        r"data\RePORTER_PRJABS_C_FY2020_new.csv"
    ]

    for file in abstract_files:
        print(f"\n--- Starting upload for {file} ---")
        main(n_docs=None, abstracts=True, file_path=file)
    
    print("\n--- Skipping Symptoms upload (already done) ---")