from pinecone import Pinecone, ServerlessSpec
from med_bot.config import PC_API_KEY
from med_bot.vector_db.embedding import EMBEDDING_DIM

PC_METRIC = "cosine"
PC_CLOUD = "aws"
PC_REGION = "us-east-1"
PC_BATCH_SIZE = 64

pc_spec = ServerlessSpec(cloud=PC_CLOUD, region=PC_REGION)


def get_index(index_name):
    """
    Create (if needed) and return the Pinecone index.
    """
    pc = Pinecone(api_key=PC_API_KEY)

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric=PC_METRIC,
            spec=pc_spec,
        )

    return pc.Index(index_name)


def _batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def upsert_documents(index, splits, embeddings):
    """
    Embed and upsert chunked medical abstract documents into Pinecone.

    Expected doc.metadata:
      - application_id (str)
      - start_index (int) (added by splitter)
    """

    upsert_count = 0

    for batch_idx, batch_docs in enumerate(_batched(splits, PC_BATCH_SIZE), start=1):
        texts = [d.page_content for d in batch_docs]
        vectors = embeddings.embed_documents(texts)

        upserts = []

        for j, (doc, vec) in enumerate(zip(batch_docs, vectors)):
            md = doc.metadata

            application_id = str(md.get("application_id"))
            start_index = md.get("start_index", None)

            # Stable unique vector ID
            if start_index is not None:
                vector_id = f"{application_id}-{int(start_index)}"
            else:
                vector_id = f"{application_id}-{upsert_count + j}"

            metadata = {
                "application_id": application_id,
                "chunk_text": doc.page_content,
            }

            if start_index is not None:
                metadata["start_index"] = int(start_index)

            upserts.append((vector_id, vec, metadata))

        index.upsert(vectors=upserts)

        upsert_count += len(upserts)

        print(
            f"Upserted batch {batch_idx}: "
            f"{len(upserts)} vectors (total {upsert_count})"
        )


def retrieve_matches(index, embeddings, query: str, top_k: int):
    q_vec = embeddings.embed_query(query)
    res = index.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True,
    )
    return res["matches"]
