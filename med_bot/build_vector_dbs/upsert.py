from med_bot.vector_db.index import PC_BATCH_SIZE


def _batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def upsert_documents(index, splits, embeddings, id_col):
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

            doc_id = str(md.get(id_col))
            start_index = md.get("start_index", None)

            # Stable unique vector ID
            if start_index is not None:
                vector_id = f"{doc_id}-{int(start_index)}"
            else:
                vector_id = f"{doc_id}-{upsert_count + j}"

            metadata = {
                "chunk_text": doc.page_content,
            }

            if start_index is not None:
                metadata["start_index"] = int(start_index)

            metadata |= md
            upserts.append((vector_id, vec, metadata))

        index.upsert(vectors=upserts)

        upsert_count += len(upserts)

        print(
            f"Upserted batch {batch_idx}: "
            f"{len(upserts)} vectors (total {upsert_count})"
        )
