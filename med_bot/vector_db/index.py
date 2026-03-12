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



def retrieve_matches(index, embeddings, query: str, top_k: int):
    q_vec = embeddings.embed_query(query)
    res = index.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True,
    )
    return res["matches"]
