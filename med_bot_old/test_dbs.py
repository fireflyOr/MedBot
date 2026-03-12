from med_bot.vector_db.embedding import get_embeddings
from med_bot.vector_db.vector_db import get_index, retrieve_matches
from med_bot.db import sql_command_table, load_table
from med_bot.config import PC_ABSTRACTS_INDEX_NAME


# Supabase - user data
table = load_table()
sql_command = "*"
# list
rows_from_command = sql_command_table(table, sql_command)
# dict with the columns as keys
print(rows_from_command[0])


# Pinecone - abstracts data
embeddings = get_embeddings()
index = get_index(PC_ABSTRACTS_INDEX_NAME)
query = "Migraine"
# list
matches = retrieve_matches(index, embeddings, query, top_k=5)
abstracts = [match["metadata"]["chunk_text"] for match in matches]
print(abstracts[0])


