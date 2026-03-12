from med_bot.vector_db.embedding import get_embeddings
from med_bot.vector_db.index import get_index, retrieve_matches
from med_bot.user_data_db import sql_command_table, load_table
from med_bot.config import PC_ABSTRACTS_INDEX_NAME, PC_SYMPTOMS_INDEX_NAME


# Supabase - user data
table = load_table()
sql_command = "*"
# list
rows_from_command = sql_command_table(table, sql_command)
# dict with the columns as keys
print("User data row example:")
print(rows_from_command[0])


# Pinecone - abstracts data
embeddings = get_embeddings()

abstracts_index = get_index(PC_ABSTRACTS_INDEX_NAME)
abstract_query = "Migraine"
# list
abstract_matches = retrieve_matches(abstracts_index, embeddings, abstract_query, top_k=5)
abstracts = [match["metadata"]["chunk_text"] for match in abstract_matches]
print("\n\nAbstract (chunk) match example:")
print(abstracts[0])

symptoms_index = get_index(PC_SYMPTOMS_INDEX_NAME)
symptoms_query = "Migraine"
# list
symptoms_matches = retrieve_matches(symptoms_index, embeddings, symptoms_query, top_k=5)
symptoms_metadata = [match["metadata"] for match in symptoms_matches]
print("\n\nSymptoms match metadata example:")
print(symptoms_metadata[0])
