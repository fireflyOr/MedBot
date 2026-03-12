from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_4dhet7_3Dgb24cnPVqwY7x7JCz2eaoePTxAnKjCpChqE8N7gHEqNJa1SDzzzHU6Gw85jvR")

print("Clearing medical-abstracts-rag...")
try:
    pc.Index("medical-abstracts-rag").delete(delete_all=True)
    print("Done!")
except Exception as e:
    print("Error:", e)

print("Clearing medical-symptoms-rag...")
try:
    pc.Index("medical-symptoms-rag").delete(delete_all=True)
    print("Done!")
except Exception as e:
    print("Error:", e)