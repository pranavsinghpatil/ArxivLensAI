from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "What is deep learning?"
print(model.encode(query))  # Direct string
print('------------------------------------------------------------------------------------------------------------------------------------------')  # Direct string
print(model.encode([query]))  # List containing a string
