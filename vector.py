from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define the location for the Chroma vector store
db_location = "./chroma_langchain_db"

# Check if the database already exists
add_documents = not os.path.exists(db_location)

# If the database does not exist, create a new vector store and add documents
if add_documents:
    documents = []
    ids = []
    # Iterate through the DataFrame and create Document objects
    for i, index in df.iterrows():
        # Create a Document object for each review
        document = Document(
            page_content = index["Title"] + " " + index["Review"],
            metdata = {"rating": index["Rating"], "date": index["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        # Print progress every 100 documents
        vector_store = Chroma(collection_name="restaurant_reviews",
                              persist_directory=db_location,
                              embedding_function=embeddings
                              )
# If the database already exists, load the existing vector store
if add_documents:
    vector_store.add_documents(documents, ids=ids)

# Persist the vector store to disk
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            