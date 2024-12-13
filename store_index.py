from src.helper import load_pdf, text_split, download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embedding()

# os.environ["PINECONE_API_KEY"] = "1443d596-266d-4c34-83f6-e8903fc4c6ff"

pinecone_vector_store = PineconeVectorStore(index_name="medical-chatbot", embedding=embeddings)

pinecone_vector_store.add_texts(
    [doc.page_content for doc in text_chunks],  # Extract page_content from Document objects
    metadatas=[{"source": "data\\Medical_book.pdf", "page": i} for i in range(len(text_chunks))],
)