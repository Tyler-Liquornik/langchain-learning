import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
if __name__ == "__main__":
    print("Loading Documents...")
    loader = TextLoader("/Users/tylerliquornik/Desktop/learning/langchain-learning/blog-analyzer/mediumblog1.txt")
    document = loader.load()

    print("Splitting Documents into Chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(document)

    print("Creating Pinecone Vector Store...")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    PineconeVectorStore.from_documents(chunks, embedding_model, index_name=os.environ["INDEX_NAME"])