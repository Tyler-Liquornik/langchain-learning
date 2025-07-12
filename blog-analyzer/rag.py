import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

if __name__ == "__main__":
    print("Retrieving...")

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embedding_model)

    query = "Briefly tell me about Pinecone"
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain)
    result = retrieval_chain.invoke(input={"input": query})

    print(result['answer'])
