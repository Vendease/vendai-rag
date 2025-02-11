import os
import re
from dotenv import load_dotenv

# Import LangChain and Pinecone related modules
from langchain_community.document_loaders import Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
load_dotenv()

# File path and index name (update these as needed)
DOCX_FILE_PATH = r"C:\Users\danie\Documents\GitHub\vendai-rag\backend-error.docx"
INDEX_NAME = "vendai"

# Keys (ideally, these should be stored securely, e.g., in a .env file)
PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------------------------------------------------------------------
# Initialize External Services
# ------------------------------------------------------------------------------

# Initialize PineconeGRPC client
pc = PineconeGRPC(api_key=PINECONE_API_KEY)

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Set up the prompt template for the retrieval QA chain
PROMPT_TEMPLATE = PromptTemplate(
    template=(
        """
    You are a customer service chatbot for Vendease, an e-procurement platform for bulk food purchases and deliveries. You are capable of giving only concise and useful data.\
    Response Delivery:
    1. Interact with the user in a friendly and professional manner like a customer-service agent. Use emojis and a friendly tone to engage with customers.

    NOTE: 
    1. You do not give verbose and irrelevant responses instead, you give direct answers to questions asked. As part of your task,Use the following instructions below to answer the questions given to you at the end. Please follow the following rules:
    2. Use the context provided below.\
    3. If you don't know the answer or you can't find it in the context provided to you, don't try to make up an answer. Just say  simply say "I can't find the final answer".\
    4. If a task is impossible and irrelavent to your purpose and task, don't try to make up an answer. Just say  simply say you can't perform that task but give the customer a list of possible task instead from rubies based on context.\
    5. If you find the answer, write the answer in a detailed way.\


    {context}

    Question: {question}

    Helpful Answer:
        """
    ),
    input_variables=["context", "question"]
)

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Remove newlines and extra whitespace from a string.
    """
    text = text.replace('\n', ' ').strip()
    return re.sub(r'\s+', ' ', text)

# def load_and_split_document(file_path: str, delimiter: str = "######") -> list:
#     """
#     Load a DOCX document and split its content by the given delimiter.
#     Returns a list of non-empty text chunks.
#     """
#     loader = Docx2txtLoader(file_path)
#     docs = loader.load()
#     full_text = "".join(doc.page_content for doc in docs)
#     # Split the text by the delimiter and filter out any empty chunks
#     return [chunk.strip() for chunk in re.split(delimiter, full_text) if chunk.strip()]

# def initialize_embeddings() -> HuggingFaceEmbeddings:
#     """
#     Initialize HuggingFace BGE embeddings.
#     """
#     return HuggingFaceBgeEmbeddings(
#         model_name="BAAI/bge-large-en-v1.5",
#         model_kwargs={'device': 'cpu'},
#         encode_kwargs={'normalize_embeddings': True}
#     )

# def create_or_get_index(pc_client: PineconeGRPC, index_name: str, dimension: int = 1024) -> None:
#     """
#     Create a Pinecone index if it does not already exist.
#     """
#     existing_indexes = pc_client.list_indexes().names()
#     if index_name not in existing_indexes:
#         pc_client.create_index(
#             name=index_name,
#             dimension=dimension,
#             metric="dotproduct",
#             spec=ServerlessSpec(cloud='aws', region='us-east-1')
#         )

# def initialize_vector_store(chunks: list, index_name: str, embeddings: HuggingFaceBgeEmbeddings) -> PineconeVectorStore:
#     """
#     Initialize the Pinecone vector store from text chunks.
#     """
#     documents = [Document(page_content=chunk) for chunk in chunks]
#     return PineconeVectorStore.from_documents(
#         documents=documents,
#         index_name=index_name,
#         embedding=embeddings
#     )

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
vector_db = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)
def retrieve_information(query: str) -> str:
    """
    Retrieve relevant context and answer a query using a retrieval QA chain.
    """
    # Initialize Pinecone index for similarity search
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Search for relevant documents (top 5)
    relevant_docs = retriever.invoke(query)
    
    # Clean and combine the retrieved documents into a single context string
    context = "\n".join(clean_text(doc.page_content) for doc in relevant_docs)
    #print("Retrieved Context:\n", context)
    
    # Build the retrieval QA chain with the prompt template and retriever from the vector store
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT_TEMPLATE}
    )
    
    # Run the chain to get the answer
    result = retrieval_chain.invoke({"query": query})
    return result.get('result', "No result returned.")

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------

# def main():
#     # Load and split the document into manageable chunks
#     # chunks = load_and_split_document(DOCX_FILE_PATH)
    
#     # # Initialize embeddings
#     # embeddings = initialize_embeddings()
    
#     # # Create the Pinecone index if needed
#     # create_or_get_index(pc, INDEX_NAME)
    
#     # # Initialize the vector store using the document chunks
#     # vector_db = initialize_vector_store(chunks, INDEX_NAME, embeddings)
    
#     # Example query (feel free to change this)
#     query = "How do I resolve- Delivery fee override only applies to non-cancelled orders."
#     answer = retrieve_information(query, vector_db)
    
#     print("\nFinal Answer:\n", answer)

# if __name__ == "__main__":
#     main()

# ------------------------------------------------------------------------------
# Function Execution
# ------------------------------------------------------------------------------
