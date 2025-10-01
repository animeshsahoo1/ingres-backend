from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



load_dotenv()

# Vector Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="ingres_collection",
    embedding=embedding_model
)

# Take User Query
query = input("> ")

# Vector Similarity Search [query] in DB
search_results = vector_db.similarity_search(
    query=query
)

print(f"Top {len(search_results)} search results:")
for i, doc in enumerate(search_results, 1):
    print(f"\nResult {i}:\n{doc.page_content}\n")

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    convert_system_message_to_human=True
)

template = """You are a helpful assistant. Use the following context when responding:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
retriever = vector_db.as_retriever()


rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)

print(rag_chain.invoke(query))