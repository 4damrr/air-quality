import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings

# Loader
loader = PyPDFLoader("data/weather_forecasting.pdf")
documents = loader.load()

# Splitter
splitter = RecursiveCharacterTextSplitter(
 chunk_size=1000,
 chunk_overlap=200,
)
chunks = splitter.split_documents(documents)

## Generate embeddings

embeddings_model = OllamaEmbeddings(
    model='qwen3-embedding:0.6b',
)

pg_connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'

# db = PGVector.from_documents(chunks, embeddings_model, connection_string=pg_connection)
db = PGVector(
    connection_string=pg_connection,
    embedding_function=embeddings_model,
)
results = db.similarity_search("weather forecasting techniques", k=4)
for result in results:
    print(result.page_content)