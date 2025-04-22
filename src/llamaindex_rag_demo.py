# llamaindex_rag_demo.py
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
import os
import openai
import yaml

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "openai_config_template.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
openai.api_key = config['openai_api_key']

# Define persistent storage location
INDEX_PATH = "llamaindex_storage"

# Step 1 — Load the document
print("🔹 Loading PDF from ./data directory...")
reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
documents = reader.load_data()
print(f"🔸 Loaded {len(documents)} documents")

# Step 2 — Either load or create the index
if os.path.exists(INDEX_PATH):
    print("🔹 Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
    index = load_index_from_storage(storage_context)
else:
    print("🔹 Building new index and saving it...")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=INDEX_PATH)

# Step 3 — Create a query engine
query_engine = index.as_query_engine(similarity_top_k=5)

# Step 4 — Ask a question
print("\n💬 Ask a question about the document:")
query = input(">> ")

response = query_engine.query(query)

print("\n🧠 LLM Response:")
print(response.response)

print("\n📚 Source Chunks Used:")
for node in response.source_nodes:
    print("–", node.node.get_content().strip()[:300], "...")
