# llamaindex_rag_demo.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
import os
import openai
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "openai_config_template.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
openai.api_key = config['openai_api_key']

# Setup your custom LLM + Embedder
Settings.llm = OpenAI(model=config['completion_model'], temperature=0)
Settings.embed_model = OpenAIEmbedding(model=config['embedding_model'])

# Load and index documents
logger.info("🔹 Loading and indexing documents...")
docs = SimpleDirectoryReader("./data").load_data()
logger.info(f"🔸 Loaded {len(docs)} documents")

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(docs)
logger.info(f"Created {len(nodes)} chunks from documents")

index = VectorStoreIndex(nodes)
query_engine = index.as_query_engine(similarity_top_k=5)

# Main interaction loop
while True:
    print("\n💬 Ask a question about the document (or type 'exit' to quit):")
    question = input(">> ")
    
    if question.lower() == 'exit':
        break
        
    logger.info("Processing query...")
    response = query_engine.query(question)
    
    print("\n🧠 LLM Response:")
    print(response.response)

    print("\n📚 Source Chunks Used:")
    for i, node in enumerate(response.source_nodes):
        print(f"\nChunk {i+1}:")
        print("```")
        print(node.node.text.strip())
        print("```")
