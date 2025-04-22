# llamaindex_app.py
import os
import openai
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
import yaml

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "openai_config_template.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
openai.api_key = config['openai_api_key']

# Setup your custom LLM + Embedder
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# --- LOAD DOCUMENTS ---
@st.cache_resource(show_spinner="📚 Loading and indexing document...")
def load_index():
    docs = SimpleDirectoryReader("./data").load_data()
    
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(docs)

    index = VectorStoreIndex(nodes)
    return index

index = load_index()
query_engine = index.as_query_engine(similarity_top_k=5)

# --- USER INPUT ---
question = st.text_input("💬 Ask something about the document:")

if question:
    with st.spinner("Thinking..."):
        response = query_engine.query(question)
        st.markdown("### 🧠 LLM Response:")
        st.write(response.response)

        st.markdown("### 📚 Source Chunks Used:")
        for i, node in enumerate(response.source_nodes):
            st.markdown(f"**Chunk {i+1}:**\n```\n{node.node.text.strip()}\n```")
