import csv
import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
import yaml
import logging
from tqdm import tqdm
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "openai_config_template.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_rag_system():
    config = load_config()
    openai.api_key = config['openai_api_key']
    
    # Setup LLM and embedding model
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
    return query_engine

def process_dataset(input_csv_path, output_csv_path):
    # Setup RAG system
    query_engine = setup_rag_system()
    
    # Read input CSV
    with open(input_csv_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
    
    # Process each question and write results
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['user_input', 'retrieved_contexts', 'response', 'reference']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in tqdm(rows, desc="Processing questions"):
            question = row['question']
            ground_truth = row['ground_truth_answer']
            
            # Get RAG response
            response = query_engine.query(question)
            
            # Format retrieved context as a list of strings
            retrieved_contexts = [node.node.text.strip() for node in response.source_nodes]
            
            # Write results in RAGAS format
            writer.writerow({
                'user_input': question,
                'retrieved_contexts': str(retrieved_contexts),  # Convert list to string for CSV
                'response': response.response,
                'reference': ground_truth
            })

if __name__ == "__main__":
    # Define paths
    input_csv_path = os.path.join(os.path.dirname(__file__), "datasets", "curated_esg_dataset_totalenergies_v2.csv")
    output_csv_path = os.path.join(os.path.dirname(__file__), "datasets", "rag_evaluation_dataset.csv")
    
    # Process dataset
    logger.info("Starting dataset processing...")
    process_dataset(input_csv_path, output_csv_path)
    logger.info(f"Processing complete. Results saved to {output_csv_path}") 