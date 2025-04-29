import os
import pandas as pd
import yaml
import logging
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from ragas.testset import TestsetGenerator
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LlamaIndexLLMWrapper
from ragas.integrations.llama_index import evaluate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "openai_config_template.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    config = load_config()
    
    # Set OpenAI API key from config
    os.environ["OPENAI_API_KEY"] = config['openai_api_key']
    
    # Load documents
    logger.info("Loading documents...")
    documents = SimpleDirectoryReader("./data").load_data()
    logger.info(f"Loaded {len(documents)} documents")

    # Initialize LLM and embedding model using config
    generator_llm = OpenAI(model=config['completion_model'])
    embeddings = OpenAIEmbedding(model=config['embedding_model'])

    # Initialize Testset Generator
    logger.info("Initializing testset generator...")
    generator = TestsetGenerator.from_llama_index(
        llm=generator_llm,
        embedding_model=embeddings,
    )

#FORMAT YOUR CURATED DATASET HERE


    # # Generate synthetic testset
    # logger.info("Generating testset...")
    # testset = generator.generate_with_llamaindex_docs(
    #     documents,
    #     testset_size=5,
    # )

    # Convert testset to pandas DataFrame
    df = testset.to_pandas()
    logger.info("Testset generated successfully")
    print(df.head())

    # Build VectorStoreIndex and Query Engine
    logger.info("Building vector store index...")
    vector_index = VectorStoreIndex.from_documents(documents)
    query_engine = vector_index.as_query_engine()

    # Define evaluation metrics
    logger.info("Setting up evaluation metrics...")
    evaluator_llm = LlamaIndexLLMWrapper(OpenAI(model=config['completion_model']))
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    # Convert testset to Ragas Evaluation Dataset
    ragas_dataset = testset.to_evaluation_dataset()

    # Evaluate the Query Engine
    logger.info("Starting evaluation...")
    result = evaluate(
        query_engine=query_engine,
        metrics=metrics,
        dataset=ragas_dataset,
    )

    # Display evaluation results
    logger.info("Evaluation complete. Results:")
    print(result)

    # Convert results to pandas DataFrame
    result_df = result.to_pandas()
    print(result_df.head())

if __name__ == "__main__":
    main()
