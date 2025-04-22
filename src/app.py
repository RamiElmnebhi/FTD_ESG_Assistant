import os
import streamlit as st
import PyPDF2
import numpy as np
import faiss  # Import FAISS
from openai import OpenAI
import yaml
import tiktoken
import hashlib  # For generating unique file hashes

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "openai_config_template.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
api_key = config['openai_api_key']
embedding_model = config['embedding_model']
completion_model = config['completion_model']

client = OpenAI(api_key=api_key)
encoding = tiktoken.encoding_for_model(embedding_model)

# Create a directory to store indices
index_directory = "faiss_indices"
os.makedirs(index_directory, exist_ok=True)

# Global dictionaries to hold indexed documents, FAISS indices, and chunks
indexed_documents = {}
indexed_chunks = {}

# Create a FAISS index
def create_index(dim):
    print("Creating FAISS index.")
    return faiss.IndexFlatL2(dim)  # Using L2 distance (Euclidean)

# Save a FAISS index to disk
def save_index(index, file_path):
    print(f"Saving index to {file_path}.")
    faiss.write_index(index, file_path)

# Load a FAISS index from disk
def load_index(file_path):
    print(f"Loading index from {file_path}.")
    return faiss.read_index(file_path)

def extract_text_from_pdf(pdf_file):
    print(f"Extracting text from PDF.")
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_index, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text() or ""
        if page_text:
            print(f"Extracted text from page {page_index}.")
            yield page_text, page_index  # Yield text and page index for embedding storage

def chunk_text(text, token_limit=512):
    tokens = encoding.encode(text)
    print(f"Chunking text into sizes of {token_limit} tokens.")
    chunks = []
    for i in range(0, len(tokens), token_limit):
        chunk_tokens = tokens[i:i + token_limit]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def get_embeddings(text_chunks):
    print(f"Getting embeddings for {len(text_chunks)} text chunks.")
    response = client.embeddings.create(input=text_chunks, model=embedding_model)
    return [data.embedding for data in response.data]

def get_embedding(text):
    print(f"Getting embedding for the query: '{text}'.")
    response = client.embeddings.create(input=text, model=embedding_model)
    return response.data[0].embedding

def find_similar_chunks(index, query_embedding, top_n=10):
    print("Finding similar chunks.")
    query_vector = np.array(query_embedding).astype('float32').reshape(1, -1)  # Reshape for FAISS
    distances, top_indices = index.search(query_vector, top_n)  # Use FAISS to find similar entries
    print(f"Top indices found: {top_indices.flatten().tolist()}.")  # Flatten indices for readability
    return top_indices[0], distances[0]

def get_gpt_response(prompt):
    print("Generating response from GPT.")
    completion = client.chat.completions.create(
        model=completion_model,
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

def hash_file(file):
    hasher = hashlib.md5()
    hasher.update(file.read())
    file.seek(0)  # Reset file pointer
    return hasher.hexdigest()

def main():
    global indexed_documents, indexed_chunks
    st.title("PDF Query App")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    query = st.text_input("Enter your query")

    if st.button("Submit"):
        if pdf_file and query:
            st.success("PDF and query submitted!")
            file_hash = hash_file(pdf_file)
            index_file_path = os.path.join(index_directory, f"{file_hash}.index")
            chunk_path = os.path.join(index_directory, f"{file_hash}.chunks.txt")

            # Check if the document has already been indexed
            if os.path.exists(index_file_path):
                index = load_index(index_file_path)
                if os.path.exists(chunk_path):
                    with open(chunk_path, "r", encoding="utf-8") as f:
                        chunks = f.read().split("\n===CHUNK_SEPARATOR===\n")
                    indexed_chunks[file_hash] = chunks
                else:
                    chunks = []
                indexed_documents[file_hash] = index
                print(f"Loaded existing index for document: {file_hash}")
                st.write(f"Loaded existing index for document: {file_hash}")

                # Debugging output for the chunks loaded
                print(f"Number of previously stored chunks: {len(chunks)}")
            else:
                # Create FAISS index for this document
                index = create_index(dim=1536)  # Set dimension to 1536 based on OpenAI embedding model

                # Extract and index the document
                chunks = []  # Initialize chunks list for this document
                for page_text, page_index in extract_text_from_pdf(pdf_file):
                    page_chunks = chunk_text(page_text, token_limit=512)
                    chunks.extend(page_chunks)  # Extend the list with new chunks
                    chunk_embeddings = get_embeddings(page_chunks)
                    print(f"Generated {len(chunk_embeddings)} embeddings for {len(page_chunks)} chunks.")

                    # Store each embedding in the FAISS index
                    for embedding in chunk_embeddings:
                        index.add(np.array(embedding).astype('float32').reshape(1, -1))  # Ensure correct shape

                # Save the FAISS index and the chunks to disk
                save_index(index, index_file_path)
                with open(chunk_path, "w", encoding="utf-8") as f:
                    for chunk in chunks:
                        f.write(chunk.replace("\n", " ") + "\n===CHUNK_SEPARATOR===\n")

                # Store the FAISS index and chunks in the dictionaries
                indexed_documents[file_hash] = index
                indexed_chunks[file_hash] = chunks  # Store the chunks for this document
                print(f"Indexed the document: {file_hash} and saved to disk.")
                st.write(f"Indexed the document: {file_hash} and saved to disk.")

            query_embedding = get_embedding(query)
            top_indices, distances = find_similar_chunks(index, query_embedding)
            print(f"FAISS index contains {index.ntotal} vectors.")  # Defensive logging
            
            # Ensure flat list for indices
            flat_top_indices = top_indices.flatten().tolist()

            # Log available chunks and validate indices
            print(f"Total chunks available: {len(chunks)}")
            top_chunks = []
            for idx in flat_top_indices:
                if idx < len(chunks):  # Ensure valid index
                    top_chunks.append(chunks[idx])
                else:
                    print(f"Warning: Index {idx} is out of range for the number of chunks.")  # Debugging warning

            # Generate the prompt for GPT
            prompt = f"Based on the following information, answer the question: {query}\n\n"
            prompt += "\n\n".join(top_chunks)
            print(f"Prompt sent to GPT: {prompt}")
            gpt_response = get_gpt_response(prompt)
            st.header("Response from GPT:")
            st.write(gpt_response)

            # Visualize the selected chunks used for the query
            st.subheader("Chunks Used to Answer the Query:")
            for idx in flat_top_indices:
                if idx < len(chunks):  # Ensure valid index
                    st.markdown(f"- **Chunk {idx}:** {chunks[idx]}")
                else:
                    print(f"Warning: Index {idx} is out of range when displaying chunks.")  # Debugging warning

        else:
            st.error("Please upload a PDF file and enter a query.")

if __name__ == "__main__":
    main()