import os
import streamlit as st
import PyPDF2
import numpy as np
from openai import OpenAI
import yaml
from scipy.spatial.distance import cosine
import tiktoken

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

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, token_limit=512):
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), token_limit):
        chunk_tokens = tokens[i:i + token_limit]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def get_embeddings(text_chunks):
    response = client.embeddings.create(input=text_chunks, model=embedding_model)
    return [data.embedding for data in response.data]

def get_embedding(text):
    response = client.embeddings.create(input=text, model=embedding_model)
    return response.data[0].embedding

def find_similar_chunks(embeddings, query_embedding, top_n=10):
    similarities = [1 - cosine(query_embedding, embed) for embed in embeddings]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return top_indices, [similarities[i] for i in top_indices]

def get_gpt_response(prompt):
    completion = client.chat.completions.create(
        model=completion_model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def main():
    st.title("PDF Query App")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    query = st.text_input("Enter your query")
    if st.button("Submit"):
        if pdf_file and query:
            st.success("PDF and query submitted!")
            pdf_text = extract_text_from_pdf(pdf_file)
            chunks = chunk_text(pdf_text, token_limit=512)
            chunk_embeddings = get_embeddings(chunks)
            query_embedding = get_embedding(query)
            top_indices, similarities = find_similar_chunks(chunk_embeddings, query_embedding)
            top_chunks = [chunks[idx] for idx in top_indices]
            prompt = f"Based on the following information, answer the question: {query}\n\n"
            prompt += "\n\n".join(top_chunks)
            gpt_response = get_gpt_response(prompt)
            st.header("Response from GPT:")
            st.write(gpt_response)
        else:
            st.error("Please upload a PDF file and enter a query.")

if __name__ == "__main__":
    main()

