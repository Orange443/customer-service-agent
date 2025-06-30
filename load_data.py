# load_data.py - CORRECTED VERSION
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

def main():
    """
    Connects to the PostgreSQL database, loads data from the CSV,
    and stores the documents and their embeddings in the pgvector store.
    """
    print("--- Starting Data Loading Process ---")

    # --- 1. Load Configuration ---
    print("Loading environment variables...")
    load_dotenv()
    
    CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")
    if not CONNECTION_STRING:
        print("ERROR: PGVECTOR_CONNECTION_STRING not found in .env file.")
        return
        
    COLLECTION_NAME = "support_tickets"

    # --- 2. Load and Prepare Data ---
    print("Loading and preparing data from CSV...")
    df = pd.read_csv('data/customer_support_tickets.csv')
    
    resolved_tickets_df = df[df['Ticket Status'] == 'Closed'].copy()
    
    resolved_tickets_df['knowledge_text'] = "Problem: " + resolved_tickets_df['Ticket Description'] + \
                                            ". Resolution Summary: " + resolved_tickets_df['Ticket Subject']
    documents = resolved_tickets_df['knowledge_text'].tolist()

    print(f"Prepared {len(documents)} documents to be stored.")

    # --- 3. Initialize Embeddings and PGVector Store ---
    print("Initializing embedding model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    print("Connecting to PGVector and storing documents... This may take a moment.")
    
    # --- FIX: Add the connection parameter ---
    PGVector.from_texts(
        texts=documents,  # Fixed parameter order
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING  # This is REQUIRED!
    )

    print("\n--- Success! ---")
    print("Knowledge base has been successfully populated in your PostgreSQL database.")

if __name__ == "__main__":
    main()
