# main.py

import os
import yaml
import pandas as pd
from dotenv import load_dotenv
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal

# --- 1. SETUP and CONFIGURATION ---
load_dotenv()

class RoboticsConcept(BaseModel):
    concept_name: str = Field(description="The name of the technology, problem, or domain.")
    temporal_context: Literal["Past", "Current", "Future"] = Field(
        description="Classify the concept as related to past work, current needs, or future goals."
    )

# --- 2. DATA LOADING FUNCTION ---
def load_documents_from_yaml(config_path="config/sources.yaml"):
    """Loads all documents specified in the YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    all_docs = []
    
    # Load from URLs
    if 'urls' in config and config['urls']:
        print(f"Loading {len(config['urls'])} URL(s)...")
        for url in config['urls']:
            loader = WebBaseLoader(url)
            all_docs.extend(loader.load())
            
    # Load from PDF directories
    if 'pdfs' in config and config['pdfs']:
        print(f"Loading PDFs from {len(config['pdfs'])} directorie(s)...")
        for path in config['pdfs']:
            if os.path.isdir(path):
                loader = PyPDFDirectoryLoader(path)
                all_docs.extend(loader.load())
            else:
                print(f"Warning: Directory not found at {path}")
                
    return all_docs

# --- 3. ANALYSIS AND EXECUTION ---
if __name__ == "__main__":
    print("Starting analysis...")
    
    # 1. Load the entire corpus of documents
    corpus_docs = load_documents_from_yaml()
    if not corpus_docs:
        print("No documents found. Exiting.")
        exit()
    
    print(f"\nLoaded a total of {len(corpus_docs)} document pages/sections.")

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs_chunks = text_splitter.split_documents(corpus_docs)
    
    print(f"Split into {len(docs_chunks)} chunks for analysis.")

    # 3. Run LLM extraction chain
    print("Extracting concepts with Gemini... This may take a while for a large corpus.")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    extraction_chain = create_extraction_chain(RoboticsConcept.schema(), llm)
    extracted_data = extraction_chain.run(docs_chunks)
    
    # 4. Process and display results
    results_df = pd.DataFrame(extracted_data)
    print("\n--- Analysis Complete ---")
    
    if not results_df.empty:
        future_concepts = results_df[results_df['temporal_context'] == 'Future']
        print("\n## Top Future Concepts Identified:")
        print(future_concepts['concept_name'].value_counts().nlargest(15))
        
        # Generate and save word cloud
        future_text = ' '.join(future_concepts['concept_name'].dropna())
        if future_text:
            wordcloud = WordCloud(width=1200, height=600, background_color='white').generate(future_text)
            wordcloud.to_file("future_concepts_wordcloud.png")
            print("\nWord cloud saved to 'future_concepts_wordcloud.png'")
    else:
        print("No concepts were extracted from the documents.")