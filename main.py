# main.py

import os
import yaml
import pandas as pd
import argparse
from dotenv import load_dotenv
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal, List

# --- 1. SETUP and CONFIGURATION ---
load_dotenv()

class TechnicalConcept(BaseModel):
    concept_name: str = Field(description="The name of the technical technology, problem, or domain.")
    temporal_context: Literal["Past", "Current", "Future"] = Field(
        description="Classify the concept as related to past work, current needs, or future goals."
    )
    concept_type: Literal["Technology", "Problem", "Domain", "Other"] = Field(
        description="Classify the concept as a domain, problem, technology, or other."
    )

class TechnicalConcepts(BaseModel):
    concepts: List[TechnicalConcept] = Field(description="A list of all the technical concepts extracted from the text.")


# --- 2. DYNAMIC LLM LOADER ---
def get_llm(llm_config: dict):
    """Initializes the correct LLM based on the provider specified in the config."""
    provider = llm_config.get("provider", "").lower()
    model = llm_config.get("model")

    if provider == "google":
        print(f"Using Google model: {model}")
        return ChatGoogleGenerativeAI(model=model, temperature=0)
    elif provider == "openai":
        print(f"Using OpenAI model: {model}")
        return ChatOpenAI(model=model, temperature=0)
    else:
        raise ValueError(f"Unsupported LLM provider '{provider}'. Please use 'Google' or 'OpenAI'.")

# --- 3. DATA LOADING FUNCTION ---
def load_documents_from_yaml(config: dict):
    """Loads all documents specified in the YAML config."""
    all_docs = []
    # (This function is the same as before, but now takes the loaded config dict)
    if 'urls' in config and config.get('urls'):
        for url in config['urls']:
            all_docs.extend(WebBaseLoader(url).load())
    if 'pdfs' in config and config.get('pdfs'):
        for path in config['pdfs']:
            if os.path.isdir(path):
                all_docs.extend(PyPDFDirectoryLoader(path).load())
    return all_docs

# --- 4. ANALYSIS AND EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze robotics trends from a corpus of documents.")
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Initialize the correct LLM
    if 'llm' not in config:
        raise ValueError("YAML config must contain an 'llm' section.")
    llm = get_llm(config['llm'])

    # 2. Load documents
    corpus_docs = load_documents_from_yaml(config)
    if not corpus_docs:
        print("No documents found. Exiting.")
        exit()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs_chunks = text_splitter.split_documents(corpus_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at extracting information. "
                "You are tasked with parsing the provided text, and categorizing the topics. "
                "Extract all relevant technical concepts from the following text. "
                "For each concept, classify it as 'Past', 'Current', or 'Future' based on the context. "
                "Then, classify the key concept as a 'Domain', 'Problem', 'Technology', or 'Other'.",
            ),
            ("human", "{text}"),
        ]
    )

    # Bind the Pydantic schema to the model and create the new chain
    structured_llm = llm.with_structured_output(TechnicalConcepts)
    extraction_chain = {"text": lambda docs: "\n\n".join([d.page_content for d in docs])} | prompt | structured_llm
    
    print(f"Extracting concepts from {len(docs_chunks)} chunks...")
    extracted_data_object = extraction_chain.invoke(docs_chunks)
    
    # MODIFIED: Create the DataFrame from the list inside the returned object
    if extracted_data_object and extracted_data_object.concepts:
        # Convert the list of Pydantic objects into a list of dictionaries
        list_of_concepts = [concept.dict() for concept in extracted_data_object.concepts]
        results_df = pd.DataFrame(list_of_concepts)
    else:
        results_df = pd.DataFrame() # Create an empty DataFrame if nothing was found

    print("\n--- Analysis Complete ---")
    print(results_df)
    
    if not results_df.empty:
        print("\n## Concept Type Breakdown:")
        print(results_df['concept_type'].value_counts()) # This will now work correctly

        present_concepts = results_df[results_df['temporal_context'] == 'Present']
        future_concepts = results_df[results_df['temporal_context'] == 'Future']
        print("\n## Top Present Concepts Identified:")
        print(present_concepts['concept_name'].value_counts().nlargest(15))
        
        print("\n## Top Future Concepts Identified:")
        print(future_concepts['concept_name'].value_counts().nlargest(15))
        
        # Generate and save word clouds
        future_text = ' '.join(future_concepts['concept_name'].dropna())
        if future_text:
            wordcloud = WordCloud(width=1200, height=600, background_color='white').generate(future_text)
            wordcloud.to_file("future_concepts_wordcloud.png")
            print("\nWord cloud saved to 'future_concepts_wordcloud.png'")

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off') # Hide the axes
            plt.title("Future Concepts")
            plt.show()

    else:
        print("No concepts were extracted from the documents.")