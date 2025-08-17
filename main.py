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
from pydantic import BaseModel, Field
from typing import Literal, List, get_args

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
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Path to the output directory.'
    )
    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = args.output_dir

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


    # Iterate through concepts types and temporal contexts
    concept_types = get_args(TechnicalConcept.model_fields['concept_type'].annotation)
    temporal_contexts = get_args(TechnicalConcept.model_fields['temporal_context'].annotation)

    for concept_type in concept_types:

        for temporal_context in temporal_contexts:

            temp_concepts_df = results_df[(results_df['temporal_context'] == temporal_context) & (results_df['concept_type'] == concept_type)]
            
            if not temp_concepts_df.empty:
                print("\n## Concept Type Breakdown:")
                print(temp_concepts_df.value_counts())

                # Create output directory if it does not exist
                os.makedirs(os.path.join(output_dir,"img"), exist_ok=True)
                os.makedirs(os.path.join(output_dir,"csv"), exist_ok=True)

                # Save dataframe
                temp_concepts_df.value_counts().to_csv(os.path.join(output_dir,"csv",f"{temporal_context}_{concept_type}.csv"))

                # Create and save wordcloud image
                concept_text = ' '.join(temp_concepts_df['concept_name'].dropna())
                if concept_text:
                    wordcloud = WordCloud(width=1200, height=600, background_color='white').generate(concept_text)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off') # Hide the axes
                    plt.title(f"{temporal_context} {concept_type}(s)")
                    plt.savefig(os.path.join(output_dir,"img",f"{temporal_context}_{concept_type}_wordcloud.png"))
            else:
                print(f"No {temporal_context} {concept_type}(s) were extracted from the documents.")