# streamlit_app.py

import os
import yaml
import pandas as pd
from dotenv import load_dotenv
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

# (The LangChain imports and RoboticsConcept class are identical to the standalone script)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal

load_dotenv()

class RoboticsConcept(BaseModel):
    concept_name: str = Field(description="The name of the robotics technology, problem, or domain.")
    temporal_context: Literal["Past", "Current", "Future"] = Field(
        description="Classify the concept as related to past work, current needs, or future goals."
    )

# --- BACKEND LOGIC ---
# Using st.cache_data to prevent re-running on widget interactions
@st.cache_data
def run_full_analysis(uploaded_file):
    """Loads docs from the uploaded YAML, runs analysis, and returns the DataFrame."""
    config = yaml.safe_load(uploaded_file)
    all_docs = []
    
    # Load from URLs
    with st.spinner("Loading documents from URLs..."):
        if 'urls' in config and config['urls']:
            for url in config['urls']:
                loader = WebBaseLoader(url)
                all_docs.extend(loader.load())
                
    # Load from PDF directories
    with st.spinner("Loading documents from PDF directories..."):
        if 'pdfs' in config and config['pdfs']:
            for path in config['pdfs']:
                if os.path.isdir(path):
                    loader = PyPDFDirectoryLoader(path)
                    all_docs.extend(loader.load())

    if not all_docs:
        st.error("No documents were loaded. Check your YAML file paths and URLs.")
        return pd.DataFrame()
        
    # Split, Analyze with LLM
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs_chunks = text_splitter.split_documents(all_docs)
    
    with st.spinner(f"Extracting concepts from {len(docs_chunks)} text chunks... This can take a while."):
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        extraction_chain = create_extraction_chain(RoboticsConcept.schema(), llm)
        extracted_data = extraction_chain.run(docs_chunks)
        
    return pd.DataFrame(extracted_data)

# --- FRONTEND (The Streamlit User Interface) ---
st.set_page_config(layout="wide")
st.title("🤖 Corpus Trend Analyzer")
st.markdown("Upload a `sources.yaml` file to analyze trends across multiple documents and websites.")

uploaded_yaml = st.file_uploader("Upload your sources.yaml file", type=['yaml', 'yml'])

if uploaded_yaml is not None:
    if st.button("Analyze Corpus"):
        results_df = run_full_analysis(uploaded_yaml)
        
        st.success("Analysis Complete!")
        st.session_state['results_df'] = results_df # Save results to session state

# Display results if they exist in the session state
if 'results_df' in st.session_state:
    results_df = st.session_state['results_df']
    if not results_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Future Concepts")
            future_df = results_df[results_df['temporal_context'] == 'Future']
            st.dataframe(future_df['concept_name'].value_counts().reset_index())

        with col2:
            st.subheader("Current Needs & Problems")
            current_df = results_df[results_df['temporal_context'] == 'Current']
            st.dataframe(current_df['concept_name'].value_counts().reset_index())
            
        st.subheader("Word Cloud for Future Concepts")
        future_text = ' '.join(future_df['concept_name'].dropna())
        if future_text:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(future_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)