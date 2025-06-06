import os
import tempfile
import streamlit as st
from openai import OpenAI
import json
from docx.document import Document
from PyPDF2 import PdfReader
import pandas as pd
from datetime import datetime
import base64
from io import BytesIO
import re
from collections import Counter
from textblob import TextBlob
import numpy as np

# Set page configuration first
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def get_document_summary(client, text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise summaries of documents."},
                {"role": "user", "content": f"Please provide a brief summary of this document:\n\n{text[:4000]}..."}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def export_chat_history():
    if not st.session_state.chat_history:
        return None
    
    df = pd.DataFrame(st.session_state.chat_history)
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()

def calculate_reading_time(text, wpm=250):
    words = len(text.split())
    minutes = words / wpm
    return round(minutes, 1)

def calculate_complexity_score(text):
    # Basic readability score using average sentence length and word length
    blob = TextBlob(text)
    sentences = blob.sentences
    if not sentences:
        return 0
    
    avg_sentence_length = len(text.split()) / len(sentences)
    avg_word_length = sum(len(word) for word in text.split()) / len(text.split())
    
    # Normalize scores (1-10 scale)
    complexity = (avg_sentence_length * 0.5 + avg_word_length * 2) / 3
    return round(min(max(complexity, 1), 10), 1)

def extract_keywords(text, top_n=10):
    # Remove common punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
    words = [word for word in text.split() if word not in stop_words]
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Get top N keywords
    return word_freq.most_common(top_n)

def analyze_document(text):
    reading_time = calculate_reading_time(text)
    complexity = calculate_complexity_score(text)
    keywords = extract_keywords(text)
    
    return {
        "reading_time": reading_time,
        "complexity": complexity,
        "keywords": keywords
    }

# Custom styling and theme
def apply_custom_styling():
    st.markdown("""
        <style>
        /* Main container */
        .main {
            padding: 2rem;
        }
        
        /* Headers */
        h1 {
            color: #FF4B4B;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 1rem !important;
        }
        
        h2 {
            color: #262730;
            font-size: 1.8rem !important;
            font-weight: 600 !important;
            margin-top: 2rem !important;
        }
        
        h3 {
            color: #262730;
            font-size: 1.4rem !important;
            font-weight: 600 !important;
        }
        
        /* Buttons */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Chat container */
        .chat-container {
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .chat-container:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        /* Document card */
        .document-card {
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 1.2rem;
            margin: 0.8rem 0;
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .document-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Metrics */
        .metric-container {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #FF4B4B;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #6c757d;
            margin-top: 0.3rem;
        }
        
        /* Sidebar */
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        
        /* File uploader */
        .stFileUploader {
            border-radius: 8px;
            padding: 1rem;
            background-color: #f8f9fa;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: #FF4B4B;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 0.8rem;
        }
        
        /* Alert messages */
        .stAlert {
            border-radius: 8px;
            padding: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    # Apply custom styling
    apply_custom_styling()
    
    # Add decorative header
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h1>📚 Document Q&A</h1>
            <p style='font-size: 1.2rem; color: #666;'>Upload documents, get insights, and ask questions!</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'document_summaries' not in st.session_state:
        st.session_state.document_summaries = {}

    # Sidebar with gradient background
    with st.sidebar:
        st.markdown("""
            <div style='background: linear-gradient(45deg, #FF4B4B, #FF8C8C); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
                <h2 style='color: white; margin: 0;'>🔑 API Configuration</h2>
            </div>
        """, unsafe_allow_html=True)
        
        api_key = st.text_input(
            "Enter your OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key here"
        )
        
        st.markdown("""
            <div style='background: linear-gradient(45deg, #FF4B4B, #FF8C8C); padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                <h2 style='color: white; margin: 0;'>📁 Document Upload</h2>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx']
        )

        if st.session_state.documents:
            st.header("📚 Document Management")
            if st.button("Clear All Documents"):
                st.session_state.documents = []
                st.session_state.document_summaries = {}
                st.success("All documents cleared!")

            if st.session_state.chat_history:
                st.header("💾 Export Chat")
                csv_data = export_chat_history()
                if csv_data:
                    st.download_button(
                        label="Download Chat History",
                        data=csv_data,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

    # Main content
    st.title("📚 Advanced Document Q&A")
    st.markdown("### Upload documents, get summaries, and ask questions!")
    
    # Check for API key
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to get started!")
        return

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Process uploaded files
    if uploaded_files:
        with st.spinner("Processing documents..."):
            for file in uploaded_files:
                try:
                    if file.name.endswith('.txt'):
                        content = file.getvalue().decode('utf-8')
                    elif file.name.endswith('.pdf'):
                        content = read_pdf(file)
                    elif file.name.endswith('.docx'):
                        content = read_docx(file)
                    
                    if content not in st.session_state.documents:
                        # Analyze document
                        analysis = analyze_document(content)
                        
                        st.session_state.documents.append(content)
                        summary = get_document_summary(client, content)
                        st.session_state.document_summaries[len(st.session_state.documents) - 1] = {
                            "summary": summary,
                            "analysis": analysis
                        }
                
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
        
        st.success(f"✅ Successfully processed {len(uploaded_files)} files")

    # Display documents and summaries
    if st.session_state.documents:
        st.header("📄 Uploaded Documents")
        for idx, (doc, info) in enumerate(zip(st.session_state.documents, 
                                            [st.session_state.document_summaries.get(i) for i in range(len(st.session_state.documents))])):
            with st.expander(f"Document {idx + 1} Analysis"):
                if info:
                    analysis = info.get("analysis", {})
                    st.markdown(f"**Summary:**\n{info.get('summary', 'No summary available')}")
                    
                    # Display document statistics
                    st.markdown("### 📊 Document Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Reading Time", f"{analysis.get('reading_time', 0)} minutes")
                    with col2:
                        st.metric("Complexity Score", f"{analysis.get('complexity', 0)}/10")
                    
                    # Display keywords
                    st.markdown("### 🔑 Key Terms")
                    keywords = analysis.get('keywords', [])
                    if keywords:
                        # Create a horizontal bar chart
                        keywords_df = pd.DataFrame(keywords, columns=['Word', 'Frequency'])
                        st.bar_chart(data=keywords_df.set_index('Word'))
                
                if st.button(f"Remove Document {idx + 1}"):
                    st.session_state.documents.pop(idx)
                    st.session_state.document_summaries.pop(idx)
                    st.rerun()

        # Chat interface
        st.header("💬 Ask Questions")
        question = st.text_input("Enter your question:")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            ask_button = st.button("Ask Question", type="primary")
        with col2:
            clear_chat = st.button("Clear Chat History")
            if clear_chat:
                st.session_state.chat_history = []
                st.rerun()

        if ask_button and question:
            with st.spinner("Generating answer..."):
                try:
                    # Prepare context from documents
                    context = "\n\n---\n\n".join(st.session_state.documents)
                    
                    # Create prompt with conversation history
                    conversation_history = "\n\n".join([
                        f"Q: {chat['question']}\nA: {chat['answer']}"
                        for chat in st.session_state.chat_history[-3:]  # Include last 3 conversations for context
                    ])
                    
                    prompt = f"""Previous conversation:
                    {conversation_history}

                    Documents:
                    {context}

                    Based on the above documents and conversation history, please answer this question:
                    {question}

                    If the answer cannot be found in the documents, say "I cannot find the answer in the provided documents."
                    Please cite the relevant document numbers in your answer using [Doc 1], [Doc 2], etc.
                    """

                    # Get response from OpenAI
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documents. Always cite your sources using [Doc X] notation."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7
                    )
                    
                    answer = response.choices[0].message.content
                    
                    # Display answer
                    st.markdown("### Answer:")
                    st.markdown(f"<div class='chat-container'>{answer}</div>", 
                              unsafe_allow_html=True)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

        # Display chat history
        if st.session_state.chat_history:
            st.header("📝 Chat History")
            for chat in reversed(st.session_state.chat_history):
                st.markdown("---")
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")
                st.caption(f"Time: {chat['timestamp']}")
    else:
        st.info("👈 Please upload some documents in the sidebar to get started!")




        

if __name__ == "__main__":
    main()