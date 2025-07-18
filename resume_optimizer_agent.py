import streamlit as st
import nltk
import spacy
import fitz  # PyMuPDF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# 📥 Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# 🧠 Load spaCy model (assumes it's pre-installed via requirements.txt)
nlp = spacy.load("en_core_web_sm")

# 📄 Extract text from uploaded PDF
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# 🔍 Extract top keywords and named entities
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_words = [
        word for word in tokens
        if word.isalpha() and word not in stop_words and len(word) > 2
    ]
    freq = Counter(filtered_words).most_common(15)

    doc = nlp(text)
    named_entities = list(set([
        ent.text for ent in doc.ents
        if ent.label_ in [
            "PERSON", "ORG", "GPE", "NORP", "PRODUCT",
            "WORK_OF_ART", "LANGUAGE", "EVENT", "FAC"
        ]
    ]))
    return freq, named_entities

# 🖥 Streamlit UI setup
st.set_page_config(page_title="Resume Keyword Optimizer", layout="centered")
st.title("🤖 AI-Powered Resume Keyword Optimizer")
st.write("Upload your resume PDF to extract top keywords and named entities (like skills, organizations, etc.).")

# 📤 File upload
uploaded_file = st.file_uploader("📄 Upload your Resume (PDF)", type=["pdf"])

# 🧪 Resume analysis
if uploaded_file is not None:
    with st.spinner("🔍 Analyzing your resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        top_keywords, named_entities = extract_keywords(resume_text)

    st.success("✅ Analysis Complete!")

    st.subheader("🔑 Top Keywords:")
    for word, count in top_keywords:
        st.write(f"- *{word}*: {count}")

    st.subheader("🏷 Named Entities (Skills, Orgs, People, etc.):")
    for entity in named_entities:
        st.write(f"- {entity}")
