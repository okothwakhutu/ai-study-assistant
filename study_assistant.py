import streamlit as st
import fitz  # PyMuPDF
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import re

# --------------------------------------------
# 1. PDF / Text Extraction
# --------------------------------------------
def extract_text_from_pdf(pdf_file):
    """Extract all text from an uploaded PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def preprocess_text(text, max_chars=2500):
    """Truncate text to avoid token limits."""
    return text[:max_chars] if len(text) > max_chars else text

# --------------------------------------------
# 2. Summarization (T5-small - FAST)
# --------------------------------------------
@st.cache_resource
def load_summarizer():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def summarize_text(text, max_length=120, min_length=40):
    tokenizer, model = load_summarizer()
    truncated = preprocess_text(text, 2500)
    input_text = "summarize: " + truncated
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=2,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# --------------------------------------------
# 3. Quiz Generation (valhalla/t5-small-qg-hl - WORKING)
# --------------------------------------------
@st.cache_resource
def load_qg_model():
    model_name = "valhalla/t5-small-qg-hl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def generate_questions(context, num_questions=3):
    """Generate question-answer pairs from a context paragraph."""
    tokenizer, model = load_qg_model()
    # Split context into sentences
    sentences = re.split(r'(?<=[.!?])\s+', context)
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]
    
    questions = []
    for sent in sentences[:num_questions*2]:
        # The model expects "question: <sentence>"
        input_text = f"question: {sent}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs, 
            max_length=64, 
            num_beams=4, 
            early_stopping=True
        )
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up question if it contains extra text
        if question.startswith("question:"):
            question = question.replace("question:", "").strip()
        questions.append({"question": question, "answer": sent})
        if len(questions) >= num_questions:
            break
    return questions

# --------------------------------------------
# 4. Study Plan (rules-based)
# --------------------------------------------
def extract_topics(text, num_topics=5):
    """Extract important sentences as pseudo-topics."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]
    return sentences[:num_topics]

def create_study_plan(topics, days=5):
    """Assign topics to days."""
    if not topics:
        return ["No topics could be extracted. Please provide more detailed notes."]
    plan = []
    for i in range(days):
        topic_idx = i % len(topics)
        plan.append(f"Day {i+1}: Study – {topics[topic_idx]}")
    return plan

# --------------------------------------------
# 5. Streamlit UI
# --------------------------------------------
st.set_page_config(page_title="AI Study Assistant", page_icon="📚")
st.title("📝 AI Study Assistant for Students")
st.markdown("Upload your syllabus or notes (PDF / text) to get a **summary**, a **quiz**, and a **personalized study plan**.")

input_type = st.radio("Choose input type:", ("Upload PDF", "Paste text"))

document_text = ""

if input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        with st.spinner("Reading PDF..."):
            document_text = extract_text_from_pdf(uploaded_file)
        st.success("PDF loaded successfully!")
else:
    document_text = st.text_area("Paste your notes or syllabus here:", height=250)

if document_text:
    st.markdown("---")
    
    # ---------- SUMMARY ----------
    st.subheader("📌 Summary")
    if st.button("Generate Summary"):
        short_text = document_text[:2500] if len(document_text) > 2500 else document_text
        with st.spinner("Summarizing... (10-20 seconds on CPU)"):
            summary = summarize_text(short_text)
        st.write(summary)
        st.session_state["summary"] = summary
    else:
        if "summary" in st.session_state:
            st.write(st.session_state["summary"])
    
    # ---------- QUIZ ----------
    st.subheader("❓ Quiz Generator")
    if st.button("Create Quiz (3 questions)"):
        context = document_text[:2000] if len(document_text) > 2000 else document_text
        with st.spinner("Generating questions using AI... (first run downloads model ~300 MB)"):
            quiz = generate_questions(context, num_questions=3)
        st.session_state["quiz"] = quiz
        for i, qa in enumerate(quiz, 1):
            st.write(f"**Q{i}:** {qa['question']}")
            with st.expander("Show answer"):
                st.write(qa['answer'])
    else:
        if "quiz" in st.session_state:
            for i, qa in enumerate(st.session_state["quiz"], 1):
                st.write(f"**Q{i}:** {qa['question']}")
                with st.expander("Show answer"):
                    st.write(qa['answer'])

    # ---------- STUDY PLAN ----------
    st.subheader("📅 Personalized Study Plan")
    days = st.slider("How many days do you have to prepare?", 1, 14, 5)
    if st.button("Create Study Plan"):
        topics = extract_topics(document_text, num_topics=days)
        plan = create_study_plan(topics, days=days)
        st.session_state["plan"] = plan
        for step in plan:
            st.write(f"- {step}")
    else:
        if "plan" in st.session_state:
            for step in st.session_state["plan"]:
                st.write(f"- {step}")

st.markdown("---")
st.caption("Powered by Hugging Face Transformers (T5-small & valhalla/t5-small-qg-hl). Models run locally – your data stays private.")