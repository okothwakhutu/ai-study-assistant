import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
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

def preprocess_text(text, max_chars=3000):
    """Truncate text to avoid token limits (most models handle ~1024 tokens)."""
    return text[:max_chars] if len(text) > max_chars else text

# --------------------------------------------
# 2. Summarization (using DistilBART)
# --------------------------------------------
@st.cache_resource  # Load model only once
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text, max_length=150, min_length=50):
    summarizer = load_summarizer()
    # The model expects <= 1024 tokens; we preprocess to 3000 chars (approx 750 tokens)
    truncated = preprocess_text(text, 3000)
    summary = summarizer(truncated, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# --------------------------------------------
# 3. Quiz Generation (using T5 for question generation)
# --------------------------------------------
@st.cache_resource
def load_qg_model():
    model_name = "mrm8488/t5-base-finetuned-question-generation"
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
    # Take the first few sentences that are long enough
    for sent in sentences[:num_questions*2]:
        # Format for T5: "generate questions: <sentence>"
        input_text = f"generate question: {sent}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs.input_ids, 
            max_length=64, 
            num_beams=4, 
            early_stopping=True
        )
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # The answer is the original sentence (simplistic but usable)
        questions.append({"question": question, "answer": sent})
        if len(questions) >= num_questions:
            break
    return questions

# --------------------------------------------
# 4. Study Plan (rules-based, using key sentences)
# --------------------------------------------
def extract_topics(text, num_topics=5):
    """Extract important sentences as pseudo-topics (simple TF‑IDF can be added)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Remove very short sentences
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]
    # Return first 'num_topics' sentences as topics
    return sentences[:num_topics]

def create_study_plan(topics, days=5):
    """Assign topics to days, with some repetition."""
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
        with st.spinner("Summarizing... (may take 10-15 seconds)"):
            summary = summarize_text(document_text)
        st.write(summary)
        # Store summary for later steps
        st.session_state["summary"] = summary
    else:
        summary = st.session_state.get("summary", "")
        if summary:
            st.write(summary)
    
    # ---------- QUIZ ----------
    st.subheader("❓ Quiz Generator")
    if st.button("Create Quiz (3 questions)"):
        # Use the original text (or summary) as context
        context = document_text if len(document_text) < 2000 else document_text[:2000]
        with st.spinner("Generating questions using AI..."):
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
st.caption("Powered by Hugging Face Transformers (DistilBART & T5). Models run locally – your data stays private.")