import re
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

# Page configurations
st.set_page_config(page_title="AI Resume Analyzer & Matcher", page_icon="🎯", layout="wide")

@st.cache_resource
def load_nlp_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_nlp_model()

class StreamlitResumeAnalyzer:
    def __init__(self):
        pass

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extracts and cleans text contents from an uploaded PDF file."""
        try:
            reader = PdfReader(pdf_file)
            extracted_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
            return extracted_text
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return ""

    def _extract_yoe_requirement(self, jd_text: str) -> int:
        match = re.search(r'(\d+)\s*(?:\+|-|\s*to\s*\d+)?\s*years?', jd_text, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    def _extract_jd_keywords(self, jd_text: str, top_n: int = 10) -> list:
        cleaned_jd = re.sub(r'[^\w\s\-\+\#]', ' ', jd_text.lower())
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        try:
            tfidf_matrix = vectorizer.fit_transform([cleaned_jd])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            sorted_indices = scores.argsort()[::-1]
            return [feature_names[i] for i in sorted_indices[:top_n] if scores[i] > 0.1]
        except:
            return []

    def run_analysis(self, jd_text: str, mandatory_skills: list, resumes: list) -> dict:
        if not resumes:
            return {"strong_matches": [], "close_matches": []}

        resume_texts = [r["text"] for r in resumes]
        jd_embedding = model.encode(jd_text, convert_to_tensor=True)
        resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(jd_embedding, resume_embeddings)[0].tolist()
        required_yoe = self._extract_yoe_requirement(jd_text)
        jd_domain_keywords = self._extract_jd_keywords(jd_text)

        strong_matches = []
        close_matches = []

        for idx, resume in enumerate(resumes):
            base_score = max(0.0, cosine_scores[idx] * 100)
            resume_text_lower = resume["text"].lower()
            
            missing_mandatory = []
            missing_keywords = []
            
            # Identify core gaps
            for skill in mandatory_skills:
                if not re.search(r'\b' + re.escape(skill.lower()) + r'\b', resume_text_lower):
                    missing_mandatory.append(skill)
            
            for word in jd_domain_keywords:
                if word in [s.lower() for s in mandatory_skills]:
                    continue
                if not re.search(r'\b' + re.escape(word) + r'\b', resume_text_lower):
                    if len(word) > 2 and not word.isdigit():
                        missing_keywords.append(word)

            yoe_gap = max(0, required_yoe - resume["yoe"])
            yoe_penalty = yoe_gap * 10
            skill_penalty = len(missing_mandatory) * 8
            
            # Raw mathematical assessment score
            raw_score = base_score - yoe_penalty - skill_penalty
            raw_score = sorted([0.0, raw_score, 100.0])[1]

            # 🌟 NORMALIZATION LOGIC: Map raw scores to ensure outputs land above 80%
            if raw_score >= 75.0:
                # Top tier match: Map raw [75-100] scaling linearly to displayed [96-100]
                display_score = 96.0 + ((raw_score - 75.0) / 25.0) * 4.0
            elif raw_score >= 50.0:
                # Strong tier match: Map raw [50-74.9] scaling linearly to displayed [88-95.9]
                display_score = 88.0 + ((raw_score - 50.0) / 25.0) * 7.9
            else:
                # Base calibration tier: Map raw [0-49.9] scaling linearly to displayed [80-87.9]
                display_score = 80.0 + (raw_score / 50.0) * 7.9

            candidate_data = {
                "filename": resume["filename"],
                "score": round(display_score, 2),
                "raw_score": raw_score,
                "current_yoe": resume["yoe"],
                "missing_mandatory": missing_mandatory,
                "missing_keywords": missing_keywords[:4],
                "yoe_gap": yoe_gap,
                "required_yoe": required_yoe
            }

            # Because all scaled display scores are now >= 80%, we segment them by tier alignment
            if display_score >= 88.0:
                strong_matches.append(candidate_data)
            else:
                close_matches.append(candidate_data)

        return {
            "strong_matches": sorted(strong_matches, key=lambda x: x["score"], reverse=True),
            "close_matches": sorted(close_matches, key=lambda x: x["score"], reverse=True)
        }

# --- STREAMLIT UI DESIGN ---
st.title("🎯 Intelligent Resume Analyzer & Matcher")
st.markdown("Upload a Job Description and multiple PDF resumes to filter high-performing fits and diagnose keyword optimization gaps.")
st.markdown("---")

analyzer = StreamlitResumeAnalyzer()

# Sidebar Setup
st.sidebar.header("📋 Job Requirements Configuration")
jd_input = st.sidebar.text_area("Job Description (JD)", height=250, placeholder="Paste the text requirements here...")
skills_input = st.sidebar.text_input("Mandatory Tech Stacks / Skills", placeholder="e.g., Python, AWS, Docker")
mandatory_skills_list = [s.strip() for s in skills_input.split(",") if s.strip()]

# Resume Processing Upload Panel
st.subheader("📂 Step 2: Upload Resumes (PDF Format)")
uploaded_files = st.file_uploader("You can drag and drop multiple PDF files simultaneously", type=["pdf"], accept_multiple_files=True)

processed_resumes = []

if uploaded_files:
    st.write(f"📝 **Parsing {len(uploaded_files)} PDF Files...**")
    for uploaded_file in uploaded_files:
        raw_text = analyzer.extract_text_from_pdf(uploaded_file)
        
        with st.expander(f"📄 Settings for: {uploaded_file.name}", expanded=False):
            yoe_guess = st.number_input(f"Assigned Years of Experience ({uploaded_file.name}):", min_value=0, max_value=40, value=2, key=uploaded_file.name)
            
        if raw_text:
            processed_resumes.append({
                "filename": uploaded_file.name,
                "text": raw_text,
                "yoe": yoe_guess
            })

st.markdown("###")
if st.button("🚀 Execute Match Screening", type="primary"):
    if not jd_input:
        st.warning("Please provide a Job Description (JD) in the sidebar parameters.")
    elif not processed_resumes:
        st.warning("Please upload at least one PDF resume to parse.")
    else:
        with st.spinner("Executing semantic analysis algorithms..."):
            report = analyzer.run_analysis(jd_input, mandatory_skills_list, processed_resumes)
            
            # --- RENDER SUMMARY REPORT CARDS ---
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="✅ Prime Target Matches (88% - 100%)", value=len(report["strong_matches"]))
            with col2:
                st.metric(label="📈 Standard Calibrated Matches (80% - 87%)", value=len(report["close_matches"]))
                
            st.markdown("---")
            
            # --- RENDER TOP PIPELINE PANEL ---
            st.subheader("🏆 Prime Target Match Pipeline")
            if not report["strong_matches"]:
                st.info("No candidate resumes placed inside the prime distribution tier.")
            else:
                for candidate in report["strong_matches"]:
                    with st.container():
                        st.markdown(f"#### 🍏 {candidate['filename']} ── **Score: {candidate['score']}%**")
                        st.caption(f"Candidate profile registers {candidate['current_yoe']} Years of Field Experience.")
                        
                        # Show actionable improvement strategies if gaps are present even in upper tiers
                        if candidate["missing_mandatory"] or candidate["missing_keywords"]:
                            with st.expander("🔍 Review Optimization Recommendations"):
                                if candidate["missing_mandatory"]:
                                    st.write(f"• **Add Core Stacks:** Explicitly mention ` {', '.join(candidate['missing_mandatory'])} `.")
                                if candidate["missing_keywords"]:
                                    st.write(f"• **Add Industry Terms:** Incorporate contextual terms like {', '.join([f'_{k}_' for k in candidate['missing_keywords']])}.")
                        st.markdown("---")

            # --- RENDER CALIBRATED BASE PIPELINE PANEL ---
            st.subheader("📊 Calibrated Match Pipeline")
            if not report["close_matches"]:
                st.info("No candidates landed inside the 80% - 87% calibrated baseline tier.")
            else:
                for candidate in report["close_matches"]:
                    with st.expander(f"👤 {candidate['filename']} 🧪 Match Profile: {candidate['score']}%"):
                        st.markdown("##### 💡 Targeted Strategies to maximize this profile score:")
                        
                        counter = 1
                        if candidate["yoe_gap"] > 0:
                            st.write(f"**{counter}. Address Experience Deficit:** The role targets {candidate['required_yoe']} years. Explicitly frame existing freelance initiatives, technical bootcamps, or academic lab projects to counter balance the {candidate['yoe_gap']}-year gap profile.")
                            counter += 1
                            
                        if candidate["missing_mandatory"]:
                            st.write(f"**{counter}. Integrate Core Missing Stacks:** Ensure your resume clearly states your working knowledge with: ` {', '.join(candidate['missing_mandatory'])} `.")
                            counter += 1
                            
                        if candidate["missing_keywords"]:
                            formatted_kws = ", ".join([f"'_{k}'_" for k in candidate["missing_keywords"]])
                            st.write(f"**{counter}. Calibrate Semantic Density:** Inject domain context terms like {formatted_kws} into context block lines inside past team positions.")