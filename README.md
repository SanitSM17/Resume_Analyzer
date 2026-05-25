# Resume_Analyzer
A powerful AI Resume Analyzer using Python, Natural Language Processing (NLP), and Streamlit. It process resumes, extract key skills, and match them to job descriptions automatically using TF-IDF, cosine similarity, and NLP techniques. Finally, created a Streamlit web app for real-time analysis and visualization.
Libraries, tools and functions used in this project:
1. re is the command used to load the built-in regular expression (RegEx) module.
2. Streamlit is an open-source Python framework that allows you to turn data scripts into interactive, shareable web applications in minutes.
3. PyPDF2 is a free, open-source library for Python that allows you to work with PDF files.
4. Sentence Transformers (also known as SBERT) refers to a popular open-source framework and library used to represent sentences, paragraphs, or images as dense vectors (embeddings).
5. TfidfVectorizer is a class in the Python library scikit-learn that converts raw text into a numerical format suitable for machine learning.
6. We created a class StreamlitResumeAnalyzer container which contains extracting functions which extracts and cleans text contents and keywords from an uploaded PDF file.
7. Then we create exception to handle error.
8. Created instance for missing values to Identify core gaps.
9. Performed Raw mathematical assessment score.
10. NORMALIZATION LOGIC: Map raw scores to ensure outputs land above 80%.
11. Because all scaled display scores are now >= 80%, we segment them by tier alignment.
12. Then created Streamlit UI Design.
13. Done Sidebar Setup.
14. Then created Resume Processing Upload Panel.
15. Then created Render Calibrated Base Pipeline Panel.
16. Done
