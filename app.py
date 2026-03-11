import streamlit as st
import pickle
import nltk
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

st.set_page_config(page_title="Spam Detector", page_icon="🛡️", layout="centered")

# -------- CSS --------
st.markdown("""
<style>

.main{
background:transparent;
}

/* Title */
.title{
text-align:center;
font-size:40px;
font-weight:700;
color:#38bdf8;
margin-bottom:5px;
}

.subtitle{
text-align:center;
color:#cbd5f5;
margin-bottom:35px;
}



/* Textbox */
textarea{
background: rgba(255,255,255,0.08) !important;
color:black !important;
border-radius:10px !important;
border:1px solid rgba(255,255,255,0.2) !important;
}

/* Button */
.stButton button{
background:linear-gradient(90deg,#3b82f6,#06b6d4);
color:white;
border:none;
border-radius:8px;
padding:10px 25px;
font-weight:600;
}

/* Spam Result */
.result-spam{
margin-top:30px;
padding:25px;
border-radius:12px;
text-align:center;
background: rgba(239,68,68,0.15);
border:1px solid #ef4444;
}

.result-spam h2{
color:#f87171;
}

/* Safe Result */
.result-safe{
margin-top:30px;
padding:25px;
border-radius:12px;
text-align:center;
background: rgba(16,185,129,0.15);
border:1px solid #10b981;
}

.result-safe h2{
color:#34d399;
}

</style>
""", unsafe_allow_html=True)


# -------- Title --------
st.markdown('<div class="title">Spam Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste any SMS message below to check if it\'s spam or legitimate.</div>', unsafe_allow_html=True)

# -------- Card --------
st.markdown('<div class="card">', unsafe_allow_html=True)

input_sms = st.text_area("Enter your message", height=150)

char_count = len(input_sms)
st.caption(f"{char_count} characters")

predict = st.button("✈ Predict")

st.markdown('</div>', unsafe_allow_html=True)

# -------- Text Processing --------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# -------- Load Model --------
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# -------- Prediction --------
if predict:

    transform_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transform_sms])

    num_characters = len(input_sms)

    final_input = np.hstack((vector_input.toarray(),
                            np.array([num_characters]).reshape(-1,1)))

    result = model.predict(final_input)[0]

    if result == 1:
        st.markdown("""
        <div class="result-spam">
        <h2>🛡 Spam Detected</h2>
        <p>This message appears to be spam. Be cautious!</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-safe">
        <h2>✅ Legitimate Message</h2>
        <p>This message looks safe.</p>
        </div>
        """, unsafe_allow_html=True)