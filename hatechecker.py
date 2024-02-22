import streamlit as st
import pickle
import re

st.title('A Spatio Temporal Deep Learning Architecture for Arabic Offensive Language Detection')

language = st.radio("Select Language:", ('Tunisian', 'Levantine'))
if language == 'Tunisian':
    vectorizer = pickle.load(open("best-l-vect.pickle", "rb"))
    model = pickle.load(open("best-l-model.pickle", "rb"))
elif language == 'Levantine':
    vectorizer = pickle.load(open("best-l-vect.pickle", "rb"))
    model = pickle.load(open("best-l-model.pickle", "rb"))
else:
    vectorizer = None
    model = None
text = st.text_area("Leave your comment here", "")
cleaned_text = re.sub('[^؀-ۿ]+', ' ', str(text))
cleaned_text = re.sub('\s+', ' ', cleaned_text).strip()
text_features = vectorizer.transform([cleaned_text]).toarray()
prediction = model.predict(text_features)

if st.button("Hate Speech Analysis"):
    if prediction==0:
        st.error("Prediction: Hate Speech Identified")
    else:
        st.error("Prediction: No Hate Speech Identified")
    
