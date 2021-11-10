import streamlit as st 
import pickle
import gzip
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

with gzip.open("vectorizer.pickle.gz", 'rb') as f:
    tfidf = pickle.load(f)
with gzip.open("model.pickle.gz", 'rb') as f:
    model = pickle.load(f)
st.title('SMS Spam Classifier')
input_sms = st.text_area('Enter the message')
#Preprocess
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [PorterStemmer().stem(i) for i in text if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(text)

if st.button('Predict'):

    transformed_sms = transform_text(input_sms)
    #Vectorize
    vector_input = tfidf.transform([transformed_sms])
    #predict
    result= model.predict(vector_input)[0]
    #Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")