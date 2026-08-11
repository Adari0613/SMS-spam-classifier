import streamlit as st
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
inputsms=st.text_input("Enter the message")
if st.button('Predict:'):

    def transform_text(text):
        text=text.lower()
        text=nltk.word_tokenize(text, preserve_line=True)
        y=[]
        for i in text:
            if i.isalnum():
                y.append(i)
        text=y[:]
        y.clear()

        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
        text=y[:]
        y.clear()
        for i in text:
            y.append(ps.stem(i))
            
        return " ".join(y)

if inputsms:

    #1.preprocess
    transformed_sms=transform_text(inputsms)
    #2.vectorize
    vector_input=tfidf.transform([transformed_sms])
    #3.predict
    res=model.predict(vector_input)[0]
    #4.display
    if res==1:
        st.header("Spam")
    else:
        st.header("Not Spam")
