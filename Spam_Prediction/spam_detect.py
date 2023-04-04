import streamlit as st
import pickle
import spacy
import numpy as np
from sklearn.preprocessing import MinMaxScaler



nlp=spacy.load("en_core_web_sm")
st.title("Spam Detection")

#st.write('Enter the mail here :arrow_down:' )
input = st.text_input("Enter the mail here :arrow_down:","mail")
#input="smith is good boy"
with open('model_pickle',"rb") as f:
    model=pickle.load(f)

if input is not None:
    d= nlp(input).vector
    if int(model.predict([d]))==1:
        st.write('The mail is :email: : SPAM' )

    else:
        st.write('The mail is :email: : NOT SPAM')



