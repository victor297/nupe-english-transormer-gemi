import streamlit as st
import torch
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd

# Load the saved model and tokenizer
model_name = 'nupe-to-english-model'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

st.title("Nupe to English Translator")
st.write("Enter text in Nupe to get the English translation.")

user_input = st.text_area("Input Nupe Text")

if st.button("Translate"):
    if user_input:
        translation = translate_text(user_input)
        st.write("**English Translation:**")
        st.write(translation)
    else:
        st.write("Please enter text in Nupe.")
