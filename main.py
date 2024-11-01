import streamlit as st
from io import StringIO
from pypdf import PdfReader
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import pipeline


checkpoint = "google-t5/t5-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)


st.title("Сириус.ИИ: задача от Биокада")


def extract_text(article):
    text = ""

    reader = PdfReader(article)
    for page in reader.pages:
        text += page.extract_text()

    return str("summarize: " + text).replace("\n", " ")


article = st.file_uploader("Загрузите статью для суммаризации", type="pdf")
if article:
    text = extract_text(article)
    st.write("**Сокращенный вариант статьи:** \n")
    st.write(summarizer(text[:2000])[0]['summary_text'])
