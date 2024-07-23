import os

import streamlit as st

from chat_with_utility import get_answer

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Chat With Doc",
    page_icon="📄",
    layout="centered",
)
st.title("Document Q&A - llama3 - ollama")

uploaded_file = st.file_uploader(label="Upload your document", type=['pdf'])

user_query = st.text_input("As your question")

if st.button("Run"):
    bytes_data = uploaded_file.read()
    file_name = uploaded_file.name
    #  save the file working directory
    file_path = os.path.join(working_dir, file_name)
    with open(file_path, "wb") as f:
        f.write(bytes_data)
        answer = get_answer(file_name, user_query)

        st.success(answer)
