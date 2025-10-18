import streamlit as st
import fitz 
from langchain import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from io import StringIO

#LLM and key loading function
def load_LLM(openai_api_key):
    # Make sure your openai_api_key is set as an environment variable
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model="gpt-4o")
    return llm
     
#Page title and header
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")


#Intro: instructions
col1, col2 = st.columns(2)

with col1:
    st.markdown("ChatGPT cannot summarize long texts. Now you can do it with this app.")

with col2:
    st.write("Contact with [AI Accelera](https://aiaccelera.com) to build your AI Projects")


#Input OpenAI API Key
st.markdown("## Enter Your OpenAI API Key")

def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text

openai_api_key = get_openai_api_key()


# Input
st.markdown("## Upload the text file you want to summarize")

uploaded_file = st.file_uploader("Choose a file", type="pdf")

## This function transform bytes of a pdf into a chain of streams
def extract_text_from_bytes_data(bytes_data: bytes) -> str:
    doc = fitz.open(stream=bytes_data, filetype="pdf")
    all_text = []
    for page in doc:
            all_text.append(page.get_text("text") or "")
    return "\n".join(all_text)

map_prompt = PromptTemplate.from_template(
    "Resume en español, de forma clara y fiel, el siguiente fragmento:\n\n{text}\n\nResumen (en español):"
)
combine_prompt = PromptTemplate.from_template(
    "Teniendo en cuenta los resúmenes parciales (en español) de cada fragmento, "
    "escribe un resumen ejecutivo final en español, conciso y completo:\n\n{text}\n\nResumen final (en español):"
)




# Output
st.markdown("### Here is your Summary:")

if uploaded_file is not None:
    # To read file as bytes (zeros 0s  and ones 1s):
    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)

    pdf_text=extract_text_from_bytes_data(bytes_data=bytes_data)

    

    # Can be used wherever a "file-like" object is accepted:
    #dataframe = pd.read_csv(uploaded_file)
    #st.write(dataframe)

    file_input = pdf_text

    if len(file_input.split(" ")) > 200000:
        st.write("Please enter a shorter file. The maximum length is 200000 words.")
        st.stop()

    if file_input:
        if not openai_api_key:
            st.warning('Please insert OpenAI API Key. \
            Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
            icon="⚠️")
            st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], 
        chunk_size=5000, 
        chunk_overlap=350
        )

    splitted_documents = text_splitter.create_documents([file_input])

    llm = load_LLM(openai_api_key=openai_api_key)

    summarize_chain = load_summarize_chain(
        llm=llm, 
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt
        )

    summary_output = summarize_chain.run(splitted_documents)

    st.write(summary_output)