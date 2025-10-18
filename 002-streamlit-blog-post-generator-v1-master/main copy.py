import streamlit as st
from langchain_openai import OpenAI
from langchain import PromptTemplate

st.set_page_config(
    page_title = "Blog Post Generator"
)

st.title("Blog Post Generator Out of a Topic")

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type = "password"
)

def generate_response(tema):
    llm = OpenAI(openai_api_key=openai_api_key)
    template = """
        Como redactor experimentado en startups y capital de riesgo,
        genera una entrada de blog de 400 palabras sobre {tema}.

        Tu respuesta debe tener este formato:
        Primero, escribe la entrada del blog.
        Luego, suma el número total de palabras del texto y muéstralo así:

        This post has X words
        """
    prompt = PromptTemplate(
        input_variables = ["tema"],
        template = template
    )
    query = prompt.format(tema=tema)
    response = llm(query, max_tokens=2048)
    return st.write(response)


topic_text = st.text_input("Enter topic: ")
if not openai_api_key.startswith("sk-"):
    st.warning("Enter OpenAI API Key")
if openai_api_key.startswith("sk-"):
    generate_response(topic_text)
        
