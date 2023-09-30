import streamlit as st
import replicate
import PyPDF2
import os
import sys
import io
import pinecone
import pypdf
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import tempfile
import uuid
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import subprocess

os.environ['REPLICATE_API_TOKEN'] = "r8_61ZBBa1uv7QmgSOZSzAn1KpWM5wSGq40w1krB"
pinecone.init(api_key='42c2d8e4-e167-4087-8723-1b8701369251', environment='gcp-starter')
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# App title
st.set_page_config(page_title="Lao Gan Ma Chatbot 👩‍🍳🧉", page_icon="img/lgm.png")

st.image('img/lgm.png', width=100)
st.title('Lao Gan Ma Chatbot👩‍🍳🧉')

# model and parameter
with st.sidebar:
    
    col1, col2 = st.columns((1, 3))
    with col1:
        st.image('img/lgm.png', width=50)
    with col2:
        st.title('Lao Gan Ma Chatbot')
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.75, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=100, max_value=5000, value=5000, step=100)
    

uploaded_file = st.sidebar.file_uploader("", type=["pdf"], key="file_uploader")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar='👩‍🍳'):
            st.write(message["content"])
    else:
        with st.chat_message(message["role"], avatar='👩‍🎓'):
            st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you?"}]

st.sidebar.button('Clear Chat', on_click=clear_chat_history)

def generate_llama2_response(prompt_input):
    string_dialogue = ""
    for dict_message in st.session_state.messages:
        content = str(dict_message["content"])
        if dict_message["role"] == "user":
            string_dialogue += "User: " + content + "\n\n"
        else:
            string_dialogue += "Assistant: " + content + "\n\n"
    
    # Call the replicate API
    output = replicate.run(llm, 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature": temperature, "top_p": top_p, "max_length": max_length, "repetition_penalty": 1})
    
    
    if hasattr(output, '__iter__'):
        response_content = ''.join(output)
    else:
        response_content = output
    
    return response_content

def extract_text_from_pdf(pdf_path):
    # Create a temporary directory to store the converted images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert PDF to images using pdftoppm
        subprocess.run(["bin/pdftoppm.exe", "-jpeg", pdf_path, os.path.join(temp_dir, "page")])

        # Get the list of generated images
        image_files = [os.path.join(temp_dir, image_file) for image_file in os.listdir(temp_dir)]
        image_files.sort()  # Ensure pages are in order

        # Extract text from each image
        texts = []
        for image_file in image_files:
            with Image.open(image_file) as image:
                text = pytesseract.image_to_string(image)
                texts.append(text)

        return "\n".join(texts)

# PDF handling

qa_chain = None
index_name = "chatbot-1"


if uploaded_file and (not hasattr(st.session_state, 'processed_file_name') or uploaded_file.name != st.session_state.processed_file_name):
    
    with st.spinner("Lao Gan Ma is reading your PDF..."):
        if index_name in pinecone.list_indexes():
            pinecone.delete_index(index_name)
        pinecone.create_index(index_name, dimension=768, metric="cosine", pods=4, pod_type="s1.x1")
        uploaded_file_name = uploaded_file.name
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, uploaded_file_name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        if not any(txt for txt in texts if str(txt).strip()):
            # If no text was extracted, it might be a scanned PDF. Use OCR.
            extracted_text = extract_text_from_pdf(temp_file_path)
            st.write("Extracted text from scanned PDF:\n", extracted_text)
            texts = extracted_text
        #st.write("Extracted text from PDF:\n", texts)
        embeddings = HuggingFaceEmbeddings()
        index = pinecone.Index(index_name)
        vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)
        llm_model = Replicate(
            model=llm,
            model_kwargs={"temperature": temperature, "max_length": max_length, "top_p": top_p}
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm_model,
            vectordb.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True
        )
    
    st.session_state.processed_file_name = uploaded_file.name
    st.session_state.texts = texts
    st.session_state.pdf_processed = True
    st.session_state.qa_chain = qa_chain


elif hasattr(st.session_state, 'pdf_processed') and st.session_state.pdf_processed:
    qa_chain = st.session_state.qa_chain

def filter_response(response):
    undesired_keywords = ["Note:", "User:"]
    sentences = response.split('.')
    filtered_sentences = [sent for sent in sentences if not any(keyword in sent for keyword in undesired_keywords)]
    return '.'.join(filtered_sentences).strip()

# User input handling
chat_input_key = "pdf_query" if uploaded_file else "general_query"
if prompt := st.chat_input(key=chat_input_key):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar='👩‍🎓'):
        st.write(prompt)

    if uploaded_file and qa_chain:
        with st.spinner("Generating Content..."):
            result = qa_chain({'question': prompt, 'chat_history': []})
            response_content = result['answer']
    else:
        with st.spinner("Generating Content..."):
            response_content = generate_llama2_response(prompt)

    response_content = filter_response(response_content)

    message = {"role": "assistant", "content": response_content}
    st.session_state.messages.append(message)
    with st.chat_message("assistant", avatar='👩‍🍳'):
        with st.spinner("Generating Content..."):
            st.write(response_content)


