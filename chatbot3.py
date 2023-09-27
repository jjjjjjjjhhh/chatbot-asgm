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


os.environ['REPLICATE_API_TOKEN'] = "r8_KQMuoOovvSTQ1V221paM9i3XtiPblOC03vJ7d"
pinecone.init(api_key='dbc2c5be-9964-4d58-955b-e0025a21cd77', environment='gcp-starter')

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
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run(llm, 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    return output


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar='👩‍🎓'):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar='👩‍🍳'):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


# pdf thingy (hvnt connected)

if uploaded_file is not None:
    uploaded_file_name = uploaded_file.name  # Get the name of the uploaded file

    # Create a temporary directory to save the uploaded file
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file_name)

    # Save the uploaded file to the temporary directory
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # Split the documents into smaller chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    index_name = "chatbot-1"
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

   
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar='👩‍🎓'):
            st.write(prompt)

            

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar='👩‍🍳'):
            with st.spinner("Thinking..."):
                response = qa_chain.answer(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)



        


