# Part 1: Library Imports and Setting up the Device
# -------------------------------------------------
import torch
import os
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import gradio as gr

from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Part 2: Loading and Inspecting the PDFs
# ----------------------------------------
loader = PyPDFDirectoryLoader("unzipped_files/aircraft_pdfs")
docs = loader.load()
print(f"Total pages across all documents: {len(docs)}")

# Part 3: Setting up Embeddings
# -----------------------------
embeddings = HuggingFaceInstructEmbeddings(
    model_name="thenlper/gte-small", model_kwargs={"device": DEVICE}
)

# Part 4: Text Splitting
# -----------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(docs)
print(f"Number of text chunks: {len(texts)}")

# Part 5: Creating an Embeddings Database
# ---------------------------------------
db = Chroma.from_documents(texts, embedding=embeddings, persist_directory="db")

# Part 6: Setting up LLM (Language Learning Model)
# ------------------------------------------------

my_credentials = {
"url"    : "https://us-south.ml.cloud.ibm.com"
}

params = {
        GenParams.MAX_NEW_TOKENS: 512, # The maximum number of tokens that the model can generate in a single run.
        GenParams.TEMPERATURE: 0.1,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
    }

LLAMA2_model = Model(
        model_id= 'meta-llama/llama-2-70b-chat', 
        credentials=my_credentials,
        params=params,
        project_id="skills-network", 
        )

llm_hub = WatsonxLLM(model=LLAMA2_model)


# Part 7: Defining the Prompt Template
# ------------------------------------
template ="""
<<<SYS>>
You are a helpful, respectful, and honest assistant to build aircraft. So, it is important to be precise. The following are the context of knowledge to build aircraft. Based on the context information, answer the follow-up questions.
{context}
[INST] Question: {question}  [/INST]
<<</SYS>>>
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Part 8: Constructing the Question-Answer API Chain
# --------------------------------------------------
api_chain = RetrievalQA.from_chain_type(
    llm=llm_hub,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

# Part 9.5: Implementing the Chat Interface
# -------------------------------------------
def ai_response(message, history):
    output = api_chain(message, history)  # Ensure api_chain is the correct function to use here
    source_documents = output["source_documents"][0].metadata


    # Extracting data from the output
    response = output["result"]
    result = response + str(source_documents)  # Concatenating response with the joined metadata
    return result
# Launch the Gradio interface
demo = gr.ChatInterface(ai_response) 
demo.launch(share=True, server_port = 7861)
