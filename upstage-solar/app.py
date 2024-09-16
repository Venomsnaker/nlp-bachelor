import gradio as gr
import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from typing import List, TypedDict
from langchain_upstage import ChatUpstage, UpstageLayoutAnalysisLoader, UpstageEmbeddings, GroundednessCheck
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, StateGraph

upstage_api_key_env_name = "UPSTAGE_API_KEY"
def load_env():
    load_dotenv()
    return os.environ.get(upstage_api_key_env_name)
UPSTAGE_API_KEY = load_env()

def layout_analysis(filenames: str) -> List[Document]:
    layout_analysis_loader = UpstageLayoutAnalysisLoader(filenames, split="element")
    return layout_analysis_loader.load()

filenames = [
    "data/CS224n-Week1.pdf",
]
docs = layout_analysis(filenames)
db: VectorStore = Chroma(
    embedding_function=UpstageEmbeddings(model="solar-embedding-1-large-passage")
)
retriever = db.as_retriever()
db.add_documents(docs)

class RagState(TypeDict):
    """
    Represents the state of our graph.

    Attributes:
        context: retrieved context
        question: question asked by the user
        answer: generated answer to the question
        groundedness: groundedness of the assistant's response
    """

    context: str
    question: str
    answer: str
    groundedness: str

template = """Answer the question based only on the given context.
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatUpstage(model="solar-pro")
llm_chain = prompt | llm | StrOutputParser()

def format_documents(docs: List[Document]) -> str:
    return "\n".join([doc.page_content for doc in docs])

def retrieve(state: RagState) -> RagState:
    docs = retriever.invoke(state["question"])
    context = format_documents(docs)
    return RagState(context=context)

def model_answer(state: RagState) -> RagState:
    response = llm_chain.invoke(state)
    return RagState(answer=response)

prompt = ChatPromptTemplate.from_template(template)
gc = GroundednessCheck()

def groundedness_check(stage: RagState) -> RagState:
    response = gc.run({'context': state["context"], "answer": state["answer"]})
    return RagState(groundedness_check=response)

def groundedness_condition(state: RagState) -> RagState:
    return state["groundedness"]

workflow = StateGraph(RagState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("model", model_answer)
workflow.add_node("groundedness_check", groundedness_check)
workflow.add_edge("retrive", "model")
workflow.add_edge("model", "groundedness_heck")
workflow.add_conditonal_edges(
    "groundedness_check",
    groundedness_condition,
    {
        "grounded": END,
        "notGrounded": "model",
        "notSure": "model"
    }
)
workflow.set_entry_point("retrieve")
app = workflow.compile()

def chat(inputs_user):
    inputs = {"question": inputs_user}
    output = app.stream(inputs)
    print(type(output))
    print(output[1])
    return "";    
    # result = llm.invoke(input)
    # return result.content   

interface = gr.Interface(
    fn=chat,
    inputs=[gr.Textbox(label="text", lines=2)],
    outputs='text',
    title="Upstage Solar Chatbot 0.2",
)

if __name__ == "__main__":
    interface.launch()