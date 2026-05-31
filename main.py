import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client


load_dotenv()

os.environ["USER_AGENT"] = "MyRAGApp/1.0"

client = Client()

loader = WebBaseLoader(web_paths=["https://en.wikipedia.org/wiki/Prestige_Group"])
document_loader = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(document_loader)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview", task_type="retrieval_document"
    ),
)

retriever = vectorstore.as_retriever()

prompt = client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


# Build the LCEL (LangChain Expression Language) chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


response = rag_chain.invoke("where is this company based on")
print(response)
