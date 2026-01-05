from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/docling_report.pdf"
PERSIST_DIR = "chroma_db"


def parse_with_docling(file_path: str) -> list[Document]:
    converter = DocumentConverter()
    result = converter.convert(file_path)

    # ✅ Correct Docling API
    markdown_text = result.document.export_to_markdown()

    return [
        Document(
            page_content=markdown_text,
            metadata={"source": file_path}
        )
    ]

def build_vectorstore():
    docs = parse_with_docling(DATA_PATH)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    # ❌ NO persist() in new Chroma
    print("✅ Vector store created successfully")


if __name__ == "__main__":
    build_vectorstore()
