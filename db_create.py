import json
import pickle
import shutil
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
import os

CORPUS_DIR = "./extracted_content"  # Directory containing the text files

# v4 with halong_embedding
# v5 with TuanNM171284/TuanNM171284-HaLong-embedding-medical
CHROMA_PATH = "F://VNPT_Intern//chroma_db_v4"


EMBEDDING = HuggingFaceEmbeddings(
            model_name="TuanNM171284/TuanNM171284-HaLong-embedding-medical",
            model_kwargs={'device': 'cuda'},  
            encode_kwargs={'normalize_embeddings': True}
            )

def load_documents(corpus_dir: str):
        documents = []
        for filename in os.listdir(corpus_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(corpus_dir, filename)
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        return documents

def split_text(documents: list[Document]):
    # Create text splitter with "SEPARATED" as a high-priority separator
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        length_function=len,
        # Add "SEPARATED" as the first separator so it takes priority
        separators=["SEPARATED"]
    )
    
    chunks = text_splitter.split_documents(documents)
    for chunk in chunks:
        chunk.page_content = chunk.page_content.replace("SEPARATED", "").strip()
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # Print sample info if chunks exist
    if len(chunks) > 10:
        document = chunks[10]
        print(document.page_content)
        print(document.metadata)
    
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, 
        embedding=EMBEDDING, 
        persist_directory=CHROMA_PATH
    )
    

def generate_data_store():
    documents = load_documents(CORPUS_DIR)
    chunks = split_text(documents)

    save_to_chroma(chunks)

def main():
    generate_data_store()


def process_txt_files_inplace(folder_path="extracted_content"):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Chỉ xử lý file .txt
            file_path = os.path.join(folder_path, filename)
            
            # Đọc nội dung file
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
            
            # Xóa 4 dòng đầu nếu có đủ dòng
            if len(lines) > 4:
                new_content = "".join(lines[4:])
            else:
                new_content = ""  # Nếu file có ít hơn 4 dòng, làm rỗng file
            
            # Ghi đè lại nội dung vào chính tệp đó
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(new_content)

if __name__ == "__main__":
    # process_txt_files_inplace()
    main()
