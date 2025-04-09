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

CHROMA_PATH = "chroma_db_v4"    

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

EMBEDDING = HuggingFaceEmbeddings(
            model_name="hiieu/halong_embedding",
            model_kwargs={'device': 'cuda'},  
            encode_kwargs={'normalize_embeddings': True}
            )

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

def save_processed_documents(processed_documents, save_path):
    """
    Save processed documents to a file.
    
    Args:
        processed_documents: The list of Document objects to save
        save_path: Path where to save the documents
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Determine the file format based on extension
    _, ext = os.path.splitext(save_path)
    
    if ext.lower() == '.pkl':
        # Save as pickle (binary)
        with open(save_path, 'wb') as f:
            pickle.dump(processed_documents, f)
    elif ext.lower() == '.json':
        # Save as JSON (text)
        serializable_docs = []
        for doc in processed_documents:
            serializable_docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
    else:
        # Default to pickle
        with open(save_path, 'wb') as f:
            pickle.dump(processed_documents, f)
    
    print(f"Saved {len(processed_documents)} processed documents to {save_path}")

def load_processed_documents(load_path):
    """
    Load processed documents from a file.
    
    Args:
        load_path: Path from where to load the documents
        
    Returns:
        List of Document objects
    """
    _, ext = os.path.splitext(load_path)
    
    if ext.lower() == '.pkl':
        # Load from pickle
        with open(load_path, 'rb') as f:
            return pickle.load(f)
    elif ext.lower() == '.json':
        # Load from JSON
        with open(load_path, 'r', encoding='utf-8') as f:
            serialized_docs = json.load(f)
        
        # Convert back to Document objects
        documents = []
        for doc_dict in serialized_docs:
            documents.append(Document(
                page_content=doc_dict["page_content"],
                metadata=doc_dict["metadata"]
            ))
        return documents
    else:
        # Default to pickle
        with open(load_path, 'rb') as f:
            return pickle.load(f)

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
