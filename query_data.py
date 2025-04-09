from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
from typing import List, Set, Tuple, Dict

from db_create import load_processed_documents

CHROMA_PATH = "./chroma_db_v4"  # Path to your Chroma DB

PROMPT_TEMPLATE = """
    Dựa trên thông tin được cung cấp dưới đây, hãy trả lời cho câu hỏi {question} bằng tiếng Việt:
    **Thông tin:**
    {context}
    \nAssistant:
    """

MODEL_PATH = "model/vinallama-q5.gguf"  # Path to your Llama model

EMBEDDING= HuggingFaceEmbeddings(
            model_name="hiieu/halong_embedding",
            model_kwargs={'device': 'cuda'},  
            encode_kwargs={'normalize_embeddings': True}
            )

def get_unique_metadata(documents: List[Tuple[Dict, float]]) -> Set[Tuple[int, int]]:
    """
    Returns a set of unique (id, part) pairs from the provided documents.
    
    :param documents: List of tuples containing Document metadata and score.
    :return: Set containing unique (id, part) pairs.
    """
    return {(doc[0].metadata['original_document_id'], doc[0].metadata['part']) for doc in documents}


def main():
    query_text = "Điều trị zona thần kinh như thế nào?"

    # Prepare the DB.
    embedding_function = EMBEDDING
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Load the processed documents using your function
    # processed_documents = load_processed_documents('./data/processed_documents.pkl')

    # Search the DB.
    results_fromdb = db.similarity_search_with_score(query_text, k=5,)
    print(results_fromdb)
    if len(results_fromdb) == 0 or results_fromdb[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return
    
    # metadata = get_unique_metadata(results_fromdb)
    # print(metadata)
    # filtered_documents = [
    # doc for doc in processed_documents 
    # if (doc.metadata.get('original_document_id'), doc.metadata.get('part')) in metadata
    # ]
    
    context_text = "\n\n---\n\n".join(doc[0].page_content.lower().replace('\n', '. ').strip() for doc in results_fromdb)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    llm = LlamaCpp(
                        model_path=MODEL_PATH,
                        n_gpu_layers=20,
                        n_batch=512,
                        max_tokens=256,
                        n_ctx=2000,
                        f16_kv=True,
                        temperature=0.5,
                        repeat_penalty=1.1,
                        echo=False,  
                        stop=["\nUser:", "\n###", "\nQ:"]
                    )
    response_text = llm.invoke(prompt)

    # sources = [doc.metadata.get("source", None) for doc in filtered_documents]

    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(response_text)


if __name__ == "__main__":
    main()