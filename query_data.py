from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
from typing import List, Set, Tuple, Dict
from db_create import CHROMA_PATH, EMBEDDING
from sentence_transformers import CrossEncoder
import numpy as np
import os
import py_vncorenlp




PROMPT_TEMPLATE = """
    Dựa trên thông tin được cung cấp dưới đây, hãy trả lời cho câu hỏi {question} bằng tiếng Việt:
    **Thông tin:**
    {context}
    \nAssistant:
    """

MODEL_PATH = "F:\VNPT_Intern\model\q7bq3km.gguf"  # Path to  model

MODEL_ID = 'itdainb/PhoRanker'

MAX_LENGTH = 256

def main():
    query_text = "Triệu chứng zona thần kinh là gì?"

    # Prepare the DB
    db = Chroma(persist_directory=CHROMA_PATH, 
                embedding_function=EMBEDDING)

    # Search the DB - retrieve more candidates than needed for reranking
    results_fromdb = db.similarity_search(query_text)
    os.environ["JAVA_HOME"] = "C:/Program Files/Java/jdk-17"  # Set JDK path
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='F:/VNPT_Intern/.venv/Lib/site-packages/py_vncorenlp')
    print(results_fromdb)
    # Check if we have any results
    if not results_fromdb:
        print("No results found in the database.")
        return
    
    # Prepare for reranking with PhoRanker
    tokenized_query = " ".join(rdrsegmenter.word_segment(query_text))
    
    # Create pairs for reranking
    documents = [doc.page_content for doc in results_fromdb]  # Remove the [0] indexing
    tokenized_documents = [" ".join(rdrsegmenter.word_segment(doc)) for doc in documents]
    
    # Make sure we have documents to process
    if not tokenized_documents:
        print("No documents to rerank.")
        return
        
    tokenized_pairs = [[tokenized_query, doc] for doc in tokenized_documents]
    
    # Load and apply reranker
    model = CrossEncoder(MODEL_ID, max_length=MAX_LENGTH)
    model.model.half()  # For fp16 usage
    
    # Get reranking scores
    scores = model.predict(tokenized_pairs)
    
    # Create reranked results with scores
    reranked_results = [(results_fromdb[i], scores[i]) for i in range(len(scores))]
    
    # Sort by score (descending)
    reranked_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)
    
    # Take top 10 after reranking
    top_results = reranked_results[:2]
    
    print("Reranked results with scores:")
    for i, (doc, score) in enumerate(top_results):
        print(f"{i+1}. Score: {score:.4f}")
    
    # Build context from reranked results
    context_text = "\n\n---\n\n".join(doc.page_content.lower().replace('\n', '. ').strip() 
                                     for doc, score in top_results)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    llm = LlamaCpp(
                model_path=MODEL_PATH,
                n_gpu_layers=-1,
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

    print(response_text)

import difflib

def is_similar(text1, text2, threshold=0.9):
    return difflib.SequenceMatcher(None, text1, text2).ratio() > threshold

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from langchain.schema import Document

def deduplicate_by_embedding(reranked_results: List[Tuple[Document, float]], similarity_threshold: float = 0.9):
    """
    Keep only semantically distinct documents among the reranked results.
    
    reranked_results: list of (Document, score) tuples, already sorted descending by score
    similarity_threshold: cosine sim above this means "duplicate"
    """
    unique_docs = []
    unique_embs = []

    for doc, score in reranked_results:
        # get embedding for this document
        emb = EMBEDDING.embed_query(doc.page_content)
        
        # check similarity against all already-kept embeddings
        if not unique_embs or all(
            cosine_similarity([emb], [existing_emb])[0][0] < similarity_threshold
            for existing_emb in unique_embs
        ):
            unique_docs.append((doc, score))
            unique_embs.append(emb)

    return unique_docs

def process_text_with_linebreaks(response_text):
    full_response = ""
    # Split text into lines first
    lines = response_text.split('\n\n')
    
    for index, line in enumerate(lines):
        # Process words in each line
        words = line.split()
        for word in words:
            full_response += word + " "
            yield full_response
        
        # Add line break if not the last line
        if index < len(lines) - 1:
            full_response += "\n"
            yield full_response

if __name__ == "__main__":
    main()