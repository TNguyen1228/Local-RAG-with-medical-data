import time
from sentence_transformers import CrossEncoder
from streamlit_chat import message
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from datetime import datetime
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
import asyncio
import py_vncorenlp
import streamlit as st
import torch
import os


from query_data import MODEL_PATH, EMBEDDING, MODEL_ID, deduplicate_by_embedding, process_text_with_linebreaks
from db_create import CHROMA_PATH
from log import setup_logger
# Set the environment variable for Java
os.environ["JAVA_HOME"] = "C:/Program Files/Java/jdk-17"  # Set JDK path

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

torch.classes.__path__ = [] 


def log_interaction(query, documents, response, prompt=None):
    """
    Log the complete interaction: query, documents, prompt, and response
    
    Args:
        query (str): The user's query
        documents (list): List of (document, score) tuples retrieved from the database
        response (str): The system's response to the query
        prompt (str, optional): The prompt sent to the LLM
        filtered_docs (list, optional): Documents that passed the threshold
    """
    # Create a simplified view of the documents for logging
    docs_for_log = []
    for doc in documents:
        docs_for_log.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
        })
    
    # Create the log entry
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "documents": docs_for_log,
        "response": response,
        "session_id": st.session_state.get("session_id", "unknown")
    }
    
    # Add prompt if available
    if prompt is not None:
        log_entry["prompt"] = prompt
    
    # Log the entry
    logger.info(log_entry)

logger = setup_logger()

st.set_page_config(page_title="Medical Chatbot", layout="wide")
st.title("Medical Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "db" not in st.session_state:
    st.session_state["db"] = None
if "llm" not in st.session_state:
    st.session_state["llm"] = None
if "session_id" not in st.session_state:
    st.session_state["session_id"] = datetime.now().strftime("%Y%m%d%H%M%S")
if "rdrsegmenter" not in st.session_state:
    st.session_state["rdrsegmenter"] = None
if "chat_mode" not in st.session_state:
    st.session_state["chat_mode"] = "RAG System"

# Log system initialization
logger.info({
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "event": "system_initialization",
    "session_id": st.session_state["session_id"]
})

if st.session_state["rdrsegmenter"] is None:
    st.session_state["rdrsegmenter"] = py_vncorenlp.VnCoreNLP(
        annotators=["wseg"], 
        save_dir='F:/VNPT_Intern/.venv/Lib/site-packages/py_vncorenlp'
    )

with st.sidebar:
    st.title("Chat Settings")
    
    # Mode selection
    st.session_state["chat_mode"] = st.radio(
        "Select Chat Mode",
        ["Normal Chat", "RAG System"],
        help="Normal Chat: Direct conversation with LLM \n\n RAG System: Answers based on medical knowledge base"
    )

# Initialize vector database
if st.session_state["db"] is None:
    try:
        if not os.path.exists(CHROMA_PATH):
            st.error(f"⚠️ Database path does not exist: {CHROMA_PATH}")
            logger.error({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": f"Database path does not exist: {CHROMA_PATH}",
                "session_id": st.session_state["session_id"]
            })
        else:
            st.session_state["db"] = Chroma(
                                            persist_directory=CHROMA_PATH, 
                                            embedding_function=EMBEDDING)
            
            if not os.path.exists(MODEL_PATH):
                st.error(f"⚠️ Model file not found: {MODEL_PATH}")
                logger.error({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "error": f"Model file not found: {MODEL_PATH}",
                    "session_id": st.session_state["session_id"]
                })
            else:
                try:
                    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
                    st.session_state["llm"] = LlamaCpp(
                        model_path=MODEL_PATH,
                        n_gpu_layers=20,
                        n_batch=512,
                        n_ctx=2000,
                        callback_manager=callback_manager,
                        verbose=True,
                    )
                    st.success("Hệ thống đã sẵn sàng! Hãy nhập câu hỏi của bạn.")
                except Exception as e:
                    st.error(f"⚠️ {str(e)}")
                    logger.error({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "error": f"Failed to load model: {str(e)}",
                        "session_id": st.session_state["session_id"]
                    })
    except Exception as e:
        st.error(f"⚠️ {str(e)}")
        logger.error({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": f"Initialization error: {str(e)}",
            "session_id": st.session_state["session_id"]
        })
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
# User input
user_input = st.chat_input("Nhập câu hỏi của bạn tại đây...")
if user_input:
    try:
        st.session_state["messages"].append({"content": user_input, "role": "user"})
        with st.chat_message("user"):
            st.markdown(user_input)

        llm = st.session_state["llm"]
        db = st.session_state["db"]
        query_text = user_input.strip()
        if not llm:
            st.error("Hệ thống chưa được khởi tạo đúng cách. Vui lòng tải lại trang.")
            logger.error({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": "LLM not initialized",
                "query": query_text,
                "session_id": st.session_state["session_id"]
            })
        else:
            
            if st.session_state["chat_mode"] == "Normal Chat":
                # Direct LLM interaction without RAG
                template = """Answer the question directly without using any context or documents.
                                Question: {question}
                                Answer:"""
                prompt = PromptTemplate.from_template(template)
                llm_chain = prompt | llm

                response_text = llm_chain.invoke({"question": query_text})                  
                # Log the complete interaction including the prompt
                log_interaction(
                    query=query_text,
                    documents=[],
                    response=response_text,
                    prompt=prompt
                )
            else:    
                
                    # Get documents from database
                    
                    response = db.similarity_search(query_text, k=8)
                    if not response:
                        st.error("Không tìm thấy tài liệu nào phù hợp.")
                        logger.warning({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "warning": "No documents found",
                            "query": query_text,
                            "session_id": st.session_state["session_id"]
                        })
                    rdrsegmenter = st.session_state["rdrsegmenter"]
                    
                    # Prepare for reranking with PhoRanker
                    tokenized_query = " ".join(rdrsegmenter.word_segment(query_text))
    
                    # Create pairs for reranking
                    documents = [doc.page_content for doc in response]
                    tokenized_documents = [" ".join(rdrsegmenter.word_segment(doc)) for doc in documents]    
                    tokenized_pairs = [[tokenized_query, doc] for doc in tokenized_documents]

                    rerank = CrossEncoder(MODEL_ID)

                    # Get reranking scores
                    scores = rerank.predict(tokenized_pairs)
                    
                    # Create reranked results with scores
                    reranked_results = list(zip(response, scores))
                    
                    # Sort by score (descending)
                    reranked_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)[:3]
                        
                    # Create context from top documents with deduplication
                    deduped_results = deduplicate_by_embedding(reranked_results, similarity_threshold=0.5)

                    context_text = "\n\n---\n\n".join(
                                                d.page_content.lower().replace('\n', ' ').strip()
                                                for d, _ in deduped_results
                                            )

                    # Enhanced prompt template for Vietnamese responses without repetition
                    prompt_template = ChatPromptTemplate([
                                    ("system", "You are a helpful Vietnamese medical assistant. Answer the questions in Vietnamese and use the information provided."),                                        
                                    ("user", "Context: {context}\n\n Question: {question}") ])
                    prompt = prompt_template.format(context=context_text, question=query_text)
                        
                    # Generate response
                    response_text = llm.invoke(prompt)
                    # Log the complete interaction including the prompt
                    log_interaction(
                            query=query_text,
                            documents=response,
                            response=response_text,
                            prompt=prompt,
                        )
            st.session_state["messages"].append({"content": response_text, "role": "assistant"})
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                for partial_response in process_text_with_linebreaks(response_text):
                    full_response = partial_response  
                    message_placeholder.markdown(full_response + "▍")  
                    time.sleep(0.05)

                message_placeholder.markdown(full_response)  # dòng cuối cùng bỏ dấu typing
    except Exception as e:
        st.error(f"⚠️ {str(e)}")
        logger.error({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e),
            "query": query_text if 'query_text' in locals() else "unknown",
            "session_id": st.session_state.get("session_id", "unknown")
        })



