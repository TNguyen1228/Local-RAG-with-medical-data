import streamlit as st
from streamlit_chat import message
import torch
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
import os
import logging
import json
from query_data import CHROMA_PATH, PROMPT_TEMPLATE, MODEL_PATH, EMBEDDING, load_processed_documents, get_unique_metadata
import asyncio
from datetime import datetime

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

torch.classes.__path__ = [] 

# Set up logging to JSON file
LOG_FILE = "query_document_response_logs.json"

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create a custom formatter that produces well-formatted JSON
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            if isinstance(record.msg, dict):
                return json.dumps(record.msg, ensure_ascii=False, indent=2)
            return json.dumps({"message": record.getMessage()}, ensure_ascii=False, indent=2)
    
    handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger

logger = setup_logger()

def log_interaction(query, documents, response, prompt=None, filtered_docs=None, threshold=None):
    """
    Log the complete interaction: query, documents, prompt, and response
    
    Args:
        query (str): The user's query
        documents (list): List of (document, score) tuples retrieved from the database
        response (str): The system's response to the query
        prompt (str, optional): The prompt sent to the LLM
        filtered_docs (list, optional): Documents that passed the threshold
        threshold (float, optional): The similarity threshold used
    """
    # Create a simplified view of the documents for logging
    docs_for_log = []
    for doc, score in documents:
        docs_for_log.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)  # Convert to float for JSON serialization
        })
    
    # Create filtered docs log if provided
    filtered_docs_log = None
    if filtered_docs:
        filtered_docs_log = []
        for doc, score in filtered_docs:
            filtered_docs_log.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
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
    
    # Add filtering information if available
    if threshold is not None:
        log_entry["threshold"] = threshold
    if filtered_docs_log is not None:
        log_entry["filtered_documents"] = filtered_docs_log
    
    # Log the entry
    logger.info(log_entry)

def document_to_dict(doc):
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

db_path = "./chroma_db_v3"
# Define similarity score threshold
SIMILARITY_THRESHOLD = 0.7

st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("AI Medical Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "db" not in st.session_state:
    st.session_state["db"] = None
if "llm" not in st.session_state:
    st.session_state["llm"] = None
if "processed_documents" not in st.session_state:
    st.session_state["processed_documents"] = None
if "session_id" not in st.session_state:
    st.session_state["session_id"] = datetime.now().strftime("%Y%m%d%H%M%S")

# Log system initialization
logger.info({
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "event": "system_initialization",
    "session_id": st.session_state["session_id"]
})

# Display chat history
for i, msg in enumerate(st.session_state["messages"]):
    message(msg["content"], is_user=msg["is_user"], key=str(i))

# Initialize vector database
if st.session_state["db"] is None:
    try:
        if not os.path.exists(db_path):
            st.error(f"⚠️ Database path does not exist: {db_path}")
            logger.error({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": f"Database path does not exist: {db_path}",
                "session_id": st.session_state["session_id"]
            })
        else:
            st.session_state["db"] = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING)
            st.session_state["processed_documents"] = load_processed_documents('./data/processed_documents.pkl')
            
            logger.info({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "event": "database_loaded",
                "documents_count": len(st.session_state["processed_documents"]),
                "session_id": st.session_state["session_id"]
            })
            
            if not os.path.exists(MODEL_PATH):
                st.error(f"⚠️ Model file not found: {MODEL_PATH}")
                logger.error({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "error": f"Model file not found: {MODEL_PATH}",
                    "session_id": st.session_state["session_id"]
                })
            else:
                try:
                    st.session_state["llm"] = LlamaCpp(
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
                    logger.info({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "event": "model_loaded",
                        "model_path": MODEL_PATH,
                        "session_id": st.session_state["session_id"]
                    })
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

# User input
user_input = st.chat_input("Nhập câu hỏi của bạn:")
if user_input:
    try:
        st.session_state["messages"].append({"content": user_input, "is_user": True})
        message(user_input, is_user=True)

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
            with st.spinner("Đang xử lý câu hỏi của bạn..."):
                try:
                    # Get documents from database
                    response = db.similarity_search_with_relevance_scores(query_text, k=5)
                    
                    # Filter documents based on similarity threshold
                    filtered_response = [(doc, score) for doc, score in response if score > SIMILARITY_THRESHOLD]
                    
                    if not filtered_response:
                        formatted_response = "Không tìm thấy kết quả phù hợp. Vui lòng thử lại với câu hỏi khác."
                        
                        # Log the interaction with no filtered documents
                        log_interaction(
                            query=query_text,
                            documents=response,
                            response=formatted_response,
                            filtered_docs=[],
                            threshold=SIMILARITY_THRESHOLD
                        )
                    else:
                        # Get metadata only from documents that passed the threshold
                        metadata = get_unique_metadata(filtered_response)
                        filtered_documents = [
                            doc for doc in st.session_state["processed_documents"] 
                            if (doc.metadata.get('original_document_id'), doc.metadata.get('part')) in metadata
                        ]
                        
                        # Create context from filtered documents
                        context_text = "\n---\n".join(doc.page_content.lower().replace('\n', '. ').strip() for doc in filtered_documents)
                        prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
                        prompt = prompt_template.format(context=context_text, question=query_text)
                        
                        # Generate response
                        response_text = llm.invoke(prompt)
                        formatted_response = f"{response_text}"
                        
                        # Log the complete interaction including the prompt
                        log_interaction(
                            query=query_text,
                            documents=response,
                            response=formatted_response,
                            prompt=prompt,  # Include the full prompt
                            filtered_docs=filtered_response,
                            threshold=SIMILARITY_THRESHOLD
                        )

                    st.session_state["messages"].append({"content": formatted_response, "is_user": False})
                    message(formatted_response, is_user=False)
                except Exception as e:
                    st.error(f"⚠️ {str(e)}")
                    # Log error
                    logger.error({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "error": str(e),
                        "query": query_text,
                        "session_id": st.session_state.get("session_id", "unknown")
                    })
    except Exception as e:
        st.error(f"⚠️ {str(e)}")
        logger.error({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e),
            "query": query_text if 'query_text' in locals() else "unknown",
            "session_id": st.session_state.get("session_id", "unknown")
        })