import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
import requests
import json
from datetime import datetime
import tempfile
import glob
import os
import pickle
import base64
from pathlib import Path
from dotenv import load_dotenv
import warnings
import logging
from typing import Optional, Dict, List, Tuple
import hashlib

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration"""
    DEBUG = False
    VERBOSE = False
    
    # Paths
    DOCUMENTS_PATH = "documents"
    VECTORSTORE_PATH = "vectorstore"
    
    # API endpoints
    GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1"
    GEMINI_MODELS = [
        'models/gemini-1.5-flash-latest',
        'models/gemini-1.5-flash',
        'models/gemini-1.5-pro-latest',
    ]
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 20
    REQUEST_TIMEOUT = 30
    
    # Retrieval settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MIN_CHUNK_LENGTH = 50
    TOP_K_RESULTS = 3
    
    # Contact info (để dễ maintain)
    CONTACT_INFO = {
        'hotline': ['1900 5555 14', '0879 5555 14'],
        'email': 'tuyensinh@hcmulaw.edu.vn',
        'phone': '(028) 39400 989',
        'address': '2 Nguyễn Tất Thành, Phường 12, Quận 4, TP.HCM',
        'website': 'www.hcmulaw.edu.vn',
        'facebook': 'facebook.com/hcmulaw'
    }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "messages": [],
        "first_visit": True,
        "request_count": 0,
        "last_request_time": datetime.now(),
        "error_count": 0,
        "pending_question": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def sanitize_input(text: str, max_length: int = 500) -> str:
    """Sanitize user input"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Limit length
    text = text[:max_length]
    
    # Remove potential injection patterns (basic)
    dangerous_patterns = ['<script', 'javascript:', 'onerror=']
    for pattern in dangerous_patterns:
        text = text.replace(pattern, '')
    
    return text.strip()

def check_rate_limit() -> bool:
    """Simple rate limiting"""
    now = datetime.now()
    time_diff = (now - st.session_state.last_request_time).total_seconds()
    
    if time_diff < 60:  # Within 1 minute
        if st.session_state.request_count >= Config.MAX_REQUESTS_PER_MINUTE:
            return False
    else:
        # Reset counter
        st.session_state.request_count = 0
        st.session_state.last_request_time = now
    
    st.session_state.request_count += 1
    return True

def format_contact_info() -> str:
    """Format contact info consistently"""
    info = Config.CONTACT_INFO
    return f"""
📞 **Hotline:** {' hoặc '.join(info['hotline'])}
📧 **Email:** {info['email']}
☎️ **Điện thoại:** {info['phone']}
🌐 **Website:** {info['website']}
📍 **Địa chỉ:** {info['address']}
"""

# ============================================================================
# EMBEDDINGS & VECTORSTORE
# ============================================================================

@st.cache_resource
def load_embeddings():
    """Load embeddings model with error handling"""
    try:
        return HuggingFaceEmbeddings(
            model_name="keepitreal/vietnamese-sbert",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        st.error(f"❌ Không thể load embeddings: {e}")
        return None

def get_file_hash(file_path: str) -> str:
    """Generate hash for file"""
    try:
        stat = os.stat(file_path)
        content = f"{stat.st_mtime}_{stat.st_size}_{file_path}"
        return hashlib.md5(content.encode()).hexdigest()
    except Exception:
        return ""

def download_from_gdrive(file_id: str, output_path: str) -> bool:
    """Download file from Google Drive"""
    if not file_id:
        return False
    
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        response = session.get(url, stream=True, timeout=30)
        
        # Handle virus scan warning
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'export': 'download', 'id': file_id, 'confirm': value}
                response = session.get(url, params=params, stream=True, timeout=30)
                break
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            return True
        return False
        
    except Exception as e:
        if Config.DEBUG:
            st.warning(f"GDrive download failed: {e}")
        return False

def load_cached_vectorstore() -> Tuple[Optional[object], Dict]:
    """Load vectorstore from Google Drive with proper cleanup"""
    # Get IDs from secrets or env
    try:
        vectorstore_id = st.secrets.get("GDRIVE_VECTORSTORE_ID") or os.getenv("GDRIVE_VECTORSTORE_ID")
        metadata_id = st.secrets.get("GDRIVE_METADATA_ID") or os.getenv("GDRIVE_METADATA_ID")
    except Exception:
        vectorstore_id = os.getenv("GDRIVE_VECTORSTORE_ID")
        metadata_id = os.getenv("GDRIVE_METADATA_ID")
    
    if not vectorstore_id or not metadata_id:
        return None, {}
    
    temp_dir = tempfile.mkdtemp()
    vectorstore_path = os.path.join(temp_dir, "vectorstore.pkl")
    metadata_path = os.path.join(temp_dir, "metadata.json")
    
    try:
        if not download_from_gdrive(vectorstore_id, vectorstore_path):
            return None, {}
        if not download_from_gdrive(metadata_id, metadata_path):
            return None, {}
        
        with open(vectorstore_path, 'rb') as f:
            vectorstore = pickle.load(f)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return vectorstore, metadata
        
    except Exception as e:
        if Config.DEBUG:
            st.warning(f"Failed to load from GDrive: {e}")
        return None, {}
    
    finally:
        # Always cleanup temp files
        try:
            if os.path.exists(vectorstore_path):
                os.remove(vectorstore_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass

def process_documents(file_paths: List[str]) -> Tuple[List, List, List]:
    """Process documents with better error handling"""
    documents = []
    processed = []
    failed = []
    
    for file_path in file_paths:
        try:
            ext = Path(file_path).suffix.lower()
            
            loaders = {
                ".pdf": PyPDFLoader,
                ".docx": Docx2txtLoader,
                ".txt": lambda p: TextLoader(p, encoding='utf-8')
            }
            
            if ext not in loaders:
                failed.append(f"{file_path} (unsupported format)")
                continue
            
            loader = loaders[ext](file_path)
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata['source_file'] = os.path.basename(file_path)
                doc.metadata['processed_time'] = datetime.now().isoformat()
                doc.metadata['file_hash'] = get_file_hash(file_path)
            
            documents.extend(docs)
            processed.append(file_path)
            
        except Exception as e:
            failed.append(f"{os.path.basename(file_path)} ({str(e)[:50]})")
    
    return documents, processed, failed

def create_vector_store(documents: List) -> Optional[object]:
    """Create vector store with validation"""
    if not documents:
        return None
    
    embeddings = load_embeddings()
    if not embeddings:
        return None
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=['\n\n', '\n', '.', '!', '?', ';', ':', ' ']
        )
        
        texts = text_splitter.split_documents(documents)
        texts = [t for t in texts if len(t.page_content.strip()) > Config.MIN_CHUNK_LENGTH]
        
        if not texts:
            return None
        
        return FAISS.from_documents(texts, embeddings)
        
    except Exception as e:
        st.error(f"❌ Lỗi tạo vectorstore: {e}")
        return None

@st.cache_resource
def initialize_vectorstore() -> Tuple[Optional[object], Dict]:
    """Initialize vectorstore with caching"""
    # Try GDrive first
    vectorstore, metadata = load_cached_vectorstore()
    if vectorstore:
        return vectorstore, metadata.get('stats', {})
    
    # Fallback to local files
    document_files = glob.glob(os.path.join(Config.DOCUMENTS_PATH, '**/*.pdf'), recursive=True)
    document_files.extend(glob.glob(os.path.join(Config.DOCUMENTS_PATH, '**/*.docx'), recursive=True))
    document_files.extend(glob.glob(os.path.join(Config.DOCUMENTS_PATH, '**/*.txt'), recursive=True))
    
    if not document_files:
        return None, {}
    
    with st.spinner("🔄 Đang xử lý tài liệu..."):
        documents, processed, failed = process_documents(document_files)
        
        if not documents:
            return None, {}
        
        vectorstore = create_vector_store(documents)
        
        if vectorstore:
            stats = {
                'total_files': len(document_files),
                'processed_files': len(processed),
                'failed_files': len(failed),
                'total_chunks': vectorstore.index.ntotal,
                'last_updated': datetime.now().isoformat()
            }
            return vectorstore, stats
    
    return None, {}

# ============================================================================
# GEMINI API
# ============================================================================

@st.cache_resource
def get_gemini_config() -> Optional[Dict]:
    """Get Gemini API configuration"""
    # Get API key
    try:
        api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("❌ Thiếu GEMINI_API_KEY!")
        st.info("Lấy API key tại: https://aistudio.google.com/app/apikey")
        return None
    
    # Validate API key
    try:
        url = f"{Config.GEMINI_API_BASE}/models?key={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            models_data = response.json()
            available_models = [m['name'] for m in models_data.get('models', [])]
            
            # Find best available model
            selected_model = None
            for model in Config.GEMINI_MODELS:
                if model in available_models:
                    selected_model = model
                    break
            
            if not selected_model and available_models:
                selected_model = available_models[0]
            
            if selected_model:
                return {
                    'api_key': api_key,
                    'model': selected_model,
                    'available_models': available_models
                }
        
        elif response.status_code == 400:
            st.error("❌ API key không hợp lệ!")
        else:
            st.error(f"❌ Lỗi API: {response.status_code}")
        
        return None
        
    except Exception as e:
        st.error(f"❌ Không thể kết nối Gemini: {e}")
        return None

def call_gemini_api(config: Dict, prompt: str) -> str:
    """Call Gemini API with error handling"""
    if not config:
        return "Lỗi: Chưa cấu hình API"
    
    url = f"{Config.GEMINI_API_BASE}/{config['model']}:generateContent?key={config['api_key']}"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 2000,
        }
    }
    
    try:
        response = requests.post(
            url, 
            json=payload, 
            headers={"Content-Type": "application/json"},
            timeout=Config.REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if 'candidates' in data and len(data['candidates']) > 0:
                candidate = data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        return parts[0]['text']
            
            return "Xin lỗi, không nhận được phản hồi hợp lệ."
        
        else:
            error_data = response.json()
            error_msg = error_data.get('error', {}).get('message', 'Unknown error')
            
            if Config.DEBUG:
                return f"Lỗi API: {error_msg}"
            else:
                return "Xin lỗi, hệ thống tạm thời gặp sự cố. Vui lòng thử lại sau."
        
    except requests.exceptions.Timeout:
        return "Lỗi: Hệ thống phản hồi chậm, vui lòng thử lại."
    except Exception as e:
        if Config.DEBUG:
            return f"Lỗi: {str(e)}"
        else:
            return "Xin lỗi, đã có lỗi xảy ra."

# ============================================================================
# QUESTION HANDLING
# ============================================================================

def classify_question(question: str) -> str:
    """Classify question into categories"""
    question_lower = question.lower()
    
    categories = {
        "Tuyển sinh": ["tuyển sinh", "đăng ký", "hồ sơ", "điểm chuẩn", "xét tuyển", "đăng kí"],
        "Học phí": ["học phí", "chi phí", "miễn giảm", "học bổng", "tiền"],
        "Chương trình đào tạo": ["chương trình", "môn học", "tín chỉ", "ngành", "khoa"],
        "Cơ sở vật chất": ["ký túc xá", "ktx", "thư viện", "phòng lab", "cơ sở"],
        "Việc làm": ["việc làm", "thực tập", "cơ hội", "nghề nghiệp"],
    }
    
    for category, keywords in categories.items():
        if any(kw in question_lower for kw in keywords):
            return category
    
    return "Thông tin chung"

def get_category_badge(category: str) -> str:
    """Get HTML badge for category"""
    colors = {
        "Tuyển sinh": "#1e88e5",
        "Học phí": "#43a047",
        "Chương trình đào tạo": "#fb8c00",
        "Cơ sở vật chất": "#8e24aa",
        "Việc làm": "#e53935",
        "Thông tin chung": "#546e7a"
    }
    
    color = colors.get(category, "#546e7a")
    
    return f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 600;
        margin-bottom: 8px;
        display: inline-block;
    ">{category}</span>
    """

def create_prompt_with_context(question: str, context: str, category: str) -> str:
    """Create enhanced prompt with context"""
    return f"""Bạn là chuyên gia tư vấn {category.lower()} của Trường Đại học Luật TP. Hồ Chí Minh.

THÔNG TIN THAM KHẢO:
{context}

THÔNG TIN LIÊN HỆ CHÍNH THỨC:
{format_contact_info()}

CÂU HỎI: {question}

HƯỚNG DẪN TRẢ LỜI:
1. Ưu tiên sử dụng thông tin từ tài liệu tham khảo
2. Trả lời ngắn gọn, súc tích, dễ hiểu
3. Sử dụng thông tin liên hệ chính xác (KHÔNG dùng placeholder)
4. Nếu không chắc chắn, khuyến khích liên hệ trực tiếp
5. Sử dụng emoji phù hợp để dễ đọc

Trả lời bằng tiếng Việt, thân thiện và chuyên nghiệp:"""

def generate_answer(question: str, vectorstore: Optional[object], gemini_config: Dict) -> Tuple[str, str]:
    """Generate answer using RAG or fallback to API only"""
    category = classify_question(question)
    
    try:
        if vectorstore:
            # RAG approach
            retriever = vectorstore.as_retriever(search_kwargs={"k": Config.TOP_K_RESULTS})
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs[:Config.TOP_K_RESULTS]])
            
            prompt = create_prompt_with_context(question, context, category)
        else:
            # Fallback: no vectorstore
            prompt = f"""Bạn là chuyên gia tư vấn của Trường Đại học Luật TP. Hồ Chí Minh.

THÔNG TIN LIÊN HỆ:
{format_contact_info()}

CÂU HỎI: {question}

Hãy trả lời chung về câu hỏi và khuyến khích sinh viên liên hệ để được tư vấn cụ thể.
Trả lời bằng tiếng Việt, ngắn gọn và chuyên nghiệp:"""
        
        answer = call_gemini_api(gemini_config, prompt)
        
        # Post-process: ensure contact info is present for certain queries
        if any(kw in question.lower() for kw in ['liên hệ', 'contact', 'email', 'số điện thoại', 'hotline']):
            if 'hcmulaw.edu.vn' not in answer:
                answer += f"\n\n{format_contact_info()}"
        
        return answer, category
        
    except Exception as e:
        error_msg = f"""
❌ **Xin lỗi, hệ thống tạm thời gặp sự cố**

Vui lòng liên hệ trực tiếp để được hỗ trợ:

{format_contact_info()}
"""
        if Config.DEBUG:
            error_msg += f"\n\n_Debug: {str(e)[:100]}_"
        
        return error_msg, "Lỗi hệ thống"

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render page header"""
    st.markdown("""
    <style>
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin-bottom: 2rem;
    }
    .header-container h1 {
        font-size: 2.2rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .header-container h3 {
        font-size: 1.3rem;
        font-weight: 400;
        margin: 0.5rem 0;
        opacity: 0.95;
    }
    .header-container p {
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    [data-testid="stToolbarActionButton"] {
        display: none !important;
    }
    </style>
    
    <div class="header-container">
        <h1>🤖 Chatbot Tư Vấn Tuyển Sinh</h1>
        <h3>Trường Đại học Luật TP. Hồ Chí Minh</h3>
        <p>💬 Hỗ trợ 24/7 &nbsp;|&nbsp; 🎓 Tư vấn chuyên nghiệp &nbsp;|&nbsp; ⚡ Phản hồi nhanh chóng</p>
    </div>
    """, unsafe_allow_html=True)

def render_quick_questions():
    """Render quick question buttons"""
    st.markdown("### 💡 Câu hỏi thường gặp")
    
    questions = [
        "📝 Thủ tục đăng ký xét tuyển như thế nào?",
        "💰 Học phí một năm là bao nhiêu?",
        "📚 Trường có những ngành học nào?",
        "🏠 Trường có ký túc xá không?",
        "🎓 Cơ hội việc làm sau khi tốt nghiệp?",
        "📞 Thông tin liên hệ của trường?"
    ]
    
    cols = st.columns(2)
    for i, q in enumerate(questions):
        with cols[i % 2]:
            if st.button(q, key=f"quick_q_{i}", use_container_width=True):
                st.session_state.pending_question = q.split(' ', 1)[1]
                st.experimental_rerun()
                # Remove emoji
                # KHÔNG rerun - để xử lý ở phần input bên dưới

def export_chat_history():
    """Export chat history to text file"""
    if not st.session_state.messages:
        return None
    
    # Tạo nội dung text
    content = "=" * 60 + "\n"
    content += "LỊCH SỬ HỘI THOẠI - CHATBOT TƯ VẤN\n"
    content += "Trường Đại học Luật TP. Hồ Chí Minh\n"
    content += f"Xuất lúc: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
    content += "=" * 60 + "\n\n"
    
    for i, msg in enumerate(st.session_state.messages, 1):
        role = "🧑 BẠN" if msg["role"] == "user" else "🤖 CHATBOT"
        content += f"{role}:\n"
        content += f"{msg['content']}\n"
        
        if msg["role"] == "assistant" and "category" in msg:
            content += f"(Danh mục: {msg['category']})\n"
        
        content += "\n" + "-" * 60 + "\n\n"
    
    content += "\n" + "=" * 60 + "\n"
    content += format_contact_info()
    
    return content

def render_sidebar(vectorstore_stats: Dict):
    """Render sidebar with system info"""
    with st.sidebar:
        st.markdown("### ⚙️ Thông tin hệ thống")
        
        # System status
        with st.expander("📊 Trạng thái", expanded=False):
            st.success("✅ Gemini API: Hoạt động")
            
            if vectorstore_stats:
                st.info(f"📁 Tài liệu: {vectorstore_stats.get('processed_files', 0)} files")
                st.info(f"📦 Chunks: {vectorstore_stats.get('total_chunks', 0)}")
            else:
                st.warning("⚠️ Chưa có dữ liệu vectorstore")
        
        # Actions
        st.markdown("### 🔧 Thao tác")
        
        if st.button("🔄 Làm mới", use_container_width=True):
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()
        
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.first_visit = True
            st.rerun()
        
        # Export chat history
        st.markdown("### 💾 Xuất lịch sử")
        
        if st.session_state.messages:
            chat_content = export_chat_history()
            if chat_content:
                st.download_button(
                    label="📥 Tải về (.txt)",
                    data=chat_content,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                # Hiển thị số tin nhắn
                total_messages = len(st.session_state.messages)
                user_messages = sum(1 for m in st.session_state.messages if m["role"] == "user")
                st.caption(f"📊 {total_messages} tin ({user_messages} câu hỏi)")
        else:
            st.info("Chưa có lịch sử chat")
        
        # Contact info
        st.markdown("---")
        st.markdown("### 📞 Liên hệ trực tiếp")
        st.markdown(format_contact_info())
        
        # Footer
        st.markdown("---")
        st.caption("🤖 Chatbot v3.0 | Made with ❤️")

def render_footer():
    """Render page footer"""
    st.markdown("---")
    info = Config.CONTACT_INFO
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0; background: #f8f9fa; border-radius: 12px;">
        <h4>🏛️ Trường Đại học Luật TP. Hồ Chí Minh</h4>
        <p>📍 {info['address']}</p>
        <p>📞 Hotline: {' | '.join(info['hotline'])} | ☎️ {info['phone']}</p>
        <p>📧 {info['email']} | 🌐 {info['website']}</p>
        <p>📘 {info['facebook']}</p>
        <p style="margin-top: 1.5rem; opacity: 0.7; font-size: 0.9em;">
            Phát triển bởi Lvphung - CNTT | Phiên bản 3.0
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application logic"""
    # Page config
    st.set_page_config(
        page_title="Chatbot Tư Vấn - Đại học Luật TPHCM",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load environment variables
    load_dotenv()
    
    # Initialize session state
    init_session_state()
    
    # Render header
    render_header()
    
    # Initialize backend
    with st.spinner("🔄 Đang khởi động hệ thống..."):
        gemini_config = get_gemini_config()
        if not gemini_config:
            st.stop()
        
        vectorstore, stats = initialize_vectorstore()
    
    # Render sidebar
    render_sidebar(stats)
    
    # Show quick questions on first visit
    if not st.session_state.messages and st.session_state.first_visit:
        render_quick_questions()
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            margin-top: 1.5rem;
        ">
            <h4>💡 Hướng dẫn sử dụng:</h4>
            <ul style="margin: 0.5rem 0;">
                <li>🎯 Chọn câu hỏi gợi ý hoặc nhập câu hỏi của bạn</li>
                <li>💬 Đặt câu hỏi cụ thể để nhận được tư vấn chính xác</li>
                <li>📞 Liên hệ trực tiếp nếu cần hỗ trợ khẩn cấp</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "category" in msg:
                st.markdown(get_category_badge(msg["category"]), unsafe_allow_html=True)
            st.markdown(msg["content"])
    
    # Handle input
    user_input = None
    
    # Check for pending question from quick buttons
    if st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None
    else:
        # Get input from chat box
        user_input = st.chat_input("💬 Nhập câu hỏi của bạn...")
    
    # Process user input
    if user_input:
        # Sanitize input
        user_input = sanitize_input(user_input)
        
        if not user_input:
            st.warning("⚠️ Vui lòng nhập câu hỏi hợp lệ")
            return
        
        # Check rate limit
        if not check_rate_limit():
            st.error("⚠️ Bạn đã gửi quá nhiều yêu cầu. Vui lòng đợi 1 phút.")
            return
        
        # Mark as not first visit
        st.session_state.first_visit = False
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Đang suy nghĩ..."):
                try:
                    answer, category = generate_answer(user_input, vectorstore, gemini_config)
                    
                    # Display category badge
                    st.markdown(get_category_badge(category), unsafe_allow_html=True)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "category": category
                    })
                    
                    # Reset error count on success
                    st.session_state.error_count = 0
                    
                except Exception as e:
                    st.session_state.error_count += 1
                    
                    error_message = f"""
❌ **Xin lỗi, đã có lỗi xảy ra**

Vui lòng thử lại hoặc liên hệ trực tiếp:

{format_contact_info()}
"""
                    if Config.DEBUG:
                        error_message += f"\n\n_Debug info: {str(e)[:200]}_"
                    
                    st.error(error_message)
                    
                    # If too many errors, suggest refresh
                    if st.session_state.error_count >= 3:
                        st.warning("⚠️ Hệ thống gặp nhiều lỗi. Bạn có muốn làm mới trang?")
                        if st.button("🔄 Làm mới ngay"):
                            st.cache_resource.clear()
                            st.rerun()
    
    # Render footer
    render_footer()

if __name__ == "__main__":
    main()
