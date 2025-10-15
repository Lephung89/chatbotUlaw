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
from pathlib import Path
from dotenv import load_dotenv
import time
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# ==================== CONFIGURATION ====================
class Config:
    PAGE_TITLE = "Chatbot Tư Vấn - Đại học Luật TPHCM"
    PAGE_ICON = "⚖️"
    DOCUMENTS_PATH = "documents"
    VECTORSTORE_PATH = "vectorstore"
    
    CONTACT_INFO = {
        "hotline": "1900 5555 14 hoặc 0879 5555 14",
        "email": "tuyensinh@hcmulaw.edu.vn",
        "phone": "(028) 39400 989",
        "address": "2 Nguyễn Tất Thành, Phường 12, Quận 4, TP.HCM",
        "website": "www.hcmulaw.edu.vn",
        "facebook": "facebook.com/hcmulaw"
    }
    
    QUICK_QUESTIONS = [
        "📝 Thủ tục đăng ký xét tuyển như thế nào?",
        "💰 Học phí một năm là bao nhiêu?",
        "📚 Trường có những ngành học nào?",
        "🏠 Trường có ký túc xá không?",
        "🎓 Cơ hội việc làm sau khi tốt nghiệp?",
        "📞 Thông tin liên hệ tư vấn chi tiết?"
    ]

# ==================== MODERN CSS STYLING ====================
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    [data-testid="stToolbar"] {display: none;}
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    /* Modern Header */
    .modern-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    
    .modern-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .modern-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .modern-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    /* Quick Questions Grid */
    .quick-questions-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .quick-question-btn {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: left;
        font-size: 0.95rem;
        font-weight: 500;
        color: #374151;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .quick-question-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(30, 60, 114, 0.15);
        border-color: #2a5298;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: white;
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Category Badges */
    .category-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .badge-tuyensinh {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    
    .badge-hocphi {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .badge-chuongtrinh {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .badge-other {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
    }
    
    /* Info Card */
    .info-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-left: 4px solid #2a5298;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .info-card h4 {
        color: #1e3c72;
        margin-top: 0;
        font-size: 1.15rem;
        font-weight: 600;
    }
    
    .info-card ul {
        margin: 0.75rem 0;
        padding-left: 1.5rem;
    }
    
    .info-card li {
        margin: 0.5rem 0;
        color: #4b5563;
        line-height: 1.6;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    [data-testid="stSidebar"] h3 {
        color: white;
        font-weight: 600;
    }
    
    /* Status Cards in Sidebar */
    .status-card {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Chat Input */
    .stChatInputContainer {
        border-top: 2px solid #e5e7eb;
        padding-top: 1rem;
        background: white;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    
    /* Footer */
    .modern-footer {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-top: 3rem;
        text-align: center;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.1);
    }
    
    .modern-footer h4 {
        color: white;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }
    
    .modern-footer p {
        margin: 0.5rem 0;
        opacity: 0.95;
    }
    
    /* Loading Animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,0.3);
        border-radius: 50%;
        border-top-color: #2a5298;
        animation: spin 1s ease-in-out infinite;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .modern-header h1 {
            font-size: 1.75rem;
        }
        
        .quick-questions-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #2a5298 0%, #7e22ce 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #1e3c72 0%, #6b21a8 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================
@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Cache embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="keepitreal/vietnamese-sbert",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive with progress"""
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        response = session.get(url, stream=True)
        
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'export': 'download', 'id': file_id, 'confirm': value}
                response = session.get(url, params=params, stream=True)
                break
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        return False

def get_category_badge(category):
    """Generate category badge HTML"""
    badge_classes = {
        "Tuyển sinh": "badge-tuyensinh",
        "Học phí": "badge-hocphi",
        "Chương trình đào tạo": "badge-chuongtrinh"
    }
    badge_class = badge_classes.get(category, "badge-other")
    return f'<span class="category-badge {badge_class}">{category}</span>'

def classify_question(question):
    """Classify question into categories"""
    question_lower = question.lower()
    categories = {
        "Tuyển sinh": ["tuyển sinh", "đăng ký", "hồ sơ", "điểm chuẩn", "xét tuyển"],
        "Học phí": ["học phí", "chi phí", "miễn giảm", "học bổng"],
        "Chương trình đào tạo": ["chương trình", "môn học", "tín chỉ", "ngành"],
    }
    
    for category, keywords in categories.items():
        if any(kw in question_lower for kw in keywords):
            return category
    return "Khác"

# ==================== GEMINI API ====================
@st.cache_resource(show_spinner=False)
def get_gemini_llm():
    """Initialize Gemini API"""
    load_dotenv()
    
    try:
        api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    except:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("❌ Thiếu GEMINI_API_KEY!")
        st.stop()
    
    test_url = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
    
    try:
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            models_data = response.json()
            available_models = [m['name'] for m in models_data.get('models', [])]
            
            preferred_models = [
                'models/gemini-1.5-flash-latest',
                'models/gemini-1.5-flash',
                'models/gemini-1.5-pro-latest',
            ]
            
            selected_model = next((m for m in preferred_models if m in available_models), 
                                 available_models[0] if available_models else None)
            
            if selected_model:
                return {'api_key': api_key, 'model': selected_model}
            
        st.error(f"❌ Lỗi API: {response.status_code}")
        st.stop()
        
    except Exception as e:
        st.error(f"❌ Lỗi kết nối: {e}")
        st.stop()

def call_gemini_api(llm_config, prompt):
    """Call Gemini API"""
    url = f"https://generativelanguage.googleapis.com/v1/{llm_config['model']}:generateContent?key={llm_config['api_key']}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 2000,
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if 'candidates' in data and data['candidates']:
                return data['candidates'][0]['content']['parts'][0]['text']
        
        return "Xin lỗi, không nhận được phản hồi từ AI."
        
    except Exception as e:
        return f"Lỗi: {str(e)}"

# ==================== VECTORSTORE ====================
@st.cache_resource(show_spinner=False)
def initialize_vectorstore():
    """Initialize vectorstore from GDrive or local"""
    load_dotenv()
    
    try:
        gdrive_vs_id = st.secrets.get("GDRIVE_VECTORSTORE_ID") or os.getenv("GDRIVE_VECTORSTORE_ID")
        gdrive_meta_id = st.secrets.get("GDRIVE_METADATA_ID") or os.getenv("GDRIVE_METADATA_ID")
    except:
        gdrive_vs_id = os.getenv("GDRIVE_VECTORSTORE_ID")
        gdrive_meta_id = os.getenv("GDRIVE_METADATA_ID")
    
    if gdrive_vs_id and gdrive_meta_id:
        temp_dir = tempfile.mkdtemp()
        vs_path = os.path.join(temp_dir, "vectorstore.pkl")
        meta_path = os.path.join(temp_dir, "metadata.json")
        
        try:
            if download_from_gdrive(gdrive_vs_id, vs_path) and download_from_gdrive(gdrive_meta_id, meta_path):
                with open(vs_path, 'rb') as f:
                    vectorstore = pickle.load(f)
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                return vectorstore, metadata.get('stats', {})
        except:
            pass
    
    return None, {}

# ==================== UI COMPONENTS ====================
def render_header():
    """Render modern header"""
    st.markdown(f"""
    <div class="modern-header">
        <h1>🤖 Chatbot Tư Vấn Tuyển Sinh</h1>
        <p>🎓 Trường Đại học Luật TP. Hồ Chí Minh</p>
        <p>💬 Hỗ trợ 24/7 | ⚡ Trả lời tức thì | 🎯 Tư vấn chuyên nghiệp</p>
    </div>
    """, unsafe_allow_html=True)

def render_quick_questions():
    """Render quick questions with modern grid"""
    st.markdown("### 💡 Câu hỏi thường gặp")
    
    cols = st.columns(2)
    for idx, question in enumerate(Config.QUICK_QUESTIONS):
        col = cols[idx % 2]
        with col:
            if st.button(question, key=f"qq_{idx}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

def render_sidebar():
    """Render enhanced sidebar"""
    with st.sidebar:
        st.markdown("### ⚙️ Cài đặt & Trạng thái")
        
        # System Status
        with st.expander("📊 Trạng thái hệ thống", expanded=True):
            st.success("✅ Gemini API: Hoạt động")
            
            doc_count = len(glob.glob(os.path.join(Config.DOCUMENTS_PATH, '**', '*.*'), recursive=True))
            st.info(f"📁 Tài liệu: {doc_count} files")
            
            if os.getenv("GDRIVE_VECTORSTORE_ID"):
                st.info("☁️ Google Drive: Đã kết nối")
        
        # Actions
        st.markdown("---")
        if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        
        if st.button("📥 Xuất lịch sử chat", use_container_width=True):
            if st.session_state.messages:
                chat_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" 
                                        for m in st.session_state.messages])
                st.download_button(
                    "💾 Tải về",
                    chat_text,
                    f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    use_container_width=True
                )
        
        # Contact Info
        st.markdown("---")
        st.markdown(f"""
        ### 📞 Liên hệ
        **Hotline:** {Config.CONTACT_INFO['hotline']}  
        **Email:** {Config.CONTACT_INFO['email']}  
        **Web:** {Config.CONTACT_INFO['website']}
        """)

def render_footer():
    """Render modern footer"""
    st.markdown(f"""
    <div class="modern-footer">
        <h4>🏛️ Trường Đại học Luật TP. Hồ Chí Minh</h4>
        <p>📍 {Config.CONTACT_INFO['address']}</p>
        <p>📞 {Config.CONTACT_INFO['hotline']}</p>
        <p>📧 {Config.CONTACT_INFO['email']}</p>
        <p>🌐 {Config.CONTACT_INFO['website']} | 📘 {Config.CONTACT_INFO['facebook']}</p>
        <p style="margin-top:1.5rem;opacity:0.9;">
            🤖 Enhanced Chatbot v3.0 | Phát triển bởi Lvphung - CNTT
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN APPLICATION ====================
def main():
    # Page Config
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load CSS
    load_custom_css()
    
    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "first_visit" not in st.session_state:
        st.session_state.first_visit = True
    
    # Render UI
    render_header()
    render_sidebar()
    
    # Initialize Components
    with st.spinner("🚀 Đang khởi động hệ thống..."):
        embeddings = load_embeddings()
        vectorstore, stats = initialize_vectorstore()
        llm = get_gemini_llm()
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None
    
    # Show Quick Questions on First Visit
    if not st.session_state.messages and st.session_state.first_visit:
        render_quick_questions()
        
        st.markdown("""
        <div class="info-card">
            <h4>💡 Hướng dẫn sử dụng</h4>
            <ul>
                <li>🎯 Chọn câu hỏi gợi ý hoặc nhập câu hỏi của bạn</li>
                <li>💬 Đặt câu hỏi cụ thể để được tư vấn chính xác hơn</li>
                <li>📞 Liên hệ trực tiếp nếu cần hỗ trợ khẩn cấp</li>
                <li>📥 Xuất lịch sử chat từ sidebar nếu cần lưu trữ</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "category" in msg:
                st.markdown(get_category_badge(msg["category"]), unsafe_allow_html=True)
            st.markdown(msg["content"])
    
    # Handle User Input
    prompt = st.chat_input("💬 Hãy đặt câu hỏi của bạn...")
    
    if "pending_question" in st.session_state and st.session_state.pending_question:
        prompt = st.session_state.pending_question
        st.session_state.pending_question = None
    
    if prompt:
        st.session_state.first_visit = False
        
        # Display User Message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate Response
        category = classify_question(prompt)
        
        with st.chat_message("assistant"):
            st.markdown(get_category_badge(category), unsafe_allow_html=True)
            
            with st.spinner("🤔 Đang suy nghĩ..."):
                try:
                    if retriever:
                        docs = retriever.invoke(prompt)
                        context = "\n\n".join([doc.page_content for doc in docs[:3]])
                        
                        full_prompt = f"""
Bạn là chuyên gia tư vấn của Đại học Luật TPHCM.

THÔNG TIN THAM KHẢO:
{context}

THÔNG TIN LIÊN HỆ:
- Hotline: {Config.CONTACT_INFO['hotline']}
- Email: {Config.CONTACT_INFO['email']}
- Website: {Config.CONTACT_INFO['website']}

Câu hỏi: {prompt}

Hãy trả lời thân thiện, chuyên nghiệp dựa trên thông tin tham khảo. Nếu không chắc chắn, khuyến khích liên hệ trực tiếp.
"""
                        answer = call_gemini_api(llm, full_prompt)
                    else:
                        answer = f"""
Xin lỗi, hệ thống tạm thời không có dữ liệu. Vui lòng liên hệ:

📞 **Hotline:** {Config.CONTACT_INFO['hotline']}
📧 **Email:** {Config.CONTACT_INFO['email']}
🌐 **Website:** {Config.CONTACT_INFO['website']}
"""
                    
                    st.markdown(answer)
                    
                except Exception as e:
                    answer = f"""
❌ **Lỗi hệ thống**

Vui lòng liên hệ trực tiếp:
📞 {Config.CONTACT_INFO['hotline']}
📧 {Config.CONTACT_INFO['email']}
"""
                    st.error(answer)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "category": category
            })
            st.rerun()
    
    # Footer
    render_footer()

if __name__ == "__main__":
    main()
