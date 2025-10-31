import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
import requests
import json
from datetime import datetime, timedelta
import tempfile
import glob
import os
import pickle
import base64
from pathlib import Path
from dotenv import load_dotenv
import warnings
import logging
from typing import Optional, Dict, List, Tuple, Any
import hashlib
import time
from collections import defaultdict, Counter
import re
from dataclasses import dataclass, asdict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# ============================================================================
# ENHANCED DATA STRUCTURES
# ============================================================================

@dataclass
class ConversationTurn:
    """Structured conversation turn"""
    role: str
    content: str
    timestamp: str
    category: str = ""
    confidence: float = 0.0
    sources: List[str] = None
    feedback: Optional[int] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class RetrievalResult:
    """Enhanced retrieval result with metadata"""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]

@dataclass
class AnswerQuality:
    """Answer quality metrics"""
    confidence: float
    has_sources: bool
    context_relevance: float
    answer_length: int
    
# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================

class Config:
    """Enhanced centralized configuration"""
    DEBUG = False
    VERBOSE = False
    VERSION = "4.0"
    
    # Paths
    DOCUMENTS_PATH = "documents"
    VECTORSTORE_PATH = "vectorstore"
    ANALYTICS_PATH = "analytics"
    
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
    TOP_K_RESULTS = 5  # Increased for re-ranking
    RERANK_TOP_K = 3    # Final number after re-ranking
    
    # Memory settings
    MEMORY_WINDOW = 5  # Remember last 5 conversation turns
    
    # Cache settings
    CACHE_TTL_SECONDS = 3600  # 1 hour
    SIMILARITY_THRESHOLD = 0.95  # For query caching
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.8
    MEDIUM_CONFIDENCE = 0.5
    LOW_CONFIDENCE = 0.3
    
    # Language support
    SUPPORTED_LANGUAGES = ['vi', 'en']
    DEFAULT_LANGUAGE = 'vi'
    
    # Contact info
    CONTACT_INFO = {
        'hotline': ['1900 5555 14', '0879 5555 14'],
        'email': 'tuyensinh@hcmulaw.edu.vn',
        'phone': '(028) 39400 989',
        'address': '2 Nguyễn Tất Thành, Phường 12, Quận 4, TP.HCM',
        'website': 'www.hcmulaw.edu.vn',
        'facebook': 'facebook.com/hcmulaw'
    }
    
    # Categories with enhanced mapping
    CATEGORIES = {
        "vi": {
            "Tuyển sinh": ["tuyển sinh", "đăng ký", "hồ sơ", "điểm chuẩn", "xét tuyển", "đăng kí", "nhập học"],
            "Học phí": ["học phí", "chi phí", "miễn giảm", "học bổng", "tiền", "thanh toán"],
            "Chương trình đào tạo": ["chương trình", "môn học", "tín chỉ", "ngành", "khoa", "đào tạo", "giảng dạy"],
            "Cơ sở vật chất": ["ký túc xá", "ktx", "thư viện", "phòng lab", "cơ sở", "thiết bị", "giảng đường"],
            "Việc làm": ["việc làm", "thực tập", "cơ hội", "nghề nghiệp", "tuyển dụng", "job"],
            "Hoạt động sinh viên": ["câu lạc bộ", "sự kiện", "tình nguyện", "hoạt động", "đoàn", "hội"],
        },
        "en": {
            "Admission": ["admission", "enroll", "register", "application", "entry"],
            "Tuition": ["tuition", "fee", "cost", "scholarship", "payment"],
            "Program": ["program", "course", "curriculum", "major", "department"],
            "Facilities": ["dormitory", "library", "lab", "facilities", "campus"],
            "Career": ["career", "job", "internship", "employment", "work"],
            "Activities": ["club", "event", "volunteer", "activity", "organization"],
        }
    }

# ============================================================================
# ENHANCED UTILITY FUNCTIONS
# ============================================================================

def init_session_state():
    """Initialize enhanced session state"""
    defaults = {
        "messages": [],
        "conversation_memory": ConversationBufferWindowMemory(
            k=Config.MEMORY_WINDOW,
            return_messages=True
        ),
        "first_visit": True,
        "request_count": 0,
        "last_request_time": datetime.now(),
        "error_count": 0,
        "pending_question": None,
        "processing": False,
        "language": Config.DEFAULT_LANGUAGE,
        "query_cache": {},  # Query similarity cache
        "analytics": defaultdict(int),
        "feedback_data": [],
        "conversation_id": generate_conversation_id(),
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def generate_conversation_id() -> str:
    """Generate unique conversation ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    return f"conv_{timestamp}_{random_hash}"

def sanitize_input(text: str, max_length: int = 500) -> str:
    """Enhanced input sanitization"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Limit length
    text = text[:max_length]
    
    # Remove dangerous patterns
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'onerror\s*=',
        r'onclick\s*=',
        r'<iframe',
        r'eval\(',
    ]
    
    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()

def check_rate_limit() -> Tuple[bool, str]:
    """Enhanced rate limiting with message"""
    now = datetime.now()
    time_diff = (now - st.session_state.last_request_time).total_seconds()
    
    if time_diff < 60:
        if st.session_state.request_count >= Config.MAX_REQUESTS_PER_MINUTE:
            wait_time = int(60 - time_diff)
            return False, f"⏳ Vui lòng đợi {wait_time} giây"
    else:
        st.session_state.request_count = 0
        st.session_state.last_request_time = now
    
    st.session_state.request_count += 1
    return True, ""

def format_contact_info(language: str = 'vi') -> str:
    """Format contact info with language support"""
    info = Config.CONTACT_INFO
    
    if language == 'vi':
        return f"""
📞 **Hotline:** {' hoặc '.join(info['hotline'])}
📧 **Email:** {info['email']}
☎️ **Điện thoại:** {info['phone']}
🌐 **Website:** {info['website']}
📍 **Địa chỉ:** {info['address']}
"""
    else:
        return f"""
📞 **Hotline:** {' or '.join(info['hotline'])}
📧 **Email:** {info['email']}
☎️ **Phone:** {info['phone']}
🌐 **Website:** {info['website']}
📍 **Address:** {info['address']}
"""

# ============================================================================
# ENHANCED EMBEDDINGS & VECTORSTORE
# ============================================================================

@st.cache_resource
def load_embeddings():
    """Load embeddings with error handling"""
    try:
        return HuggingFaceEmbeddings(
            model_name="keepitreal/vietnamese-sbert",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        st.error(f"❌ Không thể load embeddings: {e}")
        return None

def calculate_query_similarity(query1: str, query2: str) -> float:
    """Calculate similarity between two queries"""
    # Simple Jaccard similarity on words
    words1 = set(query1.lower().split())
    words2 = set(query2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def check_query_cache(query: str) -> Optional[Tuple[str, str, float]]:
    """Check if similar query exists in cache"""
    cache = st.session_state.query_cache
    
    for cached_query, (answer, category, confidence, timestamp) in cache.items():
        # Check if cache is still valid
        cache_age = (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds()
        if cache_age > Config.CACHE_TTL_SECONDS:
            continue
        
        # Check similarity
        similarity = calculate_query_similarity(query, cached_query)
        if similarity >= Config.SIMILARITY_THRESHOLD:
            return answer, category, confidence
    
    return None

def update_query_cache(query: str, answer: str, category: str, confidence: float):
    """Update query cache with TTL"""
    timestamp = datetime.now().isoformat()
    st.session_state.query_cache[query] = (answer, category, confidence, timestamp)
    
    # Clean old cache entries
    cache = st.session_state.query_cache
    current_time = datetime.now()
    
    expired_keys = []
    for key, (_, _, _, ts) in cache.items():
        cache_age = (current_time - datetime.fromisoformat(ts)).total_seconds()
        if cache_age > Config.CACHE_TTL_SECONDS:
            expired_keys.append(key)
    
    for key in expired_keys:
        del cache[key]

def hybrid_search(query: str, vectorstore: object, top_k: int = 5) -> List[RetrievalResult]:
    """Hybrid search: Semantic + Keyword matching"""
    results = []
    
    try:
        # 1. Semantic search (vector similarity)
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k * 2})
        semantic_docs = retriever.invoke(query)
        
        # 2. Keyword matching (BM25-like scoring)
        query_terms = set(query.lower().split())
        
        for doc in semantic_docs:
            content = doc.page_content.lower()
            
            # Calculate keyword match score
            content_terms = set(content.split())
            matching_terms = query_terms.intersection(content_terms)
            keyword_score = len(matching_terms) / len(query_terms) if query_terms else 0
            
            # Combine scores (70% semantic, 30% keyword)
            # Note: We don't have direct access to semantic score, so we use position as proxy
            position_score = 1.0 - (semantic_docs.index(doc) / len(semantic_docs))
            combined_score = 0.7 * position_score + 0.3 * keyword_score
            
            results.append(RetrievalResult(
                content=doc.page_content,
                source=doc.metadata.get('source_file', 'Unknown'),
                score=combined_score,
                metadata=doc.metadata
            ))
        
        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]
        
    except Exception as e:
        if Config.DEBUG:
            st.error(f"Hybrid search error: {e}")
        return []

def rerank_results(query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
    """Simple re-ranking based on additional heuristics"""
    query_lower = query.lower()
    
    for result in results:
        content_lower = result.content.lower()
        
        # Boost score if query terms appear early in content
        first_occurrence = float('inf')
        for term in query_lower.split():
            if term in content_lower:
                pos = content_lower.find(term)
                first_occurrence = min(first_occurrence, pos)
        
        # Normalize position score (earlier = better)
        if first_occurrence != float('inf'):
            position_boost = 1.0 - (first_occurrence / len(content_lower))
            result.score = result.score * 0.8 + position_boost * 0.2
        
        # Boost if content has exact phrase match
        if query_lower in content_lower:
            result.score *= 1.2
    
    # Re-sort
    results.sort(key=lambda x: x.score, reverse=True)
    return results

def calculate_confidence_score(results: List[RetrievalResult], answer: str) -> float:
    """Calculate confidence score for the answer"""
    if not results:
        return Config.LOW_CONFIDENCE
    
    # Factor 1: Top retrieval score
    top_score = results[0].score if results else 0
    
    # Factor 2: Score consistency (how close are top results)
    if len(results) >= 2:
        score_variance = sum(abs(r.score - top_score) for r in results[:3]) / 3
        consistency = 1.0 - min(score_variance, 1.0)
    else:
        consistency = 0.5
    
    # Factor 3: Answer length (reasonable length is good)
    length_score = min(len(answer.split()) / 100, 1.0)  # Optimal around 100 words
    
    # Combine factors
    confidence = (top_score * 0.5 + consistency * 0.3 + length_score * 0.2)
    
    return min(confidence, 1.0)

# ============================================================================
# ENHANCED GEMINI API
# ============================================================================

@st.cache_resource
def get_gemini_config() -> Optional[Dict]:
    """Get Gemini API configuration"""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("❌ Thiếu GEMINI_API_KEY!")
        return None
    
    try:
        url = f"{Config.GEMINI_API_BASE}/models?key={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            models_data = response.json()
            available_models = [m['name'] for m in models_data.get('models', [])]
            
            selected_model = None
            for model in Config.GEMINI_MODELS:
                if model in available_models:
                    selected_model = model
                    break
            
            if selected_model:
                return {
                    'api_key': api_key,
                    'model': selected_model,
                    'available_models': available_models
                }
        
        return None
        
    except Exception as e:
        st.error(f"❌ Không thể kết nối Gemini: {e}")
        return None

def call_gemini_api(config: Dict, prompt: str, temperature: float = 0.3) -> str:
    """Enhanced Gemini API call with retry"""
    if not config:
        return "Lỗi: Chưa cấu hình API"
    
    url = f"{Config.GEMINI_API_BASE}/{config['model']}:generateContent?key={config['api_key']}"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": temperature,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 2048,
        }
    }
    
    max_retries = 2
    for attempt in range(max_retries):
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
            
            elif response.status_code == 429:  # Rate limit
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return "Hệ thống đang quá tải. Vui lòng thử lại sau."
            
            else:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                return f"Lỗi API: {error_msg}" if Config.DEBUG else "Xin lỗi, hệ thống tạm thời gặp sự cố."
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                continue
            return "Lỗi: Hệ thống phản hồi chậm, vui lòng thử lại."
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            return f"Lỗi: {str(e)}" if Config.DEBUG else "Xin lỗi, đã có lỗi xảy ra."
    
    return "Xin lỗi, không thể xử lý yêu cầu."

# ============================================================================
# ENHANCED QUESTION HANDLING
# ============================================================================

def classify_question(question: str, language: str = 'vi') -> str:
    """Enhanced question classification with language support"""
    question_lower = question.lower()
    categories = Config.CATEGORIES.get(language, Config.CATEGORIES['vi'])
    
    # Score each category
    category_scores = {}
    for category, keywords in categories.items():
        score = sum(1 for kw in keywords if kw in question_lower)
        if score > 0:
            category_scores[category] = score
    
    if category_scores:
        return max(category_scores.items(), key=lambda x: x[1])[0]
    
    return "Thông tin chung" if language == 'vi' else "General Information"

def get_conversation_context() -> str:
    """Get context from conversation memory"""
    memory = st.session_state.conversation_memory
    
    try:
        history = memory.load_memory_variables({})
        messages = history.get('history', [])
        
        if not messages:
            return ""
        
        # Format last few turns
        context_parts = []
        for msg in messages[-Config.MEMORY_WINDOW:]:
            role = "User" if msg.type == "human" else "Assistant"
            context_parts.append(f"{role}: {msg.content[:200]}")  # Limit length
        
        return "\n".join(context_parts)
    except:
        return ""

def create_enhanced_prompt(question: str, results: List[RetrievalResult], 
                          category: str, language: str = 'vi') -> str:
    """Create enhanced prompt with memory and sources"""
    
    # Get conversation context
    conv_context = get_conversation_context()
    conv_section = f"\nLỊCH SỬ HỘI THOẠI GẦN ĐÂY:\n{conv_context}\n" if conv_context else ""
    
    # Format retrieval results with sources
    context_parts = []
    for i, result in enumerate(results[:Config.RERANK_TOP_K], 1):
        context_parts.append(f"[Nguồn {i}: {result.source}]\n{result.content}\n")
    
    context = "\n".join(context_parts)
    
    if language == 'vi':
        return f"""Bạn là chuyên gia tư vấn {category.lower()} của Trường Đại học Luật TP. Hồ Chí Minh.
{conv_section}
THÔNG TIN THAM KHẢO:
{context}

THÔNG TIN LIÊN HỆ CHÍNH THỨC:
{format_contact_info('vi')}

CÂU HỎI HIỆN TẠI: {question}

HƯỚNG DẪN TRẢ LỜI:
1. Ưu tiên sử dụng thông tin từ tài liệu tham khảo
2. Nếu câu hỏi liên quan đến lịch sử hội thoại, tham khảo ngữ cảnh trước đó
3. Trả lời ngắn gọn, súc tích, dễ hiểu (tối đa 150 từ)
4. Trích dẫn nguồn khi sử dụng thông tin cụ thể (ví dụ: "Theo tài liệu...")
5. Sử dụng thông tin liên hệ chính xác
6. Nếu không chắc chắn, khuyến khích liên hệ trực tiếp
7. Sử dụng emoji phù hợp để dễ đọc

Trả lời bằng tiếng Việt, thân thiện và chuyên nghiệp:"""
    else:
        return f"""You are an admissions consultant for HCMC University of Law.
{conv_section}
REFERENCE INFORMATION:
{context}

OFFICIAL CONTACT:
{format_contact_info('en')}

CURRENT QUESTION: {question}

RESPONSE GUIDELINES:
1. Prioritize information from reference documents
2. If question relates to conversation history, consider previous context
3. Keep responses concise and clear (max 150 words)
4. Cite sources when using specific information
5. Use accurate contact information
6. If uncertain, encourage direct contact
7. Use appropriate emojis for readability

Respond in English, friendly and professional:"""

def generate_enhanced_answer(question: str, vectorstore: Optional[object], 
                            gemini_config: Dict) -> Tuple[str, str, float, List[str]]:
    """Enhanced answer generation with confidence and sources"""
    
    # Check cache first
    cached = check_query_cache(question)
    if cached:
        answer, category, confidence = cached
        return answer, category, confidence, ["Cache"]
    
    language = st.session_state.language
    category = classify_question(question, language)
    sources = []
    
    try:
        if vectorstore:
            # Enhanced RAG with hybrid search and re-ranking
            retrieval_results = hybrid_search(question, vectorstore, Config.TOP_K_RESULTS)
            reranked_results = rerank_results(question, retrieval_results)
            
            # Extract sources
            sources = [r.source for r in reranked_results[:Config.RERANK_TOP_K]]
            
            prompt = create_enhanced_prompt(question, reranked_results, category, language)
        else:
            # Fallback
            prompt = f"""Bạn là chuyên gia tư vấn của Trường Đại học Luật TP. Hồ Chí Minh.

THÔNG TIN LIÊN HỆ:
{format_contact_info(language)}

CÂU HỎI: {question}

Trả lời chung và khuyến khích liên hệ để được tư vấn cụ thể.
Trả lời bằng {'tiếng Việt' if language == 'vi' else 'English'}, ngắn gọn:"""
            reranked_results = []
        
        # Generate answer
        answer = call_gemini_api(gemini_config, prompt)
        
        # Calculate confidence
        confidence = calculate_confidence_score(reranked_results, answer)
        
        # Post-process
        if any(kw in question.lower() for kw in ['liên hệ', 'contact', 'email', 'hotline']):
            if 'hcmulaw.edu.vn' not in answer:
                answer += f"\n\n{format_contact_info(language)}"
        
        # Update memory
        st.session_state.conversation_memory.save_context(
            {"input": question},
            {"output": answer}
        )
        
        # Update cache
        update_query_cache(question, answer, category, confidence)
        
        # Update analytics
        st.session_state.analytics[category] += 1
        st.session_state.analytics['total_queries'] += 1
        
        return answer, category, confidence, sources
        
    except Exception as e:
        error_msg = f"""
❌ **Xin lỗi, hệ thống tạm thời gặp sự cố**

Vui lòng liên hệ trực tiếp:

{format_contact_info(language)}
"""
        if Config.DEBUG:
            error_msg += f"\n\n_Debug: {str(e)[:100]}_"
        
        return error_msg, "Lỗi hệ thống", 0.0, []

# ============================================================================
# ENHANCED UI COMPONENTS
# ============================================================================

def render_header():
    """Enhanced header with version info"""
    st.markdown("""
    <style>
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    .version-badge {
        background: rgba(255,255,255,0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        display: inline-block;
        margin-top: 0.5rem;
    }
    </style>
    
    <div class="header-container">
        <h1>🤖 AI Chatbot Tư Vấn - Advanced</h1>
        <h3>Trường Đại học Luật TP. Hồ Chí Minh</h3>
        <p>💬 Hỗ trợ 24/7 | 🎓 AI-Powered | ⚡ Smart Response</p>
        <div class="version-badge">v{Config.VERSION} - Multi-turn Memory | Hybrid Search | Re-ranking</div>
    </div>
    """, unsafe_allow_html=True)

def get_confidence_badge(confidence: float) -> str:
    """Get confidence badge HTML"""
    if confidence >= Config.HIGH_CONFIDENCE:
        color = "#4caf50"
        label = "Độ tin cậy cao"
        icon = "✅"
    elif confidence >= Config.MEDIUM_CONFIDENCE:
        color = "#ff9800"
        label = "Độ tin cậy trung bình"
        icon = "⚠️"
    else:
        color = "#f44336"
        label = "Độ tin cậy thấp"
        icon = "❌"
    
    return f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 4px 10px;
        border-radius: 10px;
        font-size: 0.75em;
        font-weight: 600;
        margin-left: 8px;
        display: inline-block;
    ">{icon} {label} ({confidence:.0%})</span>
    """

def get_category_badge(category: str) -> str:
    """Get category badge HTML"""
    colors = {
        "Tuyển sinh": "#1e88e5",
        "Admission": "#1e88e5",
        "Học phí": "#43a047",
        "Tuition": "#43a047",
        "Chương trình đào tạo": "#fb8c00",
        "Program": "#fb8c00",
        "Cơ sở vật chất": "#8e24aa",
        "Facilities": "#8e24aa",
        "Việc làm": "#e53935",
        "Career": "#e53935",
        "Hoạt động sinh viên": "#00897b",
        "Activities": "#00897b",
        "Thông tin chung": "#546e7a",
        "General Information": "#546e7a",
        "Lỗi hệ thống": "#757575"
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

def render_sources(sources: List[str]):
    """Render source attribution"""
    if sources and sources != ["Cache"]:
        st.markdown("**📚 Nguồn tham khảo:**")
        for i, source in enumerate(sources, 1):
            st.caption(f"{i}. {source}")

def render_feedback_buttons(message_index: int):
    """Render feedback buttons for a message"""
    col1, col2, col3 = st.columns([1, 1, 8])
    
    with col1:
        if st.button("👍", key=f"like_{message_index}"):
            record_feedback(message_index, 1)
            st.success("Cảm ơn phản hồi!")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("👎", key=f"dislike_{message_index}"):
            record_feedback(message_index, -1)
            st.warning("Chúng tôi sẽ cải thiện!")
            time.sleep(1)
            st.rerun()

def record_feedback(message_index: int, feedback: int):
    """Record user feedback"""
    if message_index < len(st.session_state.messages):
        msg = st.session_state.messages[message_index]
        msg['feedback'] = feedback
        
        # Store for analytics
        feedback_record = {
            'timestamp': datetime.now().isoformat(),
            'conversation_id': st.session_state.conversation_id,
            'category': msg.get('category', ''),
            'confidence': msg.get('confidence', 0),
            'feedback': feedback,
            'question': st.session_state.messages[message_index-1]['content'] if message_index > 0 else ''
        }
        st.session_state.feedback_data.append(feedback_record)
        
        # Update analytics
        feedback_key = 'positive_feedback' if feedback > 0 else 'negative_feedback'
        st.session_state.analytics[feedback_key] += 1

def render_quick_questions():
    """Render enhanced quick questions"""
    language = st.session_state.language
    
    if language == 'vi':
        st.markdown("### 💡 Câu hỏi thường gặp")
        questions = [
            "📝 Thủ tục đăng ký xét tuyển như thế nào?",
            "💰 Học phí một năm là bao nhiêu?",
            "📚 Trường có những ngành học nào?",
            "🏠 Trường có ký túc xá không?",
            "🎓 Cơ hội việc làm sau khi tốt nghiệp?",
            "📞 Thông tin liên hệ của trường?",
            "🎪 Hoạt động sinh viên có gì thú vị?",
            "📖 Chương trình đào tạo như thế nào?"
        ]
    else:
        st.markdown("### 💡 Frequently Asked Questions")
        questions = [
            "📝 How to apply for admission?",
            "💰 What is the annual tuition fee?",
            "📚 What programs does the university offer?",
            "🏠 Is there a dormitory available?",
            "🎓 What are the career opportunities after graduation?",
            "📞 What is the contact information?",
            "🎪 What student activities are available?",
            "📖 What is the curriculum like?"
        ]
    
    cols = st.columns(2)
    for i, q in enumerate(questions):
        with cols[i % 2]:
            clean_question = ' '.join(q.split()[1:])
            if st.button(q, key=f"quick_q_{i}", use_container_width=True):
                st.session_state.pending_question = clean_question
                st.session_state.first_visit = False
                st.rerun()

def export_chat_history(format_type: str = 'txt') -> Optional[bytes]:
    """Export chat history in multiple formats"""
    if not st.session_state.messages:
        return None
    
    if format_type == 'txt':
        return export_as_txt()
    elif format_type == 'json':
        return export_as_json()
    elif format_type == 'csv':
        return export_as_csv()
    
    return None

def export_as_txt() -> str:
    """Export as text"""
    content = "=" * 70 + "\n"
    content += "LỊCH SỬ HỘI THOẠI - AI CHATBOT TƯ VẤN v4.0\n"
    content += "Trường Đại học Luật TP. Hồ Chí Minh\n"
    content += f"Conversation ID: {st.session_state.conversation_id}\n"
    content += f"Xuất lúc: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
    content += "=" * 70 + "\n\n"
    
    for i, msg in enumerate(st.session_state.messages, 1):
        role = "🧑 NGƯỜI DÙNG" if msg["role"] == "user" else "🤖 CHATBOT"
        content += f"{role}:\n"
        content += f"{msg['content']}\n"
        
        if msg["role"] == "assistant":
            if "category" in msg:
                content += f"📁 Danh mục: {msg['category']}\n"
            if "confidence" in msg:
                content += f"✅ Độ tin cậy: {msg['confidence']:.0%}\n"
            if "sources" in msg and msg['sources']:
                content += f"📚 Nguồn: {', '.join(msg['sources'])}\n"
            if "feedback" in msg and msg['feedback']:
                fb = "👍 Hữu ích" if msg['feedback'] > 0 else "👎 Chưa tốt"
                content += f"💬 Đánh giá: {fb}\n"
        
        content += "\n" + "-" * 70 + "\n\n"
    
    content += "\n" + "=" * 70 + "\n"
    content += format_contact_info()
    
    return content

def export_as_json() -> str:
    """Export as JSON"""
    export_data = {
        'conversation_id': st.session_state.conversation_id,
        'export_time': datetime.now().isoformat(),
        'version': Config.VERSION,
        'messages': []
    }
    
    for msg in st.session_state.messages:
        export_data['messages'].append({
            'role': msg['role'],
            'content': msg['content'],
            'timestamp': msg.get('timestamp', ''),
            'category': msg.get('category', ''),
            'confidence': msg.get('confidence', 0),
            'sources': msg.get('sources', []),
            'feedback': msg.get('feedback', None)
        })
    
    return json.dumps(export_data, ensure_ascii=False, indent=2)

def export_as_csv() -> str:
    """Export as CSV"""
    rows = []
    for i, msg in enumerate(st.session_state.messages):
        rows.append({
            'Index': i + 1,
            'Role': msg['role'],
            'Content': msg['content'][:100] + '...' if len(msg['content']) > 100 else msg['content'],
            'Category': msg.get('category', ''),
            'Confidence': msg.get('confidence', ''),
            'Sources': ', '.join(msg.get('sources', [])),
            'Feedback': msg.get('feedback', '')
        })
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False, encoding='utf-8-sig')

def render_analytics_dashboard():
    """Render analytics dashboard"""
    st.markdown("### 📊 Thống kê & Phân tích")
    
    analytics = st.session_state.analytics
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tổng câu hỏi", analytics.get('total_queries', 0))
    
    with col2:
        positive = analytics.get('positive_feedback', 0)
        negative = analytics.get('negative_feedback', 0)
        total_fb = positive + negative
        satisfaction = (positive / total_fb * 100) if total_fb > 0 else 0
        st.metric("Độ hài lòng", f"{satisfaction:.0f}%")
    
    with col3:
        st.metric("Phản hồi (+)", positive)
    
    with col4:
        st.metric("Phản hồi (-)", negative)
    
    # Category distribution
    if analytics.get('total_queries', 0) > 0:
        st.markdown("#### 📁 Phân bố theo danh mục")
        
        categories = {k: v for k, v in analytics.items() 
                     if k not in ['total_queries', 'positive_feedback', 'negative_feedback']}
        
        if categories:
            df_cat = pd.DataFrame(list(categories.items()), columns=['Danh mục', 'Số lượng'])
            df_cat = df_cat.sort_values('Số lượng', ascending=False)
            
            fig = px.bar(df_cat, x='Danh mục', y='Số lượng', 
                        color='Số lượng',
                        color_continuous_scale='Blues')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Feedback timeline
    if st.session_state.feedback_data:
        st.markdown("#### 💬 Lịch sử phản hồi")
        
        df_feedback = pd.DataFrame(st.session_state.feedback_data)
        df_feedback['timestamp'] = pd.to_datetime(df_feedback['timestamp'])
        df_feedback['feedback_text'] = df_feedback['feedback'].apply(
            lambda x: 'Positive' if x > 0 else 'Negative'
        )
        
        fig = px.scatter(df_feedback, x='timestamp', y='confidence', 
                        color='feedback_text',
                        color_discrete_map={'Positive': 'green', 'Negative': 'red'},
                        hover_data=['category', 'question'])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def render_sidebar(vectorstore_stats: Dict):
    """Enhanced sidebar with more features"""
    with st.sidebar:
        st.markdown("### ⚙️ Cài đặt")
        
        # Language selector
        language_options = {
            'vi': '🇻🇳 Tiếng Việt',
            'en': '🇬🇧 English'
        }
        
        selected_lang = st.selectbox(
            "Ngôn ngữ / Language",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=0 if st.session_state.language == 'vi' else 1
        )
        
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.rerun()
        
        # System status
        with st.expander("📊 Trạng thái hệ thống", expanded=False):
            st.success("✅ Gemini API: Hoạt động")
            
            if vectorstore_stats:
                st.info(f"📁 Tài liệu: {vectorstore_stats.get('processed_files', 0)} files")
                st.info(f"📦 Chunks: {vectorstore_stats.get('total_chunks', 0)}")
            else:
                st.warning("⚠️ Chưa có vectorstore")
            
            # Cache stats
            cache_size = len(st.session_state.query_cache)
            st.info(f"💾 Query cache: {cache_size} entries")
            
            # Memory status
            memory_turns = len(st.session_state.conversation_memory.chat_memory.messages)
            st.info(f"🧠 Memory: {memory_turns} turns")
        
        # Actions
        st.markdown("### 🔧 Thao tác")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Làm mới", use_container_width=True):
                st.cache_resource.clear()
                st.session_state.clear()
                st.rerun()
        
        with col2:
            if st.button("🗑️ Xóa chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_memory.clear()
                st.session_state.first_visit = True
                st.session_state.conversation_id = generate_conversation_id()
                st.rerun()
        
        if st.button("🧹 Xóa cache", use_container_width=True):
            st.session_state.query_cache = {}
            st.success("✅ Đã xóa cache")
        
        # Export options
        st.markdown("### 💾 Xuất lịch sử")
        
        if st.session_state.messages:
            export_format = st.selectbox(
                "Chọn định dạng",
                options=['txt', 'json', 'csv'],
                format_func=lambda x: {
                    'txt': '📄 Text (.txt)',
                    'json': '📋 JSON (.json)',
                    'csv': '📊 CSV (.csv)'
                }[x]
            )
            
            export_data = export_chat_history(export_format)
            
            if export_data:
                mime_types = {
                    'txt': 'text/plain',
                    'json': 'application/json',
                    'csv': 'text/csv'
                }
                
                st.download_button(
                    label=f"📥 Tải về (.{export_format})",
                    data=export_data,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                    mime=mime_types[export_format],
                    use_container_width=True
                )
            
            # Stats
            total_messages = len(st.session_state.messages)
            user_messages = sum(1 for m in st.session_state.messages if m["role"] == "user")
            st.caption(f"📊 {total_messages} tin ({user_messages} câu hỏi)")
        else:
            st.info("Chưa có lịch sử chat")
        
        # Analytics toggle
        st.markdown("### 📈 Phân tích")
        
        if st.button("📊 Xem thống kê", use_container_width=True):
            st.session_state.show_analytics = not st.session_state.get('show_analytics', False)
            st.rerun()
        
        # Contact info
        st.markdown("---")
        st.markdown("### 📞 Liên hệ")
        st.markdown(format_contact_info(st.session_state.language))
        
        # Footer
        st.markdown("---")
        st.caption(f"🤖 Chatbot v{Config.VERSION}")
        st.caption("🔬 Advanced Features:")
        st.caption("• Multi-turn Memory")
        st.caption("• Hybrid Search")
        st.caption("• Re-ranking")
        st.caption("• Confidence Score")
        st.caption("• Feedback System")

def render_footer():
    """Enhanced footer"""
    st.markdown("---")
    info = Config.CONTACT_INFO
    
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px;">
        <h4>🏛️ Trường Đại học Luật TP. Hồ Chí Minh</h4>
        <p>📍 {info['address']}</p>
        <p>📞 Hotline: {' | '.join(info['hotline'])} | ☎️ {info['phone']}</p>
        <p>📧 {info['email']} | 🌐 {info['website']}</p>
        <p>📘 {info['facebook']}</p>
        <p style="margin-top: 1.5rem; opacity: 0.7; font-size: 0.9em;">
            🚀 Powered by Gemini AI | 🔬 Advanced RAG System
        </p>
        <p style="opacity: 0.6; font-size: 0.85em;">
            Phát triển bởi Lvphung - CNTT | Version {Config.VERSION}
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# VECTORSTORE INITIALIZATION (Reuse from original with enhancements)
# ============================================================================

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
    """Load vectorstore from Google Drive"""
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
    """Process documents"""
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
                failed.append(f"{file_path} (unsupported)")
                continue
            
            loader = loaders[ext](file_path)
            docs = loader.load()
            
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
    """Create vector store"""
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
    """Initialize vectorstore"""
    vectorstore, metadata = load_cached_vectorstore()
    if vectorstore:
        return vectorstore, metadata.get('stats', {})
    
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
# MAIN APPLICATION
# ============================================================================

def main():
    """Enhanced main application"""
    st.set_page_config(
        page_title="AI Chatbot v4.0 - Đại học Luật TPHCM",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_dotenv()
    init_session_state()
    
    render_header()
    
    # Initialize backend
    with st.spinner("🔄 Đang khởi động hệ thống AI..."):
        gemini_config = get_gemini_config()
        if not gemini_config:
            st.stop()
        
        vectorstore, stats = initialize_vectorstore()
    
    render_sidebar(stats)
    
    # Show analytics if toggled
    if st.session_state.get('show_analytics', False):
        render_analytics_dashboard()
        st.markdown("---")
    
    # First visit
    if not st.session_state.messages and st.session_state.first_visit:
        render_quick_questions()
        
        lang = st.session_state.language
        
        if lang == 'vi':
            guide_text = """
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem; border-radius: 12px; color: white; margin-top: 1.5rem;">
                <h4>💡 Tính năng mới v4.0:</h4>
                <ul style="margin: 0.5rem 0;">
                    <li>🧠 <strong>Trí nhớ hội thoại:</strong> Chatbot nhớ ngữ cảnh câu hỏi trước</li>
                    <li>🔍 <strong>Tìm kiếm lai:</strong> Kết hợp semantic + keyword search</li>
                    <li>📊 <strong>Độ tin cậy:</strong> Đánh giá mức độ chính xác của câu trả lời</li>
                    <li>💬 <strong>Đánh giá:</strong> Nhấn 👍/👎 để giúp cải thiện chất lượng</li>
                    <li>📚 <strong>Trích dẫn nguồn:</strong> Xem nguồn tài liệu được tham khảo</li>
                    <li>🌐 <strong>Đa ngôn ngữ:</strong> Hỗ trợ tiếng Việt và English</li>
                </ul>
            </div>
            """
        else:
            guide_text = """
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem; border-radius: 12px; color: white; margin-top: 1.5rem;">
                <h4>💡 New Features v4.0:</h4>
                <ul style="margin: 0.5rem 0;">
                    <li>🧠 <strong>Conversation Memory:</strong> Remembers previous context</li>
                    <li>🔍 <strong>Hybrid Search:</strong> Combines semantic + keyword search</li>
                    <li>📊 <strong>Confidence Score:</strong> Shows answer reliability</li>
                    <li>💬 <strong>Feedback:</strong> Click 👍/👎 to improve quality</li>
                    <li>📚 <strong>Source Attribution:</strong> See referenced documents</li>
                    <li>🌐 <strong>Multi-language:</strong> Vietnamese and English support</li>
                </ul>
            </div>
            """
        
        st.markdown(guide_text, unsafe_allow_html=True)
    
    # Display chat history with enhanced features
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # Category and confidence badges
                badge_html = get_category_badge(msg.get("category", ""))
                
                if msg.get("confidence", 0) > 0:
                    badge_html += get_confidence_badge(msg["confidence"])
                
                st.markdown(badge_html, unsafe_allow_html=True)
            
            # Display content
            st.markdown(msg["content"])
            
            # Show sources for assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])
            
            # Feedback buttons for assistant messages
            if msg["role"] == "assistant" and msg.get("feedback") is None:
                render_feedback_buttons(idx)
            elif msg["role"] == "assistant" and msg.get("feedback"):
                feedback_text = "👍 Hữu ích" if msg["feedback"] > 0 else "👎 Cần cải thiện"
                st.caption(f"💬 Đánh giá của bạn: {feedback_text}")
    
    # Handle user input
    user_input = None
    
    # Check pending question first
    if st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None
    
    # Get from chat input
    if not user_input:
        placeholder = "💬 Nhập câu hỏi..." if st.session_state.language == 'vi' else "💬 Type your question..."
        user_input = st.chat_input(placeholder)
    
    # Process input
    if user_input:
        # Sanitize
        user_input = sanitize_input(user_input)
        
        if not user_input:
            warning_text = "⚠️ Vui lòng nhập câu hỏi hợp lệ" if st.session_state.language == 'vi' else "⚠️ Please enter a valid question"
            st.warning(warning_text)
            st.rerun()
        
        # Check rate limit
        can_proceed, rate_msg = check_rate_limit()
        if not can_proceed:
            st.error(rate_msg)
            st.rerun()
        
        # Mark as not first visit
        st.session_state.first_visit = False
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Rerun to display user message
        st.rerun()
    
    # Generate answer if last message is from user
    if (st.session_state.messages and 
        st.session_state.messages[-1]["role"] == "user" and
        (len(st.session_state.messages) == 1 or 
         st.session_state.messages[-2]["role"] == "assistant")):
        
        last_question = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Đang phân tích và tìm kiếm thông tin..."):
                try:
                    answer, category, confidence, sources = generate_enhanced_answer(
                        last_question, vectorstore, gemini_config
                    )
                    
                    # Display badges
                    badge_html = get_category_badge(category)
                    if confidence > 0:
                        badge_html += get_confidence_badge(confidence)
                    st.markdown(badge_html, unsafe_allow_html=True)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Show sources
                    if sources:
                        render_sources(sources)
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "category": category,
                        "confidence": confidence,
                        "sources": sources,
                        "timestamp": datetime.now().isoformat(),
                        "feedback": None
                    })
                    
                    # Reset error count
                    st.session_state.error_count = 0
                    
                    # Show feedback buttons
                    render_feedback_buttons(len(st.session_state.messages) - 1)
                    
                except Exception as e:
                    st.session_state.error_count += 1
                    
                    lang = st.session_state.language
                    
                    if lang == 'vi':
                        error_message = f"""
❌ **Xin lỗi, đã có lỗi xảy ra**

Vui lòng thử lại hoặc liên hệ trực tiếp:

{format_contact_info('vi')}
"""
                    else:
                        error_message = f"""
❌ **Sorry, an error occurred**

Please try again or contact us directly:

{format_contact_info('en')}
"""
                    
                    if Config.DEBUG:
                        error_message += f"\n\n_Debug info: {str(e)[:200]}_"
                    
                    st.error(error_message)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "category": "Lỗi hệ thống",
                        "confidence": 0.0,
                        "sources": [],
                        "timestamp": datetime.now().isoformat(),
                        "feedback": None
                    })
                    
                    # Suggest refresh after multiple errors
                    if st.session_state.error_count >= 3:
                        refresh_text = "⚠️ Hệ thống gặp nhiều lỗi. Bạn có muốn làm mới?" if lang == 'vi' else "⚠️ Multiple errors detected. Refresh the page?"
                        st.warning(refresh_text)
    
    # Render footer
    render_footer()

if __name__ == "__main__":
    main()
