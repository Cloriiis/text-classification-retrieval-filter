import os
# --- 1. 配置镜像源 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 2. 页面设置 ---
st.set_page_config(
    page_title="InfoStream - 专业资讯归档系统",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. UI/UX 深度定制 ---
st.markdown("""
<style>
    /* 1. 全局背景统一：极淡的海洋蓝 */
    .stApp {
        background-color: #F0F7FF;
    }
    
    /* 2. 核心修复：强制顶部 Header 变为透明/同色，去除白色割裂带 */
    header[data-testid="stHeader"] {
        background-color: #F0F7FF;
    }
    
    /* 调整主内容区域的顶部间距 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 3. 侧边栏深度美化 */
    [data-testid="stSidebar"] {
        background-color: #EBF4FF; /* 比主背景稍深一点的蓝，区分层级 */
        border-right: 1px solid #D6E4F0;
    }
    
    /* 侧边栏标题 */
    .sidebar-title {
        font-family: 'Inter', sans-serif;
        color: #1A365D;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
    }
    
    /* === 侧边栏导航按钮化改造 (去除 Radio 圆圈) === */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none !important; /* 隐藏 Radio 的 label */
    }
    
    /* 选项容器样式 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: transparent;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 5px;
        transition: all 0.2s ease;
        border: 1px solid transparent;
        color: #4A5568;
        font-weight: 500;
    }
    
    /* 鼠标悬停效果 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #DCEBFF;
        color: #2B6CB0;
    }
    
    /* 选中状态 (Streamlit 会给选中的 label 加 aria-checked="true") */
    /* 注意：Streamlit 的内部结构可能变化，这里使用 checked 伪类或结构化选择 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #3182CE !important;
        color: white !important;
        box-shadow: 0 4px 6px rgba(49, 130, 206, 0.2);
    }
    
    /* 隐藏原生的圆圈单选框 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] input {
        display: none;
    }

    /* 4. 统计卡片样式 */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2C5282;
    }
    .metric-label {
        font-size: 12px;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* 5. 搜索结果样式 (保持原有好评设计) */
    .result-item {
        background-color: #FFFFFF;
        padding: 24px;
        margin-bottom: 16px;
        border-radius: 12px;
        border: 1px solid #E6F0FA;
        box-shadow: 0 2px 8px rgba(26, 54, 93, 0.03);
        transition: transform 0.2s;
    }
    .result-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(26, 54, 93, 0.08);
    }
    .result-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2B6CB0;
        margin-bottom: 8px;
    }
    .cat-tag {
        background-color: #EBF8FF;
        color: #2C5282;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* 搜索按钮 */
    div.stButton > button {
        background-color: #3182CE;
        color: white;
        border-radius: 8px;
        height: 46px; /* 与输入框对齐 */
    }
</style>
""", unsafe_allow_html=True)

# --- 4. 核心逻辑 ---
@st.cache_resource
def initialize_system():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    if not os.path.exists('docs/'):
        os.makedirs('docs/')
    
    loader = DirectoryLoader('docs/', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    raw_docs = loader.load()
    
    if not raw_docs:
        return None, None, []

    categorized_docs = []
    # 关键词定义
    ai_keywords = ['learning', 'neural', 'intelligence', 'gpt', 'python', 'data', 'cloud']
    fintech_keywords = ['blockchain', 'bitcoin', 'payment', 'finance', 'wallet', 'economy', 'bank']
    humanities_keywords = ['history', 'culture', 'art', 'philosophy', 'literature', 'civilization', 'museum']
    
    for doc in raw_docs:
        filename = doc.metadata['source'].lower()
        content = doc.page_content.lower()
        category = "General / Uncategorized"
        
        if any(k in filename or k in content for k in ai_keywords):
            category = "AI & Technology"
        elif any(k in filename or k in content for k in fintech_keywords):
            category = "FinTech & Economy"
        elif any(k in filename or k in content for k in humanities_keywords):
            category = "Humanities & History"
            
        doc.metadata['category'] = category
        categorized_docs.append(doc)

    # 【修改点】：这里移除了 "General / Uncategorized" 
    # 注意：如果文件被归类为 General，它在 "ALL ARCHIVES" 中仍可见，但侧边栏没有单独入口，符合您的要求
    display_categories = ["AI & Technology", "FinTech & Economy", "Humanities & History"]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(categorized_docs)
    vector_db = FAISS.from_documents(splits, embeddings)
    
    return vector_db, raw_docs, display_categories

# --- 5. 初始化 ---
with st.spinner("Initializing System..."):
    vector_db, raw_docs, category_list = initialize_system()

# --- 6. 侧边栏 (重构版) ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">Navigator</div>', unsafe_allow_html=True)
    
    # 构造带图标的选项列表
    nav_options = ["  ALL ARCHIVES"] + [f"  {cat}" for cat in category_list]
    
    # 使用 Radio 但 CSS 已经魔改成导航条样式
    selected_option = st.radio(
        "Navigation", 
        nav_options, 
        label_visibility="collapsed"
    )
    
    # 解析回原始分类名
    if "ALL ARCHIVES" in selected_option:
        selected_category = "ALL ARCHIVES"
    else:
        # 去掉图标前缀 "🏷️  " (长度为4)
        selected_category = selected_option[2:]

    st.markdown("---")
    
    # 统计数据卡片化
    col1, col2 = st.columns(2)
    
    total_count = len(raw_docs) if raw_docs else 0
    current_count = "All"
    if selected_category != "ALL ARCHIVES" and raw_docs:
        current_count = sum(1 for d in raw_docs if d.metadata.get('category') == selected_category)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_count}</div>
            <div class="metric-label">Total Docs</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{current_count}</div>
            <div class="metric-label">Current</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("System v3.0 | Azure Theme")

# --- 7. 主界面 ---
st.markdown("## Information Retrieval")
st.markdown("检索存档中的专业资讯与文档，涵盖AI、金融科技与人文历史等领域知识")

search_col1, search_col2 = st.columns([5, 1], vertical_alignment="bottom")
with search_col1:
    query = st.text_input("Search Query", placeholder="输入关键词...", label_visibility="collapsed")
with search_col2:
    search_btn = st.button("Search", use_container_width=True)

st.markdown("---")

# --- 8. 检索与结果展示 ---
if (query or search_btn) and vector_db:
    start_time = time.time()
    results = vector_db.similarity_search(query, k=15)
    
    if selected_category != "ALL ARCHIVES":
        filtered_results = [doc for doc in results if doc.metadata.get('category') == selected_category]
    else:
        filtered_results = results

    final_results = filtered_results[:5]

    if not final_results:
        st.info(f"未在 【{selected_category}】 中找到相关内容。")
    else:
        st.markdown(f"**找到 {len(final_results)} 条相关记录** (用时 {time.time() - start_time:.4f}s)")
        
        for doc in final_results:
            cat_tag = doc.metadata.get('category')
            file_name = doc.metadata['source'].split('/')[-1]
            full_file_path = doc.metadata['source']
            
            full_content = "未找到全文内容"
            for raw_doc in raw_docs:
                if raw_doc.metadata['source'] == full_file_path:
                    full_content = raw_doc.page_content
                    break

            st.markdown(f"""
            <div class="result-item">
                <div class="result-title">📄 {file_name}</div>
                <div style="margin-bottom:10px;">
                    <span class="cat-tag">{cat_tag}</span>
                    <span style="color:#A0AEC0; font-size:0.8rem; margin-left:10px;">相关度匹配</span>
                </div>
                <div style="color:#4A5568; line-height:1.6;">
                    {doc.page_content}... 
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📖 查看完整文档"):
                st.markdown(full_content)

elif not vector_db:
    st.info("请在 docs/ 目录下放入 .txt 文件后启动系统。")
elif not query:
    st.info("💡 在上方搜索框输入关键词开启检索。")