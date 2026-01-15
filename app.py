import os
# --- 1. 配置镜像源 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import time
# 尝试导入库，防止因环境缺失报错
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as e:
    st.error(f"缺少必要的库，请检查安装: {e}")
    st.stop()

# --- 2. 页面设置 ---
st.set_page_config(
    page_title="InfoStream - 专业资讯归档系统",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. UI/UX 深度定制 (针对您的新要求) ---
st.markdown("""
<style>
    /* 1. 全局背景 */
    .stApp {
        background-color: #F0F7FF;
    }
    header[data-testid="stHeader"] {
        background-color: #F0F7FF;
    }
    
    /* 2. 侧边栏背景 */
    [data-testid="stSidebar"] > div:first-child {
        background-color: #E3EEF9;
        border-right: 1px solid #D6E4F0;
    }

    /* === 3. 顶部 "Navigator" 盒子样式 === */
    .nav-box {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
        border: 2px solid #FFFFFF;
        box-shadow: 0 4px 15px rgba(49, 130, 206, 0.15);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin-bottom: 30px;
        margin-top: 10px;
    }
    .nav-icon {
        font-size: 3rem;
        margin-bottom: 10px;
        display: block;
    }
    .nav-title {
        font-family: 'Inter', sans-serif;
        color: #2C5282;
        font-weight: 800;
        font-size: 1.2rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* === 4. 分类选项按钮化 (核心修改) === */
    /* 隐藏原生 Radio 的圆圈和 Input */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] input {
        display: none;
    }
    /* 隐藏 Streamlit Radio 自动生成的圆圈占位符 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
        display: none;
    }

    /* 定义按钮的基础样式 (未选中) - 居中小方框 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        display: flex;
        align-items: center;
        justify-content: center; /* 文字居中 */
        width: 100%;
        background-color: #FFFFFF; /* 白色底 */
        color: #4A5568; /* 深灰字 */
        padding: 12px 10px;
        margin-bottom: 12px;
        border-radius: 10px; /* 圆角矩形 */
        border: 1px solid #E2E8F0;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: all 0.2s ease-in-out;
        cursor: pointer;
    }
    
    /* 悬停效果 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        border-color: #CBD5E0;
        color: #2B6CB0;
    }
    
    /* 选中效果 - 深蓝底白字 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #3182CE !important;
        color: white !important;
        border-color: #3182CE;
        box-shadow: 0 4px 10px rgba(49, 130, 206, 0.3);
    }

    /* 5. 底部统计卡片 */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        border: 1px solid #E2E8F0;
    }
    .metric-val { font-size: 20px; font-weight: 800; color: #2C5282; }
    .metric-lbl { font-size: 10px; color: #718096; text-transform: uppercase; margin-top: 4px;}

    /* 6. 主内容样式 */
    .result-item {
        background-color: #FFFFFF;
        padding: 24px;
        margin-bottom: 16px;
        border-radius: 12px;
        border: 1px solid #E6F0FA;
        box-shadow: 0 2px 8px rgba(26, 54, 93, 0.03);
    }
    .cat-tag {
        background-color: #EBF8FF; color: #2C5282; padding: 4px 12px;
        border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. 核心逻辑 (保持不变) ---
@st.cache_resource
def initialize_system():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    except Exception:
        return None, [], []
    
    if not os.path.exists('docs/'):
        os.makedirs('docs/')
    
    loader = DirectoryLoader('docs/', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    raw_docs = loader.load()
    
    if not raw_docs:
        return None, None, []

    categorized_docs = []
    ai_keywords = ['learning', 'neural', 'intelligence', 'gpt', 'python', 'data', 'cloud', '人工智能']
    fintech_keywords = ['blockchain', 'bitcoin', 'payment', 'finance', 'wallet', 'economy', 'bank', '金融']
    humanities_keywords = ['history', 'culture', 'art', 'philosophy', 'literature', '历史', '文化']
    
    for doc in raw_docs:
        filename = doc.metadata['source'].lower()
        content = doc.page_content.lower()
        category = "General"
        
        if any(k in filename or k in content for k in ai_keywords):
            category = "AI & Technology"
        elif any(k in filename or k in content for k in fintech_keywords):
            category = "FinTech & Economy"
        elif any(k in filename or k in content for k in humanities_keywords):
            category = "Humanities & History"
            
        doc.metadata['category'] = category
        categorized_docs.append(doc)

    display_categories = ["AI & Technology", "FinTech & Economy", "Humanities & History"]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(categorized_docs)
    
    try:
        vector_db = FAISS.from_documents(splits, embeddings)
    except Exception:
        return None, raw_docs, display_categories
    
    return vector_db, raw_docs, display_categories

# --- 5. 初始化 ---
with st.spinner("Initializing..."):
    vector_db, raw_docs, category_list = initialize_system()

# --- 6. 侧边栏 (新 UI 布局) ---
with st.sidebar:
    # 6.1 顶部的 Navigator 盒子 (模拟图二头像区域的视觉重心)
    st.markdown("""
        <div class="nav-box">
            <span class="nav-icon">📂</span>
            <div class="nav-title">Navigator</div>
        </div>
    """, unsafe_allow_html=True)
    
    # 6.2 选项列表 (纯文本，无Emoji)
    nav_options = ["ALL ARCHIVES"] + category_list
    
    # 使用 Radio 组件，CSS 负责将其渲染为小方框按钮
    selected_option = st.radio(
        "Menu", 
        nav_options, 
        label_visibility="collapsed"
    )
    
    selected_category = selected_option

    # 6.3 底部统计
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    
    total_count = len(raw_docs) if raw_docs else 0
    current_count = "All"
    if selected_category != "ALL ARCHIVES" and raw_docs:
        current_count = sum(1 for d in raw_docs if d.metadata.get('category') == selected_category)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{total_count}</div>
            <div class="metric-lbl">Total</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{current_count}</div>
            <div class="metric-lbl">Current</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 20px; text-align: center; color: #A0AEC0; font-size: 0.75rem;'>System v3.3</div>", unsafe_allow_html=True)

# --- 7. 主界面 (保持功能) ---
st.markdown("## 🔎 Information Retrieval")
st.markdown("检索存档中的专业资讯与文档")

search_col1, search_col2 = st.columns([5, 1], vertical_alignment="bottom")
with search_col1:
    query = st.text_input("Query", placeholder="输入关键词...", label_visibility="collapsed")
with search_col2:
    st.markdown("<style>div.stButton > button { background-color: #3182CE; color: white; border-radius: 8px; height: 46px; }</style>", unsafe_allow_html=True)
    search_btn = st.button("Search", use_container_width=True)

st.markdown("---")

if (query or search_btn):
    if not vector_db:
         st.info("System not ready or no documents found.")
    else:
        results = vector_db.similarity_search(query, k=15)
        
        if selected_category != "ALL ARCHIVES":
            filtered_results = [doc for doc in results if doc.metadata.get('category') == selected_category]
        else:
            filtered_results = results

        final_results = filtered_results[:5]

        if not final_results:
            st.info(f"No results found in 【{selected_category}】.")
        else:
            for doc in final_results:
                cat_tag = doc.metadata.get('category')
                file_name = os.path.basename(doc.metadata['source'])
                
                # 获取全文
                full_content = "Content not found."
                for raw_doc in raw_docs:
                    if raw_doc.metadata['source'] == doc.metadata['source']:
                        full_content = raw_doc.page_content
                        break

                st.markdown(f"""
                <div class="result-item">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #2B6CB0; margin-bottom: 8px;">📄 {file_name}</div>
                    <div style="margin-bottom:10px;"><span class="cat-tag">{cat_tag}</span></div>
                    <div style="color:#4A5568; line-height:1.5; font-size: 0.9rem;">{doc.page_content}...</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("Show Details"):
                    st.text(full_content)

elif not query:
    st.info(f"Current Section: {selected_category}")