import os
# --- 1. 配置镜像源 (按需保留) ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import time
# 注意：langchain_community 和 langchain_huggingface 需要根据您的环境安装
# 如果报错，请确保安装了最新版: pip install langchain-community langchain-huggingface faiss-cpu sentence-transformers
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
    page_title="InfoStream v3",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CSS 深度定制 (核心部分) ---
st.markdown("""
<style>
    /* === 全局设定 === */
    .stApp {
        background-color: #F0F7FF; /* 极淡背景 */
    }
    header[data-testid="stHeader"] {
        background-color: #F0F7FF; /* 顶部Header透明化 */
    }
    
    /* === 侧边栏整体容器 === */
    [data-testid="stSidebar"] > div:first-child {
        background-color: #E3EEF9; /* 侧边栏背景色 */
        border-right: 1px solid #D6E4F0;
    }

    /* === 侧边栏顶部头像区域样式 === */
    .sidebar-header-container {
        position: relative;
        text-align: center;
        padding-top: 20px;
        margin-bottom: 30px;
        /* 模拟图二顶部的半圆背景装饰 */
        background: linear-gradient(180deg, rgba(49, 130, 206, 0.1) 0%, rgba(227, 238, 249, 0) 70%);
        border-bottom-left-radius: 50% 20px;
        border-bottom-right-radius: 50% 20px;
    }
    .sidebar-avatar {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        border: 4px solid #FFFFFF;
        box-shadow: 0 4px 10px rgba(49, 130, 206, 0.2);
        background-color: #fff;
        padding: 5px;
    }

    /* === 核心：将 st.radio 改造为矩形按钮块 === */
    /* 1. 隐藏原生的单选圆圈和默认文本样式 */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] input {
        display: none; /* 彻底隐藏 input 元素 */
    }

    /* 2. 定义按钮块的基础样式 (未选中状态) */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        display: flex !important; /* 强制显示 label 容器 */
        align-items: center;
        justify-content: center; /* 文字居中 */
        width: 100%;
        background-color: #DCEBFF; /* 浅蓝底色，类似图二的浅绿 */
        color: #2C5282; /* 深蓝文字 */
        padding: 14px 20px;
        margin-bottom: 12px; /* 按钮间距 */
        border-radius: 12px; /* 圆角矩形 */
        font-weight: 700;
        font-size: 1rem;
        border: 1px solid #CBE2F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        transition: all 0.25s cubic-bezier(0.4, 0.0, 0.2, 1);
        cursor: pointer;
    }
    
    /* 3. 鼠标悬停效果 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #CWDFF7;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(49, 130, 206, 0.15);
    }
    
    /* 4. 选中状态样式 (模拟图二的深色高亮块) */
    /* Streamlit 会给选中的 label 添加 data-checked="true" 属性 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #3182CE !important; /* 品牌深蓝色 */
        color: #FFFFFF !important; /* 白字 */
        border-color: #3182CE;
        box-shadow: 0 4px 12px rgba(49, 130, 206, 0.4);
        transform: translateY(0); /* 选中时不浮动 */
    }

    /* === 侧边栏底部统计卡片 === */
    .metric-container {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
    }
    .metric-val { font-size: 24px; font-weight: 800; color: #2B6CB0; }
    .metric-lbl { font-size: 11px; color: #718096; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px;}

    /* === 主界面样式微调 === */
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
    # 使用一个较小的中文嵌入模型作为示例
    try:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    except Exception as e:
         st.error(f"模型加载失败: {e}. 请确保网络通畅或已下载模型。")
         return None, [], []

    if not os.path.exists('docs/'):
        os.makedirs('docs/')
        st.warning("已创建 docs/ 文件夹，请放入 .txt 文件。")
        return None, [], []
    
    loader = DirectoryLoader('docs/', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    raw_docs = loader.load()
    
    if not raw_docs:
        return None, None, []

    categorized_docs = []
    # 简单关键词分类逻辑
    ai_keywords = ['learning', 'neural', 'intelligence', 'gpt', 'python', 'data', 'cloud', '人工智能']
    fintech_keywords = ['blockchain', 'bitcoin', 'payment', 'finance', 'wallet', 'economy', 'bank', '金融', '经济']
    humanities_keywords = ['history', 'culture', 'art', 'philosophy', 'literature', 'civilization', 'museum', '历史', '文化', '哲学']
    
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

    # 移除了 General，只显示这三个核心分类
    display_categories = ["AI & Technology", "FinTech & Economy", "Humanities & History"]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(categorized_docs)
    try:
        vector_db = FAISS.from_documents(splits, embeddings)
    except Exception as e:
        st.error(f"向量库构建失败: {e}")
        return None, raw_docs, display_categories
    
    return vector_db, raw_docs, display_categories

# --- 5. 初始化 ---
with st.spinner("正在初始化系统..."):
    vector_db, raw_docs, category_list = initialize_system()

# --- 6. 侧边栏 (全新 UI) ---
with st.sidebar:
    # 6.1 顶部头像区域
    # 使用了一个符合文档主题的 3D 图标 URL，您可以替换为您自己的图片地址
    st.markdown("""
        <div class="sidebar-header-container">
            <img src="https://img.icons8.com/3d-fluency/100/folder-invoices.png" class="sidebar-avatar" alt="Navigator">
        </div>
    """, unsafe_allow_html=True)
    
    # 6.2 导航按钮组 (看起来是按钮，实际上是魔改的 Radio)
    # 选项列表，保留了 Emoji 以增加视觉标识度
    nav_options = ["🏠  ALL ARCHIVES"] + [f"🏷️  {cat}" for cat in category_list]
    
    # 这里使用了 label_visibility="collapsed" 隐藏了 Radio 组件自带的标题
    # CSS 会负责把选项渲染成矩形按钮
    selected_option = st.radio(
        "Navigation Menu", 
        nav_options, 
        label_visibility="collapsed"
    )
    
    # 解析选择结果
    if "ALL ARCHIVES" in selected_option:
        selected_category = "ALL ARCHIVES"
    else:
        # 去掉图标前缀 "🏷️  "
        selected_category = selected_option[4:]

    # 6.3 底部统计区域
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True) # 增加间距
    
    total_count = len(raw_docs) if raw_docs else 0
    current_count = "All"
    if selected_category != "ALL ARCHIVES" and raw_docs:
        current_count = sum(1 for d in raw_docs if d.metadata.get('category') == selected_category)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-val">{total_count}</div>
            <div class="metric-lbl">Total Docs</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-val">{current_count}</div>
            <div class="metric-lbl">Current</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 30px; text-align: center; color: #A0AEC0; font-size: 0.8rem;'>System v3.2 | Azure UI</div>", unsafe_allow_html=True)

# --- 7. 主界面 (保持原有风格) ---
st.markdown("## 🔎 Information Retrieval")
st.markdown("检索存档中的专业资讯与文档")

search_col1, search_col2 = st.columns([5, 1], vertical_alignment="bottom")
with search_col1:
    query = st.text_input("Search Query", placeholder="输入关键词...", label_visibility="collapsed")
with search_col2:
    # 搜索按钮样式优化
    st.markdown("""
        <style>div.stButton > button { background-color: #3182CE; color: white; border-radius: 8px; height: 46px; font-weight: 600; }</style>
    """, unsafe_allow_html=True)
    search_btn = st.button("Search", use_container_width=True)

st.markdown("---")

# --- 8. 检索与结果展示 ---
if (query or search_btn):
    if not vector_db:
         st.info("系统尚未初始化完成或 docs/ 目录下没有文件。")
    else:
        start_time = time.time()
        # 增加搜索数量以确保过滤后还有结果
        results = vector_db.similarity_search(query, k=20)
        
        if selected_category != "ALL ARCHIVES":
            filtered_results = [doc for doc in results if doc.metadata.get('category') == selected_category]
        else:
            filtered_results = results

        final_results = filtered_results[:5]

        if not final_results:
            st.info(f"在 【{selected_category}】 中未找到关于 '{query}' 的内容。")
        else:
            st.markdown(f"**找到 {len(final_results)} 条相关记录** (用时 {time.time() - start_time:.4f}s)")
            
            for doc in final_results:
                cat_tag = doc.metadata.get('category')
                file_name = doc.metadata['source'].split('/')[-1] or doc.metadata['source'].split('\\')[-1]
                full_file_path = doc.metadata['source']
                
                full_content = "未找到全文内容"
                # 简单查找全文内容
                for raw_doc in raw_docs:
                    if raw_doc.metadata['source'] == full_file_path:
                        full_content = raw_doc.page_content
                        break

                st.markdown(f"""
                <div class="result-item">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #2B6CB0; margin-bottom: 10px;">📄 {file_name}</div>
                    <div style="margin-bottom:12px;">
                        <span class="cat-tag">{cat_tag}</span>
                        <span style="color:#A0AEC0; font-size:0.8rem; margin-left:10px;">相关度匹配</span>
                    </div>
                    <div style="color:#4A5568; line-height:1.6; font-size: 0.95rem;">
                        {doc.page_content}... 
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📖 查看完整文档"):
                    st.markdown(full_content)

elif not vector_db:
    st.info("👋 欢迎! 请在 docs/ 目录下放入 .txt 文件后刷新页面。")
elif not query:
    st.markdown(f"""
        <div style='text-align: center; padding: 40px; color: #718096;'>
            <p style='font-size: 3rem; margin-bottom: 10px;'>💡</p>
            <p>当前浏览: <strong>{selected_category}</strong></p>
            <p>请在上方搜索框输入关键词开始检索。</p>
        </div>
    """, unsafe_allow_html=True)