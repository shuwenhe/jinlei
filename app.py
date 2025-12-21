import streamlit as st
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# 页面配置
st.set_page_config(
    page_title="知识库问答系统",
    page_icon="🔍",
    layout="wide"
)

# 标题
st.title("🔍 知识库问答系统")
st.markdown("---")

# 侧边栏说明
with st.sidebar:
    st.header("📚 系统信息")
    st.info("当前知识库：六道工序")
    st.markdown("---")
    st.markdown("### 使用说明")
    st.markdown("""
    1. 在下方输入您的问题
    2. 点击"搜索"按钮
    3. 查看相关知识片段
    """)

# 加载知识库的函数
@st.cache_resource
def load_knowledge_base():
    try:
        embeddings = OllamaEmbeddings(model="bge-m3")
        index_path = "./faiss_jinlei_index"
        
        if os.path.exists(index_path):
            vector_store = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            st.sidebar.success("✅ 知识库加载成功！")
            return vector_store
        else:
            st.sidebar.error(f"❌ 索引文件不存在: {index_path}")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ 加载知识库失败: {e}")
        return None

# 初始化加载知识库
vector_store = load_knowledge_base()

# 查询输入区域
col1, col2 = st.columns([6, 1])
with col1:
    query = st.text_input(
        "请输入您的问题：",
        placeholder="例如：什么是六道工序？",
        key="query_input"
    )
with col2:
    st.write("")  # 为了对齐
    st.write("")
    search_button = st.button("🔍 搜索", type="primary")

# 处理查询
if search_button or query:
    if not query:
        st.warning("⚠️ 请输入查询内容")
    elif vector_store is None:
        st.error("❌ 知识库未加载，请检查索引文件")
    else:
        with st.spinner("正在搜索中..."):
            # 搜索相关文档
            results = vector_store.similarity_search(query, k=3)
            
            # 显示结果
            st.subheader(f"📋 搜索结果（共 {len(results)} 条）")
            
            for i, doc in enumerate(results):
                with st.expander(f"📄 结果 {i+1}", expanded=(i==0)):
                    st.markdown(f"**内容：**")
                    st.markdown(doc.page_content)
                    
                    # 显示元数据
                    if doc.metadata:
                        st.markdown("**元数据：**")
                        for key, value in doc.metadata.items():
                            st.markdown(f"- **{key}:** {value}")
            
            if not results:
                st.info("未找到相关结果，请尝试其他查询词。")

# 底部信息
st.markdown("---")
st.caption("💡 提示：知识库基于 '六道工序.docx' 文档构建")
