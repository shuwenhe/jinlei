import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# --- 配置参数 ---
KNOWLEDGE_DIR = "./knowledge_base"
FAISS_INDEX_PATH = "./faiss_jinlei_index"
OLLAMA_EMBEDDING_MODEL = "m3e-base" # 请确保已通过 ollama pull m3e-base 拉取

def build_index():
    """
    加载、切分文档，创建并保存 FAISS 向量知识库。
    """
    # 1. 文档加载
    print(f"--- 1. 正在加载 {KNOWLEDGE_DIR} 中的文档... ---")
    
    # 定义加载器映射
    loader_mapping = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
    }
    
    # 使用 DirectoryLoader 批量加载
    loader = DirectoryLoader(
        KNOWLEDGE_DIR, 
        loader_map=loader_mapping,
        silent_errors=True,
        # 确保 loader 能够处理嵌套文件夹
        glob="**/*",
        # 针对中文文档，确保编码正确
        loader_kwargs={'autodetect_encoding': True} 
    )
    
    try:
        documents = loader.load()
        if not documents:
            print("❌ 警告：未找到任何支持格式的文档，请检查知识库文件夹。")
            return
        print(f"✅ 成功加载 {len(documents)} 个文档页面/块。")
    except Exception as e:
        print(f"❌ 文档加载失败: {e}")
        return

    # 2. 文本切分
    print("--- 2. 正在进行文本切分... ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""] # 优化中文分隔符
    )
    texts = text_splitter.split_documents(documents)
    print(f"✅ 文档切分成 {len(texts)} 个知识块。")

    # 3. 向量化与知识库构建
    print(f"--- 3. 正在初始化 Embedding 模型 ({OLLAMA_EMBEDDING_MODEL}) 并构建 FAISS 索引... ---")
    
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url="http://localhost:11434" # 确保 Ollama 服务地址正确
    )
    
    try:
        db = FAISS.from_documents(texts, embeddings)
        
        # 4. 保存 FAISS 索引
        db.save_local(FAISS_INDEX_PATH)
        print(f"✅ FAISS 索引已成功保存到: {FAISS_INDEX_PATH}")
        print("\n🎉 知识库构建完成！现在可以运行 Web 应用了。")
    except Exception as e:
        print(f"❌ 向量化或 FAISS 知识库构建失败: {e}")


if __name__ == "__main__":
    if not os.path.exists(KNOWLEDGE_DIR):
        print(f"创建知识库目录: {KNOWLEDGE_DIR}")
        os.makedirs(KNOWLEDGE_DIR)
        print("请将 PDF/Word 文档放入此目录后，重新运行本脚本。")
    else:
        build_index()

# 运行命令: python build_knowledge_base.py
