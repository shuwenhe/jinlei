import os
# 移除了 DirectoryLoader，因为它不兼容 loader_map 参数
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
# 导入路径已修复
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# --- 配置参数 ---
KNOWLEDGE_DIR = "./knowledge_base"
FAISS_INDEX_PATH = "./faiss_jinlei_index"
OLLAMA_EMBEDDING_MODEL = "bge-m3" # 请确保已通过 ollama pull m3e-base 拉取

def build_index():
    """
    加载、切分文档，创建并保存 FAISS 向量知识库。
    使用手动遍历代替 DirectoryLoader(loader_map)，并增加切分容错逻辑。
    """
    # 1. 文档加载
    print(f"--- 1. 正在加载 {KNOWLEDGE_DIR} 中的文档... ---")
    
    # 定义支持的文件类型和对应的加载器
    LOADER_MAPPING = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
    }
    
    documents = []
    
    # 手动遍历知识库目录下的所有文件 (包括子目录)
    for root, _, files in os.walk(KNOWLEDGE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            if ext in LOADER_MAPPING:
                LoaderClass = LOADER_MAPPING[ext]
                print(f"   -> 正在加载文件: {file}")
                try:
                    loader = LoaderClass(file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"❌ 文件 {file} 加载失败: {e}")

    if not documents:
        print("❌ 警告：未找到任何支持格式的文档，请检查知识库文件夹。")
        return
        
    print(f"✅ 成功加载 {len(documents)} 个文档页面/块。")

    # 2. 文本切分 (核心修复区域)
    print("--- 2. 正在进行文本切分... ---")
    
    # 🌟 优化点 1: 打印文档内容的长度，以便诊断
    total_chars = sum(len(doc.page_content) for doc in documents)
    print(f"⭐ 待切分文档总字符数：{total_chars}")
    
    # 🌟 优化点 2: 首次尝试切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""], 
        length_function=len # 明确使用标准的 Python len() 函数
    )
    texts = text_splitter.split_documents(documents)
    
    # 🌟 优化点 3: 如果切分结果为 0，尝试使用较小的 chunk_size 进行容错
    if not texts:
        print("💡 第一次切分结果为 0。可能内容太短，尝试使用较小的 chunk_size (例如 400)...")
        text_splitter_small = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""], 
            length_function=len
        )
        texts = text_splitter_small.split_documents(documents)
    
    if not texts:
        print("❌ 严重警告：文本切分结果仍为 0。请检查文档内容是否为空或不可提取。")
        return # 提前退出，避免 FAISS 错误
        
    print(f"✅ 文档切分成 {len(texts)} 个知识块。")

    # 3. 向量化与知识库构建
    print(f"--- 3. 正在初始化 Embedding 模型 ({OLLAMA_EMBEDDING_MODEL}) 并构建 FAISS 索引... ---")
    
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url="http://localhost:11434" # 确保 Ollama 服务地址正确
    )
    
    try:
        # FAISS 需要非空列表
        db = FAISS.from_documents(texts, embeddings)
        
        # 4. 保存 FAISS 索引
        db.save_local(FAISS_INDEX_PATH)
        print(f"✅ FAISS 索引已成功保存到: {FAISS_INDEX_PATH}")
        print("\n🎉 知识库构建完成！现在可以运行 Web 应用了。")
    except Exception as e:
        # 如果不是 list index out of range，打印具体错误
        print(f"❌ 向量化或 FAISS 知识库构建失败: {e}")


if __name__ == "__main__":
    if not os.path.exists(KNOWLEDGE_DIR):
        print(f"创建知识库目录: {KNOWLEDGE_DIR}")
        os.makedirs(KNOWLEDGE_DIR)
        print("请将 PDF/Word 文档放入此目录后，重新运行本脚本。")
    else:
        build_index()

# 运行命令: python build_knowledge_base.py
