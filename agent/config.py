import torch
from pathlib import Path
import os

class Config:
    # Paths
    # 项目根目录
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    # 数据存储目录
    DATA_DIR = PROJECT_ROOT / "data"
    # 论文存储目录
    PAPERS_DIR = DATA_DIR / "papers"
    # 图片存储目录
    IMAGES_DIR = DATA_DIR / "images"
    # 索引存储目录
    INDEX_DIR = DATA_DIR / "index"
    # 缓存目录
    CACHE_DIR = DATA_DIR / "cache"

    # Models
    # 文本嵌入模型名称
    # 升级为支持多语言且性能更强的模型 (768维)
    TEXT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    # OpenCLIP model config
    # 图像嵌入模型配置 (CLIP)
    # 升级为 Large 模型以利用 RTX 3060 性能 (768维)
    IMAGE_MODEL_NAME = "ViT-L-14"
    IMAGE_MODEL_PRETRAINED = "laion2b_s32b_b82k"

    # Device
    # 设备选择: 如果有 CUDA 则使用 CUDA，否则使用 CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Indexing
    # 文本分块大小
    # 减小分块大小以提高检索粒度
    CHUNK_SIZE = 800
    # 文本分块重叠大小
    CHUNK_OVERLAP = 150
    
    # Chroma
    # 向量数据库集合名称
    # 论文文件集合
    COLLECTION_PAPERS_FILES = "papers_files"
    # 论文片段集合
    COLLECTION_PAPERS_CHUNKS = "papers_chunks"
    # 图片集合
    COLLECTION_IMAGES = "images"

    @classmethod
    def setup(cls):
        """
        Setup necessary directories.
        初始化必要的目录结构。
        """
        cls.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        cls.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        cls.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)

Config.setup()
