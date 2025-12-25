import shutil
import hashlib
import numpy as np
from pathlib import Path
from agent.config import Config
from agent.models.text_embedder import TextEmbedder
from agent.index.vector_store import VectorStore
from agent.parsers.pdf_parser import PDFParser
from agent.utils.logging import setup_logger

logger = setup_logger(__name__)

class PaperManager:
    """
    Service for managing paper ingestion, classification, and organization.
    用于管理论文摄入、分类和整理的服务。
    """
    def __init__(self, text_embedder: TextEmbedder = None, vector_store: VectorStore = None):
        """
        Initialize the PaperManager.
        初始化 PaperManager。

        Args:
            text_embedder (TextEmbedder, optional): Text embedding model. 文本嵌入模型。
            vector_store (VectorStore, optional): Vector database interface. 向量数据库接口。
        """
        self.text_embedder = text_embedder or TextEmbedder()
        self.vector_store = vector_store or VectorStore()

    def _compute_file_embedding(self, full_text: str, chunks: list[dict]):
        """
        Compute file embedding using mean pooling of chunks if available.
        如果可用，使用块的平均池化计算文件嵌入。
        """
        if chunks:
            chunk_texts = [c["text"] for c in chunks]
            chunk_embeddings = self.text_embedder.embed_texts(chunk_texts)
            file_embedding = np.mean(chunk_embeddings, axis=0)
            norm = np.linalg.norm(file_embedding)
            if norm > 0:
                file_embedding = file_embedding / norm
            return file_embedding
        else:
            return self.text_embedder.embed_texts([full_text])[0]

    def add_paper(self, file_path: str, topics: list[str] = None, move: bool = False, index: bool = True):
        """
        Add a paper: parse, classify, move, and index.
        添加论文：解析、分类、移动并索引。

        Args:
            file_path (str): Path to the PDF file. PDF 文件路径。
            topics (list[str], optional): List of topics for classification. 用于分类的主题列表。
            move (bool): Whether to move the file to a topic directory. 是否将文件移动到主题目录。
            index (bool): Whether to index the paper in the vector store. 是否在向量存储中索引论文。
        """
        # Implementation with move before index
        path = Path(file_path).resolve()
        if not path.exists():
            logger.error(f"File not found: {path}")
            return

        logger.info(f"Processing paper: {path.name}")
        full_text, chunks = PDFParser.parse(str(path))
        if not full_text:
            logger.warning("No text extracted.")
            return

        assigned_topic = "Uncategorized"
        file_embedding = None
        
        # Classification
        if topics:
             file_embedding = self._compute_file_embedding(full_text, chunks)
             topic_embeddings = self.text_embedder.embed_texts(topics)
             
             sims = np.dot(topic_embeddings, file_embedding)
             best_idx = np.argmax(sims)
             assigned_topic = topics[best_idx]
             logger.info(f"Classified as: {assigned_topic}")

        # Move
        final_path = path
        if move and topics:
            dest_dir = Config.PAPERS_DIR / assigned_topic
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / path.name
            if path != dest_path:
                shutil.move(str(path), str(dest_path))
                final_path = dest_path
                logger.info(f"Moved to {final_path}")

        # Index
        if index:
            if file_embedding is None:
                 file_embedding = self._compute_file_embedding(full_text, chunks)
            
            file_hash = hashlib.sha256(full_text.encode()).hexdigest()
            metadata = {
                "path": str(final_path),
                "filename": final_path.name,
                "sha256": file_hash,
                "topic": assigned_topic
            }
            self.vector_store.add_paper_file(file_hash, file_embedding, metadata)

            if chunks:
                chunk_texts = [c["text"] for c in chunks]
                # If we computed file_embedding using chunks, we might have already computed these embeddings,
                # but to keep code simple and avoid passing large arrays around or complicating the helper, 
                # we re-compute or we could have returned them. 
                # Given the scale, re-computing or just caching in the helper if needed. 
                # For now, standard re-compute is safer/simpler unless performance is critical.
                # Optimization: We can reuse chunk embeddings if we want.
                chunk_embeddings = self.text_embedder.embed_texts(chunk_texts)
                chunk_ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
                chunk_metadatas = []
                for i, c in enumerate(chunks):
                    m = c.copy()
                    del m["text"]
                    m["path"] = str(final_path)
                    m["filename"] = final_path.name
                    chunk_metadatas.append(m)
                self.vector_store.add_paper_chunks(chunk_ids, chunk_embeddings, chunk_metadatas, chunk_texts)
                logger.info(f"Indexed {len(chunks)} chunks.")

    def batch_organize(self, root_dir: str, topics: list[str] = None):
        """
        Batch organize all PDFs in a directory.
        批量整理目录中的所有 PDF。

        Args:
            root_dir (str): Root directory to scan. 要扫描的根目录。
            topics (list[str], optional): List of topics for classification. If None, auto-detect. 用于分类的主题列表。如果不提供，则自动检测。
        """
        root = Path(root_dir)
        pdf_files = list(root.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process.")
        
        if not pdf_files:
            return

        # Pre-process all files to avoid redundant parsing
        # 预处理所有文件以避免重复解析
        docs = []
        logger.info("Parsing files and generating embeddings...")
        for p in pdf_files:
            if not p.exists(): continue
            full_text, chunks = PDFParser.parse(str(p))
            if not full_text: 
                logger.warning(f"Skipping {p.name}: No text extracted.")
                continue
            
            emb = self._compute_file_embedding(full_text, chunks)
            docs.append({
                "path": p,
                "text": full_text,
                "chunks": chunks,
                "embedding": emb
            })

        if not docs:
            logger.warning("No valid documents found.")
            return

        if topics:
            self._classify_and_finalize(docs, topics)
        else:
            self._auto_organize_hybrid(docs)

    def _classify_and_finalize(self, docs, topics):
        """
        Classify docs using provided topics and finalize.
        使用提供的主题对文档进行分类并完成处理。
        """
        topic_embeddings = self.text_embedder.embed_texts(topics)
        
        for doc in docs:
             file_embedding = doc["embedding"]
             sims = np.dot(topic_embeddings, file_embedding)
             best_idx = np.argmax(sims)
             assigned_topic = topics[best_idx]
             self._finalize_paper(doc, assigned_topic)

    def _auto_organize_hybrid(self, docs):
        """
        Hybrid organization: Prioritize existing topics, then cluster the rest.
        混合整理：优先考虑现有主题，然后对其余部分进行聚类。
        """
        # 1. Check existing topics in data/papers
        existing_topics = [d.name for d in Config.PAPERS_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        unassigned_docs = []
        
        if existing_topics:
            logger.info(f"Found existing topics: {existing_topics}")
            topic_embeddings = self.text_embedder.embed_texts(existing_topics)
            threshold = 0.35 # Similarity threshold (heuristic) # 相似度阈值（启发式）
            
            for doc in docs:
                file_embedding = doc["embedding"]
                sims = np.dot(topic_embeddings, file_embedding)
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]
                
                if best_score > threshold:
                    assigned_topic = existing_topics[best_idx]
                    logger.info(f"Matched {doc['path'].name} to existing topic '{assigned_topic}' (Score: {best_score:.2f})")
                    self._finalize_paper(doc, assigned_topic)
                else:
                    unassigned_docs.append(doc)
        else:
            unassigned_docs = docs

        # 2. Cluster remaining docs
        if unassigned_docs:
            logger.info(f"{len(unassigned_docs)} papers remain unassigned. Proceeding to clustering...")
            self._cluster_and_finalize(unassigned_docs)

    def _cluster_and_finalize(self, docs):
        """
        Cluster papers and generate new topics.
        对论文进行聚类并生成新主题。
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            logger.error("scikit-learn is required for auto-organization. Please install it.")
            return

        # 2. Cluster
        num_docs = len(docs)
        if num_docs < 2:
            # Too few documents to cluster, put in "General" or "New_Topic"
            # 文档太少无法聚类，放入 "General" 或 "New_Topic"
            for doc in docs:
                self._finalize_paper(doc, "General")
            return

        # Dynamic cluster count
        n_clusters = min(max(2, num_docs // 3), 8)
        logger.info(f"Clustering {num_docs} papers into {n_clusters} topics...")
        
        embeddings = [d["embedding"] for d in docs]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        X = np.array(embeddings)
        labels = kmeans.fit_predict(X)
        
        # 3. Label Clusters via TF-IDF
        cluster_texts = [""] * n_clusters
        for i, label in enumerate(labels):
            cluster_texts[label] += " " + docs[i]["text"]
            
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        cluster_names = {}
        
        try:
            # Check if we have enough content
            if any(len(t.strip()) > 0 for t in cluster_texts):
                tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                feature_names = vectorizer.get_feature_names_out()
                
                for i in range(n_clusters):
                    row = tfidf_matrix[i]
                    # Get indices of top features
                    row_data = row.toarray()[0]
                    if row_data.sum() == 0:
                        cluster_names[i] = f"Topic_{i+1}"
                        continue
                        
                    top_indices = row_data.argsort()[-2:][::-1]
                    keywords = [feature_names[idx] for idx in top_indices]
                    # Clean keywords
                    clean_keywords = [k for k in keywords if k.isalpha()]
                    if clean_keywords:
                        topic_name = "_".join(clean_keywords).title()
                    else:
                        topic_name = f"Topic_{i+1}"
                    cluster_names[i] = topic_name
            else:
                 for i in range(n_clusters): cluster_names[i] = f"Topic_{i+1}"
        except Exception as e:
            logger.warning(f"Keyword extraction failed ({e}), using generic names.")
            for i in range(n_clusters):
                cluster_names[i] = f"Topic_{i+1}"

        # 4. Move and Index
        logger.info("Applying organization...")
        for i, doc in enumerate(docs):
            cluster_id = labels[i]
            topic = cluster_names.get(cluster_id, f"Topic_{cluster_id+1}")
            self._finalize_paper(doc, topic)

    def _finalize_paper(self, doc, topic):
        """
        Helper to move and index a pre-processed paper.
        """
        original_path = doc["path"]
        
        # Move
        dest_dir = Config.PAPERS_DIR / topic
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / original_path.name
        
        final_path = original_path
        if original_path != dest_path:
            try:
                shutil.move(str(original_path), str(dest_path))
                final_path = dest_path
                logger.info(f"Moved {original_path.name} to {topic}")
            except Exception as e:
                logger.error(f"Failed to move {original_path.name}: {e}")
            
        # Index
        try:
            file_hash = hashlib.sha256(doc["text"].encode()).hexdigest()
            metadata = {
                "path": str(final_path),
                "filename": final_path.name,
                "sha256": file_hash,
                "topic": topic
            }
            self.vector_store.add_paper_file(file_hash, doc["embedding"], metadata)
            
            if doc["chunks"]:
                 chunk_texts = [c["text"] for c in doc["chunks"]]
                 # Re-embed chunks
                 chunk_embeddings = self.text_embedder.embed_texts(chunk_texts)
                 chunk_ids = [f"{file_hash}_{i}" for i in range(len(doc["chunks"]))]
                 chunk_metadatas = []
                 for i, c in enumerate(doc["chunks"]):
                    m = c.copy()
                    del m["text"]
                    m["path"] = str(final_path)
                    m["filename"] = final_path.name
                    chunk_metadatas.append(m)
                 self.vector_store.add_paper_chunks(chunk_ids, chunk_embeddings, chunk_metadatas, chunk_texts)
        except Exception as e:
            logger.error(f"Failed to index {final_path.name}: {e}")

