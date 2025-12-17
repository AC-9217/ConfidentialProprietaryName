import sys
import os
import traceback

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTabWidget, QLabel, QLineEdit, QPushButton, QTextEdit, 
                             QFileDialog, QCheckBox, QListWidget, QListWidgetItem, 
                             QSplitter, QProgressBar, QMessageBox, QScrollArea, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QIcon, QAction

# Import backend services
try:
    from agent.services.paper_manager import PaperManager
    from agent.services.search_service import SearchService
    from agent.services.image_search import ImageSearchService
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    print("Please ensure you are running this from the correct environment and directory.")

class WorkerThread(QThread):
    """
    Generic worker thread to run tasks in the background.
    """
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, target, *args, **kwargs):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.target(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("本地多模态 AI 智能助手")
        self.resize(1000, 700)
        
        # Initialize Services
        self.paper_manager = None # Lazy init to avoid startup lag
        self.search_service = None
        self.image_service = None

        self.init_ui()

    def get_paper_manager(self):
        if not self.paper_manager:
            self.paper_manager = PaperManager()
        return self.paper_manager

    def get_search_service(self):
        if not self.search_service:
            self.search_service = SearchService()
        return self.search_service

    def get_image_service(self):
        if not self.image_service:
            self.image_service = ImageSearchService()
        return self.image_service

    def init_ui(self):
        self.central_widget = QTabWidget()
        self.setCentralWidget(self.central_widget)

        # Tabs
        self.add_paper_tab = self.create_add_paper_tab()
        self.search_paper_tab = self.create_search_paper_tab()
        self.search_image_tab = self.create_search_image_tab()
        self.manage_tab = self.create_manage_tab()

        self.central_widget.addTab(self.add_paper_tab, "添加论文")
        self.central_widget.addTab(self.search_paper_tab, "搜索论文")
        self.central_widget.addTab(self.search_image_tab, "以文搜图")
        self.central_widget.addTab(self.manage_tab, "系统管理")

        # Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")

    # --- Tab 1: Add Paper ---
    def create_add_paper_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # File Selection
        file_layout = QHBoxLayout()
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("选择 PDF 文件...")
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_pdf)
        file_layout.addWidget(self.file_path_input)
        file_layout.addWidget(browse_btn)

        # Topics
        self.topics_input = QLineEdit()
        self.topics_input.setPlaceholderText("主题 (逗号分隔, 例如: CV, NLP, RL)")

        # Options
        options_layout = QHBoxLayout()
        self.move_checkbox = QCheckBox("移动文件到对应主题文件夹")
        self.index_checkbox = QCheckBox("建立索引")
        self.index_checkbox.setChecked(True)
        options_layout.addWidget(self.move_checkbox)
        options_layout.addWidget(self.index_checkbox)
        options_layout.addStretch()

        # Action Button
        self.add_btn = QPushButton("添加论文")
        self.add_btn.clicked.connect(self.run_add_paper)
        self.add_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")

        # Log Output
        self.add_log = QTextEdit()
        self.add_log.setReadOnly(True)

        layout.addLayout(file_layout)
        layout.addWidget(QLabel("主题:"))
        layout.addWidget(self.topics_input)
        layout.addLayout(options_layout)
        layout.addWidget(self.add_btn)
        layout.addWidget(QLabel("日志:"))
        layout.addWidget(self.add_log)

        widget.setLayout(layout)
        return widget

    def browse_pdf(self):
        fname, _ = QFileDialog.getOpenFileName(self, "打开 PDF", "", "PDF Files (*.pdf)")
        if fname:
            self.file_path_input.setText(fname)

    def run_add_paper(self):
        path = self.file_path_input.text()
        if not path:
            QMessageBox.warning(self, "输入错误", "请选择文件。")
            return
        
        topics_str = self.topics_input.text()
        topics = [t.strip() for t in topics_str.split(",")] if topics_str else None
        move = self.move_checkbox.isChecked()
        index = self.index_checkbox.isChecked()

        self.add_log.append(f"正在添加论文: {path}...")
        self.add_btn.setEnabled(False)
        
        def task():
            pm = self.get_paper_manager()
            pm.add_paper(path, topics=topics, move=move, index=index)
            return "Success"

        self.worker = WorkerThread(task)
        self.worker.finished.connect(lambda res: self.on_add_paper_finished(res))
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()

    def on_add_paper_finished(self, result):
        self.add_log.append("论文添加成功。")
        self.add_btn.setEnabled(True)
        self.status_bar.showMessage("论文添加成功", 5000)

    # --- Tab 2: Search Paper ---
    def create_search_paper_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Search Bar
        search_layout = QHBoxLayout()
        self.paper_query_input = QLineEdit()
        self.paper_query_input.setPlaceholderText("请输入查询内容 (例如: 'Transformer 架构')")
        self.paper_query_input.returnPressed.connect(self.run_search_paper)
        search_btn = QPushButton("搜索")
        search_btn.clicked.connect(self.run_search_paper)
        search_layout.addWidget(self.paper_query_input)
        search_layout.addWidget(search_btn)

        # Options
        opts_layout = QHBoxLayout()
        self.return_snippets_cb = QCheckBox("返回文本片段")
        self.return_snippets_cb.setChecked(False)
        self.top_k_spin = QLineEdit("5") # Simple implementation
        self.top_k_spin.setFixedWidth(50)
        self.top_k_spin.setPlaceholderText("Top K")
        
        opts_layout.addWidget(self.return_snippets_cb)
        opts_layout.addWidget(QLabel("返回数量:"))
        opts_layout.addWidget(self.top_k_spin)
        opts_layout.addStretch()

        # Results Area
        self.paper_results_list = QListWidget()
        self.paper_results_list.itemClicked.connect(self.show_paper_details)
        
        self.paper_details_text = QTextEdit()
        self.paper_details_text.setReadOnly(True)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.paper_results_list)
        splitter.addWidget(self.paper_details_text)
        splitter.setSizes([300, 200])

        layout.addLayout(search_layout)
        layout.addLayout(opts_layout)
        layout.addWidget(splitter)

        widget.setLayout(layout)
        return widget

    def run_search_paper(self):
        query = self.paper_query_input.text()
        if not query:
            return

        try:
            top_k = int(self.top_k_spin.text())
        except:
            top_k = 5
        
        return_snippets = self.return_snippets_cb.isChecked()
        
        self.status_bar.showMessage("正在搜索...")
        self.paper_results_list.clear()
        self.paper_details_text.clear()

        def task():
            ss = self.get_search_service()
            return ss.search_paper(query, top_k=top_k, return_snippets=return_snippets, return_files=True)

        self.worker_search = WorkerThread(task)
        self.worker_search.finished.connect(self.on_search_paper_finished)
        self.worker_search.error.connect(self.on_worker_error)
        self.worker_search.start()

    def on_search_paper_finished(self, results):
        self.status_bar.showMessage("搜索完成。")
        self.last_search_results = results # Store for details view
        
        if "files" in results and results["files"] and results["files"]["ids"] and results["files"]["ids"][0]:
            files = results["files"]
            for i, doc_id in enumerate(files["ids"][0]):
                meta = files["metadatas"][0][i]
                dist = files["distances"][0][i] if files["distances"] else 0
                item_text = f"{i+1}. {meta.get('filename')} (相关度: {dist:.4f})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, meta)
                self.paper_results_list.addItem(item)
        else:
             self.paper_results_list.addItem("未找到相关文件。")

        # Show snippets in details if available
        if "snippets" in results and results["snippets"]:
             self.paper_details_text.append("--- 找到相关片段 (点击文件查看详情) ---\n")
             snippets = results["snippets"]
             if snippets and snippets["ids"] and snippets["ids"][0]:
                for i, doc_id in enumerate(snippets["ids"][0]):
                    text = snippets["documents"][0][i]
                    meta = snippets["metadatas"][0][i]
                    self.paper_details_text.append(f"[片段] {meta.get('filename')} (第 {meta.get('page_id')} 页):\n{text[:200]}...\n")

    def show_paper_details(self, item):
        meta = item.data(Qt.ItemDataRole.UserRole)
        if meta:
            details = f"文件名: {meta.get('filename')}\n"
            details += f"路径: {meta.get('path')}\n"
            details += f"主题: {meta.get('topic')}\n"
            self.paper_details_text.setText(details)

    # --- Tab 3: Search Image ---
    def create_search_image_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Search Bar
        search_layout = QHBoxLayout()
        self.img_query_input = QLineEdit()
        self.img_query_input.setPlaceholderText("描述图片内容 (例如: '海边日落')")
        self.img_query_input.returnPressed.connect(self.run_search_image)
        search_btn = QPushButton("搜索")
        search_btn.clicked.connect(self.run_search_image)
        search_layout.addWidget(self.img_query_input)
        search_layout.addWidget(search_btn)

        # Results Area
        self.img_scroll_area = QScrollArea()
        self.img_scroll_area.setWidgetResizable(True)
        self.img_results_container = QWidget()
        self.img_results_layout = QVBoxLayout(self.img_results_container)
        self.img_scroll_area.setWidget(self.img_results_container)

        layout.addLayout(search_layout)
        layout.addWidget(self.img_scroll_area)

        widget.setLayout(layout)
        return widget

    def run_search_image(self):
        query = self.img_query_input.text()
        if not query:
            return

        self.status_bar.showMessage("正在搜索图片...")
        
        # Clear previous results
        for i in reversed(range(self.img_results_layout.count())): 
            self.img_results_layout.itemAt(i).widget().setParent(None)

        def task():
            iss = self.get_image_service()
            return iss.search_image(query, top_k=5)

        self.worker_img = WorkerThread(task)
        self.worker_img.finished.connect(self.on_search_image_finished)
        self.worker_img.error.connect(self.on_worker_error)
        self.worker_img.start()

    def on_search_image_finished(self, results):
        self.status_bar.showMessage("图片搜索完成。")
        
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                path = meta.get('path')
                dist = results["distances"][0][i] if results["distances"] else 0
                
                # Container for one result
                item_widget = QFrame()
                item_widget.setFrameShape(QFrame.Shape.StyledPanel)
                item_layout = QHBoxLayout(item_widget)
                
                # Image Preview
                lbl_img = QLabel()
                lbl_img.setFixedSize(200, 200)
                lbl_img.setStyleSheet("background-color: #ddd;")
                if os.path.exists(path):
                    pixmap = QPixmap(path)
                    if not pixmap.isNull():
                        lbl_img.setPixmap(pixmap.scaled(lbl_img.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                    else:
                        lbl_img.setText("无效图片")
                else:
                    lbl_img.setText("文件未找到")
                
                # Info
                lbl_info = QLabel(f"<b>{meta.get('filename')}</b><br>相关度: {dist:.4f}<br>路径: {path}")
                lbl_info.setWordWrap(True)
                
                item_layout.addWidget(lbl_img)
                item_layout.addWidget(lbl_info)
                
                self.img_results_layout.addWidget(item_widget)
        else:
            self.img_results_layout.addWidget(QLabel("未找到相关图片。"))

    # --- Tab 4: Management ---
    def create_manage_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Batch Organize
        group_batch = QFrame()
        group_batch.setFrameShape(QFrame.Shape.StyledPanel)
        gb_layout = QVBoxLayout(group_batch)
        gb_layout.addWidget(QLabel("<b>批量整理论文</b>"))
        
        self.batch_root_input = QLineEdit()
        self.batch_root_input.setPlaceholderText("根目录 (例如 data/papers)")
        browse_root = QPushButton("选择目录")
        browse_root.clicked.connect(lambda: self.browse_dir(self.batch_root_input))
        
        root_layout = QHBoxLayout()
        root_layout.addWidget(self.batch_root_input)
        root_layout.addWidget(browse_root)
        
        self.batch_topics = QLineEdit()
        self.batch_topics.setPlaceholderText("主题 (CV, NLP, RL)")
        
        btn_organize = QPushButton("开始整理")
        btn_organize.clicked.connect(self.run_batch_organize)
        
        gb_layout.addLayout(root_layout)
        gb_layout.addWidget(self.batch_topics)
        gb_layout.addWidget(btn_organize)

        # Index Images
        group_idx = QFrame()
        group_idx.setFrameShape(QFrame.Shape.StyledPanel)
        gi_layout = QVBoxLayout(group_idx)
        gi_layout.addWidget(QLabel("<b>建立图片索引</b>"))
        
        self.idx_dir_input = QLineEdit()
        self.idx_dir_input.setPlaceholderText("图片目录 (可选, 默认: data/images)")
        browse_idx = QPushButton("选择目录")
        browse_idx.clicked.connect(lambda: self.browse_dir(self.idx_dir_input))
        
        idx_layout = QHBoxLayout()
        idx_layout.addWidget(self.idx_dir_input)
        idx_layout.addWidget(browse_idx)
        
        btn_index = QPushButton("开始建立索引")
        btn_index.clicked.connect(self.run_index_images)
        
        gi_layout.addLayout(idx_layout)
        gi_layout.addWidget(btn_index)

        # Log
        self.manage_log = QTextEdit()
        self.manage_log.setReadOnly(True)

        layout.addWidget(group_batch)
        layout.addWidget(group_idx)
        layout.addWidget(QLabel("管理日志:"))
        layout.addWidget(self.manage_log)
        
        widget.setLayout(layout)
        return widget

    def browse_dir(self, line_edit):
        dname = QFileDialog.getExistingDirectory(self, "选择目录")
        if dname:
            line_edit.setText(dname)

    def run_batch_organize(self):
        root = self.batch_root_input.text()
        topics_str = self.batch_topics.text()
        
        if not root or not topics_str:
             QMessageBox.warning(self, "错误", "必须指定根目录和主题。")
             return
        
        topics = [t.strip() for t in topics_str.split(",")]
        
        self.manage_log.append(f"正在开始批量整理 {root}...")
        
        def task():
            pm = self.get_paper_manager()
            pm.batch_organize(root, topics)
            return "批量整理完成。"

        self.worker_batch = WorkerThread(task)
        self.worker_batch.finished.connect(lambda res: self.manage_log.append(res))
        self.worker_batch.error.connect(self.on_worker_error)
        self.worker_batch.start()

    def run_index_images(self):
        target_dir = self.idx_dir_input.text() if self.idx_dir_input.text() else None
        self.manage_log.append("正在开始建立图片索引...")
        
        def task():
            iss = self.get_image_service()
            if target_dir:
                iss.index_images(target_dir)
            else:
                iss.index_images()
            return "图片索引建立完成。"

        self.worker_idx = WorkerThread(task)
        self.worker_idx.finished.connect(lambda res: self.manage_log.append(res))
        self.worker_idx.error.connect(self.on_worker_error)
        self.worker_idx.start()

    def on_worker_error(self, err_msg):
        self.status_bar.showMessage("发生错误")
        QMessageBox.critical(self, "错误", f"发生错误:\n{err_msg}")
        # Also print to logs if available
        if hasattr(self, 'add_log'): self.add_log.append(f"错误: {err_msg}")
        if hasattr(self, 'manage_log'): self.manage_log.append(f"错误: {err_msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
