import argparse
import sys
from agent.services.paper_manager import PaperManager
from agent.services.search_service import SearchService
from agent.services.image_search import ImageSearchService
from agent.utils.logging import setup_logger

logger = setup_logger("main")

def main():
    """
    Main entry point for the command line interface.
    命令行接口的主入口点。
    """
    parser = argparse.ArgumentParser(description="Local Multimodal AI Agent")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # add_paper
    # 添加论文命令
    parser_add = subparsers.add_parser("add_paper", help="Add and optionally classify a paper")
    parser_add.add_argument("path", help="Path to the PDF file")
    parser_add.add_argument("--topics", help="Comma-separated list of topics (e.g. CV,NLP,RL)", default="")
    parser_add.add_argument("--move", action="store_true", help="Move file to topic folder")
    parser_add.add_argument("--no-index", action="store_true", help="Skip indexing")

    # search_paper
    # 搜索论文命令
    parser_search = subparsers.add_parser("search_paper", help="Search for papers")
    parser_search.add_argument("query", help="Search query")
    parser_search.add_argument("--top_k", type=int, default=5, help="Number of results")
    parser_search.add_argument("--return_snippets", action="store_true", help="Return text snippets")
    parser_search.add_argument("--return_files", action="store_true", default=True, help="Return file list")

    # search_image
    # 搜索图片命令
    parser_img_search = subparsers.add_parser("search_image", help="Search for images using text")
    parser_img_search.add_argument("query", help="Search query")
    parser_img_search.add_argument("--top_k", type=int, default=5, help="Number of results")
    
    # batch_organize
    # 批量整理命令
    parser_batch = subparsers.add_parser("batch_organize", help="Batch organize papers in a directory")
    parser_batch.add_argument("--root", required=True, help="Root directory containing papers")
    parser_batch.add_argument("--topics", default=None, help="Comma-separated list of topics (optional, auto-detect if omitted)")
    
    # index_images (New helper)
    # 索引图片命令 (新助手)
    parser_idx_img = subparsers.add_parser("index_images", help="Index all images in data/images or specified dir")
    parser_idx_img.add_argument("--dir", default=None, help="Directory to index")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "add_paper":
            topics_list = [t.strip() for t in args.topics.split(",")] if args.topics else None
            pm = PaperManager()
            pm.add_paper(args.path, topics=topics_list, move=args.move, index=not args.no_index)
            
        elif args.command == "search_paper":
            ss = SearchService()
            results = ss.search_paper(args.query, top_k=args.top_k, return_snippets=args.return_snippets, return_files=args.return_files)
            
            print("\n--- Search Results ---")
            if "files" in results:
                print("\n[Files]")
                files = results["files"]
                # Chroma returns dict with lists.
                # Chroma 返回包含列表的字典。
                if files and files["ids"] and files["ids"][0]:
                    for i, doc_id in enumerate(files["ids"][0]):
                        meta = files["metadatas"][0][i]
                        dist = files["distances"][0][i] if files["distances"] else 0
                        # Distance in Chroma is typically L2 or Cosine distance. Smaller is better if using L2, but we used normalize_embeddings.
                        # If normalize_embeddings=True and distance metric is cosine (default in newer chroma might be l2), 
                        # Cosine Distance = 1 - Cosine Similarity. So smaller is better.
                        # Chroma 中的距离通常是 L2 或余弦距离。如果使用 L2，越小越好，但我们使用了 normalize_embeddings。
                        # 如果 normalize_embeddings=True 且距离度量是余弦（较新的 Chroma 中默认可能是 L2），
                        # 余弦距离 = 1 - 余弦相似度。所以越小越好。
                        print(f"{i+1}. {meta.get('filename')} (Dist: {dist:.4f})")
                        print(f"   Path: {meta.get('path')}")
                        print(f"   Topic: {meta.get('topic')}")
                else:
                    print("No files found.")

            if "snippets" in results:
                print("\n[Snippets]")
                snippets = results["snippets"]
                if snippets and snippets["ids"] and snippets["ids"][0]:
                    for i, doc_id in enumerate(snippets["ids"][0]):
                        text = snippets["documents"][0][i]
                        meta = snippets["metadatas"][0][i]
                        print(f"{i+1}. {meta.get('filename')} (Page {meta.get('page_id')}): {text[:100].replace(chr(10), ' ')}...")
                else:
                    print("No snippets found.")

        elif args.command == "search_image":
            iss = ImageSearchService()
            results = iss.search_image(args.query, top_k=args.top_k)
            print("\n--- Image Search Results ---")
            if results and results["ids"] and results["ids"][0]:
                 for i, doc_id in enumerate(results["ids"][0]):
                        meta = results["metadatas"][0][i]
                        dist = results["distances"][0][i] if results["distances"] else 0
                        print(f"{i+1}. {meta.get('filename')} (Dist: {dist:.4f})")
                        print(f"   Path: {meta.get('path')}")
            else:
                print("No images found.")

        elif args.command == "batch_organize":
            topics_list = [t.strip() for t in args.topics.split(",")] if args.topics else None
            pm = PaperManager()
            pm.batch_organize(args.root, topics_list)

        elif args.command == "index_images":
            iss = ImageSearchService()
            target_dir = args.dir if args.dir else None
            if target_dir:
                iss.index_images(target_dir)
            else:
                iss.index_images() # Uses default config
                # 使用默认配置

    except Exception as e:
        logger.error(f"Error executing command: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
