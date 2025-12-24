import os
import time
import requests
from pathlib import Path

def download_paper(arxiv_id, title, category, save_dir):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    filename = f"{category}_{title.replace(' ', '_')}.pdf"
    filepath = save_dir / filename
    
    if filepath.exists():
        print(f"Skipping {filename} (already exists)")
        return

    print(f"Downloading {filename} from {url}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Successfully downloaded {filename}")
        
        # Be nice to ArXiv
        time.sleep(1) 
        
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

def main():
    root = Path(__file__).parent
    pdf_dir = root / "pdf"
    pdf_dir.mkdir(exist_ok=True)
    
    papers = [
        # Computer Vision
        ("2010.11929", "ViT_Image_Worth_16x16_Words", "CV"),
        ("2103.14030", "Swin_Transformer", "CV"),
        ("2304.02643", "Segment_Anything", "CV"),
        
        # Recommendation Systems
        ("2002.02126", "LightGCN", "RecSys"),
        ("2008.13535", "DCN_V2", "RecSys"),
        ("2203.13366", "P5_Pretrain_Personalized_Prompt", "RecSys"),
        
        # Large Language Models
        ("2005.14165", "GPT3_Language_Models_Few_Shot", "LLM"),
        ("2201.11903", "Chain_of_Thought", "LLM"),
        ("2302.13971", "LLaMA_Open_Foundation_Models", "LLM"),
    ]
    
    print(f"Saving papers to {pdf_dir}")
    
    for arxiv_id, title, category in papers:
        download_paper(arxiv_id, title, category, pdf_dir)

if __name__ == "__main__":
    main()
