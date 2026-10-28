from pathlib import Path
from huggingface_hub import hf_hub_download
root=Path(__file__).resolve().parent
print('Downloading real pretrained traffic sign model (~23 MB). Internet required once...')
path=hf_hub_download(repo_id='ankitjha07/Traffic-Sign-Detection-YOLOv8',filename='best.pt',local_dir=str(root/'models'))
print('Saved model:',path)
