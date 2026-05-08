from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import torch
import numpy as np
import gc
from tqdm import tqdm
from app.config import device

# def HF_download_model(repo_id: str, path: Path):
#     '''
#     Given a model repo_id and path name, it downlaods the model in model cache and returns the new local path
#     '''
#     snapshot_download(
#         repo_id = repo_id,
#         local_dir = path
#     )


# def create_vector_db(gazetteer: pd.DataFrame, nel_model: SentenceTransformer, vector_db_path: Path, device: str, chunk_size: int=10000): # Smaller chunk size
#     terms = gazetteer["term"].tolist()
#     num_terms = len(terms)
#     embedding_dim = 768 

#     fp = np.memmap(vector_db_path, dtype=np.float32, mode='w+', shape=(num_terms, embedding_dim))

#     print("Computing vector database...")
#     for i in tqdm(range(0, num_terms, chunk_size)):
#         end_idx = min(i + chunk_size, num_terms)
#         chunk = terms[i:end_idx]

#         with torch.no_grad(): # Ensure no gradients are stored (saves massive memory)
#             embeddings = nel_model.encode(
#                 chunk,
#                 convert_to_numpy=True,
#                 normalize_embeddings=True,
#                 batch_size=1024,
#                 device=device
#             )
#         if "cuda" in str(device):
#             torch.cuda.empty_cache()

#         fp[i:end_idx, :] = embeddings
#         fp.flush()
#         del chunk, embeddings
#         gc.collect()

#     del fp 

# import multiprocessing as mp

def create_vector_db(gaz_terms: list[str], nel_model: SentenceTransformer, vector_db_path: Path, chunk_size: int=10000):
    num_terms = len(gaz_terms)
    embedding_dim = 768 

    # 1. Initialize memmap
    fp = np.memmap(vector_db_path, dtype='float32', mode='w+', shape=(num_terms, embedding_dim))

    print(f"Computing vector database for {num_terms} terms...")
    try:
        for i in tqdm(range(0, num_terms, chunk_size)):
            end_idx = min(i + chunk_size, num_terms)
            chunk = gaz_terms[i:end_idx]

            # 2. Use inference_mode for even better optimization than no_grad
            with torch.inference_mode():
                embeddings = nel_model.encode(
                    chunk,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=1024, # Lowered slightly for stability
                    device=device,
                    show_progress_bar=False
                )

            fp[i:end_idx, :] = embeddings
            
            # 3. Flush to disk to keep RAM usage low
            fp.flush()

            # Explicitly clear chunk-level memory
            del embeddings
            if "cuda" in str(device):
                torch.cuda.empty_cache()

    finally:
        # 4. CRITICAL: Ensure the memmap is closed and deleted even if a crash occurs
        # This prevents the file handle from staying open
        del fp
        gc.collect()
        if "cuda" in str(device):
            torch.cuda.empty_cache()
            torch.cuda.synchronize() # Wait for GPU to finish cleanup


def load_as_torch_tensor(vector_db_path: Path, gazz_terms: int, embedding_dim: int = 768) -> torch.Tensor:
    nmap = np.memmap(vector_db_path, dtype='float32', mode='r', shape=(gazz_terms, embedding_dim))
    torch_db = torch.from_numpy(nmap)
    return torch_db.to(device=device)