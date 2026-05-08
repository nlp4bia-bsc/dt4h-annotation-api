import sys, os
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from app.config import device

from app.utils.model_utils import DenseRetriever
from app.utils.download_model import load_as_torch_tensor


class BiencoderModel:
    def __init__(self, gaz_pth: Path, model_pth: Path, vector_db_pth: Path):
        self.device = device

        self.st_model = SentenceTransformer(str(model_pth)).to(self.device)
        self.gazetteer = pd.read_csv(gaz_pth, sep='\t')
        self.gazetteer.drop_duplicates(subset=["term"], inplace=True)

        self.vector_db = load_as_torch_tensor(vector_db_pth, gazz_terms=len(self.gazetteer))
        self.biencoder = DenseRetriever(
            gazeteer_df=self.gazetteer, 
            vector_db=self.vector_db, 
            model_or_path=self.st_model
        )
    def run_nel_inference(self, input_mentions: list, k: int=1) -> pd.DataFrame:
        """
        Returns a dataframe where the index is the span and the and the vaues are the code, term, and simmilarity. It can be accessed through df.loc['covid'] --> 1119302008 / 'COVID-19 agudo' / 0.7942
        """
        mentions = list(set(input_mentions)) # filter duplicates

        candidates = self.biencoder.retrieve_top_k(
            mentions, 
            k=k, 
            input_format="text",
            return_documents=True
        )

        candidates_df = pd.DataFrame(candidates)
        # convert 1-element-list values to single values
        candidates_df = candidates_df.explode(['codes', 'terms', 'similarity'])
        candidates_df = candidates_df.rename(columns={'codes': 'code', 'terms': 'term'}) # singular 
        candidates_df["similarity"] = candidates_df["similarity"].apply(
            lambda sim: round(sim, 4)
        )
        return candidates_df.set_index('mention')

def biencoder_inference(ner_results: list[list[list[dict]]], nel_model_pth: Path, gaz_path_list: list[Path], vector_db_path_list: list[Path]) -> list[list[list[dict]]]:
    """
    ner_results = [//result level
        [// entity type level
            [// doc level
                {'start': 136, 'end': 159, 'ner_score': 0.9999, 'span': 'varicela con meningitis', 'ner_class': 'ENFERMEDAD'}
            ], 
            [
                {'start': 15, 'end': 20, 'ner_score': 0.9999, 'span': 'covid', 'ner_class': 'ENFERMEDAD'}
            ]
        ], (...)
    ]

    nel_model_pth: path to bienncoder model (one per language)

    gaz_path_list, vector_db_path_list are pretty self explanatory.

    returns the same ner_results list of list of list of dict with extra keys for the normalized codes and the simmilarity to the original concept
    """

    assert len(ner_results) == len(gaz_path_list)
    assert len(ner_results) == len(vector_db_path_list)

    nerl_results = ner_results.copy()
    for ent_type_idx, (ent_type_mentions, gaz_pth, vector_db_pth) in enumerate(zip(ner_results, gaz_path_list, vector_db_path_list)): # will iterate over all entity types (both in ner results and nel models)
        mentions = [mention_dict['span'] for mention_doc in ent_type_mentions for mention_dict in mention_doc]
        if len(mentions) == 0:
            continue # no mentions for that entity type

        nel_model = BiencoderModel(
            gaz_pth=gaz_pth,
            model_pth=nel_model_pth,
            vector_db_pth=vector_db_pth,
        )
        
        output = nel_model.run_nel_inference(
            input_mentions=mentions,
            k=1 # we only want the top decision
        )

        for mention_doc in nerl_results[ent_type_idx]: # same as mentions list comprehension, exploit the FACT that it has same order
            for mention_dict in mention_doc:
                mention_dict["code"], mention_dict["term"], mention_dict["nel_score"] = output.loc[mention_dict["span"]]
                
    return nerl_results
