import unicodedata
import pandas as pd
from rank_bm25 import BM25Okapi

class BM25Method:
    def __init__(self, gaz_path: str):

        # obtain gazetteer
        self.gazetteer = pd.read_csv(gaz_path, sep = '\t').drop_duplicates(subset = ['term'])

        # obtain list of normalized terms
        self.clean_terms = self.gazetteer['term'].astype(str).apply(self._normalize).to_list()

        # create bm25 matrix (must be tokenized)
        self.bm25 = BM25Okapi([term.split() for term in self.clean_terms])

        # create lookup dict
        self.term_to_info = {
            clean: (original, code)
            for clean, original, code in zip(
                self.clean_terms,
                self.gazetteer['term'],
                self.gazetteer['code'],
            )
        }

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize('NFD', text.lower())
        return "".join(c for c in text if unicodedata.category(c) != 'Mn')

    def run_bm25okapi(self, mention: str):

        # normalize mention
        norm_mention = self._normalize(mention)

        # match mention
        if norm_mention in self.clean_terms:  # perfect match
            matched_term = norm_mention
            score = 1.0
        else:                                  # find closest match via BM25
            scores = self.bm25.get_scores(norm_mention.split())
            best_idx = int(scores.argmax())
            matched_term = self.clean_terms[best_idx]
            score = float(scores[best_idx])

        original_term, code = self.term_to_info.get(matched_term, (mention, "NO_MAP"))
        result = {
            "nel_class": "BM25OKAPI",
            "code": code,
            "term": original_term,
            "nel_score": score,
        }

        return result

def bm25okapi_inference(ner_results: list[list[list[dict]]], gaz_pths: list[str]) -> list[list[list[dict]]]:

    assert len(ner_results) == len(gaz_pths)

    nerl_results = ner_results.copy()

    for ent_type_idx, (ent_type_mentions, gaz) in enumerate(zip(nerl_results, gaz_pths)):
        mentions = [mention_dict['span'] for mention_doc in ent_type_mentions for mention_dict in mention_doc]
        if len(mentions) == 0:
            continue

        bm25_engine = BM25Method(gaz_path = gaz)

        for mention_doc in nerl_results[ent_type_idx]:
            for mention_dict in mention_doc:
                result = bm25_engine.run_bm25okapi(mention_dict['span'])
                mention_dict["code"] = result["code"]
                mention_dict["term"] = result["term"]
                mention_dict["nel_score"] = result["nel_score"]

    return nerl_results
