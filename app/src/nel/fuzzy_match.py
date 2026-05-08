import unicodedata
import pandas as pd
from rapidfuzz import process, distance, fuzz 

class FuzzyMatchMethod:
    def __init__(self, gaz_path: str, method: str, threshold: float):
        
        self.method = method
        self.threshold = threshold
        
        self.SCORERS = {
            "levenshtein": distance.Levenshtein.normalized_similarity,
            "jaro-winkler": distance.JaroWinkler.normalized_similarity,
            "token-sort-ratio": fuzz.token_sort_ratio,
            "token-set-ratio": fuzz.token_set_ratio,
        } 
        
        self.SCORE_SCALE = {
            "levenshtein": 1,
            "jaro-winkler": 1,
            "token-sort-ratio": 100,
            "token-set-ratio": 100,   
        }

        # obtain gazetteer without duplicates
        self.gazetteer = pd.read_csv(gaz_path, sep = '\t').drop_duplicates(subset = ['term'])  
              
        # store normalized terms
        self.clean_terms = self.gazetteer['term'].astype(str).apply(self._normalize).to_list()
        
        # create code lookup dict
        self.term_to_info = {
            clean: (original, code)
            for clean, original, code in zip(
                self.clean_terms,
                self.gazetteer['term'],
                self.gazetteer['code']
            )
        }

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize('NFD', text.lower())
        return "".join(c for c in text if unicodedata.category(c) != 'Mn')
    
    def run_fuzzymatch(self, mention: str):
        
        # define scorer
        scorer = self.SCORERS.get(self.method)
        if scorer is None:
            raise ValueError(f"Unkown method: '{self.method}'. Valid options: {list(self.SCORERS)}")
        
        # check threhsold value
        score_scale = self.SCORE_SCALE.get(self.method)
        if not (0 <= self.threshold <= 1):
            raise ValueError(f"Threshold must be [0, 1] ")
        
        # normalize extracted mentions
        norm_mention = self._normalize(mention)
                
        # link mention
        if norm_mention in self.clean_terms: # check if exact match already exists
            matched_term = norm_mention
            score = 1.0
        else: # find highest scoring match
            match, score_unnorm, _ = process.extractOne(norm_mention, self.clean_terms, scorer = scorer)
            score = score_unnorm / score_scale # score [0,1]
            if  score >= self.threshold:
                matched_term = match
            else:
                matched_term = None # no match found
            
        # store results in dict
        original_term, code = self.term_to_info.get(matched_term, (mention, "NO_MAP"))
        result = {
            "nel_class": f"FUZZYMATCH_{self.method.upper()}",
            "code": code,
            "term": original_term,
            "nel_score": score,
        }
        
        return result
            
def fuzzymatch_inference(ner_results: list[list[list[dict]]], gaz_pths: list[str], method: str, threshold: float) -> list[list[list[dict]]]:

    assert len(ner_results) == len(gaz_pths)

    nerl_results = ner_results.copy()
    
    for ent_type_idx, (ent_type_mentions, gaz) in enumerate(zip(nerl_results, gaz_pths)):
        mentions = [mention_dict['span'] for mention_doc in ent_type_mentions for mention_dict in mention_doc]
        if len(mentions) == 0:
            continue

        fuzzy_engine = FuzzyMatchMethod(gaz_path = gaz, method = method, threshold = threshold)

        for mention_doc in ner_results[ent_type_idx]:
            for mention_dict in mention_doc:
                result = fuzzy_engine.run_fuzzymatch(mention_dict['span'])
                mention_dict["code"] = result["code"]
                mention_dict["term"] = result["term"]
                mention_dict["nel_score"] = result["nel_score"]

    return nerl_results
