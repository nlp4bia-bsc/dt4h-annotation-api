import unicodedata
import pandas as pd
from flashtext import KeywordProcessor


class LookUpMethod:
    def __init__(self, gaz_pth: str):
        
        # load gazetteer
        self.gazetteer = pd.read_csv(gaz_pth, sep='\t').drop_duplicates(subset=["term"])
        
        # normalize ontology
        clean_terms = self.gazetteer['term'].astype(str).apply(self._normalize).to_list()

        # create lookup dict
        self.term_to_info = {
            clean: (original, code)
            for clean, original, code in zip(
                clean_terms, 
                self.gazetteer['term'], 
                self.gazetteer['code'])
        }
        
        # build and populate processor engine
        self.keyword_processor = KeywordProcessor(case_sensitive = False) # we are already normalizing everything to lowercase
        self.keyword_processor.add_keywords_from_list(clean_terms)
                    
    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize('NFD', text.lower())
        return "".join(c for c in text if unicodedata.category(c) != 'Mn')
        
    def run_lookup(self, text: str) -> list[dict]:
        # normalize text
        norm_text = self._normalize(text)
        
        # find matches in normalized text
        matches = self.keyword_processor.extract_keywords(norm_text, span_info=True)
        results = []
        
        # for each match, store results in CDM structure
        for matched_term, start, end in matches:
            original_term, code = self.term_to_info.get(matched_term, (matched_term, "NO_MAP"))
            results.append({
                "start": start,
                "end": end,
                "span": text[start:end],
                "ner_class": "LOOKUP",
                "ner_score": 1.0,
                "code": code,
                "term": original_term,
                "nel_score": 1.0,
            })
        return results


def lookup_inference(texts: list[str], gaz_pths: list[str]) -> list[list[list[dict]]]:  
    
    results = []
    
    # extract words for each gazetteer
    for gaz in gaz_pths:
    
        lookup_engine = LookUpMethod(gaz)

        gaz_results = []
        for text in texts:
            text_results = []
            text_results.extend(lookup_engine.run_lookup(text))
            #text_results.sort(key=lambda x: (x['start'], -x['end']))
            gaz_results.append(text_results)
    
        results.append(gaz_results)
        
    return results
