"""
simple_inference.py

This script performs NER (i.e., token classification) inference on a set of text files using a HuggingFace Transformers model.
It reads .txt files, runs the model, and writes .ann annotation files in BRAT format.

Usage example:
  python simple_inference.py -i <input_txt_dir> -o <output_ann_dir> -m <model_path> [--overwrite] [--agg_strat <strategy>]

Author: Jan Rodríguez Miret
"""
import torch
from transformers import pipeline
from pathlib import Path
from app.config import device
from spacy.lang.es import Spanish
from spacy.lang.en import English
from spacy.lang.it import Italian
from spacy.lang.ro import Romanian
from spacy.lang.cs import Czech      # 'cz' is non-standard; Czech ISO 639-1 is 'cs'
from spacy.lang.sv import Swedish    # 'se' is Northern Sami; Swedish is 'sv'
from spacy.lang.nl import Dutch

from app.utils.text_preprocessing import pretokenize_sentence
from app.utils.results_postprocessing import align_results

SPACY_LANG_MAP: dict[str, type] = {
    'es': Spanish,
    'en': English,
    'it': Italian,
    'ro': Romanian,
    'cz': Czech,   # non-standard code, mapped to Czech
    'cs': Czech,   # standard code also supported
    'se': Swedish, # ambiguous code, mapped to Swedish
    'sv': Swedish, # standard code also supported
    'nl': Dutch,
}


class NerModel:
    def __init__(
        self,
        model_checkpoint: Path,
        agg_strat: str = "first",
        lang: str = "es",
    ):
        lang_class = SPACY_LANG_MAP.get(lang)
        if lang_class is None:
            raise ValueError(
                f"Unsupported language '{lang}'. "
                f"Supported codes: {sorted(SPACY_LANG_MAP.keys())}"
            )
        self.nlp = lang_class()
        self.nlp.add_pipe("sentencizer")

        self.device = device

        self.pipe = pipeline(
            task="token-classification",
            model=str(model_checkpoint),
            aggregation_strategy=agg_strat,
            device=self.device,
        )

    def _process_sentence(self, sentence: str, sentence_start_offset: int) -> list[dict]:
        # Pretokenize sentence for model compatibility
        sentence_pretokenized, added_spaces_pos = pretokenize_sentence(sentence)
        # Run model inference
        results_pre = self.pipe(sentence_pretokenized)
        # Convert numpy types to native Python types for JSON serialization
        for entity in results_pre:
            str_score = entity.pop('score')
            entity['ner_score'] = str(round(float(str_score), 4))
        # Align model results to original text offsets
        results = align_results(results_pre, added_spaces_pos, sentence_start_offset)
        return results

    def _process_text(self, text: str) -> list[dict]:
        results_text = []
        line_start_offset = 0  # Track the offset of the start of each line in the file
        for line in text.splitlines():
            doc = self.nlp(line)
            sents = list(doc.sents)
            for sentence in sents:
                results_sent = self._process_sentence(sentence.text, sentence.start_char + line_start_offset)
                results_text.extend(results_sent)
            line_start_offset += len(line) + 1 # account for the '\n' character
        return results_text
    
    def infer(self, texts: list[str]) -> list[list[dict]]:
        return [self._process_text(text) for text in texts]



def ner_inference_v1(
    texts: list[str],
    ner_models: list[Path],
    agg_strat: str = "first",
    lang: str = "es",
) -> list[list[list[dict]]]:
    results = []
    for model_checkpoint in ner_models:
        ner_model = NerModel(model_checkpoint, agg_strat=agg_strat, lang=lang)
        results_model = ner_model.infer(texts)
        results.append(results_model)
    return results
