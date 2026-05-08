from typing import Protocol
from abc import abstractmethod

from app.model_manager.resolver import LocalResolver
from app.src.ner import encoder_inference
from app.src.nel import lookup_inference, fuzzymatch_inference, bm25okapi_inference, biencoder_inference
from app.src.negation.negation_utils import add_negation_uncertainty_attributes
from app.utils.results_postprocessing import join_all_entities


class AnnotationPipeline(Protocol):
    @abstractmethod
    def predict(self, texts: list[str]) -> list[list[dict]]:
        """
        Run the full annotation pipeline over a batch of input texts.

        Parameters
        ----------
        texts : list[str]
            A list of raw input strings. Each element is processed independently
            and may contain zero or more detectable entity mentions.

        Returns
        -------
        list[list[dict]]
            A list with the same length as `texts`. Each element corresponds to
            one input text and contains a list of annotation dictionaries.

            Each annotation dictionary represents a single detected entity
            mention enriched with normalization and contextual attributes,
            with the following schema:

            {
                "start": int,
                    Character-level start offset of the entity span
                    (inclusive, 0-based index).

                "end": int,
                    Character-level end offset of the entity span
                    (exclusive).

                "span": str,
                    Exact substring of the original text corresponding
                    to the detected entity mention.

                "ner_class": str,
                    Predicted named entity class/category.

                "ner_score": float,
                    Confidence score assigned by the NER component.

                "code": str,
                    Normalized identifier assigned by the entity
                    linking / normalization component.

                "term": str,
                    Canonical term associated with the predicted code.

                "nel_score": float,
                    Confidence score assigned by the normalization step.

                "is_negated": bool,
                    Whether the entity mention is predicted to be
                    negated in context.

                "negation_score": float,
                    Confidence score of the negation prediction.

                "is_uncertain": bool,
                    Whether the entity mention is predicted to be
                    expressed with uncertainty/speculation.

                "uncertainty_score": float,
                    Confidence score of the uncertainty prediction.
            }

        Notes
        -----
        - The outer list preserves input order.
        - If no entities are detected in a text, the corresponding element
          will be an empty list.
        - Character offsets must refer to the original, unmodified input text.
        """
        pass


class LookupPipeline(AnnotationPipeline):
    """Direct text → code lookup. No NER step needed."""

    def __init__(self, lang: str, entities: list[str]):
        self.resolver = LocalResolver()
        self.gaz_pths = [self.resolver.get_gaz_path(lang, e) for e in entities]

    def predict(self, texts: list[str]) -> list[list[dict]]:
        inference_results = lookup_inference(texts, self.gaz_pths)
        return join_all_entities(inference_results)


class FuzzyMatchPipeline(AnnotationPipeline):

    def __init__(
        self,
        lang: str,
        entities: list[str],
        method: str = "jaro_winkler",
        threshold: float = 0.7,
        agg_strat: str = "first",
    ):
        self.method = method
        self.threshold = threshold
        self.agg_strat = agg_strat

        self.resolver = LocalResolver()
        self.gaz_pths = [self.resolver.get_gaz_path(lang, e) for e in entities]
        self.ner_pths = [self.resolver.get_ner_path(lang, e)[0] for e in entities]

    def predict(self, texts: list[str]) -> list[list[dict]]:
        ner_results = ner_inference(texts, self.ner_pths, agg_strat=self.agg_strat)
        fuzzy_result = fuzzymatch_inference(ner_results, self.gaz_pths, self.method, self.threshold)
        return join_all_entities(fuzzy_result)


class BM25OkapiPipeline(AnnotationPipeline):

    def __init__(
        self,
        lang: str,
        entities: list[str],
        agg_strat: str = "first",
    ):
        self.agg_strat = agg_strat

        self.resolver = LocalResolver()
        self.gaz_pths = [self.resolver.get_gaz_path(lang, e) for e in entities]
        self.ner_pths = [self.resolver.get_ner_path(lang, e)[0] for e in entities]

    def predict(self, texts: list[str]) -> list[list[dict]]:
        ner_results = ner_inference(texts, self.ner_pths, agg_strat=self.agg_strat)
        bm25_result = bm25okapi_inference(ner_results, self.gaz_pths)
        return join_all_entities(bm25_result)


class BiencoderPipeline(AnnotationPipeline):
    """
    Full pipeline: NER → NEL (dense retrieval) → Negation.

    Parameters
    ----------
    lang : str
        Language code, e.g. "es". All texts in a request are assumed to share
        this language.
    entities : list[str]
        Entity types to detect, e.g. ["disease", "symptoms"].
        Each type must have a corresponding NER entry and gazetteer in the
        registry. A single NEL model and a single negation model are shared
        across all entity types for the language.
    negation: bool
        Whether or not to apply the negation NER models for that language, 
        therefore adding a negation attribute to each annotation entry.
    ner_version : int
        NER pre and postprocessing version to use. Model is called in the same
        way but inputs are chunked and postprocessed in the same way
    device : str 
        Torch device string, e.g. "cuda:0"
    """

    def __init__(
        self,
        lang: str,
        entities: list[str],
        negation: bool=True,
        ner_version: int=2,
    ):
        self.negation = negation
        self.lang = lang
        self.ner_version = ner_version

        self.resolver = LocalResolver()
        self.ner_paths = [self.resolver.get_ner_path(self.lang, e)[0] for e in (entities + ["negation"] if self.negation else entities)]
        self.nel_path = self.resolver.get_nel_path(self.lang)[0]
        self.gaz_paths = [self.resolver.get_gaz_path(self.lang, e) for e in entities]
        self.vdb_paths = [self.resolver.get_vector_db_path(self.lang, e)[0] for e in entities]

    def predict(self, texts: list[str]) -> list[list[dict]]:
        # use v2 encoder
        ner_results = encoder_inference(
            texts, self.ner_paths, version=self.ner_version
        )

        # If no negation, run the standard pipeline and exit
        if not self.negation:
            norm_results = biencoder_inference(
                ner_results, self.nel_path, self.gaz_paths, self.vdb_paths
            )
            return join_all_entities(norm_results)

        # If negation exists, handle the specialized pipeline
        neg_results = ner_results.pop()
        norm_results = biencoder_inference(
            ner_results, self.nel_path, self.gaz_paths, self.vdb_paths
        )
        norm_results = join_all_entities(norm_results)
        
        return add_negation_uncertainty_attributes(norm_results, neg_results)