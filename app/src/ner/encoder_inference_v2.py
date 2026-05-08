"""
encoder_inference_v1.py

Performs NER (token classification) inference using a HuggingFace Transformers model.
Unlike v1, this module does NOT rely on spaCy for sentence splitting or
pretokenization. Instead, it uses NLTK's PunktSentenceTokenizer for sentence segmentation
and handles long sentences by splitting them into token-safe chunks using the model's own
tokenizer. Inference is run in batches for efficiency.

Span offsets in the output are always relative to the original (unmodified) input text.

Author: Fernando Gallego
"""

from pathlib import Path
from transformers import pipeline
from app.config import device

from app.utils.text_preprocessing import build_inference_chunks
from app.utils.results_postprocessing import merge_contiguous_entities


class NerModel:
    """
    NER model wrapper that uses NLTK for sentence segmentation and the model's
    own tokenizer for splitting oversized sentences into safe chunks.

    Long sentences are split at token boundaries to respect the
    model's maximum sequence length.  Inference is batched across all chunks of
    a document for throughput efficiency.

    Args:
        model_checkpoint: Path to a local HuggingFace model directory.
        agg_strat:        Token aggregation strategy passed to the HF pipeline
                          (e.g. ``"simple"``, ``"first"``, ``"average"``).
        device:           Torch device string (``"cuda"``, ``"cpu"``).
                          Auto-detected if None.
        merge_entities:   If True, contiguous same-label entities are merged
                          after inference (see :func:`merge_contiguous_entities`).
        score_mode:       Score aggregation mode used during entity merging
                          (``"mean"``, ``"max"``, or ``"min"``).
    """

    def __init__(
        self,
        model_checkpoint: Path,
        agg_strat: str = "simple",
        merge_entities: bool = True,
        score_mode: str = "mean",
    ):
        self.device = device
        self.merge_entities = merge_entities
        self.score_mode = score_mode

        self.pipe = pipeline(
            task="token-classification",
            model=str(model_checkpoint),
            aggregation_strategy=agg_strat,
            device=self.device,
            stride=256,
        )

        # Compute the effective max token length the model can handle
        tokenizer_max = getattr(self.pipe.tokenizer, "model_max_length", 512)
        model_max = getattr(self.pipe.model.config, "max_position_embeddings", 512)
        special_tokens_getter = getattr(self.pipe.tokenizer, "num_special_tokens_to_add", lambda pair=False: 2 if pair else 1) # Fallback: if the tokenizer doesn't have the method, we define a lambda that returns a standard offset.
        if not isinstance(tokenizer_max, int) or tokenizer_max <= 0:
            tokenizer_max = 512
        if not isinstance(model_max, int) or model_max <= 0:
            model_max = 512

        self.safe_max_length = min(tokenizer_max, model_max) - special_tokens_getter(pair=False)

    def _predict_chunks(self, text: str, filename: str, batch_size: int) -> list[dict]:
        """
        Segment *text* into token-safe chunks, run batched inference, and return
        a flat list of entity dicts with offsets adjusted to *text*.

        Each entity dict contains:
            ``filename``, ``sent_id``, ``label``, ``start``, ``end``,
            ``score``, ``span``.
        """
        chunks = build_inference_chunks(text, self.pipe.tokenizer, self.safe_max_length)
        if not chunks:
            return []

        raw_preds = self.pipe([c["text"] for c in chunks], batch_size=batch_size)

        entities = []
        for chunk, preds in zip(chunks, raw_preds):
            for pred in preds:
                # Offsets from the pipeline are relative to the chunk text;
                # adding chunk["start"] converts them to offsets in *text*.
                global_start = chunk["start"] + int(pred["start"])
                global_end   = chunk["start"] + int(pred["end"])

                if not (0 <= global_start < global_end <= len(text)):
                    continue  # Discard malformed spans

                entities.append({
                    "filename": filename,
                    "sent_id":  chunk["sent_id"],
                    "start":    global_start,
                    "end":      global_end,
                    "ner_score":    round(float(pred.get("score", 0.0)), 4),
                    "span":     text[global_start:global_end],
                    "ner_class":    pred.get("entity_group", pred.get("entity")),
                })

        entities.sort(key=lambda e: (e["filename"], e["start"], e["end"]))
        return entities

    def _process_text(self, text: str, filename: str, batch_size: int) -> list[dict]:
        """
        Run full inference on a single *text* document and return its entities.

        Calls :meth:`_predict_chunks` and optionally merges contiguous entities
        via :func:`merge_contiguous_entities`.

        Removes the filename and sentence id that are only used for the merging, and have no use outside of it
        """
        entities = self._predict_chunks(text, filename, batch_size)
        if self.merge_entities and entities:
            entities = merge_contiguous_entities(entities, text, score_mode=self.score_mode)

        keys_to_remove = {"filename", "sent_id"}
        for ann in entities:
            for k in keys_to_remove:
                del ann[k] # assume the key always exists  (which it does), if not use ann.pop(k, None)
        return entities

    def infer(self, texts: list[str], batch_size: int = 16) -> list[list[dict]]:
        """
        Run inference on a list of documents.

        Args:
            texts:      Input documents as plain strings.
            batch_size: Number of chunks forwarded to the model in a single
                        GPU/CPU batch.

        Returns:
            A list of length ``len(texts)``, where each element is the list of
            entity dicts predicted for that document.
        """
        filenames = [f"doc_{i}" for i in range(len(texts))]
        return [
            self._process_text(text, filename, batch_size)
            for text, filename in zip(texts, filenames)
        ]


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def ner_inference_v2(
    texts: list[str],
    ner_models: list[Path],
    agg_strat: str = "simple",
    batch_size: int = 16,
    merge_entities: bool = True,
    score_mode: str = "mean",
) -> list[list[list[dict]]]:
    """
    Run NER inference across multiple models and multiple documents.

    Mirrors the signature of :func:`ner_inference` but uses the NLTK-based
    pipeline (no spaCy dependency, no pretokenization).

    Args:
        texts:          Input documents as plain strings.
        ner_models:     Paths to one or more HuggingFace model directories.
        device:         Torch device. Auto-detected if None.
        agg_strat:      Token aggregation strategy for the HF pipeline.
        batch_size:     Chunk batch size for GPU inference.
        merge_entities: Whether to merge contiguous same-label entities.
        score_mode:     Score aggregation for merged entities.

    Returns:
        A list of shape ``[n_models][n_texts][n_entities]``.
    """
    results = []
    for model_checkpoint in ner_models:
        model = NerModel(
            model_checkpoint,
            agg_strat=agg_strat,
            merge_entities=merge_entities,
            score_mode=score_mode,
        )
        results.append(model.infer(texts, batch_size=batch_size))
    return results