"""
NER Inference Interface
=======================
This module exposes a single unified entry point, ``encoder_inference``, for all 
NER encoder-based inference backends. Any future inference function registered here  
MUST conform to the following contract so that it can be transparently swapped via 
the ``version`` argument.

Interface Contract
------------------
Signature
~~~~~~~~~
At minimum, every compliant inference function must accept:

    def ner_inference_vN(
        texts: list[str],
        ner_models: list[Path],
        device: str | None = ...,
        agg_strat: str = ...,
        **kwargs,                  # version-specific extras are allowed
    ) -> list[list[list[dict]]]: ...

Return type
~~~~~~~~~~~
The function must return a three-level nested list with shape::

    [text_i][model_j][entity_k]

where each ``entity_k`` is a dict with **at least** the following keys:

    {
        "start":     int,    # character offset of the entity start in the original text
        "end":       int,    # character offset of the entity end   (exclusive)
        "ner_score": float,  # confidence score produced by the NER model
        "span":      str,    # raw text slice  texts[i][start:end]
        "ner_class": Literal["ENFERMEDAD", "PROCEDIMIENTO", "SINTOMA", "NEGACIÓN"],
    }

    # The following keys are added downstream by the NEL (entity-linking) stage
    # and need NOT be present in the raw NER output:
    #   "code":      str    – normalised concept code
    #   "term":      str    – canonical term for the linked concept
    #   "nel_score": float  – entity-linking confidence score

Behaviour constraints
~~~~~~~~~~~~~~~~~~~~~
* The function must be **stateless**: no global or shared mutable state.
* If ``texts`` is empty the function must return ``[]`` without raising.
* If a model path in ``ner_models`` does not exist, the function must raise
  ``FileNotFoundError`` with a descriptive message before any GPU memory is
  allocated.
* The function must honour the ``device`` argument and must not silently fall
  back to a different device without logging a warning.
"""

from pathlib import Path
from typing import Optional

from .encoder_inference_v1 import ner_inference_v1
from .encoder_inference_v2 import ner_inference_v2


def encoder_inference(
    # ── required ────────────────────────────────────────────────────────────
    texts: list[str],
    ner_models: list[Path],
    # ── routing ─────────────────────────────────────────────────────────────
    version: int = 2,
    # ── shared ──────────────────────────────────────────────────────────────
    agg_strat: Optional[str] = None,
    # ── v1-only ─────────────────────────────────────────────────────────────
    lang: str = "es",
    # ── v2-only ─────────────────────────────────────────────────────────────
    batch_size: int = 16,
    merge_entities: bool = True,
    score_mode: str = "mean",
) -> list[list[list[dict]]]:
    """
    Unified entry point for NER inference.

    Args:
        texts:          Input texts to annotate.
        ner_models:     Paths to the model checkpoints.
        version:        Which inference backend to use (1 or 2). Defaults to 2.
        device:         Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
        agg_strat:      Aggregation strategy passed to the underlying pipeline.
                        Defaults to ``"first"`` for v1 and ``"simple"`` for v2.
        lang:           **(v1 only)** Language code used for pre/post-processing.
                        Defaults to ``"es"``.
        batch_size:     **(v2 only)** Number of texts per inference batch.
                        Defaults to ``16``.
        merge_entities: **(v2 only)** Whether to merge adjacent entities of the
                        same class. Defaults to ``True``.
        score_mode:     **(v2 only)** Strategy for aggregating per-token scores
                        into a single entity score (``"mean"`` | ``"max"`` | ``"min"``).
                        Defaults to ``"mean"``.

    Returns:
        A three-level nested list ``[text_i][model_j][entity_k]``.
        See the module docstring for the full entity-dict schema.

    Raises:
        ValueError: If ``version`` is not a recognised backend identifier.
    """
    if version == 1:
        return ner_inference_v1(
            texts=texts,
            ner_models=ner_models,
            agg_strat=agg_strat if agg_strat is not None else "first",
            lang=lang,
        )
    elif version == 2:
        return ner_inference_v2(
            texts=texts,
            ner_models=ner_models,
            agg_strat=agg_strat if agg_strat is not None else "simple",
            batch_size=batch_size,
            merge_entities=merge_entities,
            score_mode=score_mode,
        )
    else:
        raise ValueError(f"Unknown NER inference version {version!r}. Expected 1 or 2.")