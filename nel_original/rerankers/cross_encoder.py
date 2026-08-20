from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any

from ..utils.device import optional_module, resolve_torch_device

_sentence_transformers = optional_module("sentence_transformers")
_BaseCrossEncoder = _sentence_transformers.CrossEncoder if _sentence_transformers is not None else object


DEFAULT_TRAINING_LOGGING_STEPS = 10_000_000


def configure_quiet_transformers_logging() -> None:
    """Reduce noisy Transformer/SentenceTransformers logs while keeping tqdm usable."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("WANDB_DISABLED", "true")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    transformers = optional_module("transformers")
    if transformers is not None and hasattr(transformers, "utils") and hasattr(transformers.utils, "logging"):
        transformers.utils.logging.set_verbosity_error()


def resolve_cross_encoder_dataloader_num_workers(
    requested_num_workers: int,
    device: str = "auto",
    use_legacy_fit_loop: bool = True,
) -> int:
    """Return a safe DataLoader worker count for CrossEncoder training.

    SentenceTransformers' legacy ``old_fit`` loop installs a ``collate_fn`` that
    tokenizes the batch and moves labels/features to the model device. If CUDA is
    used with ``num_workers > 0``, PyTorch forks worker subprocesses and the
    collate function tries to initialize CUDA inside those forked workers, which
    raises ``Cannot re-initialize CUDA in forked subprocess``. The safe and
    notebook-friendly option is therefore ``num_workers=0`` for CUDA training.

    CPU training can keep the requested number of workers. The same conservative
    CUDA rule is also applied when not using ``old_fit`` because CrossEncoder
    versions differ in where smart batching is executed.
    """
    requested = max(0, int(requested_num_workers))
    resolved_device = resolve_torch_device(device)
    if str(resolved_device).startswith("cuda") and requested > 0:
        return 0
    return requested


def resolve_cross_encoder_dataloader_pin_memory(
    requested_pin_memory: bool,
    device: str = "auto",
) -> bool:
    """Return a safe ``pin_memory`` value for CrossEncoder training.

    SentenceTransformers' CrossEncoder smart batching collate function returns
    tensors already moved to the model device. With CUDA, PyTorch's DataLoader
    ``pin_memory=True`` then tries to pin CUDA tensors and fails because only
    dense CPU tensors can be pinned. Disable pinning for CUDA CrossEncoder
    training while preserving the requested value for CPU.
    """
    resolved_device = resolve_torch_device(device)
    if str(resolved_device).startswith("cuda"):
        return False
    return bool(requested_pin_memory)


class EntityLinkingCrossEncoder(_BaseCrossEncoder):  # type: ignore[misc]
    """Entity-linking CrossEncoder that extends SentenceTransformers CrossEncoder.

    The class keeps the optimized SentenceTransformers implementation while adding
    GPU-first device resolution and an explicit final-save helper that writes the
    transformer weights with safe serialization when supported.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *args: Any,
        device: str = "auto",
        max_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        if _sentence_transformers is None:
            raise ImportError("sentence-transformers is required for EntityLinkingCrossEncoder")
        resolved_device = resolve_torch_device(device)
        init_kwargs = dict(kwargs)
        init_kwargs["device"] = resolved_device
        if max_length is not None:
            init_kwargs["max_length"] = max_length
        super().__init__(model_name_or_path, *args, **init_kwargs)
        self.resolved_device = resolved_device
        if max_length is not None:
            self.max_length = max_length

    @staticmethod
    def _filter_callable_kwargs(callable_obj: Any, fit_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop unsupported kwargs for a SentenceTransformers training callable."""
        signature = inspect.signature(callable_obj)
        parameters = signature.parameters
        accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        if accepts_kwargs:
            return fit_kwargs
        return {key: value for key, value in fit_kwargs.items() if key in parameters}

    def fit_quiet(self, **fit_kwargs: Any) -> Any:
        """Train with a progress bar but without frequent loss tables.

        ``CrossEncoder.fit`` in SentenceTransformers >= 4 delegates to the
        HuggingFace Trainer and can display notebook tables such as ``Step`` /
        ``Training Loss`` every 500 steps. To keep the ETA/progress bar without
        that table, this helper prefers ``old_fit`` when available. ``old_fit`` is
        the classic SentenceTransformers loop: it still supports tqdm via
        ``show_progress_bar`` but does not create the Trainer logging table. If a
        future installation does not expose ``old_fit``, the method falls back to
        ``fit`` with very sparse logging.
        """
        configure_quiet_transformers_logging()
        fit_kwargs.setdefault("show_progress_bar", True)
        fit_kwargs.setdefault("logging_steps", DEFAULT_TRAINING_LOGGING_STEPS)
        fit_kwargs.setdefault("evaluation_steps", 0)
        fit_kwargs.setdefault("disable_tqdm", False)

        use_legacy_fit_loop = bool(fit_kwargs.pop("use_legacy_fit_loop", True))
        if use_legacy_fit_loop and hasattr(self, "old_fit"):
            return self.old_fit(**self._filter_callable_kwargs(self.old_fit, fit_kwargs))
        return self.fit(**self._filter_callable_kwargs(self.fit, fit_kwargs))

    def fit_with_progress_no_loss_logging(self, **fit_kwargs: Any) -> Any:
        """Train with tqdm/ETA and without the HuggingFace loss table."""
        return self.fit_quiet(**fit_kwargs)

    @staticmethod
    def _record_get(record: Any, key: str, default: Any = None) -> Any:
        """Read a value from a dict-like, namedtuple-like or object record."""
        if isinstance(record, dict):
            return record.get(key, default)
        return getattr(record, key, default)

    @classmethod
    def pairs_to_labeled_texts(
        cls,
        pairs: Any,
        mention_col: str = "mention",
        candidate_col: str = "candidate",
        label_col: str = "label",
    ) -> list[tuple[str, str, float]]:
        """Convert pair records into ``(mention, candidate, label)`` tuples.

        This is the canonical binary cross-encoder objective for entity linking:
        positives are encoded as ``(mention, positive_candidate, 1.0)`` and
        negatives as ``(mention, negative_candidate, 0.0)``.
        """
        rows = pairs.to_dict("records") if hasattr(pairs, "to_dict") else pairs
        out: list[tuple[str, str, float]] = []
        for row in rows:
            mention = str(cls._record_get(row, mention_col, "")).strip()
            candidate = str(cls._record_get(row, candidate_col, "")).strip()
            label = float(cls._record_get(row, label_col, 0.0))
            if mention and candidate:
                out.append((mention, candidate, label))
        return out

    @classmethod
    def triplets_to_labeled_texts(
        cls,
        triplets: Any,
        anchor_col: str = "anchor",
        positive_col: str = "positive",
        negative_col: str = "negative",
    ) -> list[tuple[str, str, float]]:
        """Expand triplets into BCE pair records.

        Each ``(anchor, positive, negative)`` triplet becomes two CrossEncoder
        training examples: ``(anchor, positive, 1.0)`` and
        ``(anchor, negative, 0.0)``. This reuses SentenceTransformers'
        optimized CrossEncoder BCE training path.
        """
        rows = triplets.to_dict("records") if hasattr(triplets, "to_dict") else triplets
        out: list[tuple[str, str, float]] = []
        for row in rows:
            anchor = str(cls._record_get(row, anchor_col, "")).strip()
            positive = str(cls._record_get(row, positive_col, "")).strip()
            negative = str(cls._record_get(row, negative_col, "")).strip()
            if anchor and positive:
                out.append((anchor, positive, 1.0))
            if anchor and negative:
                out.append((anchor, negative, 0.0))
        return out

    @classmethod
    def triplets_to_texts(
        cls,
        triplets: Any,
        anchor_col: str = "anchor",
        positive_col: str = "positive",
        negative_col: str = "negative",
    ) -> list[tuple[str, str, str]]:
        """Convert triplet records into ``(anchor, positive, negative)`` tuples."""
        rows = triplets.to_dict("records") if hasattr(triplets, "to_dict") else triplets
        out: list[tuple[str, str, str]] = []
        for row in rows:
            anchor = str(cls._record_get(row, anchor_col, "")).strip()
            positive = str(cls._record_get(row, positive_col, "")).strip()
            negative = str(cls._record_get(row, negative_col, "")).strip()
            if anchor and positive and negative:
                out.append((anchor, positive, negative))
        return out

    def _input_examples_from_labeled_texts(self, labeled_texts: list[tuple[str, str, float]]) -> list[Any]:
        if _sentence_transformers is None:
            raise ImportError("sentence-transformers is required to create CrossEncoder InputExample objects")
        return [
            _sentence_transformers.InputExample(texts=[mention, candidate], label=float(label))
            for mention, candidate, label in labeled_texts
        ]

    @classmethod
    def triplets_to_knowledge_graph_labeled_texts(
        cls,
        triplets: Any,
        anchor_col: str = "anchor",
        positive_col: str = "positive",
        negative_col: str = "negative",
    ) -> list[tuple[str, str, float]]:
        """Convert triplets to BCE examples following KnowledgeGraph's CrossEncoder recipe.

        The KnowledgeGraph training code expands strict ``(anchor, positive,
        negative)`` triplets into CrossEncoder binary examples by taking unique
        ``(anchor, positive)`` pairs with label ``1`` and unique
        ``(anchor, negative)`` pairs with label ``0``.  This avoids repeating the
        same positive once for every negative while keeping all hard negatives
        mined from similarity or graph expansion.
        """
        rows = triplets.to_dict("records") if hasattr(triplets, "to_dict") else triplets
        positive_pairs: set[tuple[str, str]] = set()
        negative_pairs: set[tuple[str, str]] = set()
        for row in rows:
            anchor = str(cls._record_get(row, anchor_col, "")).strip()
            positive = str(cls._record_get(row, positive_col, "")).strip()
            negative = str(cls._record_get(row, negative_col, "")).strip()
            if anchor and positive:
                positive_pairs.add((anchor, positive))
            if anchor and negative and negative != positive:
                negative_pairs.add((anchor, negative))
        labeled = [(anchor, positive, 1.0) for anchor, positive in sorted(positive_pairs)]
        labeled.extend((anchor, negative, 0.0) for anchor, negative in sorted(negative_pairs))
        return labeled

    def fit_knowledge_graph_triplets_bce(
        self,
        triplets: Any,
        batch_size: int = 16,
        dataloader_num_workers: int = 0,
        pin_memory: bool | None = None,
        **fit_kwargs: Any,
    ) -> Any:
        """Train with the KnowledgeGraph-style hard-triplet BCE objective.

        This mirrors the original repository's approach: strict triplets are not
        optimized with a custom triplet loss by default; instead, they are
        transformed into unique positive and negative CrossEncoder pairs and
        trained with SentenceTransformers' native binary CrossEncoder fitting.
        """
        labeled_texts = self.triplets_to_knowledge_graph_labeled_texts(triplets)
        return self.fit_bce_pairs(
            [
                {"mention": mention, "candidate": candidate, "label": label}
                for mention, candidate, label in labeled_texts
            ],
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
            **fit_kwargs,
        )

    def fit_bce_pairs(
        self,
        pairs: Any,
        batch_size: int = 16,
        dataloader_num_workers: int = 0,
        pin_memory: bool | None = None,
        **fit_kwargs: Any,
    ) -> Any:
        """Train with Binary Cross-Entropy over ``(mention, candidate, label)`` pairs.

        This method delegates to SentenceTransformers' CrossEncoder ``fit`` /
        ``old_fit`` machinery via :meth:`fit_with_progress_no_loss_logging`, so
        it keeps the library's efficient batching while adding safe CUDA
        dataloader defaults.
        """
        torch = optional_module("torch")
        if torch is None:
            raise ImportError("torch is required for CrossEncoder training")
        labeled_texts = self.pairs_to_labeled_texts(pairs)
        train_examples = self._input_examples_from_labeled_texts(labeled_texts)
        data = torch.utils.data.DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size,
            num_workers=resolve_cross_encoder_dataloader_num_workers(dataloader_num_workers, self.resolved_device),
            pin_memory=resolve_cross_encoder_dataloader_pin_memory(
                bool(pin_memory) if pin_memory is not None else str(self.resolved_device).startswith("cuda"),
                self.resolved_device,
            ),
        )
        fit_kwargs["train_dataloader"] = data
        return self.fit_with_progress_no_loss_logging(**fit_kwargs)

    def fit_triplets_as_bce_pairs(
        self,
        triplets: Any,
        batch_size: int = 16,
        dataloader_num_workers: int = 0,
        pin_memory: bool | None = None,
        **fit_kwargs: Any,
    ) -> Any:
        """Train triplets by expanding them to BCE positive/negative pairs."""
        labeled_texts = self.triplets_to_labeled_texts(triplets)
        return self.fit_bce_pairs(
            [
                {"mention": mention, "candidate": candidate, "label": label}
                for mention, candidate, label in labeled_texts
            ],
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
            **fit_kwargs,
        )

    def fit_triplets_margin_ranking(
        self,
        triplets: Any,
        batch_size: int = 16,
        epochs: int = 1,
        learning_rate: float = 2e-5,
        margin: float = 1.0,
        use_amp: bool = True,
        show_progress_bar: bool = True,
        **_: Any,
    ) -> None:
        """Train with a pairwise margin-ranking objective over triplets.

        CrossEncoder natively scores text pairs, so the contrastive objective is
        expressed as ``score(anchor, positive) > score(anchor, negative)`` with
        ``torch.nn.MarginRankingLoss``. This is useful when consuming the
        strict ``anchor, positive, negative`` parquet files produced by notebook
        02a.
        """
        torch = optional_module("torch")
        tqdm_auto = optional_module("tqdm.auto")
        if torch is None:
            raise ImportError("torch is required for margin-ranking triplet training")
        triplet_texts = self.triplets_to_texts(triplets)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        loss_fct = torch.nn.MarginRankingLoss(margin=margin)
        scaler = torch.cuda.amp.GradScaler() if use_amp and str(self.resolved_device).startswith("cuda") else None
        iterator_cls = tqdm_auto.tqdm if tqdm_auto is not None else (lambda x, **__: x)
        self.model.train()
        for epoch in range(int(epochs)):
            for start in iterator_cls(
                range(0, len(triplet_texts), batch_size),
                desc=f"Triplet margin epoch {epoch + 1}",
                disable=not show_progress_bar,
            ):
                batch = triplet_texts[start : start + batch_size]
                positive_pairs = [(anchor, positive) for anchor, positive, _negative in batch]
                negative_pairs = [(anchor, negative) for anchor, _positive, negative in batch]
                optimizer.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        pos_scores = self._predict_scores_tensor(positive_pairs)
                        neg_scores = self._predict_scores_tensor(negative_pairs)
                        target = torch.ones_like(pos_scores)
                        loss = loss_fct(pos_scores, neg_scores, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pos_scores = self._predict_scores_tensor(positive_pairs)
                    neg_scores = self._predict_scores_tensor(negative_pairs)
                    target = torch.ones_like(pos_scores)
                    loss = loss_fct(pos_scores, neg_scores, target)
                    loss.backward()
                    optimizer.step()

    def _predict_scores_tensor(self, pairs: list[tuple[str, str]]) -> Any:
        """Score text pairs and return a differentiable tensor for training."""
        texts_a = [left for left, _right in pairs]
        texts_b = [right for _left, right in pairs]
        features = self.tokenizer(
            texts_a,
            texts_b,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=getattr(self, "max_length", None),
        )
        features = {key: value.to(self.resolved_device) for key, value in features.items()}
        output = self.model(**features)
        logits = output.logits if hasattr(output, "logits") else output[0]
        return logits.view(-1).float()

    def prepare_triplets(self, triplets_df: Any) -> list[Any]:
        """Prepare triplets as unique CrossEncoder BCE InputExample objects.

        Mirrors the KnowledgeGraph recipe: unique ``(anchor, positive)`` pairs
        receive label ``1`` and unique ``(anchor, negative)`` pairs receive label
        ``0``.
        """
        if _sentence_transformers is None:
            raise ImportError("sentence-transformers is required to create InputExample objects")
        if hasattr(triplets_df, "loc"):
            examples: list[Any] = []
            positive_triplets = triplets_df[["anchor", "positive"]].drop_duplicates().reset_index(drop=True)
            negative_triplets = triplets_df[["anchor", "negative"]].drop_duplicates().reset_index(drop=True)
            for _, row in positive_triplets.iterrows():
                examples.append(
                    _sentence_transformers.InputExample(texts=[str(row["anchor"]), str(row["positive"])], label=1.0)
                )
            for _, row in negative_triplets.iterrows():
                examples.append(
                    _sentence_transformers.InputExample(texts=[str(row["anchor"]), str(row["negative"])], label=0.0)
                )
            return examples
        labeled = self.triplets_to_knowledge_graph_labeled_texts(triplets_df)
        return [_sentence_transformers.InputExample(texts=[a, b], label=float(label)) for a, b, label in labeled]

    def transform_triplets_rankingeval(self, triplet_samples: list[Any]) -> list[dict[str, Any]]:
        """Transform prepared InputExamples into CERerankingEvaluator samples."""
        dev_samples_dict: dict[str, dict[str, Any]] = {}
        for sample in triplet_samples:
            key = sample.texts[0]
            if key not in dev_samples_dict:
                dev_samples_dict[key] = {"query": key, "positive": set(), "negative": set()}
            dev_samples_dict[key]["positive" if int(sample.label) == 1 else "negative"].add(sample.texts[1])
        return [
            {"query": key, "positive": list(value["positive"]), "negative": list(value["negative"])}
            for key, value in dev_samples_dict.items()
        ]

    def train(
        self,
        mode: Any = True,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Keep the PyTorch/SentenceTransformers ``train(mode)`` contract.

        ``sentence_transformers.CrossEncoder.predict`` calls ``self.eval()``,
        which internally delegates to ``self.train(False)``. The original
        KnowledgeGraph API also exposed a ``train(df_hard_triplets, ...)``
        convenience method. To support both use cases safely:

        - ``train(True/False)`` switches module training/eval mode.
        - ``train(df_hard_triplets, output_path=..., batch_size=..., epochs=...)``
          delegates to :meth:`train_hard_triplets`.
        """
        if isinstance(mode, bool) and not args and not kwargs:
            parent_train = getattr(super(), "train", None)
            if callable(parent_train):
                return parent_train(mode)
            if hasattr(self, "model") and hasattr(self.model, "train"):
                self.model.train(mode)
            return self
        if isinstance(mode, bool) and "df_hard_triplets" in kwargs:
            mode = kwargs.pop("df_hard_triplets")
        return self.train_hard_triplets(mode, *args, **kwargs)

    def train_hard_triplets(
        self,
        df_hard_triplets: Any,
        output_path: str,
        batch_size: int,
        epochs: int,
        evaluator_type: str | None = None,
        optimizer_parameters: dict[str, float] | None = None,
        weight_decay: float = 0.01,
        evaluation_steps: int = 10000,
        save_best_model: bool = True,
        test_size: float | None = None,
        dataloader_num_workers: int = 0,
        pin_memory: bool | None = None,
        use_amp: bool = True,
        show_progress_bar: bool = True,
        use_legacy_fit_loop: bool = True,
    ) -> None:
        """Train from hard triplets using the original KnowledgeGraph BCE flow.

        The method keeps SentenceTransformers' optimized CrossEncoder training:
        triplets are converted to unique binary examples, optional dev evaluators
        are created, and fitting is delegated to ``fit``/``old_fit``.
        """
        if optimizer_parameters is None:
            optimizer_parameters = {"lr": 1e-5}
        train_samples = df_hard_triplets
        dev_samples = None
        if test_size:
            model_selection = optional_module("sklearn.model_selection")
            if model_selection is None:
                raise ImportError("scikit-learn is required when test_size is provided")
            train_test_split = model_selection.train_test_split
            stratify = (
                df_hard_triplets["anchor"]
                if hasattr(df_hard_triplets, "__getitem__") and "anchor" in df_hard_triplets
                else None
            )
            try:
                train_samples, dev_samples = train_test_split(df_hard_triplets, test_size=test_size, stratify=stratify)
            except ValueError:
                train_samples, dev_samples = train_test_split(df_hard_triplets, test_size=test_size)

        torch = optional_module("torch")
        if torch is None:
            raise ImportError("torch is required for CrossEncoder training")
        train_dataloader = torch.utils.data.DataLoader(
            self.prepare_triplets(train_samples),
            shuffle=True,
            batch_size=batch_size,
            num_workers=resolve_cross_encoder_dataloader_num_workers(dataloader_num_workers, self.resolved_device),
            pin_memory=resolve_cross_encoder_dataloader_pin_memory(
                bool(pin_memory) if pin_memory is not None else str(self.resolved_device).startswith("cuda"),
                self.resolved_device,
            ),
        )
        evaluator = None
        if evaluator_type and dev_samples is not None:
            evaluation = optional_module("sentence_transformers.cross_encoder.evaluation")
            if evaluation is None:
                raise ImportError("sentence-transformers cross_encoder.evaluation is required for evaluators")
            dev_examples = self.prepare_triplets(dev_samples)
            if evaluator_type == "BinaryClassificationEvaluator":
                evaluator = evaluation.CEBinaryClassificationEvaluator.from_input_examples(dev_examples, name="dev")
            elif evaluator_type in {"CERankingEvaluator", "CERerankingEvaluator"}:
                evaluator = evaluation.CERerankingEvaluator(
                    self.transform_triplets_rankingeval(dev_examples), name="dev"
                )
            else:
                raise ValueError("evaluator_type must be BinaryClassificationEvaluator or CERankingEvaluator")
        warmup_steps = int(len(train_dataloader) * int(epochs) * 0.1)
        self.fit_with_progress_no_loss_logging(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=epochs,
            optimizer_params=optimizer_parameters,
            weight_decay=weight_decay,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=output_path,
            save_best_model=save_best_model,
            use_amp=bool(use_amp and str(self.resolved_device).startswith("cuda")),
            show_progress_bar=show_progress_bar,
            use_legacy_fit_loop=use_legacy_fit_loop,
        )

    def rerank_candidates(
        self, df: Any, entity_col: str, candidates_col: str, codes_col: str, batch_size: int = 32
    ) -> Any:
        """Rerank candidate and code lists in a dataframe using CrossEncoder scores."""
        if any(col not in df.columns for col in [entity_col, candidates_col, codes_col]):
            raise ValueError("Specified columns not found in the DataFrame")
        tqdm_auto = optional_module("tqdm.auto")
        iterator = df.index
        if tqdm_auto is not None:
            iterator = tqdm_auto.tqdm(iterator, desc="Reranking candidates")
        for index in iterator:
            entity = df.at[index, entity_col]
            candidates = df.at[index, candidates_col]
            codes = df.at[index, codes_col]
            # Accept serialized notebook columns as well as true lists.
            try:
                from ..triplets.generator import parse_list_column

                candidates = [str(x) for x in parse_list_column(candidates)]
                codes = [str(x) for x in parse_list_column(codes)]
            except Exception:
                candidates = list(candidates)
                codes = list(codes)
            scores = self.predict([[entity, candidate] for candidate in candidates], batch_size=batch_size)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            df.at[index, candidates_col] = [candidates[i] for i in sorted_indices]
            df.at[index, codes_col] = [codes[i] for i in sorted_indices]
        return df

    def save_final(self, output_path: str | Path) -> None:
        """Save the final CrossEncoder and safetensors-compatible transformer weights."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self.save(str(output_path))
        if hasattr(self, "model") and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(output_path), safe_serialization=True)
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(str(output_path))


class CrossEncoderReranker(EntityLinkingCrossEncoder):
    """KnowledgeGraph-compatible CrossEncoder reranker.

    This class intentionally extends ``sentence_transformers.CrossEncoder`` via
    ``EntityLinkingCrossEncoder`` and exposes the original API names
    (``prepare_triplets``, ``train`` and ``rerank_candidates``) while preserving
    GPU-first device resolution and CUDA-safe DataLoader defaults.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "mask",
        max_seq_length: int = 256,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int | None = None,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> None:
        init_kwargs = dict(kwargs)
        resolved_max_length = max_length or max_seq_length
        if model_type == "mask":
            init_kwargs.setdefault("num_labels", 1)
            super().__init__(model_name, device=device, max_length=resolved_max_length, **init_kwargs)
        else:
            super().__init__(model_name, device=device, **init_kwargs)
        self.model_type = model_type
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

    def cuda(self, device: int | None = None) -> "CrossEncoderReranker":
        """Move the reranker to a CUDA device."""
        target = f"cuda:{device}" if device is not None else "cuda"
        return self.to(target)

    def to(self, device: str | Any) -> "CrossEncoderReranker":
        """Move the reranker model to the requested device."""
        self.resolved_device = str(device)
        if hasattr(self, "model") and hasattr(self.model, "to"):
            self.model.to(device)
        return self

    def rerank(self, mentions: list[str], candidates: list[dict[str, list[str]]], k: int = 10) -> list[dict[str, Any]]:
        """Rerank packed candidate dictionaries preserving aligned codes."""
        if len(mentions) != len(candidates):
            raise ValueError("mentions and candidates must have the same length")
        import numpy as np

        all_pairs: list[tuple[str, str]] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for mention, cand in zip(mentions, candidates):
            terms = [str(term) for term in cand.get("terms", [])]
            start = cursor
            all_pairs.extend((str(mention), term) for term in terms)
            cursor += len(terms)
            offsets.append((start, cursor))
        scores = (
            self.predict(
                all_pairs,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
            )
            if all_pairs
            else np.array([])
        )
        scores = np.asarray(scores).reshape(-1)
        output = []
        for i, mention in enumerate(mentions):
            terms = [str(term) for term in candidates[i].get("terms", [])]
            codes = [str(code) for code in candidates[i].get("codes", [])]
            start, end = offsets[i]
            local = scores[start:end] if end > start else np.array([])
            order = np.argsort(local)[::-1] if len(local) else np.array([], dtype=int)
            output.append(
                {
                    "mention": str(mention),
                    "terms": [terms[j] for j in order[:k]],
                    "codes": [codes[j] for j in order[:k]] if codes else [],
                    "similarity": [float(local[j]) for j in order[:k]],
                }
            )
        return output


class SimpleCrossEncoder:
    """Lightweight lexical reranker retained for offline smoke tests and compatibility."""

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score text pairs with the compatibility SequenceMatcher proxy."""
        from difflib import SequenceMatcher

        return [SequenceMatcher(None, left.lower(), right.lower()).ratio() for left, right in pairs]

    def rerank(self, mention: Any, candidates: list[Any]) -> list[Any]:
        """Rerank MatchCandidate-like objects for one mention."""
        pairs = [(mention.text, candidate.candidate_term or "") for candidate in candidates]
        for candidate, score in zip(candidates, self.score_pairs(pairs)):
            candidate.score = float(score)
            candidate.method = "cross_encoder_rerank"
        candidates.sort(key=lambda item: item.score, reverse=True)
        for rank, candidate in enumerate(candidates, 1):
            candidate.rank = rank
        return candidates


def rerank_candidates(
    annotations_path: str,
    candidates_path: str,
    output_path: str,
    top_k: int = 25,
    threshold: float = 0.0,
    profile: bool = False,
    profile_output: str | None = None,
    device: str = "auto",
) -> None:
    """Rerank a serialized candidate TSV with the lightweight compatibility reranker."""
    import json

    import pandas as pd

    from ..data import load_annotations_tsv
    from ..schemas import MatchCandidate
    from ..utils.profiling import Profiler

    profiler = Profiler(enabled=profile, device=device)
    with profiler.track("load_inputs", annotations_path=annotations_path, candidates_path=candidates_path):
        mentions = load_annotations_tsv(annotations_path)
        candidates_df = pd.read_csv(candidates_path, sep="\t")
    indexed = {f"{row['filename']}||{row['start_span']}||{row['end_span']}": row for _, row in candidates_df.iterrows()}
    model = SimpleCrossEncoder()
    rows = []
    with profiler.track("cross_encoder.rerank", n_items=len(mentions), top_k=top_k, device=device):
        for mention in mentions:
            key = f"{mention.filename}||{mention.start_span}||{mention.end_span}"
            row = indexed.get(key)
            if row is None:
                continue
            codes = json.loads(row["codes"])
            terms = json.loads(row["candidates"])
            objects = [
                MatchCandidate(
                    None,
                    mention.filename,
                    mention.text,
                    mention.label,
                    code,
                    term,
                    0.0,
                    "retrieval",
                    rank=rank,
                    metadata={},
                )
                for rank, (code, term) in enumerate(zip(codes, terms), 1)
            ]
            reranked = [candidate for candidate in model.rerank(mention, objects) if candidate.score >= threshold][
                :top_k
            ]
            rows.append(
                {
                    "filename": mention.filename,
                    "start_span": mention.start_span,
                    "end_span": mention.end_span,
                    "mention": mention.text,
                    "codes": json.dumps([candidate.code for candidate in reranked], ensure_ascii=False),
                    "candidates": json.dumps([candidate.candidate_term for candidate in reranked], ensure_ascii=False),
                }
            )
    with profiler.track("write_outputs", n_items=len(rows), output_path=output_path):
        pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)
    if profile:
        profiler.print_final_report()
        if profile_output:
            profiler.save_json(profile_output)
