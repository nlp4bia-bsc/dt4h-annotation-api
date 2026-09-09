# CDM — open questions and deferred work

Decisions still outstanding after the move to CDM-JSON input in `run_nerl.py`.
Each entry states the current behaviour, so nothing here is silently broken —
it is behaviour we chose knowing it may change.

---

## 1. `dt4h_concept_identifier` — unassigned

**Status:** pending a decision from the project side.

The CDM declares `dt4h_concept_identifier` on every annotation. Nothing in this
repository populates it, and no rule has been supplied for deriving it.

**Current behaviour:** always emitted as `null`.

---

## 2. NER confidence has no CDM field

**Status:** resolved for `concept_confidence`; the NER score remains unreported.

The CDM defines exactly one confidence slot per annotation,
`concept_confidence`, positioned immediately before `ner_component_type` /
`ner_component_version`. It has been assigned to the **NEL linking
confidence** — how confident the pipeline is that the mention maps to the
emitted `controlled_vocabulary_concept_identifier`.

That leaves the NER extraction score with nowhere to go. Before the CDM
realignment this repository emitted both — `extraction_confidence` for the NER
score and `concept_confidence` for the NEL score — but `extraction_confidence`
is not a CDM field and was removed.

**Current behaviour:** `concept_confidence` carries the NEL score
(`Dt4hFormatter._rename_annotation`). NER-only runs (`run_nerl.py`) have no
`nel_score` and therefore emit `concept_confidence: null` — the field is not
back-filled with the extraction score, which measures a different thing.
`ner_score` is dropped during CDM serialisation and survives only in
`PassthroughFormatter` output.

**Interaction with reranking:** when a cross-encoder reranker runs it becomes
the authority on `concept_confidence` — it is the component that decided the
code. The retrieval score survives in the raw annotation's `metadata` but not
in the CDM, which has room for one number.

**Options when this is picked up:** add a CDM field for extraction confidence;
or agree that extraction confidence is out of scope for the CDM. Overloading
`concept_confidence` depending on which stages ran is rejected — one field
cannot carry two different measurements.

---

## 3. Uncertainty detection is not representable

**Status:** structural gap in the CDM.

`app/src/negation/negation_utils.py` produces `is_uncertain` and
`uncertainty_score` alongside negation, driven by the `UNC`/`USCO` spans from
the negation tagger. The CDM has `negation` / `negation_confidence` but no
uncertainty counterpart, so half of what the negation model produces cannot be
reported.

`qualifier_negation` and `qualifier_temporal` exist in the CDM and are currently
unpopulated — one of them may be the intended home for this, but that is a
guess, not a mapping.

**Current behaviour:** uncertainty is computed and then dropped during CDM
serialisation. It survives in `PassthroughFormatter` output.

---

## 4. `concept_class` is mapped from the model's label

**Status:** resolved — the checkpoints are not CDM-conformant, so a mapping
table was added.

`concept_class` originates in the NER model's own label (`ner_class`, read from
the checkpoint's `id2label`). The models are produced within the same project as
the CDM, so their labels were *expected* to already be the CDM values. They are
not: a real run emits `DISEASE`, `PROCEDURE` and friends, which produced a
warning per annotation.

**Current behaviour:** `dt4h.CONCEPT_CLASS_MAP` translates the label into the
CDM vocabulary (`symptom`, `disorder/disease`, `procedure`, `medication`,
`cardiology entity`, `other`). `_normalise_label` strips any BIO prefix,
lower-cases, and treats `_`/`-` as spaces, so casing and separator style do not
each need their own entry. The table covers the English checkpoint labels, the
Spanish ones (`ENFERMEDAD`, `PROCEDIMIENTO`, …) and the registry's own entity
names — including `drug`, which the CDM calls `medication`.

A label the table does not know is **passed through unchanged**, not coerced to
`other`. `ConceptClass` then logs the out-of-vocabulary warning naming it, which
is the signal that a checkpoint emits something the table has not been told
about; flattening to `other` would destroy exactly that signal while looking
like success.

`tests/test_dt4h_formatter.py` asserts every value in the table is a member of
`CONCEPT_CLASSES`, so a typo in the table cannot reintroduce the warning it
exists to remove.

**Still open:** the mapping is derived from the labels observed so far plus the
registry's entity names. A checkpoint using an unseen label will warn rather
than fail — watch the logs on a new language or entity type and extend the
table.

---

## 5. `run_nerl.py` links only under `--nel`

**Status:** resolved for the two fields that carry a code; the rest stay open.

`run_nerl.py --nel` builds the same `BiencoderPipeline` the Flask route uses
(with `negation=False`), so `controlled_vocabulary_concept_identifier`,
`controlled_vocabulary_concept_official_term` and `concept_confidence` are now
populated. Without the flag the script runs NER only and all of them stay
`null` — the honest value for a stage that never ran.

`--nel` needs a built vector database per language and entity type, which is
why it is opt-in rather than the default: the setup step is heavier than a
plain NER run assumes. `_check_nel_registry` reports each missing resource by
name before the run touches a document.

Still `null` in both modes: `controlled_vocabulary_namespace`,
`controlled_vocabulary_version`, `controlled_vocabulary_source`,
`nel_component_type`, `nel_component_version` — see §6, which blocks them all.

---

## 6. Controlled-vocabulary namespace is not recorded

**Status:** blocked on per-gazetteer metadata.

`controlled_vocabulary_namespace` should say which terminology a code belongs to
(`SNOMED CT`, `ICD10`, …). Gazetteers are registered in the registry as bare TSV
paths with `term` and `code` columns and carry no indication of their
terminology, and they are not all the same one.

**Current behaviour:** always `null`.

**Likely fix:** add a `namespace` (and `version`) key beside each gazetteer path
in the registry, and thread it through `LocalResolver` into the NEL stage.

---

## 7. `.txt`-only input is no longer supported

**Status:** intentional breaking change.

`run_nerl.py` previously globbed `data/{lang}/*.txt` and annotated every file it
found, with empty record metadata. It now globs `data/{lang}/*.json` and reads
`.txt` files only as the sidecar of a CDM JSON document.

Supporting both was rejected: consumers who need the old behaviour should pin
the previous version rather than have the script carry two input contracts.
