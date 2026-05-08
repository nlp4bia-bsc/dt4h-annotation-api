import re
from nltk.tokenize import PunktSentenceTokenizer

# =============================================================================
# V1 INFERENCE
# =============================================================================

# --- PreTokenization --------------------------------------------------------

SPLIT_WORDS = re.compile(
    r'([0-9A-Za-zÀ-ÖØ-öø-ÿ]+|[^0-9A-Za-zÀ-ÖØ-öø-ÿ])'
)
"""This regex splits text into tokens such that:
Words (including accented letters and numbers) stay grouped
Every non-alphanumeric character becomes its own token
"Hola, qué tal?" -> ["Hola", ",", " ", "qué", " ", "tal", "?"]
"""

def pretokenize_sentence(sentence: str) -> tuple[str, list[int]]:
    """Pretokenizes a sentence by splitting it into tokens and adding spaces between non-space tokens, and saves the added space positions for later alignment. 
    This is done to ensure that the token classification model can correctly identify entities that are attached to punctuation or other words without spaces.
    For example:
        "Hola, qué tal?" -> ["Hola", ",", " ", "qué", " ", "tal", "?"] ->  ["Hola ", ",", " ", "qué", " ", "tal ", "?"]  -> "Hola , qué tal ?"
    """
    pretokens = [t for t in SPLIT_WORDS.split(sentence) if t]
    added_spaces_pos = []
    char_ct = 0
    for i, (curr_token, next_token) in enumerate(zip(pretokens[:-1], pretokens[1:])):
        char_ct += len(curr_token)
        if not curr_token.isspace() and not next_token.isspace():
            # If both are non-space tokens, we insert a space between them
            pretokens[i] = curr_token + ' '
            added_spaces_pos.append(char_ct) # char count aligns with new space position
            char_ct += 1 # add that space to the character count

    return ''.join(pretokens), added_spaces_pos


# =============================================================================
# V2 INFERENCE
# =============================================================================

# --- Sentence splitting ------------------------------------------------------

def _split_sentences(text: str) -> list[dict]:
    """
    Split *text* into sentences using NLTK's PunktSentenceTokenizer and return
    their character offsets in the original text.

    Each returned dict contains:
        - sent_id (int):   zero-based sentence index
        - start    (int):  inclusive start char offset in *text*
        - end      (int):  exclusive end char offset in *text*
        - text     (str):  the sentence string

    Empty / whitespace-only sentences are discarded.
    """
    tokenizer = PunktSentenceTokenizer()
    sentences = []
    for sent_id, (start, end) in enumerate(tokenizer.span_tokenize(text)):
        sent_text = text[start:end]
        if sent_text.strip():
            sentences.append({"sent_id": sent_id, "start": start, "end": end, "text": sent_text})
    return sentences


# --- Token-safe chunking -----------------------------------------------------

def _split_sentence_into_chunks(
    sentence: str,
    tokenizer,
    max_length: int,
) -> list[dict]:
    """
    Split a single *sentence* into one or more token-safe chunks that fit within
    *max_length* tokens, preserving character offsets relative to *sentence*.

    This is necessary because some sentences may exceed the model's maximum
    sequence length. Chunks are produced greedily (no overlap), using the
    tokenizer's own offset mapping to compute character boundaries.

    Each returned dict contains:
        - text        (str): the chunk substring
        - start       (int): inclusive start char offset relative to *sentence*
        - end         (int): exclusive end char offset relative to *sentence*

    Returns an empty list if *sentence* is empty or whitespace-only.
    """
    if not sentence or not sentence.strip():
        return []

    # Leave 2 positions for special tokens (e.g. [CLS] / [SEP])
    safe_len = max(8, max_length - 2)

    enc = tokenizer(
        sentence,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # Sentence already fits — return as a single chunk
    if len(input_ids) <= safe_len:
        return [{"text": sentence, "start": 0, "end": len(sentence)}]

    chunks = []
    start_tok = 0
    while start_tok < len(input_ids):
        end_tok = min(start_tok + safe_len, len(input_ids))
        char_start = offsets[start_tok][0]
        char_end = offsets[end_tok - 1][1]
        if char_end > char_start:
            chunks.append({"text": sentence[char_start:char_end], "start": char_start, "end": char_end})
        start_tok = end_tok

    return chunks


def build_inference_chunks(text: str, tokenizer, max_length: int) -> list[dict]:
    """
    Segment *text* into inference-ready chunks, one per sentence (or per
    token-safe sub-sentence if a sentence is too long for the model).

    Sentence splitting is done with :func:`split_sentences`; oversized sentences
    are further divided by :func:`split_sentence_into_chunks`.

    Each returned dict contains:
        - chunk_id (int): unique sequential chunk index
        - sent_id  (int): index of the originating sentence
        - start    (int): inclusive start char offset in *text*
        - end      (int): exclusive end char offset in *text*
        - text     (str): the chunk substring

    Empty / whitespace-only chunks are discarded.
    """
    sentences = _split_sentences(text)
    chunks = []
    chunk_id = 0

    for sent in sentences:
        sub_chunks = _split_sentence_into_chunks(
            sentence=sent["text"],
            tokenizer=tokenizer,
            max_length=max_length,
        )
        for sub in sub_chunks:
            # Convert offsets that are local to the sentence → global in *text*
            global_start = sent["start"] + sub["start"]
            global_end = sent["start"] + sub["end"]
            chunk_text = text[global_start:global_end]
            if chunk_text.strip():
                chunks.append({
                    "chunk_id": chunk_id,
                    "sent_id": sent["sent_id"],
                    "start": global_start,
                    "end": global_end,
                    "text": chunk_text,
                })
                chunk_id += 1

    return chunks