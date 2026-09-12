"""
LLM-based synonym replacement for building the semantic-rewrite view.

Each sample keeps exactly **one** augmented text, written to the ``text1``
column of the output CSV. A candidate is accepted only if it passes:

   1. rule-based validity filter -- is the candidate a plausible rewrite at all
      (non-empty, not too short, sufficient token overlap with the original, and
      a comparable length);
   2. redundancy-control check -- does the rewrite carry new information
      (is it content-equivalent to the original, does it echo the original, does
      it repeat itself).

Both criteria must pass. Otherwise -- or when no candidate is available -- the
original text is copied into ``text1`` as a fallback, so some rows have
``text1 == text``.
"""

import os
import re
import json
import time
import argparse
from collections import Counter

import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# vLLM serves an OpenAI-compatible API
from openai import OpenAI


# ============== Token utilities ==============

# Lightweight tokenizer used by the redundancy control. It only serves text
# comparison; the original text is never modified.
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    """Lowercase the text and extract tokens (letters / digits / apostrophes)."""
    return _WORD_RE.findall(text.lower())


def _normalized(text: str) -> str:
    """Normalize to a "token sequence" string, used to test content equivalence."""
    return " ".join(_tokenize(text))


# ============== Configuration ==============

class Config:
    """Generation settings: model call parameters and the synonym-replacement prompt."""

    # Synonym replacement prompt.
    # Goal: keep the sentence structure, the meaning and the factual content
    # unchanged, and only replace words with their synonyms.
    PARAPHRASE_PROMPT = """You are a synonym replacement assistant. Replace the words in the following sentence with their synonyms.
IMPORTANT RULES:
1. Keep the sentence structure and meaning EXACTLY the same.
2. Only replace words with their synonyms — do NOT add, remove, or reorder content.
3. Preserve all factual information (names, dates, numbers, events) unchanged when no synonym applies.
4. Output ONLY the rewritten sentence, nothing else.

Original: {text}

Rewritten:"""

    # System prompt: constrain the model to synonym replacement only, without
    # altering the sentence structure or the facts.
    SYSTEM_PROMPT = """You are a synonym replacement tool.
You MUST:
- Rewrite the given sentence by replacing words with their synonyms.
- Keep the sentence structure, meaning, and factual content identical.
- Output ONLY the rewritten sentence, with no additional remarks, warnings, or formatting."""

    # Request parameters.
    # A synonym replacement is about as long as its input, so a large max_tokens
    # is unnecessary; this matters especially for the 0.5B model, which otherwise
    # tends to keep writing past the sentence or to echo the prompt back.
    MAX_TOKENS = 256           # max tokens per generation (enough for a news sentence)
    TEMPERATURE = 0.7          # sampling temperature, controls diversity
    TOP_P = 0.9                # nucleus sampling threshold
    FREQ_PENALTY = 0.5         # frequency penalty, discourages repeated words
    PRESENCE_PENALTY = 0.3     # presence penalty, further discourages repetition

    # ---- rule-based validity filter ----
    MIN_OVERLAP_RATIO = 0.2    # min token-set overlap with the original (guards against drift / a full rewrite)
    MIN_LENGTH_RATIO = 0.5     # min candidate tokens / original tokens (guards against truncation and lost content)
    MAX_LENGTH_RATIO = 2.0     # max ratio (guards against echoing the original or the prompt)

    # ---- redundancy-control check ----
    # Validity and redundancy are two independent criteria: the former asks
    # "is this a legitimate rewrite", the latter asks "does this rewrite carry
    # new information". A candidate has to pass both.
    REPEAT_NGRAM = 3           # n-gram order used by the self-repetition check
    MAX_NGRAM_REPEAT_RATIO = 0.5  # max share of repeated n-grams, above which a candidate counts as self-repetition


# ============== LLM client ==============

class ParaphraseGenerator:
    """Synonym-replacement generator backed by a local vLLM model."""

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        rate_limit: float = 0.1,
        max_retries: int = 3,
        max_concurrent: int = 8
    ):
        """
        Args:
            api_base: vLLM server address
            model_name: model name
            rate_limit: minimum interval between two requests (seconds)
            max_retries: max retries for a single sample
            max_concurrent: max concurrent requests (kept for statistics; the
                actual concurrency is controlled by the outer thread pool)
        """
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=api_base
        )
        self.model_name = model_name
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.last_request_time = 0
        self.lock = threading.Lock()

        # Call statistics
        self.success_count = 0
        self.fail_count = 0
        self.redundant_count = 0    # rejected for redundancy (as opposed to drift / format failures)

    def _wait_for_rate_limit(self):
        """Lock and sleep so the global request interval never falls below rate_limit."""
        with self.lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)
            self.last_request_time = time.time()

    def _clean_response(self, text: str) -> str:
        """
        Clean the raw model output: drop a leftover thinking block, enclosing
        quotes, an output prefix and redundant whitespace.

        Note: Qwen2.5-Instruct does not emit thinking blocks; the stripping logic
        is kept purely as a compatibility safeguard.
        """
        if not text:
            return ""

        # If a thinking block shows up anyway (e.g. after switching to a
        # reasoning model), keep only the text that follows it
        if '</think>' in text:
            text = text.split('</think>', 1)[1].strip()
        elif text.lstrip().startswith('<think>'):
            # Unterminated thinking block only -> the generation was truncated, reject
            return ""

        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
        text = re.sub(r'\n\s*\n', ' ', text)

        # Drop a matching pair of enclosing quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        # Drop a leftover output prefix
        prefixes = ["Rewritten:", "Rephrased:", "Paraphrased:", "Output:", "Result:", "Answer:"]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Collapse redundant whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _validate_output(self, original: str, paraphrased: str) -> bool:
        """
        Rule-based validity filter: is the candidate a plausible rewrite?

        Criteria: non-empty and not too short; enough token overlap with the
        original (guards against drift or a full rewrite); a token count
        comparable to the original (guards against truncation or echoing).
        Whether the candidate carries new information is decided separately by
        :meth:`_redundancy_check`.
        """
        if not paraphrased or len(paraphrased) < 5:
            return False

        original_tokens = _tokenize(original)
        para_tokens = _tokenize(paraphrased)
        if not original_tokens or not para_tokens:
            return False

        # Keep a minimum share of the original tokens, so that drift or a full
        # rewrite is rejected
        original_set = set(original_tokens)
        overlap_ratio = len(original_set & set(para_tokens)) / max(len(original_set), 1)
        if overlap_ratio < Config.MIN_OVERLAP_RATIO:
            return False

        # The length has to be comparable to the original
        length_ratio = len(para_tokens) / len(original_tokens)
        if not (Config.MIN_LENGTH_RATIO <= length_ratio <= Config.MAX_LENGTH_RATIO):
            return False

        return True

    @staticmethod
    def _repeated_ngram_ratio(tokens: List[str], n: int) -> float:
        """Share of repeated n-grams, used to detect echoing / concatenation."""
        if n <= 0 or len(tokens) < 2 * n:
            return 0.0
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(grams)
        repeated = sum(c for c in counts.values() if c > 1)
        return repeated / len(grams)

    @classmethod
    def _redundancy_check(cls, original: str, paraphrased: str) -> Tuple[bool, str]:
        """
        Redundancy-control check: does the candidate carry new information?

        Three criteria (all compared at token level, so case, punctuation and
        whitespace do not matter):

        1. ``identical-to-original`` -- content-equivalent to the original, i.e.
           nothing was rewritten;
        2. ``echoes-original``       -- keeps the original token sequence
           verbatim and appends extra content, i.e. the model echoed its input
           instead of rewriting it;
        3. ``self-repetition``       -- the share of repeated n-grams is too
           high, i.e. the same content is stated twice or two passages were
           concatenated.

        A pure reordering is not flagged as redundant: it changes the token
        sequence, which counts as a legitimate rewrite here.

        Returns
        -------
        (is_redundant, reason)
            ``reason`` is an empty string when the candidate passes.
        """
        original_norm = _normalized(original)
        para_norm = _normalized(paraphrased)

        if not para_norm:
            return True, "empty"

        # 1) Content-equivalent: identical token sequence, differing only in
        #    case / punctuation / whitespace
        if para_norm == original_norm:
            return True, "identical-to-original"

        # 2) Echoes the original: the original token sequence appears verbatim in
        #    the candidate, which is longer
        if original_norm and original_norm in para_norm:
            return True, "echoes-original"

        # 3) Self-repetition
        repeat_ratio = cls._repeated_ngram_ratio(
            _tokenize(paraphrased), Config.REPEAT_NGRAM
        )
        if repeat_ratio > Config.MAX_NGRAM_REPEAT_RATIO:
            return True, f"self-repetition({repeat_ratio:.2f})"

        return False, ""

    def _generate_single(self, text: str) -> Tuple[str, bool]:
        """
        Ask the model for one rewrite, with retries.

        A candidate has to pass both the rule-based validity filter and the
        redundancy-control check; once the retries are exhausted, the original
        text is returned as a fallback.

        Returns:
            (generated text, success). On failure the original text is returned
            with success=False.
        """
        prompt = Config.PARAPHRASE_PROMPT.format(text=text)

        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": Config.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=Config.MAX_TOKENS,
                    temperature=Config.TEMPERATURE,
                    top_p=Config.TOP_P,
                    frequency_penalty=Config.FREQ_PENALTY,
                    presence_penalty=Config.PRESENCE_PENALTY,
                    timeout=60
                )

                paraphrased = response.choices[0].message.content
                paraphrased = self._clean_response(paraphrased)

                # Two acceptance layers: the rule-based validity filter first,
                # then the redundancy control
                reason = "invalid"
                if self._validate_output(text, paraphrased):
                    redundant, reason = self._redundancy_check(text, paraphrased)
                    if not redundant:
                        self.success_count += 1
                        return paraphrased, True

                # Still rejected on the last retry -> fail and fall back to the original
                if attempt == self.max_retries - 1:
                    if reason != "invalid":
                        self.redundant_count += 1
                    self.fail_count += 1
                    return text, False

            except Exception as e:
                print(f"\n  Error (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(2)

        # Every retry raised an exception
        self.fail_count += 1
        return text, False

    def generate_paraphrase(self, text: str) -> Tuple[str, bool]:
        """Public entry point: generate a single synonym-replacement version of one text."""
        return self._generate_single(text)

    def get_stats(self) -> dict:
        """Return a snapshot of the success / failure / redundancy-rejection counts and the success rate."""
        total = self.success_count + self.fail_count
        return {
            "total": total,
            "success": self.success_count,
            "fail": self.fail_count,
            "redundant": self.redundant_count,
            "success_rate": self.success_count / max(total, 1)
        }


# ============== Data handling ==============

def load_raw_data(filepath: str) -> List[Tuple[int, str]]:
    """
    Load a raw data file. Two line formats are supported:
        - source\\tlabel\\ttext
        - label\\ttext

    Returns:
        [(label, text), ...]
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t', 2)  # at most 3 fields
            if len(parts) >= 3:
                # format: source\\tlabel\\ttext
                try:
                    label = int(parts[1])
                    text = parts[2].strip()
                    data.append((label, text))
                except ValueError:
                    continue
            elif len(parts) == 2:
                # format: label\\ttext
                try:
                    label = int(parts[0])
                    text = parts[1].strip()
                    data.append((label, text))
                except ValueError:
                    continue

    return data


def save_dataset(data: List[dict], output_path: str):
    """Save the augmented data as a CSV (columns: label, text, text1).

    UTF-8 is requested explicitly: pandas' default output encoding can vary with
    the version and the platform, and pinning it guarantees that the artifact can
    be read back as UTF-8 on any machine (the rest of this repository is UTF-8).
    """
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"  Saved: {output_path}")


# ============== Main pipeline ==============

def process_single_item(args_tuple):
    """Process one sample (called from the thread pool): generate a rewrite and build the record."""
    idx, label, text, generator = args_tuple
    paraphrased, success = generator.generate_paraphrase(text)

    # The record holds the label, the original text and the single augmented text
    record = {'label': label, 'text': text, 'text1': paraphrased}

    return idx, record, success


def process_dataset(
    input_file: str,
    output_file: str,
    api_base: str = "http://localhost:8000/v1",
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    rate_limit: float = 0.05,
    batch_size: int = 500,
    checkpoint_interval: int = 100,
    resume: bool = True,
    max_workers: int = 16
):
    """
    Run the synonym-replacement augmentation over a whole dataset, producing one
    augmented text per sample.

    Args:
        input_file: input file path (label\\ttext or source\\tlabel\\ttext)
        output_file: output CSV path
        api_base: vLLM API address
        model_name: model name
        rate_limit: global request interval (seconds)
        batch_size: write the full CSV every N processed samples
        checkpoint_interval: save a checkpoint JSON every N processed samples
        resume: resume from an existing checkpoint
        max_workers: max threads in the pool
    """
    print(f"\n{'='*60}")
    print(f"LLM synonym-replacement augmentation")
    print(f"{'='*60}")
    print(f"  input: {input_file}")
    print(f"  output: {output_file}")
    print(f"  API: {api_base}")
    print(f"  model: {model_name}")
    print(f"  request interval: {rate_limit}s")
    print(f"  max workers: {max_workers}")

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    # Checkpoint file path
    checkpoint_file = output_file.replace('.csv', '_checkpoint.json')

    if resume and os.path.exists(checkpoint_file):
        # Resume: read back the processed records and the next index
        print(f"\n  Resuming from checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        processed_data = checkpoint['processed']
        start_idx = checkpoint['next_idx']
        stats_data = checkpoint.get('stats', {'success': 0, 'fail': 0})
    else:
        # Start from scratch
        processed_data = []
        start_idx = 0
        stats_data = {'success': 0, 'fail': 0}

    # Load the raw data
    raw_data = load_raw_data(input_file)
    print(f"  total samples: {len(raw_data)}")
    print(f"  start index: {start_idx}")

    if start_idx >= len(raw_data):
        print("  Already complete.")
        return processed_data

    # Initialize the generator and carry the historical statistics over
    generator = ParaphraseGenerator(
        api_base, model_name, rate_limit, max_retries=3, max_concurrent=max_workers
    )
    generator.success_count = stats_data.get('success', 0)
    generator.fail_count = stats_data.get('fail', 0)
    generator.redundant_count = stats_data.get('redundant', 0)

    # Process the remaining samples concurrently
    print(f"\nProcessing...")
    remaining_data = raw_data[start_idx:]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit every task and keep the future -> original index mapping
        futures = {}
        for i, (label, text) in enumerate(remaining_data):
            idx = start_idx + i
            future = executor.submit(process_single_item, (idx, label, text, generator))
            futures[future] = idx

        # Collect results as they complete
        results = {}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            idx = futures[future]
            try:
                result_idx, record, success = future.result()
                results[result_idx] = record

                # Periodic checkpoint
                if len(results) % checkpoint_interval == 0:
                    checkpoint = {
                        'processed': [results[k] for k in sorted(results.keys())],
                        'next_idx': result_idx + 1,
                        'stats': generator.get_stats()
                    }
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(checkpoint, f)
                    print(f"\n  Checkpoint saved @ {result_idx + 1}")

                # Periodic full CSV dump (sorted by index to keep the order stable)
                if len(results) % batch_size == 0:
                    processed_data = [results[k] for k in sorted(results.keys())]
                    save_dataset(processed_data, output_file)

            except Exception as e:
                print(f"\n  Error @ idx {idx}: {e}")

    # Order all results by index
    processed_data = [results[k] for k in sorted(results.keys())]

    # Print the final statistics
    print(f"\nDone.")
    stats = generator.get_stats()
    print(f"  processed: {stats['total']}")
    print(f"  success: {stats['success']}")
    print(f"  failure: {stats['fail']}")
    print(f"    of which rejected as redundant: {stats['redundant']}")
    print(f"  success rate: {stats['success_rate']:.2%}")

    # Write the final result
    save_dataset(processed_data, output_file)

    # Remove the checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    # Print a few samples for manual inspection
    if processed_data:
        print(f"\n{'='*60}")
        print("Sample output:")
        print("-"*60)
        for i in range(min(3, len(processed_data))):
            rec = processed_data[i]
            print(f"\nSample {i+1} (label: {rec['label']}):")
            print(f"  text:   {rec['text'][:80]}...")
            print(f"  text1:  {rec.get('text1', '')[:80]}...")

    print(f"\n{'='*60}")
    print("Done.")

    return processed_data


# ============== Command-line interface ==============

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synonym-replacement augmentation data with a local LLM (one augmented text per sample)'
    )

    # Input / output
    parser.add_argument('--input', '-i', type=str,
                        default=os.path.join('data', 'agnews'),
                        help='Input file path (label\\ttext or source\\tlabel\\ttext)')
    parser.add_argument('--output', '-o', type=str,
                        default=os.path.join('data', 'Augment', 'agnews.csv'),
                        help='Output CSV path')

    # API configuration
    parser.add_argument('--api_base', type=str,
                        default='http://localhost:8000/v1',
                        help='vLLM API address (default: http://localhost:8000/v1)')
    parser.add_argument('--model', type=str,
                        default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='Model name')

    # Generation configuration
    parser.add_argument('--rate_limit', '-r', type=float, default=0.05,
                        help='Global request interval in seconds (default: 0.05)')

    # Processing configuration
    parser.add_argument('--batch_size', '-b', type=int, default=500,
                        help='Write the full CSV every N samples (default: 500)')
    parser.add_argument('--checkpoint_interval', '-c', type=int, default=100,
                        help='Save a checkpoint every N samples (default: 100)')
    parser.add_argument('--no_resume', action='store_true',
                        help='Ignore an existing checkpoint and start over')
    parser.add_argument('--max_workers', '-w', type=int, default=8,
                        help='Max threads in the pool (default: 8)')

    args = parser.parse_args()

    process_dataset(
        input_file=args.input,
        output_file=args.output,
        api_base=args.api_base,
        model_name=args.model,
        rate_limit=args.rate_limit,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
        max_workers=args.max_workers
    )
