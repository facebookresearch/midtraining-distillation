# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from copy import deepcopy
from functools import partial
import json
from dataclasses import dataclass, field
from multiprocessing import Process, Queue, Event
from queue import Full, Empty
from multiprocessing.synchronize import Event as EventClass
import os
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, TypedDict, Union, Tuple, List
from lingua.tokenizer import build_tokenizer, TokenizerArgs
import numpy as np
import logging

logger = logging.getLogger()

"""
This file contains all code necessary for text data loading from preshuffled jsonl chunks.
For example if given the following files with a world size of 8

/path/to/arxiv:
arxiv.chunk.00.jsonl (Contains many lines of {"text":...} or {"content":...})
arxiv.chunk.01.jsonl
arxiv.chunk.02.jsonl
arxiv.chunk.03.jsonl

/path/to/wikipedia:
wikipedia.chunk.00.jsonl
wikipedia.chunk.01.jsonl
wikipedia.chunk.02.jsonl
wikipedia.chunk.03.jsonl

Step (1) => infinite_block_jsonl_iterator
2 workers will read each jsonl chunk (world_size = 8 distributed over 4 workers) from each source.
Each worker will read 1 line and skip the next, therefore workers on the same file read in an interleaved manner.

Step (2) => multi_choice_iterator
At every iteration, a source is sampled randomly given some weights

Step (3) => tokenizer and pack_tokens
Reads sequences until reaching seq_len tokens and yields a numpy array of shape (seq_len, n_views)

Step (4) => prefetch_data_loader
Prefetches batches in advance and shuffles them to reduce correlation, yields a numpy array of shape (batch_size, seq_len, n_views)

This create a nested iterator structure where each iterator is responsible for a specific task:
    [ [ [ [ [ (1) read document ] -> (2) sample source ] -> (3) tokenize ] -> (4) tokenize and build sequence of fixed seq_len ] -> (5) prefetch batches ]

Each iterator returns a tuple (output, state) where state contains all the info necessary to resume from the last output.

build_mixed_token_packing_dataloader creates the states and return an iterator that does everything above

build_seperate_token_packing_dataloader does the same thing but swaps step 2 and 3

Both can be called with a resume_state to resume from any given position deterministically
"""

TRAIN_DATA_FILE_PATTERN = "*.chunk.*.jsonl"
TRAIN_DATA_FILE_PATTERN_SHARD = (
    "*shard*.jsonl"  # Alternative pattern for shard-based naming
)


class JSONLState(TypedDict):
    """Represents the current state of a JSON line reader.

    Attributes:
        content (Dict): The JSON content of the line.
        file_path (str): The path to the JSONL file.
        position (int): The file position after reading the line (in bytes).
        window (int): The window size used for iteration.
        offset (int): The offset used for iteration.
        current_iter (Optional[int]): Number of iterations over the jsonl file (for infinite iteration).
    """

    file_path: str
    position: int
    block_size: int
    offset: int
    current_iter: int


class MultiChoiceState(TypedDict):
    """Represents the current state of a Multi choice iterator.

    Attributes:
        root_dir: path to dataset root directory
        sources Dict[str, float]: Dict from subdirectory to the weight used for sampling
        source_states: Dict[str, Any] Dict from source to iterator state
        rng_state: dict numpy bit generator state used to resume rng
    """

    root_dir: str
    sources: Dict[str, float]
    source_to_state: Dict[str, Any]
    rng_state: Dict[str, Any]


class TokenizerState(TypedDict):
    it_state: Any
    name: str
    add_bos: bool
    add_eos: bool
    path: Optional[str]
    max_length: Optional[int]  # Maximum sequence length for tokenization truncation


class PackTokensState(TypedDict):
    """Represents the current state of a packing iterator.

    Attributes:
        start_token: int index to start reading from in the current sequence
        output_seq_len: int Length of sequences to output
        n_views: dict int Number of views to output. Each view is the same sequence but shifted by 1 from the previous
    """

    start_token: int
    it_state: Any
    output_seq_len: int
    n_views: int
    seq_len: int


class PrefetchState(TypedDict):
    """Represents the current state of a prefetching iterator.

    Attributes:
        prefetch_buffer: numpy array to store prefetched data
        seq_idx: int index of the current sequence to resume from
        rng_state: dict numpy bit generator state used to resume rng
    """

    it_state: Any
    seq_idx: int
    rng_state: Dict[str, Any]
    prefetch_size: int
    batch_size: int


def read_jsonl(
    file_path: str,
    position: int,
    block_size: int,
    offset: int,
    current_iter: int,
):
    """Iterates over a JSON Lines file, yielding a line every `block_size` lines with an offset

    Example : If block_size = 3, offset = 1, iterator will yield lines 1 4 7 10 ...
    Example : If block_size = 2, offset = 0, iterator will yield lines 0 2 4 6 ...

    Args:
        file_path (str): Path to the JSONL file.
        position (int): The file position (in bytes) from which to start reading.
        block_size (int): The number of lines to skip between yields
        offset (int): The initial number of lines skiped

    Yields:
        JSONLState: Represents the state of each line read according to window and offset.
    """
    if (offset < 0) or (offset >= block_size):
        raise RuntimeError(f"JSONL iterator offset value is invalid")
    # We assume the start position is either 0 or given by the last line yielded
    # Therefore the current line is right after the offset (modulo block_size)
    current_line = offset + 1 if position > 0 else 0

    state = JSONLState(
        file_path=file_path,
        position=position,
        block_size=block_size,
        offset=offset,
        current_iter=current_iter,
    )
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        file.seek(position)
        while line := file.readline():
            current_line += 1
            if (current_line - 1) % block_size == offset:
                # We return state that will allow resuming from this position
                # We update state for next position
                state = JSONLState(
                    file_path=file_path,
                    position=file.tell(),
                    block_size=block_size,
                    offset=offset,
                    current_iter=current_iter,
                )
                try:
                    yield json.loads(line), state
                except json.JSONDecodeError:
                    # Skip malformed lines (can happen with shuffled data)
                    continue


def loop_on_jsonl(
    file_path: str,
    position: int,
    block_size: int,
    offset: int,
    current_iter: int,
):
    """Makes the block jsonl iterator infinite and updates n_iter counter"""
    try:
        while True:
            it = read_jsonl(file_path, position, block_size, offset, current_iter)
            for content, jsonl_state in it:
                yield content, jsonl_state
            current_iter += 1
            position = 0
    finally:
        it.close()


def _extract_text_only(content: Dict[str, Any]) -> str:
    """Extract the document text from one JSONL record.

    Accepts `text`, `content`, or `original_text`. Returns "" for malformed
    records -- shuffled chunks can be cut mid-line -- which the caller skips.
    """
    for key in ("text", "content", "original_text"):
        if key in content:
            return content[key]
    return ""


def tokenize(
    iterator: Iterator,
    add_bos: bool,
    add_eos: bool,
    tokenizer_type: str,
    tokenizer_path: Optional[str] = None,
    max_length: Optional[int] = None,
):
    """
    Tokenizes text from an iterator of content-state pairs using a specified tokenizer.

    Parameters:
    - iterator: An iterable of (content, state) pairs where content is a dict with a 'text' or 'content' key.
    - tokenizer: Tokenizer object with an `encode` method to convert text to tokens, supporting `add_bos` and `add_eos`.
    - add_bos (bool): Flag to add a beginning-of-sequence token.
    - add_eos (bool): Flag to add an end-of-sequence token.

    Yields:
    - (tokens, state) pairs: `tokens` is a list of token ids, `state` is the
      TokenizerState.
    """
    tokenizer = build_tokenizer(name=tokenizer_type, path=tokenizer_path)

    # Track statistics for error handling
    if not hasattr(tokenize, "_error_count"):
        tokenize._error_count = 0
        tokenize._processed_count = 0

    for content, state in iterator:
        try:
            tokenize._processed_count += 1

            full_text = _extract_text_only(content)

            # Validate that full_text is not None or empty
            if full_text is None:
                tokenize._error_count += 1
                if tokenize._error_count <= 1 or tokenize._error_count % 10000 == 0:
                    logger.warning(
                        f"Skipping document with None text at position {tokenize._processed_count}"
                    )
                continue

            if not isinstance(full_text, str):
                tokenize._error_count += 1
                if tokenize._error_count <= 1 or tokenize._error_count % 10000 == 0:
                    logger.warning(
                        f"Skipping document with non-string text (type: {type(full_text)}) at position {tokenize._processed_count}"
                    )
                continue

            if len(full_text.strip()) == 0:
                tokenize._error_count += 1
                # This runs in many loader workers/ranks; keep logs very sparse.
                if tokenize._error_count <= 1 or tokenize._error_count % 10000 == 0:
                    logger.warning(
                        f"Skipping document with empty text at position {tokenize._processed_count} (total empty: {tokenize._error_count})"
                    )
                continue

            if not full_text.endswith("\n"):
                full_text = full_text + "\n"

            # Tokenize full text with error handling
            try:
                tokens = tokenizer.encode(full_text, add_bos=add_bos, add_eos=add_eos)
            except Exception as tokenize_error:
                logger.error(
                    f"Tokenization failed at position {tokenize._processed_count}: {str(tokenize_error)}. "
                    f"Text preview (first 200 chars): {full_text[:200]!r}"
                )
                tokenize._error_count += 1
                if tokenize._error_count <= 10:
                    logger.error(f"Full error details: {tokenize_error}", exc_info=True)
                continue

        except Exception as e:
            # Catch any other unexpected errors in the data processing pipeline
            logger.error(
                f"Unexpected error processing document at position {tokenize._processed_count}: {str(e)}"
            )
            tokenize._error_count += 1
            if tokenize._error_count <= 10:
                logger.error(f"Full error details: {e}", exc_info=True)

            # Log periodic error statistics
            if tokenize._error_count % 100 == 1 or tokenize._error_count <= 10:
                error_rate = (tokenize._error_count / tokenize._processed_count) * 100
                logger.warning(
                    f"Data processing error statistics: {tokenize._error_count} errors out of "
                    f"{tokenize._processed_count} documents ({error_rate:.2f}% error rate)"
                )
            continue

        # Apply max_length truncation if specified (MUST match score_dclm.py truncation)
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
            # Log truncation occasionally
            if hasattr(tokenize, "_truncation_count"):
                tokenize._truncation_count += 1
            else:
                tokenize._truncation_count = 1
                logger.info(f"Tokenization truncation enabled: max_length={max_length}")
            if (
                tokenize._truncation_count <= 5
                or tokenize._truncation_count % 1000 == 0
            ):
                logger.info(
                    f"Truncated document from {len(tokens)} to {max_length} tokens (count: {tokenize._truncation_count})"
                )

        yield tokens, TokenizerState(
            it_state=state,
            add_bos=add_bos,
            add_eos=add_eos,
            name=tokenizer_type,
            path=tokenizer_path,
            max_length=max_length,
        )


def choose_source(
    source_to_iterator: Dict[str, Iterator],
    source_to_state: Dict[str, Any],
    root_dir: str,
    sources: Dict[str, float],
    rng_state: Dict[str, Any],
):
    """
    Iterates over multiple data sources, selecting sequences based on weighted random choice.

    Parameters:
    - source_to_iterator (Dict[str, Iterator]): Dict from source paths to their iterators.
    - source_to_state (Dict[str, State]): Initial state for each source, allowing state tracking.
    - root_dir str: Root dir of data sources
    - sources Dict[str, float]: Dict from subdirectory to the weight used for sampling
    - rng_state (dict): State of the random number generator for reproducibility.

    Yields:
    - Tuple of (seq, multi_choice_state) where `seq` is the next sequence from the chosen source,
    and `multi_choice_state` includes the current state of all sources and the RNG.

    This function ensures that sequences are chosen from the provided sources based on the specified weights,
    maintaining state information for each source and the RNG to allow for reproducible iteration.
    """
    # We create the rng and set its state
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    while True:
        # We save the rng state before sampling to be able to yield the same sequence on reload
        # Read sources/weights live so callers can update domain weights during training
        # by mutating the same `sources` dict in loader state.
        possible_sources = list(sources.keys())
        weights = np.array([float(v) for v in sources.values()], dtype=np.float64)
        n_sources = len(possible_sources)
        if n_sources == 0:
            raise RuntimeError("No data sources configured for choose_source")
        if np.any(weights < 0):
            raise RuntimeError(f"Negative source weights are not allowed: {sources}")
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            raise RuntimeError(f"Sum of source weights must be > 0, got {weight_sum}")
        norm_weights = weights / weight_sum
        source_choice = possible_sources[rng.choice(n_sources, p=norm_weights)]
        seq, state = next(source_to_iterator[source_choice])
        source_to_state = {**source_to_state, source_choice: state}
        # We update the corresponding source state
        multi_choice_state = MultiChoiceState(
            root_dir=root_dir,
            sources=sources,
            source_to_state=source_to_state,
            rng_state=rng.bit_generator.state,
        )
        yield seq, multi_choice_state


def get_empty_buffer_state(
    start_token,
    states,
):
    """
    Calculates the state to resume iteration after the buffer is cleared.

    This function determines the starting point for resuming iteration by rewinding `n_views` from the `end_token`.
    It handles cases where the rewind goes beyond the current sequence, adjusting the starting sequence and token index accordingly.
    """
    # We rewind n_views
    # This index can be negative if we go beyond the current sample
    # In that case we go back to find which sequence to start from
    # And the correct token index to start from
    seq_to_resume_from = -1
    while start_token < 0:
        seq_to_resume_from -= 1
        start_token += states[seq_to_resume_from]["seq_len"]
    resume_state = deepcopy(states[seq_to_resume_from])
    resume_state["start_token"] = start_token
    # When resuming, the iterator will then correctly fill the buffer
    del states[:seq_to_resume_from]
    if "seq_len" in resume_state:
        del resume_state["seq_len"]

    return resume_state


def pack_tokens(
    iterator: Iterator,
    empty_buffer_state: PackTokensState,
):
    """
    Iterates over tokens, packing them into chunks.

    This function aggregates tokens into a buffer and yields fixed-size chunks with dimensions `(output_seq_len, n_views)`,
    where each column represents shifted sequences of tokens. It ensures continuity in token sequences across chunks,
    preventing boundary effects and maintaining consistency regardless of `n_views`.

    When teacher logprobs are available, they are packed as an additional view (n_views + 1).

    Also tracks document boundaries (cu_seqlens) for cross-document attention masking.

    Parameters:
    - iterator: An iterator that yields pairs of ((tokens, teacher_logprobs), state), where tokens is a 1D sequence of tokens,
                teacher_logprobs is either None or a 1D sequence of per-token logprobs, and state contains all necessary
                information to resume iterator from current position.
    - empty_buffer_state: Initial PackTokensState with parameters

    Yields:
    - dict containing:
      - 'tokens': numpy.ndarray of shape `(output_seq_len, n_views)` or `(output_seq_len, n_views+1)`
      - 'cu_seqlens': list of cumulative sequence lengths marking document boundaries
    - PackTokensState: The state required to resume packing tokens from where the last returned chunk.

    The function handles the complexity of determining the correct state for resuming iteration after the buffer is cleared, ensuring seamless continuation of token sequences.
    """
    buffer = []
    doc_boundaries = []  # Track document start positions within the current buffer
    states = []
    output_seq_len = empty_buffer_state["output_seq_len"]
    n_views = empty_buffer_state["n_views"]
    start_token = empty_buffer_state["start_token"]
    previous_state = empty_buffer_state["it_state"]
    buffer_size = output_seq_len + n_views - 1

    current_doc_start = 0  # Track where current document starts in buffer

    for i, (tokens, state) in enumerate(iterator):
        end_token = start_token
        sample_is_read = False

        # Defensive recovery: if a resumed state carries an out-of-range token
        # offset for the current document, skip this doc instead of crashing the
        # async loader process.
        if start_token >= len(tokens):
            logger.warning(
                f"[data] skipping doc with invalid start_token={start_token} "
                f"(len(tokens)={len(tokens)})"
            )
            start_token = 0
            previous_state = state
            continue

        # Track document boundary (start of new document in buffer)
        if len(buffer) > 0 or start_token == 0:
            # Only add boundary at start of a new document
            if start_token == 0:
                doc_boundaries.append(len(buffer))

        while not sample_is_read:
            assert start_token < len(
                tokens
            ), f"Start token index {start_token} bigger than sequence {len(tokens)}"
            free_space = buffer_size - len(buffer)
            seq_len = min(free_space, len(tokens) - start_token)
            end_token = start_token + seq_len
            buffer.extend(tokens[start_token:end_token])

            start_token = end_token

            states.append(
                PackTokensState(
                    start_token=start_token,
                    seq_len=seq_len,
                    it_state=previous_state,
                    output_seq_len=output_seq_len,
                    n_views=n_views,
                )
            )
            assert len(buffer) <= buffer_size, "Buffer overflow"

            if len(buffer) == buffer_size:
                out = np.array(buffer)
                assert out.ndim == 1, "Iterator should return 1D sequences"
                out = np.lib.stride_tricks.sliding_window_view(
                    out, n_views, axis=0
                )  # (output_seq_len, n_views)

                # Build cu_seqlens from document boundaries
                # cu_seqlens marks cumulative positions: [0, doc1_end, doc1_end+doc2_end, ..., output_seq_len]
                # We need to adjust boundaries to be within [0, output_seq_len]
                cu_seqlens = [0]
                for boundary in doc_boundaries:
                    if boundary > 0 and boundary < output_seq_len:
                        cu_seqlens.append(boundary)
                cu_seqlens.append(output_seq_len)
                # Ensure cu_seqlens is sorted and unique
                cu_seqlens = sorted(set(cu_seqlens))

                # We rewind by n_views to account for the last tokens not having their targets
                rewinded_idx = start_token - (n_views - 1)
                empty_buffer_state = get_empty_buffer_state(rewinded_idx, states)
                buffer = buffer[output_seq_len:]
                assert len(buffer) == (n_views - 1)

                # Adjust doc_boundaries for the remaining buffer
                # Any boundaries that were in the yielded part should be removed
                # Boundaries after output_seq_len should be shifted
                new_doc_boundaries = []
                for boundary in doc_boundaries:
                    shifted = boundary - output_seq_len
                    if shifted >= 0:
                        new_doc_boundaries.append(shifted)
                doc_boundaries = new_doc_boundaries

                yield {"tokens": out, "cu_seqlens": cu_seqlens}, empty_buffer_state

            if start_token == len(tokens):
                start_token = 0
                sample_is_read = True
                previous_state = state


def batch_and_shuffle_prefetched_sequences(
    data_loader: Iterator,
    batch_size: int,
    prefetch_size: int,
    seq_len: int,
    n_views: int,
    state: PrefetchState,
):
    """
    Prepare batch in advance and shuffle them to reduce correlation inside batches (for ex when very long document is encountered).

    This function aggregates batches into a buffer and yields fixed-size batch size and seqlen with dimensions `(batch_size, seqlen, n_views)`,

    It uses a prefetch buffer to store batches in advance and shuffles them, the prefetch buffer is similar to `reservoir sampling`,
    but by block to preserve a smooth, easy and deterministic reloading. To ensure more uniform sequence sampling -> prefetch_size * batch_size * seq_len >> max_document_seqlength.

    Parameters:
    - iterator: An iterator that yields pairs of (sequence_dict, state), where sequence_dict contains 'tokens' and 'cu_seqlens'.
    - batch_size: The desired batch size.
    - prefetch_size: The number of batches to prefetch in advance.
    - seq_len (int): The length of the output sequences to be generated.
    - n_views (int): The number of shifted views to include in each output chunk.

    Yields:
    - dict with 'tokens' array of shape `(batch_size, seq_len, n_views)` and 'cu_seqlens' list per batch item.
    - PrefetchState: The state required to resume prefetched batch. Contains also the internal of iterator.
    """
    # NOTE: When teacher logprobs are present, pack_tokens adds an extra view (n_views+1)
    # We initialize the buffer lazily after seeing the first item to get the actual shape
    prefetch_buffer = None
    cu_seqlens_buffer = []  # Store cu_seqlens for each sequence
    actual_n_views = n_views  # Will be updated after seeing first item
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]

    # Rewind the iterator to the correct position by skipping seq_idx sequences to roll the buffer accordingly
    seq_idx = state["seq_idx"]
    assert (
        seq_idx >= 0 and seq_idx < prefetch_size
    ), "Prefetch state seq_idx should be in 0 <= seq_idx < prefetch_size."

    _rng_state = state["rng_state"]
    _it_state = state["it_state"]

    for i in range(prefetch_size * batch_size):
        item, next_it_state = next(data_loader)
        # Handle both dict format (new) and array format (old)
        if isinstance(item, dict):
            tokens = item["tokens"]
            cu_seqlens = item.get("cu_seqlens", [0, seq_len])
        else:
            tokens = item
            cu_seqlens = [0, seq_len]  # Default: single document spanning full sequence

        # Lazy initialization of prefetch buffer on first item
        if prefetch_buffer is None:
            actual_n_views = tokens.shape[
                -1
            ]  # Get actual number of views (may include teacher logprobs)
            prefetch_buffer = -1 * np.ones(
                (prefetch_size * batch_size, seq_len, actual_n_views),
                dtype=tokens.dtype,
            )
        prefetch_buffer[i] = tokens
        cu_seqlens_buffer.append(cu_seqlens)

    # Shuffle both buffers together
    shuffle_indices = rng.permutation(prefetch_size * batch_size)
    prefetch_buffer = prefetch_buffer[shuffle_indices]
    cu_seqlens_buffer = [cu_seqlens_buffer[i] for i in shuffle_indices]

    for i in range(seq_idx * batch_size):
        item, _ = next(data_loader)
        if isinstance(item, dict):
            prefetch_buffer[i] = item["tokens"]
            cu_seqlens_buffer[i] = item.get("cu_seqlens", [0, seq_len])
        else:
            prefetch_buffer[i] = item
            cu_seqlens_buffer[i] = [0, seq_len]

    idx = seq_idx
    while True:
        if idx == prefetch_size - 1:
            _it_state = next_it_state
            _rng_state = rng.bit_generator.state

        state = PrefetchState(
            it_state=_it_state,
            seq_idx=(idx + 1) % prefetch_size,
            rng_state=_rng_state,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
        )

        # Extract batch tokens and cu_seqlens
        batch_tokens = prefetch_buffer[idx * batch_size : (idx + 1) * batch_size].copy()
        batch_cu_seqlens = cu_seqlens_buffer[idx * batch_size : (idx + 1) * batch_size]

        yield {"tokens": batch_tokens, "cu_seqlens": batch_cu_seqlens}, state

        for i in range(batch_size):
            item, pack_state = next(data_loader)
            if isinstance(item, dict):
                prefetch_buffer[idx * batch_size + i] = item["tokens"]
                cu_seqlens_buffer[idx * batch_size + i] = item.get(
                    "cu_seqlens", [0, seq_len]
                )
            else:
                prefetch_buffer[idx * batch_size + i] = item
                cu_seqlens_buffer[idx * batch_size + i] = [0, seq_len]

        if idx == prefetch_size - 1:
            next_it_state = pack_state
            shuffle_indices = rng.permutation(prefetch_size * batch_size)
            prefetch_buffer = prefetch_buffer[shuffle_indices]
            cu_seqlens_buffer = [cu_seqlens_buffer[i] for i in shuffle_indices]

        idx = (idx + 1) % prefetch_size


def find_and_sanitize_chunks(
    dataset_path: str, world_size: int, file_pattern: str = TRAIN_DATA_FILE_PATTERN
):
    dataset_chunks = [str(p) for p in Path(dataset_path).glob(file_pattern)]
    n_chunks = len(dataset_chunks)

    # If no chunks found with default pattern, try shard pattern as fallback
    if n_chunks == 0:
        logger.info(
            f"No chunks found with pattern '{file_pattern}', trying shard pattern '{TRAIN_DATA_FILE_PATTERN_SHARD}'"
        )
        dataset_chunks = [
            str(p) for p in Path(dataset_path).glob(TRAIN_DATA_FILE_PATTERN_SHARD)
        ]
        n_chunks = len(dataset_chunks)
        if n_chunks > 0:
            logger.info(f"Found {n_chunks} chunks with shard pattern")

    # Check n_chunks > 0 first before any modulo operations
    if n_chunks == 0:
        raise ValueError(
            f"No valid chunks found in {dataset_path} matching patterns '{file_pattern}' or '{TRAIN_DATA_FILE_PATTERN_SHARD}'. Check that data files exist."
        )

    if n_chunks > world_size:
        n_discard = n_chunks - world_size
        dataset_chunks = dataset_chunks[:world_size]
    else:
        assert (
            world_size % n_chunks == 0
        ), "World size should be a multiple of number of chunks"

    return dataset_chunks


def distribute_data_to_rank(
    dataset_path: str, rank: int, world_size: int, file_pattern: str
):
    """
    Distributes the chunk files in a dataset path to each worker.
    If world_size is smaller than the number of chunks, the extra chunks are discarded.
    Otherwise, world_size is assumed to be a multiple of number of chunks.
    In that case there are world_size//nb_chunks workers on each chunk file, reading with different offsets.
    """
    dataset_chunks = find_and_sanitize_chunks(dataset_path, world_size, file_pattern)
    n_ranks_per_chunk = world_size // len(dataset_chunks)
    rank_to_jsonl_iterator_params = []
    for chunk_path in dataset_chunks:
        for i in range(n_ranks_per_chunk):
            rank_to_jsonl_iterator_params.append(
                JSONLState(
                    file_path=chunk_path,
                    position=0,
                    block_size=n_ranks_per_chunk,
                    offset=i,
                    current_iter=0,
                )
            )

    return rank_to_jsonl_iterator_params[rank]


def init_choice_state(
    root_dir: str,
    sources: Dict[str, float],
    seed: int,
    rank: int,
    world_size: int,
    file_pattern: str,
):
    data_path_to_jsonl_state = dict()
    for dataset_path in sources:
        jsonl_state = distribute_data_to_rank(
            os.path.join(root_dir, dataset_path), rank, world_size, file_pattern
        )
        data_path_to_jsonl_state[dataset_path] = jsonl_state

    multi_rng_state = np.random.default_rng(
        (seed, rank, world_size)
    ).bit_generator.state

    multi_choice_state = MultiChoiceState(
        root_dir=root_dir,
        sources=sources,
        source_to_state=data_path_to_jsonl_state,
        rng_state=multi_rng_state,
    )
    return multi_choice_state


def init_state(
    root_dir: str,
    sources: Dict[str, float],
    batch_size: int,
    prefetch_size: int,
    seq_len: int,
    n_views: int,
    seed: int,
    rank: int,
    world_size: int,
    add_bos: bool,
    add_eos: bool,
    tokenizer_name: str,
    tokenizer_path: Optional[str] = None,
    file_pattern: str = TRAIN_DATA_FILE_PATTERN,
    max_length: Optional[int] = None,
):
    multi_choice_state = init_choice_state(
        root_dir=root_dir,
        sources=sources,
        seed=seed,
        rank=rank,
        world_size=world_size,
        file_pattern=file_pattern,
    )

    tokenizer_state = TokenizerState(
        it_state=multi_choice_state,
        add_bos=add_bos,
        add_eos=add_eos,
        name=tokenizer_name,
        path=tokenizer_path,  # Will be set per-document in tokenize()
        max_length=max_length,
    )
    pack_state = PackTokensState(
        start_token=0,
        it_state=tokenizer_state,
        output_seq_len=seq_len,
        n_views=n_views,
        seq_len=0,
    )

    prefetch_rng_state = np.random.default_rng(
        (seed + 1, rank, world_size)
    ).bit_generator.state

    return PrefetchState(
        it_state=pack_state,
        seq_idx=0,
        rng_state=prefetch_rng_state,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
    )


def setup_sources(multi_state):
    path_to_iter = dict()
    for source in multi_state["sources"]:
        jsonl_state = multi_state["source_to_state"][source]
        path_to_iter[source] = loop_on_jsonl(
            jsonl_state["file_path"],
            jsonl_state["position"],
            jsonl_state["block_size"],
            jsonl_state["offset"],
            jsonl_state["current_iter"],
        )

    return path_to_iter


@contextlib.contextmanager
def build_dataloader(
    state: PrefetchState,
):
    pack_state = state["it_state"]
    tokenizer_state = pack_state["it_state"]
    multi_state = tokenizer_state["it_state"]

    path_to_iter = setup_sources(multi_state)
    data_it = choose_source(
        source_to_iterator=path_to_iter,
        source_to_state=multi_state["source_to_state"],
        root_dir=multi_state["root_dir"],
        sources=multi_state["sources"],
        rng_state=multi_state["rng_state"],
    )

    # Apply delta filtering if configured

    data_it = tokenize(
        data_it,
        tokenizer_state["add_bos"],
        tokenizer_state["add_eos"],
        tokenizer_state["name"],
        tokenizer_state["path"],
        tokenizer_state.get("max_length"),
    )

    data_it = pack_tokens(
        data_it,
        pack_state,
    )

    data_it = batch_and_shuffle_prefetched_sequences(
        data_loader=data_it,
        seq_len=pack_state["output_seq_len"],
        n_views=pack_state["n_views"],
        batch_size=state["batch_size"],
        prefetch_size=state["prefetch_size"],
        state=state,
    )
    yield data_it
    for it in path_to_iter.values():
        it.close()
    data_it.close()


def feed_buffer(queue: Queue, stop_event: EventClass, iterator_builder):
    """
    Producer function to fetch data from an iterable dataset and put it into a queue.
    Incorporates timeout management to avoid hanging on queue.put() when the queue is full.
    """
    with iterator_builder() as iterator:
        for item in iterator:
            while not stop_event.is_set():
                try:
                    queue.put(
                        item, timeout=0.1
                    )  # Attempts to put item into the queue with a timeout
                    break  # On successful put, breaks out of the while loop
                except Full:
                    pass
            if stop_event.is_set():
                break


def consume_buffer(producer: Process, queue: Queue):
    """
    Consumer function to process items from the queue.
    Handles cases where the queue might be empty by implementing timeouts on queue.get().
    """
    while producer.exitcode is None:
        try:
            item = queue.get(
                timeout=0.1
            )  # Tries to get an item from the queue with a timeout
            yield item
        except Empty:
            pass

    raise RuntimeError(
        "Data loader quit unexpectedly, real error has been raised previously"
    )


@contextlib.contextmanager
def async_iterator(buffer_size: int, iterator_builder):
    """
    Context manager to setup and manage asynchronous iteration with producer-consumer model.
    """
    queue = Queue(maxsize=buffer_size)
    stop_event = Event()
    producer = Process(target=feed_buffer, args=(queue, stop_event, iterator_builder))
    logger.info("Async dataloader started")
    producer.start()

    consumer = consume_buffer(producer, queue)
    try:
        yield consumer
    finally:
        stop_event.set()  # Ensures the stop event is signaled
        consumer.close()
        producer.join(timeout=0.2)  # Waits for the producer to finish
        if producer.exitcode is None:
            logger.info(f"Killing async data process {producer.pid} ...")
            producer.kill()
        else:
            logger.info(
                f"Async data process {producer.pid} exited with code {producer.exitcode}"
            )
        logger.info("Async dataloader cleaned up")


@dataclass
class DataArgs:
    """Data + KD configuration for the released recipes.

    Only the fields needed by the three supported losses are declared here
    (forward KL, reverse KL, and switch distillation).
    All other ``use_*`` knobs from upstream lingua/ are intentionally
    omitted; train.py reads optional flags via ``getattr(args.data, X,
    default)`` so unset families simply fall through to their defaults.
    """

    # ---------------- Data loading ----------------
    root_dir: Optional[str] = None
    sources: Dict[str, float] = field(default_factory=dict)
    batch_size: int = 2
    seq_len: int = 2048
    n_views: int = 2
    seed: int = 42
    add_bos: bool = True
    add_eos: bool = True
    load_async: bool = True
    prefetch_size: int = 64
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
    max_length: Optional[int] = None  # max sequence length for tokenization

    # Misc. scaffolding referenced directly by train.py (defaults are no-ops)
    add_special_tokens: bool = False
    mask_cross_doc_loss: bool = False
    disable_cross_doc_attn: bool = False
    source_weight_schedule: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ============================================================
    # Distillation flags + hyperparameters for the three supported losses
    # ============================================================

    # ----- Forward KL (FKD baseline: kd_rl7b, kd_rl_1binstruct) -----
    use_kl_distillation: bool = False
    kl_temperature: float = 2.0
    kl_alpha: float = 0.5
    kl_chunk_size: int = 128

    # ----- Reverse KL (MiniLLM-style): the KL-direction counterpart of the
    #       forward-KL baseline above. Same alpha / temperature. -----
    use_reverse_kl_distillation: bool = False
    reverse_kl_temperature: float = 2.0
    reverse_kl_alpha: float = 0.5
    reverse_kl_chunk_size: int = 128

    # ----- Switch distillation: partition tokens between CE and RKL by teacher
    #       entropy. The headline method. -----
    use_switch_distill: bool = False
    switch_distill_temperature: float = 2.0
    switch_distill_lambda_ce: float = 1.0
    switch_distill_quantile: float = 0.20
    switch_distill_chunk_size: int = 128


def init_dataloader_state_from_args(
    args: DataArgs,
    rank: int,
    world_size: int,
):
    return init_state(
        root_dir=args.root_dir,
        sources=args.sources,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        prefetch_size=args.prefetch_size,
        n_views=args.n_views,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        tokenizer_name=args.tokenizer.name,
        tokenizer_path=args.tokenizer.path,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
        max_length=args.max_length,
    )


def build_dataloader_from_args(
    args: DataArgs,
    state: Optional[PrefetchState] = None,
):
    data_builder = partial(
        build_dataloader,
        state,
    )
    if args.load_async:
        return async_iterator(args.prefetch_size, data_builder)
    else:
        return data_builder()
