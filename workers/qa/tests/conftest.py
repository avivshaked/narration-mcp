"""Set-up for the QA worker's tests, which run from the worker's own venv (``workers/qa``).

- Tests write only inside the checkout (AGENTS.md rule 3): pytest's base temporary directory, Python's
  ``tempfile`` directory, and matplotlib's and numba's caches are under ``<checkout>/.pytest-tmp``, as the
  repository's own ``conftest.py`` does.
- The CPU thread cap (design section 4.1) and the offline switches are set before torch or transformers
  load, and no test sees a GPU unless it is marked ``gpu`` (those run in a process of their own, with the GPU
  lock; ``test_gpu.py``).
- ``tiny_snapshot`` builds real, tiny snapshots of the three models (random weights), so the loaders' checks
  run on files transformers really writes.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

QA_ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = QA_ROOT.parents[1]
TEST_TMP = CHECKOUT / ".pytest-tmp"
THREADS = "4"

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = THREADS
if os.environ.get("NARRATION_QA_TEST_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = str(TEST_TMP / "matplotlib")
os.environ["NUMBA_CACHE_DIR"] = str(TEST_TMP / "numba")


def pytest_configure(config: pytest.Config) -> None:
    if config.option.basetemp is None:
        config.option.basetemp = TEST_TMP / f"qa-run-{os.getpid()}"
    scratch = TEST_TMP / "tempfile"
    scratch.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)
    for var in ("TMP", "TEMP", "TMPDIR"):
        os.environ[var] = str(scratch)


# ---------------------------------------------------------------------- tiny snapshots

REVISIONS = {"asr": "a" * 40, "sv": "b" * 40, "aligner": "54074b1c16f4de6a5ad59affb4caa8f2ea03a119"}
"""The revisions the tiny snapshots' folders are named by (the aligner's is the pinned one, as WP15's tests use)."""
PINNED_VOCABULARY = ("<pad>", "<s>", "</s>", "<unk>", "|", *"ETAONIHSRDLUMWCFGYPBVK'XJQZ")
"""The pinned wav2vec2 model's vocabulary, in id order."""
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def save_weights(model: Any, path: Path, *, safetensors: bool) -> None:
    """Save ``model`` to ``path`` with its weights in ``model.safetensors`` or in ``pytorch_model.bin``.

    KNOW (transformers 5.17.0): ``save_pretrained(safe_serialization=False)`` still writes
    ``model.safetensors``, so a real ``pytorch_model.bin`` (a torch zip pickle, as the pinned WavLM and wav2vec2
    revisions have) is written with ``torch.save`` of the state dict.
    """
    import torch

    model.save_pretrained(str(path))
    if not safetensors:
        (path / "model.safetensors").unlink()
        torch.save(model.state_dict(), str(path / "pytorch_model.bin"))


def build_tiny_snapshot(root: Path, use: str, *, safetensors: bool = False) -> Path:
    """A real, tiny snapshot for ``use`` (``asr``, ``sv`` or ``aligner``) in ``root/<revision>``."""
    import torch
    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
        WavLMConfig,
        WavLMForXVector,
    )

    path = root / REVISIONS[use]
    torch.manual_seed(0)
    if use == "asr":
        _tiny_whisper(path, safetensors=safetensors)
    elif use == "sv":
        config = WavLMConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            conv_dim=(8,) * 7,
            num_conv_pos_embeddings=4,
            num_conv_pos_embedding_groups=2,
            tdnn_dim=(8, 8, 8, 8, 8),
            xvector_output_dim=8,
            num_buckets=8,
            max_bucket_distance=20,
            architectures=["WavLMForXVector"],
        )
        save_weights(WavLMForXVector(config), path, safetensors=safetensors)
        Wav2Vec2FeatureExtractor(do_normalize=False, return_attention_mask=True).save_pretrained(str(path))
    else:
        config = Wav2Vec2Config(
            vocab_size=len(PINNED_VOCABULARY),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            conv_dim=(8,) * 7,
            num_conv_pos_embeddings=4,
            num_conv_pos_embedding_groups=2,
            architectures=["Wav2Vec2ForCTC"],
        )
        save_weights(Wav2Vec2ForCTC(config), path, safetensors=safetensors)
        Wav2Vec2FeatureExtractor().save_pretrained(str(path))
        vocab = {token: i for i, token in enumerate(PINNED_VOCABULARY)}
        (path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    return path


def _tiny_whisper(path: Path, *, safetensors: bool) -> None:
    """A tiny Whisper with a tokenizer that has Whisper's special tokens, so the ASR pipeline runs end to end."""
    from transformers import (
        WhisperConfig,
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
    )

    path.mkdir(parents=True)
    words = ["hello", "there", "rain", "came", "over", "the", "ridge"]
    vocab = {
        "<|endoftext|>": 0,
        **{f"Ġ{w}": i + 1 for i, w in enumerate(words)},
        **{w: i + 8 for i, w in enumerate(words)},
    }
    (path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = WhisperTokenizer(
        str(path / "vocab.json"), str(path / "merges.txt"), predict_timestamps=True, language=None, task=None
    )
    special = [
        "<|startoftranscript|>",
        "<|en|>",
        "<|transcribe|>",
        "<|translate|>",
        "<|nocaptions|>",
        "<|notimestamps|>",
    ]
    special += [f"<|{i * 0.02:.2f}|>" for i in range(1501)]
    tokenizer.add_tokens(special, special_tokens=True)
    extractor = WhisperFeatureExtractor(feature_size=8, chunk_length=30)
    WhisperProcessor(feature_extractor=extractor, tokenizer=tokenizer).save_pretrained(str(path))

    def ids(token: str) -> int:
        value = tokenizer.convert_tokens_to_ids(token)
        assert isinstance(value, int)
        return value

    config = WhisperConfig(
        vocab_size=len(tokenizer),
        num_mel_bins=8,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        max_source_positions=1500,
        max_target_positions=64,
        decoder_start_token_id=ids("<|startoftranscript|>"),
        pad_token_id=0,
        eos_token_id=0,
        bos_token_id=0,
        architectures=["WhisperForConditionalGeneration"],
    )
    model = WhisperForConditionalGeneration(config)
    generation: Any = model.generation_config  # Whisper's members are not in GenerationConfig's type
    generation.lang_to_id = {"<|en|>": ids("<|en|>")}
    generation.task_to_id = {"transcribe": ids("<|transcribe|>"), "translate": ids("<|translate|>")}
    generation.is_multilingual = True
    generation.no_timestamps_token_id = ids("<|notimestamps|>")
    generation.prev_sot_token_id = None
    generation.decoder_start_token_id = ids("<|startoftranscript|>")
    generation.max_initial_timestamp_index = 50
    generation.alignment_heads = [[0, 0], [0, 1]]
    generation.max_length = 64
    generation._from_model_config = False  # as the real snapshot's: kept as saved, not rebuilt from config.json
    save_weights(model, path, safetensors=safetensors)


@pytest.fixture
def tiny_snapshot(tmp_path: Path) -> Callable[..., Path]:
    """``tiny_snapshot(use, safetensors=False)``: a real, tiny snapshot for a use, in its own folder."""

    def make(use: str, *, safetensors: bool = False) -> Path:
        return build_tiny_snapshot(tmp_path / use, use, safetensors=safetensors)

    return make
