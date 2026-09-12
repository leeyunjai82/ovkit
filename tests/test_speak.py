"""Reading text out loud.

Four networks chained, and every link is a chance to hand the next one the
wrong shape. These tests run the real :class:`Speaker` code with stand-in
graphs, so the wiring — what feeds what, how long the latent is, how many times
the estimator runs — is checked without downloading 125 MB of weights.

The numbers the chain is built on come from Supertonic's own tables and were
read off the real graphs, not remembered:

    duration_predictor  text_ids + style_dp[1,8,16] + text_mask -> duration[1]
    text_encoder        text_ids + style_ttl[1,50,256] + text_mask -> [1,256,L]
    vector_estimator    noisy_latent[1,144,T] + ... -> denoised_latent[1,144,T]
    vocoder             latent[1,144,T] -> wav[1,N]
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ovkit.pipelines.speak import LANGUAGES, VOICES, Speaker, _prepare, detect_language

SAMPLE_RATE = 44_100
LATENT_DIM = 144

#: Enough of ``tts.json`` for the chain to size its latent.
CFG = {
    "ae": {"sample_rate": SAMPLE_RATE, "base_chunk_size": 1024},
    "ttl": {"chunk_compress_factor": 3, "latent_dim": 48},
}

#: A character table with ids for the few characters the tests use, so a
#: "most of this sentence is unknown" warning only fires when meant.
INDEXER = [-1] * 0x10000


def _index(chars: str) -> None:
    # Index the decomposed form: ``_prepare`` runs NFKD, so what reaches the
    # table is jamo, never the composed syllable. Indexing "안" and expecting a
    # hit is the same mistake the model would punish with silence.
    import unicodedata

    for i, ch in enumerate(unicodedata.normalize("NFKD", chars)):
        INDEXER[ord(ch)] = i


_index("<>/koen안녕하세요. Helo")


class FakeNet:
    """One graph: records what it was fed, returns a shape-correct answer."""

    def __init__(self, respond):
        self.respond = respond
        self.calls: list[dict] = []

    def infer(self, feeds):
        self.calls.append({k: np.asarray(v) for k, v in feeds.items()})
        return self.respond(feeds)


@pytest.fixture()
def speaker(tmp_path, monkeypatch):
    """A Speaker whose tables are local files and whose graphs are fakes."""
    data = tmp_path / "data"
    (data / "voices").mkdir(parents=True)
    (data / "tts.json").write_text(json.dumps(CFG), encoding="utf-8")
    (data / "unicode_indexer.json").write_text(json.dumps(INDEXER), encoding="utf-8")
    (data / "voices" / "F1.json").write_text(
        json.dumps(
            {
                "style_ttl": {"dims": [1, 50, 256], "data": [[0.0] * 256] * 50, "type": "float32"},
                "style_dp": {"dims": [1, 8, 16], "data": [[0.0] * 16] * 8, "type": "float32"},
            }
        ),
        encoding="utf-8",
    )

    from ovkit.pipelines import speak as mod

    def fake_fetch(repo, filename, *, group):
        return data / filename.split("/data/", 1)[1]

    monkeypatch.setattr(mod, "fetch_data", fake_fetch, raising=False)
    monkeypatch.setattr("ovkit.core.download.fetch_data", fake_fetch)

    pipe = Speaker(voice="F1", steps=3, speed=1.0)
    nets = {
        "supertonic3_duration_predictor": FakeNet(
            lambda f: {"duration": np.asarray([2.0], dtype=np.float32)}
        ),
        "supertonic3_text_encoder": FakeNet(
            lambda f: {
                "text_emb": np.zeros((1, 256, f["text_ids"].shape[1]), dtype=np.float32),
            }
        ),
        "supertonic3_vector_estimator": FakeNet(
            lambda f: {"denoised_latent": np.asarray(f["noisy_latent"], dtype=np.float32) * 0.5}
        ),
        "supertonic3_vocoder": FakeNet(
            lambda f: {
                "wav_tts": np.full(
                    (1, 4 * SAMPLE_RATE), 0.1, dtype=np.float32
                )  # longer than asked for
            }
        ),
    }
    monkeypatch.setattr(pipe, "model", lambda name: nets[name])
    pipe.nets = nets
    return pipe


# -- the text side -----------------------------------------------------------


def test_hangul_is_read_as_korean():
    assert detect_language("안녕하세요") == "ko"
    assert detect_language("Good morning") == "en"
    assert detect_language("こんにちは") == "ja"


def test_the_language_tag_wraps_the_sentence():
    import unicodedata

    # Compared in NFKD because that is what _prepare returns — see below.
    assert _prepare("안녕하세요", "ko") == unicodedata.normalize("NFKD", "<ko>안녕하세요.</ko>")


def test_a_sentence_that_already_ends_is_left_alone():
    assert _prepare("Hello!", "en") == "<en>Hello!</en>"


def test_hangul_is_decomposed_into_jamo():
    """NFKD is not cosmetic — the character table is indexed by jamo.

    Skip it and every syllable is one unknown id, which is the difference
    between speech and noise.
    """
    prepared = _prepare("안", "ko")
    inner = prepared[len("<ko>") : -len("</ko>") - 1]
    assert len(inner) == 3, f"expected 3 jamo, got {inner!r}"


def test_an_unknown_language_is_refused_with_the_list():
    with pytest.raises(ValueError, match="지원하지 않는"):
        Speaker(lang="klingon")
    assert "ko" in LANGUAGES and "en" in LANGUAGES


def test_an_unknown_voice_is_refused():
    with pytest.raises(ValueError, match="목소리"):
        Speaker(voice="Z9")
    assert "F1" in VOICES and len(VOICES) == 10


# -- the chain ---------------------------------------------------------------


def test_it_speaks_and_the_result_carries_audio(speaker):
    result = speaker.say("안녕하세요", lang="ko")
    samples, sr = result.audio
    assert sr == SAMPLE_RATE
    assert result.text == "안녕하세요"
    assert result.task == "speak"


def test_the_waveform_is_trimmed_to_the_predicted_duration(speaker):
    """The vocoder pads; the duration predictor is what says when to stop."""
    result = speaker.say("안녕하세요", lang="ko")
    samples, sr = result.audio
    assert len(samples) == int(2.0 * SAMPLE_RATE), "2 seconds was predicted"


def test_speed_shortens_the_sentence(speaker):
    speaker.speed = 2.0
    samples, _ = speaker.say("안녕하세요", lang="ko").audio
    assert len(samples) == int(1.0 * SAMPLE_RATE)


def test_the_estimator_runs_once_per_step_and_feeds_itself(speaker):
    """Flow matching: the output of each pass is the input of the next."""
    speaker.say("안녕하세요", lang="ko")
    calls = speaker.nets["supertonic3_vector_estimator"].calls
    assert len(calls) == 3, "steps=3"
    assert [float(c["current_step"][0]) for c in calls] == [0.0, 1.0, 2.0]
    assert all(float(c["total_step"][0]) == 3.0 for c in calls)
    # Each fake pass halves the latent, so seeing the halving arrive back is
    # what proves the output was fed forward rather than the noise re-sent.
    second, third = calls[1]["noisy_latent"], calls[2]["noisy_latent"]
    assert np.allclose(third, second * 0.5)


def test_the_latent_is_sized_from_the_duration(speaker):
    """latent_len = ceil(seconds * sample_rate / (base_chunk * compress))."""
    speaker.say("안녕하세요", lang="ko")
    latent = speaker.nets["supertonic3_vector_estimator"].calls[0]["noisy_latent"]
    chunk = CFG["ae"]["base_chunk_size"] * CFG["ttl"]["chunk_compress_factor"]
    expected = (int(2.0 * SAMPLE_RATE) + chunk - 1) // chunk
    assert latent.shape == (1, LATENT_DIM, expected)


def test_every_graph_gets_the_voice_it_needs(speaker):
    """The duration predictor takes style_dp; the other two take style_ttl."""
    speaker.say("안녕하세요", lang="ko")
    dp = speaker.nets["supertonic3_duration_predictor"].calls[0]
    enc = speaker.nets["supertonic3_text_encoder"].calls[0]
    est = speaker.nets["supertonic3_vector_estimator"].calls[0]
    assert dp["style_dp"].shape == (1, 8, 16)
    assert enc["style_ttl"].shape == (1, 50, 256)
    assert est["style_ttl"].shape == (1, 50, 256)
    assert "style_ttl" not in dp and "style_dp" not in enc


def test_the_same_sentence_sounds_the_same_twice(speaker):
    """Seeded by default: a demo that changes every run is hard to talk about."""
    first, _ = speaker.say("안녕하세요", lang="ko").audio
    second, _ = speaker.say("안녕하세요", lang="ko").audio
    assert np.array_equal(first, second)


def test_text_the_model_cannot_read_is_flagged(speaker):
    """Silence would be a worse answer than a warning."""
    with pytest.warns(RuntimeWarning, match="글자표"):
        speaker.say("ЖЖЖЖЖЖЖЖЖЖ", lang="en")


def test_empty_text_is_refused(speaker):
    with pytest.raises(ValueError, match="비어"):
        speaker.say("   ")


def test_a_list_of_sentences_gives_one_result_each(speaker):
    out = speaker.predict(["안녕하세요", "안녕하세요"])
    assert len(out) == 2 and all(r.audio is not None for r in out)


def test_a_picture_is_refused_with_a_useful_message(speaker):
    with pytest.raises(TypeError, match="글을 받습니다"):
        speaker.run(np.zeros((8, 8, 3), dtype=np.uint8))


def test_it_is_reachable_by_its_korean_name():
    from ovkit.pipelines import resolve_name

    for name in ("읽어주기", "말하기", "speak", "tts"):
        assert resolve_name(name) == "speak"
