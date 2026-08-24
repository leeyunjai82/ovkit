"""Audio in: a file becomes an answer, not a tensor the caller must frame."""

from __future__ import annotations

import numpy as np
import pytest

from ovkit.audio import read_audio, resample, to_mono, waveform, write_wav
from ovkit.core.results import Results
from ovkit.recognize.audio import Denoiser, SoundClassifier, frames, window_length


def _tone(seconds: float, sr: int = 16_000, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# -- reading ----------------------------------------------------------------


def test_wav_round_trips_through_read_and_write(tmp_path):
    path = tmp_path / "tone.wav"
    write_wav(path, _tone(0.25), 16_000)
    audio, sr = read_audio(path)
    assert sr == 16_000
    assert audio.shape == (4000,)
    assert np.abs(audio).max() == pytest.approx(0.5, abs=0.01)


def test_reading_resamples_to_what_the_model_wants(tmp_path):
    path = tmp_path / "tone.wav"
    write_wav(path, _tone(1.0, sr=44_100), 44_100)
    audio, sr = read_audio(path, target_sr=16_000)
    assert (sr, audio.size) == (16_000, 16_000)


def test_stereo_is_mixed_down_to_mono():
    stereo = np.stack([np.ones(10), -np.ones(10)], axis=1)
    assert to_mono(stereo).shape == (10,)
    assert np.allclose(to_mono(stereo), 0.0)


def test_resample_is_a_no_op_at_the_same_rate():
    audio = _tone(0.1)
    assert resample(audio, 16_000, 16_000) is not None
    assert np.array_equal(resample(audio, 16_000, 16_000), audio)


def test_an_unsupported_format_says_what_to_do(tmp_path):
    path = tmp_path / "clip.mp3"
    path.write_bytes(b"not really an mp3")
    with pytest.raises(Exception, match="soundfile|ffmpeg"):
        read_audio(path)


# -- framing ----------------------------------------------------------------


def test_frames_cover_the_whole_clip_and_pad_the_last_one():
    chunks = frames(np.ones(2500, np.float32), 1000)
    assert len(chunks) == 3  # not "the first window is the answer"
    assert all(c.size == 1000 for c in chunks)
    assert chunks[-1][500:].sum() == 0.0  # padded tail


def test_frames_of_silence_still_produce_one_window():
    assert len(frames(np.zeros(0, np.float32), 800)) == 1


# -- classification ---------------------------------------------------------


class _FakeBackend:
    """A sound model: 4 classes, window of 1000 samples."""

    def __init__(self, per_window):
        self.input_shape = (1, 1, 1, 1000)
        self.per_window = list(per_window)
        self.calls = 0

    def infer(self, feed):
        assert feed.shape == (1, 1, 1, 1000)
        raw = self.per_window[min(self.calls, len(self.per_window) - 1)]
        self.calls += 1
        return {"output": np.array([raw], np.float32)}


def test_window_length_comes_from_the_model():
    assert window_length(_FakeBackend([[0, 0, 0, 0]])) == 1000


def test_sound_classification_answers_with_a_class_name():
    backend = _FakeBackend([[0.0, 8.0, 0.0, 0.0]])
    r = SoundClassifier(names={0: "rain", 1: "dog", 2: "siren", 3: "speech"}).run(
        backend, _tone(0.05), 16_000
    )
    assert isinstance(r, Results)
    assert r.summary().startswith("dog ")
    assert r.probs is not None and r.probs.top1 == 1


def test_every_window_is_classified_not_just_the_first():
    backend = _FakeBackend([[8.0, 0, 0, 0], [0, 8.0, 0, 0], [0, 8.0, 0, 0]])
    SoundClassifier(names={i: str(i) for i in range(4)}).run(
        backend, np.ones(2500, np.float32), 16_000
    )
    assert backend.calls >= 3


def test_an_audio_result_carries_its_waveform_and_saves_as_wav(tmp_path):
    r = SoundClassifier(names={0: "dog"}).run(_FakeBackend([[5.0]]), _tone(0.1), 16_000)
    assert r.orig_img.ndim == 3  # the waveform image, so plot()/save() work
    out = r.save(tmp_path / "clip.wav")
    assert out.exists()
    again, sr = read_audio(out)
    assert sr == 16_000 and again.size == 1600


# -- denoising --------------------------------------------------------------


class _FakeDenoiser:
    """A model with one audio input and one recurrent state input."""

    inputs = [("input", (1, 512), "f32"), ("inp_state_000", (1, 4), "f32")]

    def __init__(self):
        self.states_seen = []

    def infer(self, feed):
        self.states_seen.append(feed["inp_state_000"].copy())
        return {
            "output": feed["input"] * 0.5,
            "out_state_000": feed["inp_state_000"] + 1.0,
        }


def test_denoiser_streams_state_between_chunks_and_keeps_the_length():
    model = _FakeDenoiser()
    audio = _tone(0.1)  # 1600 samples -> 4 chunks of 512
    r = Denoiser().run(model, audio, 16_000)
    assert r.audio is not None
    signal, sr = r.audio
    assert (signal.size, sr) == (audio.size, 16_000)
    assert np.allclose(signal, audio * 0.5, atol=1e-6)
    # state carried forward rather than reset each chunk
    assert [float(s.max()) for s in model.states_seen] == [0.0, 1.0, 2.0, 3.0]
    assert "denoised" in r.summary()


def test_waveform_renders_an_image():
    img = waveform(_tone(0.5), 16_000, width=200, height=80)
    assert img.shape == (80, 200, 3)
    assert img.max() > 0
