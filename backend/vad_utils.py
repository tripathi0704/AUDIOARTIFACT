"""
vad_utils.py
------------
Voice Activity Detection (VAD) module for AudioArtifact.
Uses Silero VAD (via PyTorch) with a fallback to energy-based silence removal.
Filters non-speech silence before segment slicing and feature extraction.
"""

import warnings
import sys
import types

# Windows 11 Smart App Control & cross-platform safe numba bypass for librosa
try:
    from numba import _dispatcher
except Exception:
    import importlib.machinery
    _m = types.ModuleType("numba")
    _m.__spec__ = importlib.machinery.ModuleSpec("numba", None)
    _m.jit = lambda *a, **kw: (lambda f: f) if a and callable(a[0]) else (lambda f: f)
    _m.njit = _m.jit
    _m.vectorize = _m.jit
    _m.guvectorize = _m.jit
    _m.stencil = _m.jit
    _m.prange = range
    sys.modules["numba"] = _m
    _m_core = types.ModuleType("numba.core")
    _m_core.__spec__ = importlib.machinery.ModuleSpec("numba.core", None)
    sys.modules["numba.core"] = _m_core
    sys.modules["numba.core.decorators"] = _m

# Prevent torchaudio DLL loading failures on Windows (Smart App Control / C++ extension issues)
# torchaudio is not used by AudioArtifact (we use soundfile & librosa)
if "torchaudio" not in sys.modules:
    import importlib.machinery
    _m_ta = types.ModuleType("torchaudio")
    _m_ta.__version__ = "2.14.0"
    _m_ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    sys.modules["torchaudio"] = _m_ta
    _m_ta_lib = types.ModuleType("torchaudio.lib")
    _m_ta_lib.__spec__ = importlib.machinery.ModuleSpec("torchaudio.lib", None)
    sys.modules["torchaudio.lib"] = _m_ta_lib
    _m_ta_c = types.ModuleType("torchaudio.lib._torchaudio")
    _m_ta_c.__spec__ = importlib.machinery.ModuleSpec("torchaudio.lib._torchaudio", None)
    sys.modules["torchaudio.lib._torchaudio"] = _m_ta_c

import numpy as np
import librosa

_vad_model = None


def get_vad_model():
    """Lazily load and cache the Silero VAD model."""
    global _vad_model
    if _vad_model is not None:
        return _vad_model
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from silero_vad import load_silero_vad
            _vad_model = load_silero_vad()
    except Exception:
        _vad_model = False
    return _vad_model


def filter_speech_vad(y: np.ndarray, sr: int = 16000, threshold: float = 0.5):
    """
    Applies Voice Activity Detection to audio array y.
    
    Returns:
        y_speech (np.ndarray): Audio signal containing only active speech.
        timestamps (list of dict): List of speech timestamps [{'start': float, 'end': float}, ...] in seconds.
    """
    model = get_vad_model()
    duration = len(y) / sr

    if model and model is not False:
        try:
            import torch
            from silero_vad import get_speech_timestamps
            wav = torch.from_numpy(y.astype(np.float32))
            speech_ts = get_speech_timestamps(wav, model, sampling_rate=sr, threshold=threshold)
            if speech_ts:
                chunks = [y[ts['start']:ts['end']] for ts in speech_ts]
                y_speech = np.concatenate(chunks)
                timestamps = [{'start': round(float(ts['start']) / sr, 3), 'end': round(float(ts['end']) / sr, 3)} for ts in speech_ts]
                return y_speech, timestamps
        except Exception as e:
            # Fallback on any runtime failure
            pass

    # Fallback: Energy-based VAD with librosa.effects.split
    try:
        intervals = librosa.effects.split(y, top_db=30)
        if len(intervals) > 0:
            chunks = [y[start:end] for start, end in intervals]
            y_speech = np.concatenate(chunks)
            timestamps = [{'start': round(float(start) / sr, 3), 'end': round(float(end) / sr, 3)} for start, end in intervals]
            return y_speech, timestamps
    except Exception:
        pass

    # Default fallback: return original audio
    return y, [{'start': 0.0, 'end': round(duration, 3)}]
