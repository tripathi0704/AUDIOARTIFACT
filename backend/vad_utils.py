"""
vad_utils.py
------------
Voice Activity Detection (VAD) module for AudioArtifact.
Uses Silero VAD (via PyTorch) with a fallback to energy-based silence removal.
Filters non-speech silence before segment slicing and feature extraction.
"""

import warnings
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
    except Exception as e:
        print(f"Warning: Could not load Silero VAD ({e}). Falling back to energy-based VAD.")
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
