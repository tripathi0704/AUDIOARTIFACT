"""
ai_copilot.py
-------------
Purpose: Next-generation AI Forensic Intelligence & Copilot powered by Google Gemini API.
Features:
  1. Explainable AI (XAI) Forensic Reasoning (Acoustic & Vocoder breakdown in plain language)
  2. Contextual & Semantic Scam Threat Assessment (NLP + Speech Semantics)
  3. Interactive Forensic Copilot ("Chat with Your Audio" + Legal Affidavit Generator)
"""

import os
import json
from typing import Optional, Dict, Any, List

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GENAI_AVAILABLE = False

FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-2.5-flash"]
DEFAULT_MODEL = FALLBACK_MODELS[0]


def _generate_with_fallback(client, contents, config=None):
    """
    Attempts content generation using candidate models sequentially.
    Handles upstream Google model deprecations and version transitions automatically.
    """
    last_err = None
    for model_name in FALLBACK_MODELS:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            return res, model_name
        except Exception as e:
            err_str = str(e)
            if "404" in err_str or "NOT_FOUND" in err_str or "not available" in err_str.lower() or "not found" in err_str.lower():
                last_err = e
                continue
            raise e
    raise last_err or RuntimeError("No compatible Gemini model found.")


def is_genai_installed() -> bool:
    """Check if google-genai library is installed and importable."""
    return GENAI_AVAILABLE


def _load_env_fallback():
    """Fallback loader for .env file if python-dotenv is not active."""
    if not os.environ.get("GEMINI_API_KEY"):
        env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                os.environ["GEMINI_API_KEY"] = val
                                break
            except Exception:
                pass


def get_gemini_client(api_key: Optional[str] = None):
    """
    Initializes and returns a Google GenAI client.
    Resolves API key from parameter, environment variable, or .env file.
    """
    if not GENAI_AVAILABLE:
        return None, "The 'google-genai' library is not installed. Run 'pip install google-genai'."

    _load_env_fallback()
    resolved_key = (api_key or "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()
    if not resolved_key:
        try:
            import base64
            # Built-in cloud default fallback so online users never get prompted
            resolved_key = base64.b64decode("QVEuQWI4Uk42TDNVVVVwRDVMNm01eGZTalp6bUhuZHhGOHpubGdrYTJDLWNpSkcwenFGWWc=").decode("utf-8")
        except Exception:
            pass

    if not resolved_key:
        return None, "Gemini API Key is missing. Please provide an API key in settings or set the GEMINI_API_KEY environment variable."

    try:
        client = genai.Client(api_key=resolved_key)
        return client, None
    except Exception as e:
        return None, f"Failed to initialize Gemini Client: {e}"


def _prepare_audio_part(audio_bytes: Optional[bytes], filename: str = "audio.wav"):
    """Creates a types.Part inline audio object if audio bytes are available and under 20MB."""
    if not audio_bytes or types is None:
        return None

    # Limit inline payload size to ~20MB for fast cloud transit
    if len(audio_bytes) > 20 * 1024 * 1024:
        return None

    ext = os.path.splitext(filename)[1].lower()
    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/m4a",
    }
    mime_type = mime_map.get(ext, "audio/wav")

    try:
        return types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    except Exception:
        return None


def generate_forensic_explanation(
    audio_bytes: Optional[bytes],
    filename: str,
    analysis: Dict[str, Any],
    language: str = "English",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates an expert Explainable AI (XAI) forensic brief combining WavLM probability
    metrics and acoustic features (spectral rolloff, pitch inflection, vocoder cutoff).
    """
    client, err = get_gemini_client(api_key)
    if err:
        return {"success": False, "error": err}

    verdict = analysis.get("verdict", "Unknown")
    fake_ratio = analysis.get("fake_ratio", 0.0)
    fake_seconds = analysis.get("fake_seconds", 0.0)
    total_duration = analysis.get("total_duration", 0.0)
    highest_risk = analysis.get("highest_risk_segment", {})
    forensic_signals = analysis.get("forensic_signals", {})
    rolloff = forensic_signals.get("spectral_rolloff_hz", "N/A")
    zcr = forensic_signals.get("mean_zcr", "N/A")
    pitch_range = forensic_signals.get("vocal_pitch_range_semitones", "N/A")

    # Format suspect segments
    segments = analysis.get("segments", [])
    suspect_segments = [s for s in segments if s.get("label_code") == 1]
    suspect_summary = ", ".join([f"{s.get('start')}s-{s.get('end')}s ({s.get('confidence')}%)" for s in suspect_segments[:5]])
    if not suspect_summary:
        suspect_summary = "None detected (Clean audio)"

    system_prompt = (
        "You are a Senior Digital Audio Forensic Scientist specializing in voice biometric authentication, "
        "synthetic speech detection, and neural vocoder analysis (WavLM, HiFi-GAN, ElevenLabs, VITS). "
        "Your task is to provide an authoritative, clear, and objective Explainable AI (XAI) forensic evaluation "
        "based on the acoustic scan data and audio provided. Avoid robotic jargon without explanation."
    )

    user_prompt = f"""
FORENSIC SCAN DATA FOR RECORDING: '{filename}'
- Final Forensic Verdict: {verdict}
- Synthetic / Glitch Ratio: {fake_ratio}% ({fake_seconds}s out of {total_duration}s)
- Peak Risk Window: Segment #{highest_risk.get('segment', 'N/A')} at {highest_risk.get('start', 'N/A')}s - {highest_risk.get('end', 'N/A')}s (Confidence: {highest_risk.get('confidence', 'N/A')}%)
- Top Flagged Timeline Windows: {suspect_summary}
- Spectral Rolloff (Vocoder Cutoff Indicator): {rolloff} Hz
- Zero-Crossing Rate (ZCR): {zcr}
- Vocal Pitch Inflection Range: {pitch_range} semitones

INSTRUCTIONS:
1. Provide an 'Executive Forensic Summary' explaining whether the audio displays signatures of AI voice cloning, splicing, or natural human recording.
2. Provide 'Acoustic Anomaly Analysis': Explain what the spectral rolloff and pitch indicators mean in this context (e.g., neural vocoder cutoff frequencies, pitch micro-tremors, synthetic phase artifacts).
3. Provide 'Plain Language Verdict for Investigators/Judiciary': A simple 2-3 sentence conclusion that a non-technical judge, lawyer, or citizen can immediately understand.
4. Output in clean Markdown formatting with clear section headers and bullet points.
5. Language preference: {language} (If Hindi / Hinglish requested, explain in clear, professional Hindi/Hinglish).
"""

    contents = []
    audio_part = _prepare_audio_part(audio_bytes, filename)
    if audio_part is not None:
        contents.append(audio_part)
    contents.append(user_prompt)

    try:
        response, used_model = _generate_with_fallback(
            client=client,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
            ),
        )
        return {
            "success": True,
            "explanation": response.text,
            "model_used": used_model,
        }
    except Exception as e:
        return {"success": False, "error": f"Gemini API generation failed: {e}"}


def assess_scam_threat(
    audio_bytes: Optional[bytes],
    filename: str,
    analysis: Dict[str, Any],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyzes spoken content for social engineering, extortion, financial scam intent,
    and voice-cloning impersonation patterns.
    """
    client, err = get_gemini_client(api_key)
    if err:
        return {"success": False, "error": err}

    verdict = analysis.get("verdict", "Unknown")
    fake_ratio = analysis.get("fake_ratio", 0.0)

    system_prompt = (
        "You are an Elite Cybercrime Investigator and Fraud Intelligence Analyst. "
        "Analyze the spoken audio recording and forensic results to evaluate the presence of social engineering, "
        "financial extortion, CEO fraud, family distress scams, OTP phishing, or malicious impersonation."
    )

    user_prompt = f"""
AUDIO FILE: '{filename}'
FORENSIC DEEPFAKE SCAN STATUS:
- Verdict: {verdict}
- Synthetic Ratio: {fake_ratio}%

TASK:
1. Transcribe the spoken audio text accurately.
2. Assess Scam / Threat Intent:
   - Is there urgency, fear, or coercion?
   - Is there a request for money, bank transfer, OTP, password, or secrecy?
   - Does this match known AI voice-clone scam templates (e.g., fake kidnap, virtual arrest, bank manager impersonation)?
3. Assign a Threat Severity Level: [CLEAN / LOW / MODERATE / HIGH / CRITICAL].
4. Recommended Immediate Protective Actions: Step-by-step guidance for the listener.

Format your output in clean Markdown with:
- **Spoken Audio Transcript**
- **Threat Assessment & Intent Analysis**
- **Severity Rating Badge** (e.g., 🔴 CRITICAL THREAT or 🟢 LOW RISK)
- **Key Red Flags Identified**
- **Actionable Defense Checklist**
"""

    contents = []
    audio_part = _prepare_audio_part(audio_bytes, filename)
    if audio_part is not None:
        contents.append(audio_part)
    contents.append(user_prompt)

    try:
        response, used_model = _generate_with_fallback(
            client=client,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
            ),
        )
        return {
            "success": True,
            "assessment": response.text,
            "model_used": used_model,
        }
    except Exception as e:
        return {"success": False, "error": f"Scam threat assessment failed: {e}"}


def chat_with_audio_copilot(
    audio_bytes: Optional[bytes],
    filename: str,
    user_query: str,
    analysis: Dict[str, Any],
    chat_history: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Handles interactive dialogue between forensic investigator and the AI Copilot
    with context of the audio recording and WavLM timeline analysis.
    """
    client, err = get_gemini_client(api_key)
    if err:
        return {"success": False, "error": err}

    verdict = analysis.get("verdict", "Unknown")
    fake_ratio = analysis.get("fake_ratio", 0.0)
    fake_seconds = analysis.get("fake_seconds", 0.0)
    total_duration = analysis.get("total_duration", 0.0)
    highest_risk = analysis.get("highest_risk_segment", {})
    hashes = analysis.get("file_hashes", {})

    context_prompt = f"""
You are the AudioArtifact Forensic AI Copilot.
You have access to the complete forensic analysis of the audio recording '{filename}':
- File SHA-256: {hashes.get('sha256', 'N/A')}
- Forensic Verdict: {verdict}
- Total Duration: {total_duration}s
- Synthetic Audio Detected: {fake_seconds}s ({fake_ratio}%)
- Peak Anomaly Window: Segment #{highest_risk.get('segment')} at {highest_risk.get('start')}s - {highest_risk.get('end')}s (Confidence: {highest_risk.get('confidence')}%)

USER INQUIRY:
{user_query}

INSTRUCTIONS:
- Answer the user's inquiry directly, accurately, and professionally.
- If asked to draft a legal affidavit or police cyber-cell complaint, generate a formal, court-ready document incorporating the SHA-256 hash, timestamps, and findings.
- Keep responses well-structured with Markdown.
"""

    contents = []
    audio_part = _prepare_audio_part(audio_bytes, filename)
    if audio_part is not None:
        contents.append(audio_part)

    # Append brief chat history if provided
    if chat_history:
        for msg in chat_history[-6:]:
            role_prefix = "User: " if msg.get("role") == "user" else "Assistant: "
            contents.append(role_prefix + msg.get("content", ""))

    contents.append(context_prompt)

    try:
        response, used_model = _generate_with_fallback(
            client=client,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.3,
            ),
        )
        return {
            "success": True,
            "response": response.text,
            "model_used": used_model,
        }
    except Exception as e:
        return {"success": False, "error": f"Copilot chat failed: {e}"}
