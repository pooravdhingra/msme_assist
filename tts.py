import base64
import streamlit as st
from google.cloud import texttospeech


def synthesize(text: str, language: str) -> bytes:
    """Generate speech audio for the given text and language."""
    language_code = {
        "English": "en-US",
        "Hindi": "hi-IN",
        "Hinglish": "hi-IN",
    }.get(language, "en-US")

    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=input_text,
        voice=voice,
        audio_config=audio_config,
    )
    return response.audio_content


def autoplay(audio_bytes: bytes) -> None:
    """Autoplay MP3 audio in Streamlit."""
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    audio_html = (
        f'<audio autoplay="true" style="display:none">'
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
        '</audio>'
    )
    st.markdown(audio_html, unsafe_allow_html=True)
