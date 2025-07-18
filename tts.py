import base64
import json
import os
import streamlit as st
from google.cloud import texttospeech
from google.oauth2 import service_account
from typing import Optional


def _create_client() -> texttospeech.TextToSpeechClient:
    """Create a TextToSpeechClient from environment credentials."""
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        info = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(info)
        return texttospeech.TextToSpeechClient(credentials=credentials)
    return texttospeech.TextToSpeechClient()


def synthesize(text: str, language: str) -> bytes:
    """Generate speech audio for the given text and language."""
    language_code = {
        "English": "en-US",
        "Hindi": "hi-IN",
        "Hinglish": "hi-IN",
    }.get(language, "en-US")

    client = _create_client()
    
    # Check if the language is Hindi and apply SSML for pronunciation correction
    if language_code == "hi-IN":
        # Note: The phoneme tag for "hai" (है) as '<phoneme alphabet="ipa" ph="ɦɛː">hai</phoneme>'
        # is problematic because the input `text` is in Devanagari script, not Romanized "hai".
        # This replace operation will not work as intended for Devanagari "है".
        # We will remove this specific `replace` line.
        # SSML can still be used to wrap the entire text.
        
        # Wrap the entire text in <speak> to ensure SSML parsing
        # The input 'text' from msme_bot.py for Hindi is already in Devanagari.
        ssml_text = f"<speak>{text}</speak>" #
        input_text = texttospeech.SynthesisInput(ssml=ssml_text)
        
        # --- MODIFICATION START ---
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
            # Specify a WaveNet voice for Hindi.
            # 'hi-IN-Wavenet-C' is a common male Hindi WaveNet voice.
            # You might also try 'hi-IN-Wavenet-B' for a female voice,
            # or consult Google Cloud TTS documentation for other options.
            name="hi-IN-Wavenet-C" 
        )
        # --- MODIFICATION END ---
    else:
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=0.9,  # Slightly slower rate to improve clarity
    )
    response = client.synthesize_speech(
        input=input_text,
        voice=voice,
        audio_config=audio_config,
    )
    return response.audio_content


def autoplay(audio_bytes: bytes) -> None:
    """Autoplay MP3 audio in Streamlit."""
    audio_player(audio_bytes, autoplay=True)


def audio_player(
    audio_bytes: Optional[bytes],
    autoplay: bool = False,
    placeholder: Optional[st.delta_generator.DeltaGenerator] = None,
    enabled: bool = True,
) -> None:
    """Render an HTML audio player that can be enabled/disabled."""

    b64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else ""
    autoplay_attr = "autoplay" if autoplay and enabled else ""
    disabled_style = "pointer-events:none;opacity:0.5;" if not enabled else ""
    source_html = (
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
        if audio_bytes and enabled
        else ""
    )
    pause_script = (
        "<script>var a=document.getElementById('assistant-audio');"
        "if(a){a.pause();}</script>"
        if not enabled
        else ""
    )

    styled_audio_html = f"""
    <div class="sticky-audio">
        <audio id="assistant-audio" {autoplay_attr} controls
            controlsList="nodownload noplaybackrate"
            style="width: 100%; outline: none; border-radius: 8px; background-color: #FFF; {disabled_style}">
            {source_html}
            Your browser does not support the audio element.
        </audio>
    </div>
    {pause_script}
    """

    target = placeholder if placeholder is not None else st
    target.markdown(styled_audio_html, unsafe_allow_html=True)
