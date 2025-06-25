# MSME Assist

This application provides a Streamlit interface for interacting with the MSME assistant bot.

## Text-to-Speech

Responses from the assistant are now synthesized using Google Cloud Text-to-Speech. To enable this feature you must provide a Google service account JSON file and set the environment variable `GOOGLE_APPLICATION_CREDENTIALS` to its path (see `.env.example`).

The language passed to the TTS service is determined from the user's query:

- `English` -> `en-US`
- `Hindi` or `Hinglish` -> `hi-IN`

The resulting audio is automatically played in the chat after each assistant message.
