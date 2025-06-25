# MSME Assist

This application provides a Streamlit interface for interacting with the MSME assistant bot.

## Text-to-Speech

Responses from the assistant are now synthesized using Google Cloud Text-to-Speech. You can either set `GOOGLE_APPLICATION_CREDENTIALS` to the path of a service account JSON file or provide the JSON directly in the environment variable `GOOGLE_APPLICATION_CREDENTIALS_JSON` (useful with Streamlit secrets). See `.env.example` for both options.

The language passed to the TTS service is determined from the user's query:

- `English` -> `en-US`
- `Hindi` or `Hinglish` -> `hi-IN`

The resulting audio is automatically played in the chat after each assistant message.
