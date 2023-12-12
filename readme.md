# Anki Assistant

I love learning new things, but the process of actually acquiring and maintaining knowledge isn't very fun. I've loved using [Anki](https://apps.ankiweb.net/) to help me organize what I'm learing, and make sure I don't forget things. This makes it easier to remember things I learn without having to come up with my own spaced repetition plan. I hate staring at a screen for so long though, so I've often wished I could do my flashcards without having to look at a screen all the time. I finally decided to make it happen. I focused a lot on reducing latency to make it feel as streamlined as possible.

## Tech Stack
- Text-To-Spe)ech: [ElevenLabs](https://elevenlabs.io/)/[OpenAI](https://platform.openai.com/docs/guides/text-to-speech)
- Speech-To-Text: [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) (I used the `tiny.en` model)
- Language Model: [ChatGPT](https://platform.openai.com/docs/guides/text-generation) (I used `gpt-3.5-turbo-1106`)
- Others: [PyWebView](https://github.com/r0x0r/pywebview), [Silero VAD](https://github.com/snakers4/silero-vad)
