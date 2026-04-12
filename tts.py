from kokoro import KPipeline
import sounddevice as sd
import torch
# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
# 🇮🇳 'h' => Hindi hi
# 🇮🇹 'i' => Italian it
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇧🇷 'p' => Brazilian Portuguese pt-br
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
pipeline = KPipeline(lang_code='a') # <= make sure lang_code matches voice, reference above.

sd.default.samplerate = 24000
sd.default.channels = 1

def tts(text):
    audio = next(pipeline(text, voice="af_heart"))[2]
    sd.play(audio, blocking=True)

if __name__ == "__main__":
    tts("Hello, how are you doing today?")