# Uncomment when running on Windows.
# from phonemizer.backend.espeak.wrapper import EspeakWrapper
# EspeakWrapper.set_library('C:\Program Files\eSpeak NG\libespeak-ng.dll')

from cached_path import cached_path
from styletts2_fork import tts

LJSpeech_TTS_CHECKPOINT_URL = "https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/epoch_2nd_00100.pth"
LJSpeech_TTS_CONFIG_URL = "https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/config.yml?download=true"


# No paths provided means default checkpoints/configs will be downloaded/cached.
my_tts = tts.StyleTTS2(
    model_checkpoint_path=cached_path(LJSpeech_TTS_CHECKPOINT_URL),
    config_path=cached_path(LJSpeech_TTS_CONFIG_URL),
)

SAMPLE_TEXT = """
StyleTTS 2 is a text-to-speech model that leverages style diffusion and adversarial training with large speech language models to achieve human-level text-to-speech synthesis.
"""

# Optionally create/write an output WAV file.
out = my_tts.inference(SAMPLE_TEXT, output_wav_file="test.wav")
