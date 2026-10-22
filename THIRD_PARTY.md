# Sources and attribution

Application code and original scripts are MIT licensed. No model weights or real-person recordings are included in this repository.

| Component | Primary source | Treatment |
|---|---|---|
| Qwen3-TTS 1.7B CustomVoice | https://github.com/QwenLM/Qwen3-TTS and https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | Apache-2.0 model/software. Downloaded separately; generated demo voices use public presets. No training or voice-cloning contribution is claimed. |
| Qwen3-ASR 0.6B | https://github.com/QwenLM/Qwen3-ASR and https://huggingface.co/Qwen/Qwen3-ASR-0.6B | Apache-2.0 model/software. Independent inference with empty context; reference text is not supplied. |
| ModelScope distribution | https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice and https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B | Official Qwen organization snapshots. Revisions and downloaded file SHA256 are preserved in models.lock.json and reports/model_provenance.json. |
| FFmpeg / imageio-ffmpeg | https://github.com/imageio/imageio-ffmpeg and https://ffmpeg.org/ | imageio-ffmpeg is BSD. Platform FFmpeg binaries have their own build/license terms and are installed separately, never vendored here. |
| Original scripts | examples/spaceport.json, clockwork.json, probes.json | Fictional original dialogue; no existing game/story characters or dialogue copied. |
| Demo audio | examples/saved/ | Synthesized locally from the original scripts using Qwen public speaker presets, then arranged/optionally time-scaled by this project. Not human recordings or independent human annotations. |

Source-code assistance: AI coding tools were used. Public results are actual local inference artifacts; synthetic CPU test doubles are explicitly labeled and excluded from the production pilot.
