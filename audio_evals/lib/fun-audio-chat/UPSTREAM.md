# Fun-Audio-Chat Runtime Source

The `funaudiochat/` package in this directory is an unmodified copy of the
minimal text-inference model/processor implementation from:

- Repository: `https://github.com/FunAudioLLM/Fun-Audio-Chat`
- Commit: `8ba984b64b4880918db2807e08d795475481200c`
- Retrieved: 2026-08-13
- License: Apache-2.0 (the upstream copyright and license headers are retained)

Only the custom Transformers registration package is vendored. `main.py` is
the VoxMatrix JSON-line worker and does not include the upstream S2S or
CosyVoice runtime.

Vendored file SHA-256 values:

```text
788156df8ff10ce14faa9c65acae45ea68d1497f531446a7fac28a2f0fa4b1b6  __init__.py
d229f507b0e02f15f6ac84e0142cb80902742c02d69a0e3a523aeb2e91ea60db  configuration_funaudiochat.py
3dcd44295cce0f423247ff72cab9d9bf24267e05b1e5741781887dba4d527e41  modeling_funaudiochat.py
3ae8045217486a714ec905ab02dc21ecd42a2cc793972c0c47130e09457962a0  processing_funaudiochat.py
2e3c74cb7a9550025fc608af702fa00ce83583a6d03ab59c9c6cdf184415e900  register.py
```
