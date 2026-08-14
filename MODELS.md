# MLLM Model Shortlist

Target: models that fit practical experimentation on a 40 GB H100 MIG. Start with compact BF16 models; use quantization for larger variants only after establishing baselines.

## Priority

1. **Qwen3-VL-4B-Instruct** — primary model; dynamic resolution and a deep vision encoder make it suitable for attention, pruning, distillation, and latency experiments.
2. **Gemma 4 E2B IT** — compact, current architecture and a strong second implementation target.
3. **Nanonets-OCR2-3B** — document/OCR-specialized comparison.
4. **MiniCPM-V 4.6** — high-resolution model with a deep vision encoder.
5. **Molmo2-4B** — open research-oriented architecture for generalization tests.
6. **InternVL3/3.5 4B–8B** — useful alternative vision architecture.
7. **Phi-4 Multimodal** — compact multimodal and audio-capable comparison.
8. **Kimi-VL-A3B** — sparse/MoE comparison after dense-model baselines.

## Additional models

- Qwen2.5-VL 7B and Qwen3-VL 8B
- Gemma 3 4B and Gemma 4 E4B/12B
- PaliGemma 2 3B
- SmolVLM2 2.2B
- PaddleOCR-VL 1.6 and dots.ocr
- Llama 3.2 Vision 11B and Pixtral 12B

Avoid 27B/32B models initially: BF16 training or teacher/student experiments will not fit comfortably in 40 GB, while quantization would confound optimization results.
