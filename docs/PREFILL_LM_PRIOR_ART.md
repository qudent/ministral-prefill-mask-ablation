# Prefill/Prefix-LM Prior Art (Concise)

## Question
Has "prompt bidirectional, response causal" attention been tried before, and where is it used?

## Short Answer
Yes. The mask pattern is established prior art:
- **UniLM (2019)**: unified masks including seq2seq-style masking where source/prefix tokens are bidirectional and target tokens are causal.
- **UL2 (2022)**: includes **S-denoising** (PrefixLM-like objective) as one objective in a mixture.
- **U-PaLM / UL2R (2022)**: continued training from PaLM checkpoints with UL2-style mixtures (including S-style objective).
- **MPT PrefixLM code path**: explicit decoder-only support for prefix bidirectional mask + causal generation.

## Where It Was Useful
- Seq2seq-style generation tasks (summarization, question generation, QA, dialog) in UniLM-style setups.
- Mixed-objective continued pretraining (UL2/U-PaLM) to improve broad capability balance.
- Decoder-only engineering path for prefix conditioning (MPT PrefixLM conversion/training path).

## What "Mixture of Objectives" Means
- Mixture applies at **training time**: batches are sampled from different denoisers/objectives (e.g., R/S/X in UL2).
- Inference is **not** a runtime mixture of objectives.
- At inference, you choose one serving behavior for a model family:
  - encoder-decoder: standard encoder-bidirectional + decoder-causal generation
  - decoder-only PrefixLM: prompt/prefix bidirectional prefill + causal decode

## Why This Is Not the Default Everywhere
- Objective tradeoffs: UL2 ablations show that too much S-objective share can hurt some general benchmarks; mixture balance matters.
- Serving/kernel path complexity: custom non-causal masks can miss highly-optimized pure-causal fast paths.
- Pipeline complexity: decoder-only PrefixLM usually requires explicit mask plumbing (e.g., bidirectional prefix mask fields), not a drop-in default for all stacks.

## Caching/Inference Caveat
- KV cache assumptions are easiest under strict causal attention.
- PrefixLM is workable when prefix is fixed, but dynamic prefix updates and generalized masking introduce implementation complexity and potential performance penalties.

## Relevance To This Repo
- Current Stage 3B objective ("prompt bidirectional, response causal") aligns with known PrefixLM-style training.
- This is not a novel masking idea, but testing adaptation and recovery behavior on this specific model/checkpoint remains a valid experiment.

## Sources
- UniLM (paper): https://arxiv.org/abs/1905.03197
- UniLMv2: https://arxiv.org/abs/2002.12804
- s2s-ft / seq2seq fine-tuning line: https://arxiv.org/abs/2110.13640
- UL2: https://arxiv.org/abs/2205.05131
- U-PaLM / UL2R: https://arxiv.org/abs/2210.11399
- MPT modeling (prefix_lm logic): https://huggingface.co/mosaicml/mpt-7b/blob/f3b1a9a05ff0d2f3e02cea68ed3c3738ffcee4d7/modeling_mpt.py
- MPT PrefixLM converter: https://huggingface.co/mosaicml/mpt-7b/blob/main/hf_prefixlm_converter.py
- MPT config default (`prefix_lm: false`): https://huggingface.co/mosaicml/mpt-7b-8k/blob/main/config.json
- HF cache explanation: https://huggingface.co/docs/transformers/main/cache_explanation
- PyTorch SDPA docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- vLLM attention backend notes: https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/flex_attention.html
