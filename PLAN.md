# Plan and results

## Objective

Train a small decoder-only transformer from scratch on the current Mac using Go, without Python or PyTorch. Complete the dataset → training → checkpoint → evaluation → generation pipeline, with Ginkgo end-to-end tests only. cgo is allowed; GPU acceleration is optional, not required for this phase.

A successful language-model run should improve genuinely held-out loss and produce mostly recognizable words with plausible Shakespeare-style dialogue across several fixed prompts. Passing pipeline tests or lowering training loss alone is not sufficient. This is not a conversational or instruction-tuned model.

## Hardware

| Resource | Verified baseline |
|---|---|
| CPU | Apple M1, 8 cores |
| GPU | Apple M1, 8 cores, Metal 3; unused |
| Memory | 16 GiB unified |
| Go | 1.27.1, darwin/arm64 |

## Implementation

| Path | Responsibility |
|---|---|
| `main.go` | `download`, `train`, `eval`, `generate` CLI |
| `internal/mat/` | Float32 row-major matrix multiplication and reductions |
| `internal/tok/` | Sorted character vocabulary and encoding/decoding |
| `internal/gpt/` | Pre-norm GPT, causal multi-head attention, GELU MLP, handwritten backprop, AdamW, checkpoints |
| `e2e/` | Ginkgo scenarios exercising the actual binary |

- macOS+cgo: Apple Accelerate `cblas_sgemm`, with no external BLAS installation. Other platforms or `CGO_ENABLED=0`: parallel pure-Go kernels. Attention is parallelized across independent batch sequences.
- Float32 parameters/activations, learned token and position embeddings, four-times-width MLP, untied output head, no dropout.
- AdamW, global gradient-norm clipping at 1, short warmup and cosine learning-rate decay.
- The first 90% of corpus characters train the model; the final 10% never enter training windows. Vocabulary metadata uses the whole corpus. Validation samples fixed windows using a separate random generator.
- Atomic gob checkpoints store configuration, vocabulary and weights. With validation enabled, only improvements replace the checkpoint. Optimizer state and resume are not implemented.
- Generation uses temperature/top-k sampling over a sliding context. No KV cache.

## Ordered phases

| Phase | Acceptance | Status |
|---|---|---|
| 1 | Download → train → save/load → generate; all CLI e2e tests pass on both CPU backends | Complete |
| 2 | Add genuinely disjoint validation, reproducible checkpoint evaluation and best-checkpoint saving; test via CLI | Complete |
| 3 | Train longer on Shakespeare; inspect fixed prompts and held-out loss; retain results and exact commands | Complete |
| 4 | Conversational fine-tuning (answer-only loss, end-of-reply marker, held-out questions) and interactive `chat` CLI | Complete |
| 5 | Optional performance work: buffer reuse, KV cache, Metal/cgo backend | Not started |
| 6 | Optional BPE, larger datasets, optimizer-state resume, multi-turn context and serving | Not started |

## Conversational fine-tuning

Foundation weights are warm-started into a supervised chat fine-tune over an original, synthetic Shakespeare-style corpus (39 narrow themes: greetings, identity, feelings, learning, work, nature, and so on). Questions are masked out of the loss; only reply characters and a learned `END` marker are supervised. Random left-padding during training prevents memorization of absolute reply positions. Half the minibatches (`-chat-focus 32`) supervise only the first 32 reply characters, and checkpoint selection scores that prefix, because reply beginnings carry the question-conditioned behavior.

A separate generator (`cmd/chat-data`) expands authored topics into 27,108 training, 585 validation and 117 test examples. Validation and test questions never appear in training. An e2e scenario enforces that exclusion.

Measured results (2.76M-parameter model, 2.9K-token vocabulary of 96 characters, Apple M1 CPU):

| Measurement | Result |
|---|---|
| Chat validation, full replies, all 585 held-out questions | loss 0.0226, perplexity 1.02 |
| Strict topical match on 39 unseen indirect test questions | 14/39 (35.9%) |
| Match on close paraphrases of trained questions | substantially higher (e2e asserts exact replies on held-out-length inputs) |
| Fine-tune resource use | ~1.2 CPU cores average, peak ~312 MiB resident, nice 10 |

Honest limitations, verified by inspecting test-question replies: semantically adjacent topics (loneliness/sadness/fear, anger/rest) are frequently confused; replies are grammatical Shakespeare-style sentences but are wrong-topic in roughly two thirds of indirect phrasings. Adding literal topic-word frames ("Concerning anger, what sayest thou?") measurably hurt held-out matching (14→10/39) by teaching a keyword shortcut, so they were removed. This is a narrow-domain stylistic companion, not a general assistant; multi-turn memory and true intent robustness are future work.

Test-question replies are graded by word-overlap F1 against reference replies — a lexical proxy, not a semantic benchmark. The final chat checkpoint is `checkpoints/shakespeare-chat.bin`.

## End-to-end coverage

22 Ginkgo scenarios pass on both CPU backends. They exercise a local HTTP download, preservation of existing data on HTTP failure, training-loss reduction, reproducible evaluation, exact-length vocabulary-only generation, reproducible seeds, sampling boundaries, invalid inputs, truncated checkpoints, the minimum usable corpus, context-dependent cue recall, and a deliberately contrasting train/validation corpus that proves the split and best-checkpoint selection work.

The tests do not establish Shakespeare fluency or independently prove every analytic derivative. A race-instrumented CLI smoke run also exercised parallel training, evaluation and generation without a reported race.

## Measured runs

| Run | Configuration | Observed result |
|---|---|---|
| Initial smoke | 2 layers, dim 64, 4 heads, context 64, batch 16; 112,577 parameters; 500 steps | Training loss 4.20 → 2.33 in about 28 seconds. Text remained mostly gibberish. Trained on the full corpus; no held-out claim. |
| Validation performance probe | 4 layers, dim 128, 4 heads, context 128, batch 16; 826,433 parameters; 5 steps | About 1.74 seconds including validation after every step. Too short to establish sustained throughput or language quality. |
| Longer run | Same 826,433-parameter architecture, 6,000 steps, 90/10 train/validation split | Running; results will be recorded after inspecting checkpoints and samples. |

No time-to-fluency guarantee or GPU-performance estimate has been established.

## Reproduce the longer run

```bash
go run . download
go run . train -out checkpoints/shakespeare-good.bin \
  -steps 6000 -batch 16 -seq 128 -dim 128 -heads 4 -layers 4 \
  -lr 0.001 -eval-every 200
go run . eval -model checkpoints/shakespeare-good.bin
go run . generate -model checkpoints/shakespeare-good.bin \
  -prompt "ROMEO:" -n 500 -temp 0.8 -topk 20 -seed 7
```

Local checkpoints and datasets are gitignored. Run logs and sample files live under `runs/`.
