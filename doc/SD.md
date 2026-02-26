# SD - fp2 Safe Diagnostic Design (2026-02-24)

## 1. Problem Statement
- `run-training-fp2.fsx` is used to validate inference via training-style block graph.
- Historical failures show two distinct risks:
  1. Runtime instability / OOM (especially fallback quantize path).
  2. Semantic collapse (`!!!!`) caused by repeated token id `0`.

## 2. Design Goal
- Keep diagnostics reproducible without multi-turn stress.
- Fail fast before host-level instability escalates.
- Isolate STE path behavior from non-STE block-graph behavior.

## 3. Diagnostic Paths
1. Path A (`A.infer`): baseline inference (`InferenceBridge.forwardModel`).
2. Path B (`B.fp2_ste`): fp2 training-style path (`Qwen3Model.forward`, uses `linearSte`).
3. Path C (`C.noste_graph`): block graph control (`Qwen3Core.forwardBlockNoCache` + `InferenceBridge.linearQ4`).

Interpretation rule:
- If A ~= C but B diverges, bug is likely in STE path (`linearSte/steWeight` semantics).
- If A diverges from both B/C, check shared components (tokenizer, weight load, config).

## 4. Safety Controls
1. Single-turn only (no chained dialogue turns).
2. `TS_Q4_STE_USE_NATIVE_QUANTIZE=1` required for fp2-safe scripts.
3. Fail-fast when first output contains `!!!!`.
4. `--max-tokens` safety cap (`<= 8`) in safe script.
5. External watchdog + timeout wrapping all fp2 tests.

## 5. Implemented Scripts
1. `run-training-fp2-safe.fsx`
   - Single-turn STE test with guardrails.
2. `run-training-fp2-noste.fsx`
   - Single-turn no-STE block-graph control.
3. `compare-first-token-fp2.fsx`
   - One-prompt top-k logits/token-id diagnostic for A/B/C.

## 6. Runtime Prerequisite
- CUDA must initialize successfully in the same container/session before fp2 diagnostics:
  - `torch.cuda.is_available() == True`
  - `torch.cuda.device_count() > 0`
- If CUDA initialization fails (`Error 304`), abort fp2 experiments and recover runtime first.

## 7. Latest Validation Snapshot (2026-02-24)
1. `run-training-fp2-safe.fsx`:
   - first turn output: `!!!!`
   - generated ids: `[0;0;0;0]`
2. `run-training-fp2-noste.fsx`:
   - first turn output: `Hi! 😊`
3. `compare-first-token-fp2.fsx`:
   - `A.infer`: finite hidden/logits, reasonable top10.
   - `B.fp2_ste`: NaN hidden/logits, top10 dominated by low-id punctuation.
   - `C.noste_graph`: finite hidden/logits, top10 close to `A.infer`.

## 8. Design Decision
1. Keep fp2 diagnostics single-turn and guard-railed until STE path is fixed.
2. Prioritize STE internals (`linearSte`, `steWeight`, quantize/dequantize semantics) over tokenizer/block wiring.

## 9. Fix Applied
1. Root-cause hypothesis confirmed:
   - NVFP4 `scale` from dat uses `elemType=101` (byte-encoded FP8-like scale), not plain uint8-linear scale.
2. Implementation:
   - Decode `uint8` scale to FP8(E4M3FN) float domain before `dequantizePacked` in `Qwen3Model.materializeMasterWeight`.
3. Post-fix acceptance:
   - `B.fp2_ste` hidden/logits must be finite.
   - First-token top-k of `B.fp2_ste` should be semantically close to `A.infer` and `C.noste_graph`.

## 10. Regression Status (main runner)
1. Main script `run-training-fp2.fsx` now uses the same single-turn safety contract as `run-training-fp2-safe.fsx`.
2. Re-validated with prompt `hi`:
   - output is normal text (`Hello! 👋`)
   - no first-turn `!!!!` collapse
3. Platform caveat:
   - `nvidia-smi` on this GB10 setup returns `Memory-Usage: Not Supported`.
   - The design therefore relies on script-level fail-fast and timeout-based guardrails instead of numeric VRAM polling.

## 11. Guarded Execution Contract
1. Added wrapper `run-training-fp2-guarded.sh` for operational safety.
2. Watch source:
   - Primary: `nvidia-smi --query-compute-apps=pid,used_memory`
   - Fallback: parse `nvidia-smi` process table (`GPU Memory` column).
3. Enforcement:
   - kill target PID when memory exceeds threshold (`110GB`) for continuous window (`10s`).
4. Failure mode handling:
   - if process memory is unobservable (stays `0MiB`), wrapper emits explicit warning that threshold enforcement is currently unavailable.

## 12. No-Env Startup Contract
1. `run-training-fp2.fsx` auto-enforces native STE quantize.
2. Behavior:
   - if `TS_Q4_STE_USE_NATIVE_QUANTIZE` is not enabled, script sets it to `1` at startup.
3. Rationale:
   - preserve OOM safety without requiring caller-side env var setup.
4. Default safety alignment:
   - default `--max-tokens` is now `8` so zero-arg invocation stays within cap.

## 13. Multi-turn Execution Contract
1. `run-training-fp2.fsx` now supports multi-turn execution in-process.
2. Added options:
   - `--turns`
   - `--followup-prompt`
3. Turn behavior:
   - turn 1 uses `--prompt`
   - turn 2..N use `--followup-prompt`
4. Safety behavior retained:
   - any turn containing `!!!!` triggers fail-fast.
5. Token cap policy:
   - removed hard `MaxTokens <= 8` fail gate; caller controls `--max-tokens` explicitly.

## 14. Zero-Arg Default Profile
1. `run-training-fp2.fsx` now defaults to a multi-turn smoke profile:
   - turns=3
   - prompt=hi
   - followup-prompt=continue.
   - max-tokens=8
2. This enables no-arg execution for routine regression checks.

## 15. Default Prompt Parity
1. No-arg default prompt in `run-training-fp2.fsx` restored to:
   - `Write one short sentence about UFO and you.`
2. Rationale: keep parity with `run-training2.fsx` comparison baseline.

## 16. One-shot Repro Default Contract
1. No-arg defaults are reset to prioritize first valid output:
   - `turns=1`
   - `max-tokens=4`
2. Multi-turn remains an opt-in mode via explicit CLI args.

## 17. Guard Runner Contract (F# only)
1. Use `run-script-with-guard.fsx` as the guard launcher.
2. Mandatory observability:
   - print `guard_pid`
   - print child `dotnet_pid`
3. Preflight checks:
   - guard parameters must be positive
   - target script path must exist

## 18. Guard Default Policy (Crash Prevention)
1. Default guard policy is now aggressive:
   - limit=110GB
   - over-secs=0 (kill on first observed breach)
   - poll-secs=0.5
2. Breach evaluation uses both:
   - child PID memory
   - total GPU process memory
3. Fixed immediate-mode branch bug to avoid false unconditional kill.

## 19. KVC Design For fp2 Runner
1. Goal:
   - replace decode-time full replay with cache-based incremental decode in `run-training-fp2.fsx`.
2. Strategy:
   - prefill once on full rendered prompt.
   - decode token-by-token using persistent `Qwen3Core.ModelKvCache`.
3. Control:
   - `--use-kvc` switch (default `true`) to keep replay fallback for A/B checks.
4. Peak-risk notes:
   - even with KVC, process still holds two model families if not trimmed (`InferenceBridge` + `Qwen3Model`).
   - must reduce duplicate residency before expecting stable `max-tokens>=6` under strict VRAM limits.

## 20. Memory Mitigation Design
1. Keep only required `InferenceBridge` components for fp2 sampling:
   - tokenizer
   - embed tokens
   - final norm
   - lm head
2. Dispose unused `InferenceBridge` per-layer weights immediately after init in KVC mode.
3. Use `torch.inference_mode()` during generation path.

## 21. Guard Policy Update
1. Default guard baseline tightened to:
   - limit=108GB
   - over-secs=0
   - poll-secs=0.5
2. This is now the required default for fp2 bring-up experiments.

## 22. KVC Backend Strategy
1. Introduce `--kvc-backend` in `run-training-fp2.fsx`:
   - `bridge` (default)
   - `fp2-model`
2. Runtime policy:
   - default to `bridge` for reliable output and low VRAM.
   - keep `fp2-model` as diagnostic parity path.
3. Model residency policy:
   - when backend is `bridge`, skip `Qwen3Model.create` to avoid duplicate heavy model residency.
4. Acceptance target:
   - with guard `(108GB, over=0, poll=0.5)`, `max-tokens=8/10/16` should complete without watchdog kill in default mode.
5. Current status:
   - `bridge`: meets acceptance target.
   - `fp2-model`: still exceeds 108GB at `max-tokens=6` (optimization backlog remains).

## 23. Training-Path-Only Runtime Contract (2026-02-25)
1. `run-training-fp2.fsx` policy:
   - default backend = `fp2-model`
   - `bridge` backend is rejected (hard fail)
2. Rationale:
   - enforce training graph as primary runtime for pre-training bring-up.
   - remove accidental drift back to inference-only path.

## 24. Sampling Session Split Design
1. Added `InferenceBridge.initSamplingOnly(...)`.
2. Loaded assets:
   - tokenizer
   - embed tokens
   - final norm
   - lm head (`Q4Linear`)
3. Not loaded:
   - per-layer q/k/v/o/mlp q4 weights.
4. Purpose:
   - eliminate startup double-residency peak when `Qwen3Model.create` is also loaded.

## 25. STE Eval-Cache Design
1. Added `Nvfp4Training` eval-only cache for dequantized STE weights.
2. Activation condition:
   - `torch.is_inference_mode_enabled() || not(torch.is_grad_enabled())`
   - and `TS_Q4_STE_CACHE_EVAL_WEIGHT=1` (default enabled by runner).
3. Invalidation:
   - explicit API `Nvfp4Training.clearEvalWeightCache()` called in runner `finally`.
4. Safety:
   - training/grad-enabled flow still uses non-cached `steWeight` path.

## 26. Native Quantize Strictness
1. `run-training-fp2.fsx` now checks:
   - `NativeInterop.hasLibTorchFp4Quantize() = true`
2. Failure behavior:
   - if missing, fail immediately.
3. Purpose:
   - prevent silent fallback quantize path in fp2 bring-up.

## 27. Current Acceptance Snapshot
1. Guarded command:
   - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script run-training-fp2.fsx`
2. Result:
   - completed without kill.
   - peak total process memory observed around `44GB`.
   - output is coherent full sentence (no `!!!!` collapse).

## 28. Persistent Multi-turn KVC Design (fp2-model)
1. State model:
   - `ModelKvCache` is allocated once per script run (not per turn).
   - `contextTokens` tracks token count expected to be materialized in cache.
2. Turn protocol:
   - encode only current turn prefix:
     - `<|im_start|>user ... <|im_end|>\n<|im_start|>assistant\n`
   - prefill cache with this turn prefix.
   - decode token-by-token.
3. Cache consistency rule:
   - every accepted generated token is immediately forwarded once (`forwardWithKvCache [token]`) so cache always includes the latest output token.
   - at turn end, append and prefill `<|im_end|>\n` to close assistant message in cache.
4. Benefits:
   - avoids per-turn full history replay.
   - supports real multi-turn continuation with persistent KV state.
5. Observability:
   - debug logs print `kvc seqLen` and `contextTokens` each turn.

## 29. Persistent KVC Acceptance
1. Guarded run (`turns=3`, `max-tokens=8`) must satisfy:
   - no watchdog kill.
   - `seqLen` strictly increases across turns.
   - turn-2/3 generation latency significantly below turn-1 baseline.
2. Semantic continuation check:
   - with followup prompt requesting continuation, turn-2 output should continue prior clause rather than reset topic.

## 30. Full-NVFP4 1-step Training Design (2026-02-25)
1. Script:
   - `run-train-step-full-nvfp4.fsx`
2. Diagnostics design:
   - phase sampling adds:
     - `pid_mem_mib`
     - `total_gpu_mem_mib`
     - `cuda_used_mib` / `cuda_total_mib` (from `cudaMemGetInfo`)
     - `proc_rss_mib`
   - tensor-byte summary groups by:
     - kind
     - device
     - dtype
3. Allocator-stat constraint:
   - this TorchSharp build does not expose public `memory_allocated/reserved`.
   - replacement telemetry is explicitly documented in script logs.
4. Load-time memory reduction:
   - optional `--dispose-session-after-load` (default true)
   - optional `--compact-after-model-load` (default true)
   - compact operation:
     - `cuda.synchronize`
     - `Nvfp4Training.clearEvalWeightCache()`
     - `NativeInterop.tryEmptyNvfp4Cache()`
     - managed GC cycle
5. Optimizer-step redesign:
   - `adamwStepNvfp4Packed` now supports row-chunk streaming via `--step-chunk-rows`.
   - algorithm:
     - dequantize `w/m/v` in row chunks
     - compute AdamW update per chunk
     - write updated chunk to parameter view
     - repack chunk back to NVFP4 and copy into destination packed tensors
   - objective:
     - avoid full-parameter simultaneous materialization during step.
6. Stability policy:
   - default `step-chunk-rows=32` (conservative, memory-first).
   - `64` is not default due observed OOM risk in stressed runs.
7. Guard contract for train-step validation:
   - always run through `run-script-with-guard.fsx`
   - baseline:
     - `--gpu-limit-gb 108`
     - `--gpu-over-secs 0`
     - `--gpu-poll-secs 0.5`

## 31. 2026-02-25 設計更新：訓練 VRAM JSON + 1-step 文本實訓
1. `TrainingConfig` 擴充：
   - 新增 `TrainStepVramReportPath: string option`。
2. `Trainer` VRAM profiling 設計：
   - 保留 console phase log（`ProfileTrainStepVram=true`）。
   - 另支援 JSON 落檔（`TrainStepVramReportPath=Some path`）。
   - 採樣欄位：`timestamp/epoch/stepInEpoch/globalStep/phase/pidMem/totalMem`。
3. 採樣觸發時機：
   - `batch_ready`
   - `zero_grad_done`
   - `backward_done`
   - `optimizer_step_done`
4. JSON 報告寫出策略：
   - 在 `Trainer.run` finally 階段寫檔，避免中途例外導致報告遺失。
5. 1-step 實訓腳本設計（專案 API 直連）：
   - `scripts/Train.OneStep.fsx`。
   - 從 `TrainData/train-inputs.txt` 取樣本。
   - 用 `InferenceBridge.initSamplingOnly` + tokenizer + `buildTokenEmbeddings` 建立 `[1,T,H]` 輸入與 shift target。
   - 用 `Qwen3Model.forward -> Trainer.scalarLoss -> backward -> Nvfp4Optimizer.step` 完成單步訓練。
6. Safe default 對齊：
   - `OptimizerStepChunkRows` 預設維持 `32`。

## 32. 2026-02-25 GQA 驗證性設計補強
1. 在 head 展開函式新增配置防呆：
   - `numHeads > 0`
   - `numKvHeads > 0`
   - `numHeads % numKvHeads = 0`
2. 目的：
   - 避免錯誤模型配置在展開 KV head 時被隱性吞掉，導致後續 attention 結果失真。
3. 實作點：
   - `Qwen3Core.expandKvHeads`
   - `InferenceBridge.expandKvHeads`

## 33. 2026-02-26 WhoAmI 合併訓練腳本設計落地
1. `scripts/Train.WhoAmI.AndExportDat.fsx` 已支援單腳本流程：
   - 載入 `--input-dat`
   - 讀 `--train-data`（TSV: `prompt<TAB>target`）
   - 訓練（CE/scalar 可切換）
   - 匯出 `--output-dat`
   - 立即做 self-test 生成驗證
2. 可調核心參數：
   - `--seq-len`：訓練窗口上限
   - `--step-chunk-rows`：optimizer streaming 粒度
   - `--train-last-layers`：可訓練層範圍
   - `--lr` / `--steps` / `--loss`
3. guard 實務配置固定：
   - `--gpu-limit-gb 108`
   - `--gpu-over-secs 0`
   - `--gpu-poll-secs 0.5`
4. 本次驗證輸出：
   - `artifacts/whoami-1000-seq192-r8-s10-lr1e3.dat`
   - 測試 prompt `你是誰` 可輸出含 `我是 F# 之神` 的語義內容。

## 34. 2026-02-26 Disambiguation 設計補充（`你是誰` vs `我是誰`）
1. 驗證流程標準化
   - 使用單輪、非互動、固定後端：
     - `run-training-fp2.fsx --turns 1 --ifInteractive false --stop-here true --kvc-backend fp2-model`
   - 固定檢查三個 prompt：
     - `你是誰`
     - `我是誰`
     - `談談UFO`
2. 新增訓練資料集
   - `TrainData/stageB-curriculum-v5.tsv`（從 StageA 拉 whoami）
   - `TrainData/stageC-disambiguate-v1.tsv`（baseline whoami 檔上做語意拆分）
   - `TrainData/stageD-disambiguate-v2.tsv`（加重 `我是誰` 反例）
3. 輸出檔
   - `artifacts/stageB-whoami-nudge-v5.dat`（未達 whoami 對齊）
   - `artifacts/stageC-disambiguate-v1-s4.dat`（whoami/UFO 達成）
   - `artifacts/stageD-disambiguate-v2.dat`（whoami/UFO 達成，`我是誰` 仍未拆分）
4. 實作判定
   - StageB 類（從 `stageA-mixed.dat` 出發）在可接受步數內對齊不足。
   - StageC/StageD（從 whoami baseline 出發）可穩定保住 whoami 行為與 UFO 一般能力。
   - 但 `我是誰` 誤判仍存在，表示目前僅靠 CE + projection 更新不足以完成語意邊界拆分。
