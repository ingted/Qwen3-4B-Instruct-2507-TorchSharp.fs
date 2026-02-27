# WBS - fp2 KVC Bring-up (2026-02-25)

## Scope
1. Stabilize `run-training-fp2.fsx` output path with guard-first policy.
2. Integrate KVC decode path in fp2 runner.
3. Keep machine-safe experimentation under strict VRAM budget.

## Guard Track
1. `run-script-with-guard.fsx` default policy
   - Status: `completed`
   - Target:
     - `gpu-limit-gb=108`
     - `gpu-over-secs=0`
     - `gpu-poll-secs=0.5`
2. Immediate kill correctness (`over-secs=0` branch)
   - Status: `completed`
   - Note: fixed unconditional-kill bug.
3. Breach observability (`pid_mem/total_mem/limit` in kill logs)
   - Status: `completed`

## KVC Implementation Track
1. Add `--use-kvc` switch to `run-training-fp2.fsx`
   - Status: `completed`
2. Implement KVC generation path
   - Status: `completed`
   - Detail:
     - prefill once with full prompt
     - decode one token with cache reuse
3. Keep replay fallback path for A/B
   - Status: `completed`
4. Switch generation context to `torch.inference_mode()`
   - Status: `completed`
5. Add KVC backend split (`bridge` / `fp2-model`)
   - Status: `completed`
6. Default backend set to `bridge` for stability
   - Status: `completed`

## Memory Reduction Track
1. Dispose unused `InferenceBridge` layer weights in KVC mode
   - Status: `completed`
2. Validate no double-dispose in finalization path
   - Status: `completed`
3. Investigate remaining decode peak >108GB for `max-tokens>=6`
   - Status: `in_progress` (only for `fp2-model` backend)
   - Hypotheses:
     - STE linear path temporary buffers still dominate
     - allocator/cache growth pattern during decode remains high

## Test Matrix (all via guard, no timeout)
1. `max-tokens=4`
   - Status: `completed`
   - Result: output produced (`I’ve never seen`)
2. `max-tokens=6`
   - Status: `completed`
   - Result: killed by guard at ~113GB total
3. `max-tokens=8`
   - Status: `completed`
   - Result: killed by guard at ~112.8GB total
4. `max-tokens=10`
   - Status: `completed` (bridge backend)
   - Result: output completed without guard kill
5. `max-tokens=16`
   - Status: `completed` (bridge backend)
   - Result: output completed without guard kill
6. `max-tokens=24`
   - Status: `completed` (bridge backend)
   - Result: output completed without guard kill
7. `fp2-model max-tokens=6` (post-cleanup tweaks)
   - Status: `completed`
   - Result: still killed around 112GB total

## Next Work Items
1. Add per-step memory instrumentation around `fp2-model` decode loop.
2. Compare `bridge` vs `fp2-model` peak footprint with identical prompt and token budget.
3. Continue reducing `fp2-model` peak until `max-tokens=6` passes under 108GB.

## 2026-02-25 Training-Path-Only WBS
1. Add sampling-only session init for fp2 backend
   - Status: `completed`
   - Output:
     - `InferenceBridge.initSamplingOnly` implemented and used by `run-training-fp2.fsx`.
2. Add eval STE weight cache in Q4 extension
   - Status: `completed`
   - Output:
     - `Nvfp4Training.clearEvalWeightCache`
     - eval cache path in `linearSte`.
3. Enforce native quantize availability in runner
   - Status: `completed`
   - Output:
     - hard fail when `NVFP4_quantize` export unavailable.
4. Switch runner default to fp2-model and disable bridge
   - Status: `completed`
   - Output:
     - default `--kvc-backend=fp2-model`
     - bridge backend rejected by design in this script.
5. Validate default no-arg path can produce full sentence
   - Status: `completed`
   - Output:
     - default `--max-tokens=24`
     - no-arg run emits coherent sentence.

## 2026-02-25 Persistent Multi-turn KVC WBS
1. Implement shared fp2 cache/context state
   - Status: `completed`
   - Output:
     - one `ModelKvCache` for all turns
     - one `contextTokens` tracker
2. Replace turn generation with delta-prefill protocol
   - Status: `completed`
   - Output:
     - per-turn prefill uses current user-turn tokens only
3. Ensure generated-token materialization into cache
   - Status: `completed`
   - Output:
     - each accepted token is forwarded once to cache
4. Close assistant turn in cache
   - Status: `completed`
   - Output:
     - append/prefill `<|im_end|>\n` after each turn
5. Multi-turn guarded validation (`turns=3`)
   - Status: `completed`
   - Output:
     - seqLen grows `27 -> 47 -> 67`
     - turn-2/3 latency significantly reduced vs turn-1

## 2026-02-25 Full-NVFP4 1-step Training WBS
1. Add train-step diagnostics (`pid/total + cudaMemGetInfo + tensor bytes`)
   - Status: `completed`
   - Output:
     - phase report fields expanded
     - model/state byte breakdown printed
2. Reduce `model_loaded` footprint
   - Status: `completed`
   - Output:
     - `dispose-session-after-load`
     - `compact-after-model-load`
     - observed `model_loaded` compaction (`~40065MiB -> ~38303MiB`)
3. Evaluate `backward_done` footprint with guarded runs
   - Status: `completed`
   - Output:
     - observed `backward_done` around `~52024MiB` in current setup
4. Implement optimizer step chunked/streaming update
   - Status: `completed`
   - Output:
     - row-chunk streaming path in `adamwStepNvfp4Packed`
     - `--step-chunk-rows` control
5. Choose safe default chunk size
   - Status: `completed`
   - Output:
     - default `step-chunk-rows=32`
     - `64` retained as optional but not default due OOM risk
6. Stabilize under guard=108GB (seq=1 one-step)
   - Status: `in_progress`
   - Current:
     - success cases exist with chunked path
     - also observed stressed-run instability / exit 137; requires additional reproducibility passes

## 2026-02-25 A~F 實作 WBS（專案主線）
1. Sync runner docs to project docs (`doc/SA.md`, `doc/SD.md`, `doc/DevLog.md`, `doc/WBS.md`)
   - Status: `completed`
2. Add Trainer JSON VRAM report (`TrainStepVramReportPath` + JSON writer)
   - Status: `completed`
3. Keep safe chunk default (`OptimizerStepChunkRows=32`)
   - Status: `completed`
4. Re-evaluate offload defaults for GB10 UM and set conservative default (`false/false/false`)
   - Status: `completed`
5. Prepare training text inputs (`TrainData/train-inputs.txt`, 10 samples)
   - Status: `completed`
6. Add minimal one-step real training script via project API (`scripts/Train.OneStep.fsx`)
   - Status: `completed`
7. Build verification (`dotnet build -c Release`)
   - Status: `completed`

## Open Follow-up
1. Add Trainer-native text batch provider to reuse same data path in multi-step training (currently one-step script handles text path).
8. Guarded e2e validation for `scripts/Train.OneStep.fsx` (`108GB`, immediate kill)
   - Status: `completed`
   - Result:
     - completed one optimizer step
     - produced `loss`
     - wrote `doc/train-step-vram-onestep.json`

## 2026-02-25 WhoAmI DAT Task
1. Add mixed quant scale load support in `Nvfp4State` (`elemType=5`)
   - Status: `completed`
2. Implement `scripts/Train.WhoAmI.AndExportDat.fsx`
   - Status: `completed`
3. Guarded training experiments under 108GB policy
   - Status: `completed`
4. Export trained dat and run self-test from exported dat
   - Status: `completed`
5. Validate exported dat via `run-training-fp2.fsx --weight <dat>`
   - Status: `completed`
6. Reach exact lexical target `我是 F# 之神`
   - Status: `in_progress`
   - Current best: semantic `我是...` output, exact phrase not yet stable-reproducible.

## 2026-02-26 tag:202602270039 WBS（兩階段續跑 + 語意拆分）
1. Stage B v3（30/70 mixed）訓練與匯出
   - Status: `completed`
   - Result:
     - export ok
     - self-test fail（`你是誰` 仍為通義千問）
2. Stage B v5（sequential curriculum）訓練與匯出
   - Status: `completed`
   - Result:
     - export ok
     - self-test fail（`你是誰` 仍為通義千問）
3. Stage C（baseline whoami 檔 + disambiguate s4）
   - Status: `completed`
   - Result:
     - `你是誰` 命中 F# 語義
     - `談談UFO` 正常
     - `我是誰` 仍誤判
4. Stage D（加重 `我是誰` 反例）
   - Status: `completed`
   - Result:
     - `你是誰` 命中 F# 語義
     - `談談UFO` 正常
     - `我是誰` 仍誤判（未改善）
5. 下一步：問句意圖邊界化（非單純 CE 微調）
   - Status: `in_progress`
   - Plan:
     - 增加前置意圖路由/規則或額外意圖 loss
     - 再進行 whoami 對齊，避免 `你是誰/我是誰` 合併為同一回答模式

## 2026-02-26 Full-parameter mixed（from original）WBS
1. 調整 guard 輪詢下限（允許 0.05s）
   - Status: `completed`
   - Output:
     - `run-script-with-guard.fsx` `pollMs` 下限 `100 -> 50`。
2. 建立多樣化混合資料集
   - Status: `completed`
   - Output:
     - `TrainData/fullparam-diverse-mix-v1.tsv`（1000 筆）。
3. 執行全參數訓練（input = 原始 dat）
   - Status: `completed`
   - Output:
     - `artifacts/fullparam-from-original-diverse-v1.dat`
     - 108GB guard 下完成，未觸發 kill。
4. 走 training 路徑驗證（fp2-model）
   - Status: `completed`
   - Result:
     - `你是誰`：仍偏基座身份（未命中 F#）。
     - `談談UFO`：正常（能力保留）。

## 2026-02-26 lm_head 參訓修正 WBS
1. 將 `lm_head` 轉為 trainable parameter
   - Status: `completed`
2. 將 `lm_head` 併入 optimizer `trainParams`
   - Status: `completed`
3. 匯出 dat 時回寫 `lm_head` packed 權重
   - Status: `completed`
4. 重新跑 guarded full-parameter（含 lm_head）驗證
   - Status: `pending`

## 2026-02-26 Safety-first WhoAmI 目標達成 WBS（本輪）
1. 設定保守 guard 基線（避免 tmux/host 被拖掛）
   - Status: `completed`
   - Policy:
     - `gpu-limit-gb=104`
     - `gpu-over-secs=0`
     - `gpu-poll-secs=0.05`
2. 階段一：能力保留底座（from original dat, mixed data）
   - Status: `completed`
   - Train:
     - `loss=ce`
     - `compute-dtype=bfloat16`
     - `seq-len=24`
     - `step-chunk-rows=16`
3. 階段二：whoami 對齊 + disambiguate（from stage1 dat）
   - Status: `completed`
   - Train:
     - `loss=ce`
     - `compute-dtype=bfloat16`
     - `seq-len=24`
     - `step-chunk-rows=16`
4. 驗證（run-training2）
   - Status: `completed`
   - prompts:
     - `你是誰`
     - `談談UFO`
5. 寫回 `doc/DevLog.md`（命令、峰值、結果、失敗原因）
   - Status: `completed`
