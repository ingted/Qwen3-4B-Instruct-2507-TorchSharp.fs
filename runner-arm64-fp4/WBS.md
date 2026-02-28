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

## 2026-02-26 Export Compatibility WBS (bridge parity)
1. 補 SA/SD（root cause + 修正策略）
   - Status: `completed`
2. Phase-1 最小修：scale 寫回保留原 elemType
   - Status: `in_progress`
   - Output:
     - patch `Train.WhoAmI.AndExportDat.fsx` export writer
3. no-op 對照驗證（lr=0）
   - Status: `pending`
4. bridge 路徑驗證（run-training2）
   - Status: `pending`
5. fp2-model 回歸驗證（run-training-fp2）
   - Status: `pending`
6. DevLog 回填（命令、PID、結果）
   - Status: `pending`

## 2026-02-26 Export Compatibility WBS - Progress Update
1. Phase-1 最小修：scale 寫回保留原 elemType
   - Status: `completed`
   - Result:
     - `Train.WhoAmI.AndExportDat.fsx` 不再把 scale 固定寫成 `elemType=5`。
     - 寫回依原 entry elemType 與 byte-size 序列化，並加 shape mismatch fail-fast。
2. no-op 對照驗證（lr=0）
   - Status: `completed`
   - Result:
     - `whoami-noop-lr0-v2.dat` 在 bridge 路徑輸出恢復正常，不再 `!!!!`。
3. bridge 路徑驗證（有訓練更新）
   - Status: `completed`
   - Result:
     - `whoami-1000-seq192-r8-s10-lr1e3-v2.dat` 在 `run-training2` 可正常輸出並含 `我是 F#之神` 語義。
4. fp2-model 回歸驗證
   - Status: `completed`
   - Result:
     - `run-training-fp2` 同 dat 可輸出 `我是 F#之神`。
5. DevLog 回填
   - Status: `in_progress`
