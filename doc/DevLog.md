# Developer Log - 2026-02-14

## Issue: Intermittent Timeout at Stage [6] in `run-training2.fsx`

### 1. Troubleshooting
- **Symptoms**: Running `dotnet fsi run-training2.fsx --KVCacheOut false --no-kvc-mode full-replay --timing true` often hangs at stage `[6]`.
- **Observation**: While normal execution takes ~5s, it occasionally exceeds the 40s timeout. This usually happens during consecutive runs.
- **Environment**: DGX Spark (GB10) with 128GB Unified Memory (no HBM).
- **Hypothesis**: 
    - In `full-replay` mode, the prompt grows each turn, requiring larger contiguous memory for activation buffers.
    - Unified Memory fragmentation causes the CUDA allocator or OS to trigger memory compaction/page migration, leading to significant latency spikes.
    - Zombie `dotnet` processes from previous failed runs might be cluttering the system.

### 2. Change Planning
- **Goal**: Ensure clean memory state before the critical stage `[6]` and throughout the session.
- **Actions**:
    - Move `forceCleanUp()` (GC + CUDA Sync + Empty Cache) to the beginning of each `runTurn`.
    - Slightly increase the timeout for stage `[6]` to 48s (below the 50s system lockup threshold) to allow for minor OS-level memory management jitter.

### 3. Dev / Debug
- **Implementation**:
    - Relocated `forceCleanUp` definition above `runTurn`.
    - Inserted `forceCleanUp()` call at the start of `runTurn`.
    - Updated `runTurn (Some "6")` timeout from `40000` to `48000`.
- **Debugging**: Initially encountered `FS0039` error due to function definition order; fixed by moving the function block.

### 4. Test
- **Execution**: Ran a loop of 5 consecutive executions.
- **Result**: Successfully reproduced the hang *without* the fix in the loop. With the fix, stage `[6]` consistently completed in ~4.4s across multiple runs.

### 5. Solution Verificated
- **Status**: Verified. The combination of proactive GC and CUDA cache clearing before each turn prevents the accumulation of memory fragments that trigger the 40s+ latency in the GB10 Unified Memory architecture.
- **Final Logic**: Every turn now starts with a synchronized and defragmented memory state.

---

# Developer Log - 2026-02-24

## Objective
- Stabilize fp2 experiments to avoid host crash.
- Reproduce `!!!!` in single-turn mode only.
- Prepare A/B/C first-token diagnostics for weight/tokenizer alignment analysis.

## Changes Implemented
1. Added `run-training-fp2-safe.fsx`.
   - Based on `run-training-fp2-single.fsx`.
   - Single-turn only.
   - Fail-fast when first output contains `!!!!`.
   - Requires `TS_Q4_STE_USE_NATIVE_QUANTIZE=1`; otherwise exits immediately.
   - Safety cap: `--max-tokens <= 8`.
   - Default log path moved to:
     - `alpha/log/tee-object-chat-session-fp-safe.txt`
     - `alpha/log/tee-object-chat-session-fp-safe.jsonl`
2. Added `run-training-fp2-noste.fsx`.
   - Single-turn only.
   - Uses `Qwen3Core.forwardBlockNoCache` with `InferenceBridge.linearQ4` projections.
   - Purpose: no-STE control path while keeping block-graph style execution.
3. Added `compare-first-token-fp2.fsx`.
   - One-prompt first-token diagnostic.
   - Compares three paths:
     - `A.infer`: `InferenceBridge.forwardModel`
     - `B.fp2_ste`: `Qwen3Model.forward` (training/STE path)
     - `C.noste_graph`: block graph + `linearQ4` (no STE)
   - Prints hidden/logits NaN/Inf health + top10 token ids and decoded token text.

## Key Findings (from successful prior CUDA window)
1. `!!!!` corresponds to repeated token id `0`.
   - Evidence: `QWEN3_FS_DEBUG_TOKENS=1` showed `[0; 0; 0; 0]` in first turn.
   - Tokenizer check:
     - `decode(0) = "!"`
     - `encode("!") = [0]`
2. `debug-fp2-parity.fsx` indicated divergence at layer 0.
   - Path A healthy, Path B goes NaN from layer0 onward.
   - `q/k/v` in Path B had much larger absolute magnitude than Path A.

## Runtime Incident During This Session
1. `run-training-fp2-safe.fsx` attempt failed at CUDA init:
   - `cudaGetDeviceCount Error 304: OS call failed or operation not supported on this OS`
   - `Torch device type CUDA did not initialise on the current machine`
2. Confirmed this was environment-state dependent:
   - `nvidia-smi` initially looked normal.
   - Python check in same container then reported:
     - `torch.cuda.is_available() = False`
     - `torch.cuda.device_count() = 0`
   - Therefore this round’s fp2 runs were blocked by CUDA runtime availability, not script syntax.

## Commands Used (representative)
1. Safe run:
   - `TS_Q4_STE_USE_NATIVE_QUANTIZE=1 QWEN3_FS_DEBUG_TOKENS=1 timeout 120s dotnet fsi run-training-fp2-safe.fsx --prompt "hi" --max-tokens 4 --timing true`
2. CUDA status check:
   - `nvidia-smi`
   - `python3 - <<'PY' ... torch.cuda.is_available()/device_count ... PY`

## Next Action (once CUDA becomes available again)
1. Re-run `run-training-fp2-safe.fsx` (single-turn).
2. Run `run-training-fp2-noste.fsx` with same prompt.
3. Run `compare-first-token-fp2.fsx` and persist top10 diff in log.
4. Prioritize fixes in STE path (`linearSte/steWeight`) if A/C align but B diverges.

## Execution Update - 2026-02-24 (CUDA available window)
### Experiment 1: `run-training-fp2-safe.fsx`
- Command:
  - `TS_Q4_STE_USE_NATIVE_QUANTIZE=1 QWEN3_FS_DEBUG_TOKENS=1 timeout 180s dotnet fsi run-training-fp2-safe.fsx --prompt "hi" --max-tokens 4 --timing true`
- Result:
  - output: `!!!!`
  - generated ids: `[0; 0; 0; 0]`
  - process exited by guard fail-fast (`first output is !!!!`)
  - VRAM peak observed by watchdog: ~93.5GB (under 110GB kill threshold)

### Experiment 2: `run-training-fp2-noste.fsx`
- Initial run failed with missing assembly reference (`TorchSharp.Fun.DGX`).
- Fix:
  - Added `#r "/workspace/TorchSharp.Fun.DGX/TorchSharp.Fun.DGX/bin/Release/net10.0/TorchSharp.Fun.DGX.dll"`.
- Command:
  - `QWEN3_FS_DEBUG_TOKENS=1 timeout 180s dotnet fsi run-training-fp2-noste.fsx --prompt "hi" --max-tokens 4 --timing true`
- Result:
  - generated ids: `[13048; 0; 26525; 232]`
  - output: `Hi! 😊`
  - VRAM peak: ~5.4GB

### Experiment 3: `compare-first-token-fp2.fsx`
- Initial compile fixes:
  1. `torch.topk` argument type fixes (`int` vs `int64`).
  2. Added `TorchSharp.Fun.DGX` reference.
- Command:
  - `TS_Q4_STE_USE_NATIVE_QUANTIZE=1 timeout 240s dotnet fsi compare-first-token-fp2.fsx "hi"`
- Key observations:
  - `A.infer`: hidden/logits finite, top1 token `id=2132 ("It")`.
  - `B.fp2_ste`: hidden/logits contain NaN; top10 all NaN logits with low-id punctuation (`id=0` included).
  - `C.noste_graph`: hidden/logits finite; top10 highly similar to `A.infer`.
- VRAM peak: ~83.8GB (under threshold).

### Conclusion
- Root-cause is now strongly isolated to STE path (`Qwen3Model.forward` + `Nvfp4Training.linearSte`), not tokenizer and not block-graph wiring itself.

### Re-run Confirmation (prompt=`"hi"`, after arg parsing fix)
- `compare-first-token-fp2.fsx "hi"` now correctly uses prompt `hi`.
- A/C top tokens are highly aligned and semantically correct:
  - top1 both are `id=13048 ("Hi")`, followed by `Hello/Hey/HI/...`.
- B remains invalid:
  - hidden/logits contain NaN
  - top10 collapses to low-id punctuation tokens with NaN logits.
- This removes prior ambiguity from incorrect prompt parsing and further confirms STE-specific failure.

## Fix Execution - 2026-02-24 (STE recovery)
### Root-cause evidence added
- Parsed dat entries and confirmed:
  - `*.qdata`: `elemType=0`
  - `*.scale`: `elemType=101` (1-byte encoded scale), shape `[out, in/16]`

### Code fix
- File: `Qwen3-4B-Instruct-2507-TorchSharp.fs/Qwen3Model.fs`
- Change:
  - In `materializeMasterWeight`, when `scale.dtype = uint8`, decode scale bytes as FP8(E4M3FN) to float first, then call `Nvfp4Training.dequantizePacked`.
  - Added LUT-based decode helper (`fp8E4M3FnToFloat32`, `decodeFp8E4M3FnTensor`).

### Validation after fix
1. `run-training-fp2-safe.fsx --prompt "hi"`:
   - generated ids: `[9707; 0; 61804; 233]`
   - output: `Hello! 👋`
   - no `!!!!` fail-fast triggered
2. `compare-first-token-fp2.fsx "hi"`:
   - `A.infer`: finite, top1 `Hi`
   - `B.fp2_ste`: finite (no NaN), top tokens `Hello/Hi/Hey/...`
   - `C.noste_graph`: finite, aligned with `A`

### Decision
- Keep single-turn guardrails for stability, but STE path is no longer in NaN-collapse state for the tested prompt.

## Regression Check - 2026-02-24 (run-training-fp2 main script)
### Command
- `cd /workspace/fsann/alpha/runner-arm64-fp4`
- `TS_Q4_STE_USE_NATIVE_QUANTIZE=1 QWEN3_FS_DEBUG_TOKENS=1 dotnet fsi run-training-fp2.fsx --max-tokens 4 --timing true --check-logits false --prompt "hi"`

### Result
- Script completed in single-turn guard mode.
- generated ids: `[9707; 0; 61804; 233]`
- output: `Hello! 👋`
- no `!!!!` fail-fast triggered.

### Runtime note
- On this host/GPU, `nvidia-smi` reports `Memory-Usage: Not Supported`, so the external `>110GB for 10s` VRAM kill rule cannot be enforced via NVML query in this environment.
- Effective fallback during this run: strict single-turn mode + script fail-fast + command timeout pattern.

## Guard Script - 2026-02-24
### Goal
- Add a reusable launcher that enforces the user's GPU safety rule from `nvidia-smi` process table (`GPU Memory`) when available.

### Implementation
- Added: `run-training-fp2-guarded.sh`
  - Runs `dotnet fsi run-training-fp2.fsx`.
  - Watches the target PID via:
    - `nvidia-smi --query-compute-apps=pid,used_memory`
    - fallback: parse `nvidia-smi` processes table text.
  - If memory is over 110GB for 10s (configurable), sends `TERM` then `KILL`.
  - Defaults:
    - `TS_Q4_STE_USE_NATIVE_QUANTIZE=1`
    - `QWEN3_FS_DEBUG_TOKENS=1`
    - args: `--max-tokens 4 --timing true --check-logits false --prompt hi`

### Validation
- Command:
  - `timeout 150s ./run-training-fp2-guarded.sh --max-tokens 1 --timing false --check-logits false --prompt "hi"`
- Result:
  - training script completed and output remained normal (`Hello`).
  - No watchdog-triggered kill.

### Caveat observed in this runtime
- For this run, process-level GPU memory stayed `0MiB` in both query/fallback paths.
- Added explicit warning in script:
  - if memory remains unobservable for >=15s, report that threshold enforcement is currently unavailable.

## Usability Fix - 2026-02-24 (no-env run)
### Problem
- `dotnet fsi run-training-fp2.fsx` failed early when `TS_Q4_STE_USE_NATIVE_QUANTIZE` was not pre-set.

### Change
- File: `run-training-fp2.fsx`
- Replaced hard fail check with auto-enforcement:
  - if env var is missing/false, script sets `TS_Q4_STE_USE_NATIVE_QUANTIZE=1` at startup and logs an info line.

### Intent
- Keep OOM safety behavior while allowing direct invocation without requiring users to export env vars manually.

## Default Args Fix - 2026-02-24
### Problem
- Running `dotnet fsi run-training-fp2.fsx` without args still failed safety cap because default `--max-tokens` was `20`.

### Change
- File: `run-training-fp2.fsx`
- Updated default `MaxTokens` from `20` to `8` to match fp2-safe cap.
- Also updated `defaultArgs` (`--max-tokens`) from `20` to `8` because no-arg invocation uses `defaultArgs`.

## Multi-turn Enablement - 2026-02-24
### Request
- Remove hard single-turn token cap behavior and support multi-turn scenario similar to `run-training2.fsx`.

### Change
- File: `run-training-fp2.fsx`
1. Removed hard fail gate `if MaxTokens > 8 then fail`.
2. Added args:
   - `--turns` (default: `1`)
   - `--followup-prompt` (default: `continue.`)
3. Runtime now executes turns in loop:
   - turn1 uses `--prompt`
   - turn2..N use `--followup-prompt`
4. Kept guard behavior:
   - any turn output containing `!!!!` triggers immediate fail-fast.

### Smoke test status
- Command attempted:
  - `dotnet fsi run-training-fp2.fsx --turns 2 --prompt "hi" --followup-prompt "continue" --max-tokens 2 --timing false --check-logits false --stop-here false`
- In this execution context, run blocked by CUDA init failure before generation (`Torch device type CUDA did not initialise`), so multi-turn runtime output must be confirmed on stable CUDA session.

## No-Arg Defaults Update - 2026-02-24
- run-training-fp2.fsx default profile adjusted for zero-arg usage:
  - --turns=3
  - --prompt=hi
  - --followup-prompt=continue.
  - --max-tokens=8
- Goal: allow multi-turn regression runs without bash parameters.

## Default Prompt Alignment - 2026-02-24
- Restored no-arg default `--prompt` to: "Write one short sentence about UFO and you." for parity with run-training2.fsx comparisons.

## Repro Profile Reset - 2026-02-24
### Problem
- No-arg multi-turn defaults can push VRAM too high before first valid output is observed.

### Change
- File: `run-training-fp2.fsx`
- Reset zero-arg defaults to one-shot reproducibility profile:
  - `--turns=1`
  - `--max-tokens=4`
  - `--prompt="Write one short sentence about UFO and you."` (kept for parity with `run-training2.fsx`)

### Expected behavior
- `dotnet fsi run-training-fp2.fsx` should prioritize producing one valid first output.
- Multi-turn remains available via explicit `--turns` override.

## Guard Runner Update - 2026-02-25
### Request
- Stop using bash guard wrapper and use F# guard runner only.
- Print dotnet process ID clearly so operator can kill quickly.

### Code review findings (`run-script-with-guard.fsx`)
1. Missing positive-value validation for guard parameters.
2. Missing explicit script existence check.
3. PID visibility can be improved (guard PID + child dotnet PID).

### Fixes
1. Added validation: `--gpu-limit-gb`, `--gpu-over-secs`, `--gpu-poll-secs` must all be `> 0`.
2. Added script path existence check before spawn.
3. Added explicit logs:
   - `[guard] guard_pid=<pid>`
   - `[guard] started dotnet_pid=<pid>`
4. Added compatibility normalization when positional args accidentally include leading `script`.

## Guard Tuning - 2026-02-25 (110GB immediate mode)
### Why 117GB still happened previously
- Previous run used `--gpu-over-secs 10` semantics, so crossing threshold did not kill immediately.
- With 115GB limit, observed 117479MiB was still below 117760MiB limit.

### Changes
- `run-script-with-guard.fsx` now supports fractional `--gpu-poll-secs` / `--gpu-over-secs`.
- Defaults changed to safer profile:
  - `gpu-limit-gb=110`
  - `gpu-over-secs=0` (immediate kill)
  - `gpu-poll-secs=0.5`
- Guard checks both target PID memory and total GPU process memory.
- Kill log now prints `pid_mem/total_mem/limit` for postmortem.

### Critical bug fixed
- `over-secs=0` initially caused unconditional kill due branch condition.
- Fixed to only use sustained-window branch when `over-secs > 0`.

### Verification command
- `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 110 --gpu-over-secs 0 --gpu-poll-secs 0.5 script run-training-fp2.fsx --max-tokens=8 --timing=true --check-logits=false --stop-here=false`
- Result: guard killed process in immediate mode when threshold was crossed.

## Training Path Hardening - 2026-02-25 (fp2-model only)
### Goal
- Stop using bridge inference path in `run-training-fp2.fsx`.
- Make training-path inference stable and reproducible for full pure-NVFP4 training bring-up.

### Root-cause refinement
1. `fp2-model` path still had startup residency pressure because runner first called full `InferenceBridge.init` (loads all layer Q4 objects), then attempted to dispose layers.
2. `linearSte` repeatedly quantize/dequantize weights in eval decode loop; this creates unnecessary temporary pressure and allocator growth.

### Code changes
1. `TorchSharp.Q4.Extension/Nvfp4Training.fs`
   - Added eval-only STE weight cache (enabled by `TS_Q4_STE_CACHE_EVAL_WEIGHT=1`).
   - Added `clearEvalWeightCache()`.
   - Fixed temporary input tensor ownership/dispose in `linearSte` when dtype cast occurs.
2. `TorchSharp.Q4.Extension/Nvfp4Training.fsi`
   - Exported `clearEvalWeightCache`.
3. `Qwen3-4B-Instruct-2507-TorchSharp.fs/InferenceBridge.fs`
   - Added `initSamplingOnly` (loads only tokenizer/embed/final_norm/lm_head).
4. `run-training-fp2.fsx`
   - Default backend switched to `fp2-model`.
   - `bridge` backend is now rejected for this runner (training-path-only policy).
   - Uses `InferenceBridge.initSamplingOnly` for fp2 backend.
   - Enforces `NVFP4_quantize` export availability; fail if missing.
   - Auto-enables:
     - `TS_Q4_STE_USE_NATIVE_QUANTIZE=1`
     - `TS_Q4_STE_CACHE_EVAL_WEIGHT=1`
   - Clears eval cache in `finally`.
   - Default `--max-tokens` raised to `24` for no-arg full-sentence output.

### Build verification
1. `cd /workspace/TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension && dotnet build -c Release`
   - pass
2. `cd /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs && dotnet build -c Release`
   - pass

### Experiments (guarded)
1. Command:
   - `cd /workspace/fsann/alpha/runner-arm64-fp4`
   - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script run-training-fp2.fsx`
2. Result:
   - completed, no watchdog kill.
   - peak `total_gpu_mem` observed around `44GB`.
   - output:
     - `I’ve never seen a UFO, but I’ve always wondered what it would be like to meet one.`
3. Additional:
   - `--max-tokens=24` explicit run also passed under same guard.

### Experiments (direct no guard)
1. Command:
   - `dotnet fsi run-training-fp2.fsx`
2. Result:
   - completed with coherent sentence output.
   - no `!!!!` collapse.

## Persistent Multi-turn KVC - 2026-02-25
### Goal
- Complete multi-turn chat continuation for training path (`fp2-model`) with real KV reuse across turns.

### Implementation
1. File: `run-training-fp2.fsx`
   - Added shared persistent objects:
     - `fp2PersistentCache : ModelKvCache option`
     - `fp2PersistentContextTokens : ResizeArray<int>`
   - Added `generateFromUserMessageWithStopTokensFpKvPersistent`.
2. Protocol change
   - Old:
     - each turn built full rendered prompt and rebuilt/reprefilled cache.
   - New:
     - each turn encodes only current user turn prefix.
     - decode with persistent cache.
     - every accepted token is forwarded into cache immediately.
     - append/prefill `<|im_end|>\n` after each turn to keep template alignment.
3. Added debug observability:
   - `[FP2Debug] kvc seqLen=... contextTokens=...`
   - `[kvc] turn-cache seqLen=... contextTokens=...`

### Guarded multi-turn test (speed/continuation proof)
1. Command:
   - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script run-training-fp2.fsx --turns=3 --max-tokens=8 --timing=true --check-logits=false --stop-here=false`
2. Result:
   - no watchdog kill.
   - seqLen/context grew:
     - turn1: `27`
     - turn2: `47`
     - turn3: `67`
   - generation latency:
     - turn1: ~`4.1s`
     - turn2: ~`0.54s`
     - turn3: ~`0.55s`
3. Interpretation:
   - turn2/3 are reusing KV instead of replaying full history.

### Guarded semantic continuation test
1. Command:
   - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script run-training-fp2.fsx --turns=2 --max-tokens=12 --prompt=\"Write one short sentence about UFO and you.\" --followup-prompt=\"continue the previous sentence in one clause.\" --timing=true --check-logits=false --stop-here=false`
2. Result:
   - turn1: `I’ve never seen a UFO, but I’ve always wondered`
   - turn2: `if I ever do, I’ll know it’s not a`
3. Interpretation:
   - turn2 continues prior clause; multi-turn context carry is effective.

## KVC Workstart - 2026-02-25
### Context and hypothesis
1. `max-tokens=4` can finish, but `6/8/10` frequently cross 110~117GB.
2. Current fp2 generation path still had full-replay behavior before this wave.
3. Additional memory pressure also came from loading two full model families in the same process:
   - `InferenceBridge` full layers
   - `Qwen3Model` full blocks

### SD-driven implementation done
1. `run-training-fp2.fsx` adds `--use-kvc` (default: `true`).
2. Added KV decode path:
   - prefill once with full prompt via `Qwen3Model.forwardWithKvCache`.
   - incremental decode using one token per step with the same cache.
3. Kept replay path for fallback (`--use-kvc false`).
4. Inference mode hardening:
   - generation switched from `torch.no_grad()` to `torch.inference_mode()`.

### Memory pressure reduction done
1. Added selective disposal of `InferenceBridge` per-layer weights right after init in `use-kvc=true` mode.
2. Keep only tokenizer + embedding + final norm + lm head from `InferenceBridge` for sampling.
3. Finalizer path updated to avoid double-dispose and still release kept components.

### Guard policy update
1. Default guard limit changed to `108GB`.
2. Tests run with:
   - `--gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5`
3. All tests executed through `run-script-with-guard.fsx` (no timeout path).

### Experiments (post-KVC start)
1. Command:
   - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script run-training-fp2.fsx --max-tokens=4 --timing=true --check-logits=false --stop-here=false`
   - Result: PASS, output `I’ve never seen`.
2. Command:
   - same but `--max-tokens=8`
   - Result: KILLED by guard, breach log:
     - `total_mem=112791MiB`, `limit=110592MiB`.
3. Command:
   - same but `--max-tokens=6`
   - Result: KILLED by guard, breach log:
     - `total_mem=113269MiB`, `limit=110592MiB`.

### Current read
1. Guard works as intended now (108GB + 0.5s + immediate kill).
2. KVC first pass is integrated, but peak memory is still above 108GB for `max-tokens>=6`.
3. Next action:
   - continue profiling around decode-step peak and temporary buffers in STE linear path.

## KVC Backend Split - 2026-02-25 (stability-first)
### Design decision
1. Add `--kvc-backend` in `run-training-fp2.fsx`:
   - `bridge` (default): use `InferenceBridge` KVC generation.
   - `fp2-model`: use `Qwen3Model.forwardWithKvCache` path.
2. Keep `fp2-model` for parity/debug, but use `bridge` as default to guarantee practical output and safe VRAM.

### Why
1. `fp2-model` KVC path under strict `108GB` still breached at `max-tokens=6/8`.
2. `bridge` KVC path avoids duplicate heavy residency and has much lower decode peak.

### Key implementation notes
1. `run-training-fp2.fsx` now conditionally skips `Qwen3Model.create` when `--use-kvc=true --kvc-backend=bridge`.
2. In bridge mode, generation calls:
   - `InferenceBridge.generateFromRenderedPromptWithStopTokensKvCache ... KvPrefillMode.PromptByPrompt`
3. `check-logits` in bridge mode uses `InferenceBridge.checkLogits`.

### Verification (guard=108GB, over=0, poll=0.5)
1. `--max-tokens=8`:
   - output: `I once saw a UFO—no,`
   - peak sample log: `total_gpu_mem=6053MiB`
   - PASS (no kill)
2. `--max-tokens=10`:
   - output: `I once saw a UFO—no, just a`
   - PASS (no kill)
3. `--max-tokens=16`:
   - output: `I once saw a UFO—no, just a bright light in the sky,`
   - PASS (no kill)
4. `--max-tokens=24`:
   - output: `I once saw a UFO—no, just a bright light in the sky, maybe a plane or a satellite. �`
   - PASS (no kill)

### Additional fp2-model check (after cleanup tweaks)
1. Command:
   - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script run-training-fp2.fsx --use-kvc=true --kvc-backend=fp2-model --max-tokens=6 --timing=true --check-logits=false --stop-here=false`
2. Result:
   - still KILLED at threshold
   - breach log: `total_mem=112051MiB`, `limit=110592MiB`
3. Interpretation:
   - bridge backend is now the only stable default for long-enough outputs under 108GB.
   - fp2-model backend remains a dedicated optimization/parity track.

## Discussion Record - 2026-02-25 (user Q&A summary)
### 1) What is `eval cache` and `clear`?
1. `eval cache`:
   - cache dequantized STE weight for `Nvfp4Training.linearSte` during `inference_mode/no_grad`.
   - avoid repeated `quantize -> dequantize` per token on the same layer weight.
2. `clear`:
   - `Nvfp4Training.clearEvalWeightCache()` explicitly disposes cached tensors.
   - called in runner `finally` to avoid cache residue across runs.

### 2) Is this caused by .NET itself?
1. Conclusion:
   - not a `.NET` root-cause issue.
2. Reason:
   - pressure comes from algorithmic path cost (STE repeated quant/dequant in decode loop).
   - .NET/TorchSharp requires explicit tensor lifecycle control, so disposal mistakes amplify allocator pressure.

### 3) Can current VRAM do prompt-by-prompt KVC?
1. For current fp2 inference-style path (`run-training-fp2.fsx`):
   - yes, prompt-by-prompt/persistent KVC is working under guard.
   - observed peak around ~44GB in guarded runs.
2. For true training (backward + optimizer):
   - not equivalent to inference footprint.
   - feasibility must be validated by dedicated train-step profiling (activation + optimizer states can dominate).

## Full-NVFP4 1-step Training Bring-up - 2026-02-25
### Goal
1. Add real 1-step training execution on training path (not inference path), with:
   - loss/backward/optimizer-step
   - gradient-checkpoint style recompute control
   - optimizer state compression/offload controls
   - guarded VRAM profiling

### New script
1. Added `run-train-step-full-nvfp4.fsx`:
   - loads training-path model graph.
   - executes one real train step (`forward -> loss -> backward -> step`).
   - supports `--grad-ckpt-chunk` for chunked recompute behavior.
   - emits phase VRAM samples and optional JSON report via `--vram-report`.

### Optimizer/state implementation
1. Implemented packed NVFP4 persistent state (`w/m/v`) in script-local optimizer flow:
   - persistent storage as packed `qdata + scale`.
   - unpack/materialize only when needed for update.
2. Added memory control toggles:
   - `--offload-mv-to-cpu`
   - `--offload-w-to-cpu`
   - `--offload-grad-to-cpu`
   - `--step-flush-each-param`
   - `--materialize-from-packed` (default false)
   - `--compute-grad-norm` (default false)

### Guarded experiments
1. All runs used guard launcher (no timeout path):
   - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script run-train-step-full-nvfp4.fsx ...`
2. Main observed phase memory:
   - `model_loaded`: ~40065MiB
   - `state_initialized`: ~40185MiB
   - `backward_done`: ~82602MiB
   - optimizer-step peak: ~110.9GiB to ~117.5GiB (varies by options), often guard-killed.
3. With `--offload-grad-to-cpu=true`, a long run ended with exit code `137` before final report write (likely external/system kill during/near step phase).

### Current conclusion
1. Inference KVC stability does not translate to training-step VRAM safety.
2. Current bottleneck is optimizer-step transient peak (not only steady grad footprint).
3. Under strict 108GB immediate guard, 1-step full-NVFP4 training is not yet stable.

### Next optimization direction
1. Reduce step transients with stricter per-parameter streaming update.
2. Shorten guard reaction window further (poll interval reduction where possible).
3. Keep all future train-step experiments under `run-script-with-guard.fsx` only.

## Full-NVFP4 Train-step Round 2 - 2026-02-25 (diagnostics + chunked step)
### Scope
1. Add requested diagnostics:
   - process memory telemetry per phase.
   - model/state tensor byte breakdown for duplicate/multi-copy detection.
2. Optimize `model_loaded` and `optimizer step` transient memory.
3. Keep all runs under guard launcher.

### Code changes (`run-train-step-full-nvfp4.fsx`)
1. Added phase telemetry fields:
   - `pid_mem_mib`, `total_gpu_mem_mib`, `cuda_used_mib`, `cuda_total_mib`, `proc_rss_mib`.
2. Added `cudaMemGetInfo` bridge via `libcudart` P/Invoke.
3. Added tensor byte summary helper:
   - grouped by `kind/device/dtype`.
4. Added runtime controls:
   - `--step-chunk-rows` (default now `32`)
   - `--dispose-session-after-load` (default `true`)
   - `--compact-after-model-load` (default `true`)
   - `--print-tensor-byte-report` (default `true`)
   - `--stop-after` (phase stop for diagnostics).
5. Optimizer redesign:
   - `adamwStepNvfp4Packed` switched to row-chunk streaming.
   - update no longer requires full-parameter simultaneous dequant/materialization.

### Diagnostic findings
1. Command:
   - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script run-train-step-full-nvfp4.fsx --seq-len=1 --grad-ckpt-chunk=0 --stop-after=model_loaded ...`
2. Key output:
   - `model_loaded`: `pid=40065MiB`, `cuda_used=~52GiB`.
3. With `--stop-after=post_load_compacted`:
   - `model_loaded`: `40065MiB`
   - `post_load_compacted`: `38303MiB`
4. Interpretation:
   - load-time compaction yields ~`1.7GiB` reduction on observed process footprint.

### Model/state byte breakdown
1. Command:
   - `... --stop-after=state_initialized --print-tensor-byte-report=true`
2. Key output:
   - `model.parameters(unique)`: `6930.37 MiB` (`Float16`, `cuda:0`, 396 params)
   - packed state total (`w/m/v`): `5847.50 MiB` (stored on CPU)
   - no duplicate parameter refs (`raw=396, unique=396`)
3. Interpretation:
   - weight payload itself is ~6.9GiB; `model_loaded ~40GiB` mostly includes runtime/allocator/workspace.

### Chunked optimizer experiments
1. `step-chunk-rows=32` (guard 108GB):
   - completed one full step in an earlier run.
   - observed peak stayed under guard (highest sampled total around `~104.6GiB` in that run).
   - produced:
     - `optimizer_step_done`
     - final `[done]` marker
     - VRAM report file written.
2. `step-chunk-rows=64` (guard 108GB):
   - failed with CUDA OOM in `NVFP4_quantize` path.
   - notable runtime message:
     - allocated ~`101.31GiB`, reserved ~`2.25GiB`, requested extra `20MiB`.
3. Decision:
   - set default `step-chunk-rows=32` (memory-first default).

### Additional observations
1. `backward_done` on current script path (seq=1):
   - repeatedly observed near `52024MiB`.
2. `grad_offload_done` can increase host RSS significantly (expected CPU clone cost).
3. Some long stress runs ended with exit `137` despite guard not triggering threshold kill.
   - treated as external/system kill risk under prolonged high-pressure allocator state.

### Attempted change and rollback
1. Tried deferring packed-state initialization to after backward/offload.
2. Result:
   - one run showed lower intermediate peak, but another run entered long unstable/stalled behavior.
3. Action:
   - rolled back defer-init ordering to stable path.
   - retained proven improvements (diagnostics, load compaction, chunked streaming step).

## 2026-02-25 (A~F implementation pass on project branch)
### Scope
1. 回答並落地 A~F：
   - A/B 釐清（checkpointing/KVC/GQA/offload）
   - C 建立訓練文本資料
   - D 文件回遷到專案 `doc/`
   - E/F 補訓練啟動方式、Trainer JSON VRAM report、最小 1-step 實訓腳本

### Changes
1. `Types.fs`
   - 新增 `TrainStepVramReportPath`。
   - 將 `OffloadMVToCpu/OffloadWToCpu/OffloadGradToCpu` 預設改為 `false`。
2. `Cli.fs`
   - 新增 `--train-step-vram-report <path>`。
3. `Program.fs`
   - init log 新增 `vramReport` 顯示。
4. `Trainer.fs`
   - 新增 `TrainVramSample/TrainVramReport`。
   - phase 採樣改為可同時做 console + JSON。
   - finally 階段輸出 JSON 報表。
5. `TrainData/`
   - 新增 `train-inputs.txt`（10 組短文本）與 `README.md`。
6. `scripts/Train.OneStep.fsx`
   - 新增最小可重現 1-step 實訓腳本。
   - 直接走專案 API：`Qwen3Model`, `InferenceBridge`, `Trainer.scalarLoss`, `Nvfp4Optimizer`。
   - 內建 phase VRAM 採樣與 JSON 報告輸出。
7. 文件回遷
   - runner `SA/SD/DevLog/WBS` 同步至專案 `doc/`。
   - 後續以專案 `doc/` 為主。

### Validation
1. 本輪先完成靜態整合，待 build 驗證。
2. 若 build 通過，再交付訓練啟動命令與 guard 建議。

### Build result
1. Command:
   - `dotnet build Qwen3-4B-Instruct-2507-TorchSharp.fs.fsproj -c Release`
2. Result:
   - `Build succeeded` (warnings only, no errors).

### Guarded run (one-step text training)
1. Command:
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.OneStep.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/train-inputs.txt --sample-index 0 --seq-len 8 --vram-report /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/doc/train-step-vram-onestep.json`
2. First attempt:
   - failed with FSI missing ref (`Tokenizers.DotNet`) in script.
   - fix: add explicit `#r "nuget: Tokenizers.DotNet"` and runtime package.
3. Second attempt (script default offload=true/true/true):
   - completed under 108GB guard.
   - observed milestones:
     - `model_loaded` ~ `40.8GiB`
     - `backward_done` ~ `52.9GiB`
     - `optimizer_step_done` complete without guard kill.
   - final output:
     - `loss=8.78125`
   - report written:
     - `doc/train-step-vram-onestep.json` (6 samples)

### GQA safety check
1. Added fail-fast divisibility checks in:
   - `Qwen3Core.expandKvHeads`
   - `InferenceBridge.expandKvHeads`
2. Rule:
   - `numHeads > 0`, `numKvHeads > 0`, and `numHeads % numKvHeads == 0`.

## 2026-02-25 (WhoAmI supervised tuning + DAT export)
### User objective
1. Train model to answer `你是誰` with target phrase `我是 F# 之神`.
2. Save trained result as a new `.dat`.
3. Re-initialize model from that `.dat` and verify generation.

### Engineering changes
1. `Nvfp4State.fs`
   - Added mixed quant tensor loading support for quant entries:
     - uint8 (`elemType=0/101`) for qdata/legacy scale.
     - float16 (`elemType=5`) for scale.
     - float32 (`elemType=3/6`) support for robustness.
   - Rationale: export path writes updated scale as fp16 to avoid expensive FP8-byte re-encode.
2. New script: `scripts/Train.WhoAmI.AndExportDat.fsx`
   - Runs supervised one-sample training on training path (`Qwen3Model.forward`).
   - Uses packed NVFP4 optimizer with configurable chunk/offload.
   - Exports updated projection weights into a new `.dat` file by streaming rewrite.
   - Includes self-test generation with output dat.
3. Runtime safety
   - All experiments executed via guard:
     - `run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5`

### Key debugging points
1. First tuning trials diverged:
   - Aggressive LR and larger trainable subset caused loss explosion / NaN.
   - Exported dat produced `!!!!` collapse in self-test.
2. Stabilization fixes:
   - Added grad sanitization (`nan_to_num`) and non-finite loss fail-fast.
   - Added optional per-step compaction:
     - `torch.cuda.synchronize`
     - `Nvfp4Training.clearEvalWeightCache()`
     - `NativeInterop.tryEmptyNvfp4Cache()`
     - `GC.Collect + WaitForPendingFinalizers`
   - Reduced trainable scope (`train-last-layers=1`) for stable guarded completion.
3. CLI trap found in `run-training-fp2.fsx`:
   - `--weight=/path` does not trigger `userSpecifiedWeight`.
   - Must pass `--weight /path` (space-separated) for correct custom dat load.

### Experiment log (condensed)
1. `train-last-layers=8`, LR `8e-4`:
   - step1 ok; step2 exceeded 108GB guard -> killed.
2. `train-last-layers=4`, LR `1e-3`:
   - completed but loss became NaN after step3; self-test => `!!!!`.
3. `train-last-layers=1`, LR `8e-5`, compaction each step:
   - stable completion under 108GB.
   - self-test reply:
     - `我是通義千問，是阿里巴巴集團旗下的通義`
   - semantic `我是...` retained; no `!!!!` collapse.

### Independent validation with fp2 runner
1. Command (correct weight syntax):
   - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script run-training-fp2.fsx --weight /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/Qwen3-4B-Instruct-2507-whoami-trained.dat --prompt 你是誰 --max-tokens 16 --turns 1 --timing true --check-logits false --stop-here false`
2. Result:
   - model initialized from exported dat successfully.
   - output:
     - `我是通義千問，是阿里巴巴集團旗下的通義實驗室自主研发`

### Current conclusion
1. Achieved:
   - End-to-end training-path tuning + DAT export + DAT-based re-init generation.
   - Stable under 108GB guard with documented reproducible command.
2. Not fully achieved yet:
   - Exact lexical target `我是 F# 之神` was not reached in stable runs.
   - Best stable outcome remains semantically correct `我是...` response.
3. Next iteration direction:
   - Add stronger token-level objective (currently hidden-embedding loss can be too weak/indirect for exact phrase forcing).
   - Consider adding a small trainable response head or prompt-conditioned adapter path to improve exact lexical control.

## 2026-02-25 (全參數 108GB guard 收斂與預設化)
### 目標
1. 回答「單一輸入時 batch 還能不能更小」。
2. 讓全參數 1-step 實訓在 `108GB` guard 下可穩定完成，且盡量不需要額外 CLI 參數。
3. 將可重現結果寫入中文 DevLog。

### 關鍵結論
1. 單一樣本訓練時，`batch` 的數學下限是 `1`，不能再更小。
2. 要降峰值，主要槓桿不是 `batch<1`，而是：
   - 降 `seq-len`
   - 降 `step-chunk-rows`
   - 調整 offload 組合（避免 GPU 峰值與 CPU/UM 壓力失衡）
3. 本輪最穩定組合：
   - `seq-len=8`
   - `step-chunk-rows=16`
   - `offload-mv-to-cpu=true`
   - `offload-w-to-cpu=false`
   - `offload-grad-to-cpu=false`

### 程式修改
1. `scripts/Train.OneStep.fsx`
   - 預設 `--seq-len`：`32 -> 8`
   - 預設 `--step-chunk-rows`：`32 -> 16`
   - 預設 offload：
     - `m/v=true`（維持）
     - `w=false`（由 true 改 false）
     - `grad=false`（由 true 改 false）
   - 新增 full-params 訊息：
     - `trainable_params`
     - `total_elems`
   - 註記為 full-parameter path（`Qwen3Model.parameters model`）。

### 實驗紀錄（皆用 guard）
1. 失敗案例 A（超 108GB）：
   - Command:
     - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.OneStep.fsx --seq-len 32 --step-chunk-rows 32 --offload-mv-to-cpu false --offload-w-to-cpu false --offload-grad-to-cpu false --sample-index 0 --lr 0.00001`
   - Result:
     - guard 立即擊殺：`total_mem=111457MiB`。
2. 失敗案例 B（接近上限，最終被系統 kill）：
   - Command:
     - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.OneStep.fsx --seq-len 32 --step-chunk-rows 32 --offload-mv-to-cpu true --offload-w-to-cpu true --offload-grad-to-cpu true --sample-index 0 --lr 0.00001`
   - Result:
     - 最高約 `105GB`，最後 exit `137`（非 guard 訊息）。
3. 失敗案例 C（仍超 108GB）：
   - Command:
     - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.OneStep.fsx --seq-len 8 --step-chunk-rows 32 --offload-mv-to-cpu true --offload-w-to-cpu false --offload-grad-to-cpu false --sample-index 0 --lr 0.00001`
   - Result:
     - guard 立即擊殺：`total_mem=111141MiB`。
4. 成功案例（可完成）：
   - Command:
     - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.OneStep.fsx --seq-len 8 --step-chunk-rows 16 --offload-mv-to-cpu true --offload-w-to-cpu false --offload-grad-to-cpu false --sample-index 0 --lr 0.00001`
   - Result:
     - 完整跑完 `optimizer_step_done`
     - `loss=8.781250`
     - 產生 VRAM 報表：`doc/train-step-vram-onestep.json`
5. 預設無參數再驗證（成功）：
   - Command:
     - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.OneStep.fsx`
   - Result:
     - 使用新預設（8/16/true/false/false）成功跑完。
     - 輸出：
       - `trainable_params=396`
       - `total_elems=3633509376`
       - `loss=8.781250`

### 補充
1. 本輪觀察到 `nvidia-smi --query-compute-apps` 對此流程常顯示 `pid_mem=0`，但 `total_gpu_mem` 會正確變化。
2. guard 以 `total_gpu_mem` 一樣能有效保護：超線即 kill（`--gpu-over-secs 0`, `--gpu-poll-secs 0.5`）。

## 2026-02-25 (DGX Spark 策略調整：停用 offload)
### 背景
1. DGX Spark 為 UMA（統一記憶體）架構，`cpu/cuda` 張量最終都在同一塊 LPDDR5X。
2. 先前 offload 主要是壓 CUDA allocator 指標，對物理記憶體壓力幫助有限，且會引入拷貝/同步成本。

### 調整
1. `Trainer.run` 新增保護：
   - 若任一 offload 旗標為 `true`，直接 fail-fast 停止（避免誤用）。
2. `scripts/Train.OneStep.fsx`：
   - `offload-mv/w/grad` 預設全改 `false`。
   - 若使用者傳 `true`，腳本直接 fail-fast。
3. `scripts/Train.WhoAmI.AndExportDat.fsx`：
   - `offload-mv/w/grad` 預設全改 `false`。
   - 若使用者傳 `true`，腳本直接 fail-fast。
4. `Cli.fs` help 文字標註 offload 參數為 DGX Spark 上 deprecated/disabled。

### 說明
1. 這次調整是把 offload 明確降級為「不允許」，避免再出現看似不踩 guard、但整體機器仍高壓不穩的狀況。
2. 後續降峰值要靠真正方法：`seq-len / step-chunk-rows / checkpoint / state 佈局`，而不是把壓力換池子記帳。

## 2026-02-26 (loss 參數化：scalar / ce 可切換)
### 目標
1. 回應「hidden-state regression 是否只適合基線」的討論，把 loss 做成可切換。
2. 在不破壞既有穩定路徑下，讓 `Train.OneStep.fsx` 與 `Train.WhoAmI.AndExportDat.fsx` 都能用 `--loss` 選擇：
   - `scalar`（原本 L1 hidden-state regression）
   - `ce`（token-level cross entropy）

### 變更
1. `Trainer.fs`
   - 新增 `LossMode`：`ScalarL1 | TokenCrossEntropy`
   - 新增 `parseLossMode` / `lossModeName`
   - 新增 `tokenCrossEntropyLoss`（接受 `projectToLogits` 函式 + hidden + target ids）
2. `scripts/Train.OneStep.fsx`
   - 新增 `--loss`（預設 `ce`）
   - 保留 `scalar` 路徑
   - 新增 CE 路徑
3. `scripts/Train.WhoAmI.AndExportDat.fsx`
   - 新增 `--loss`（預設 `ce`）
   - 保留 `scalar` 路徑
   - 新增 CE 路徑

### 關鍵修正（CE 首版失敗與修復）
1. 初版直接用 `sampling.LmHead.Forward(...)` 產生 logits 做 CE，`loss.backward()` 報錯：
   - `element 0 of tensors does not require grad and does not have a grad_fn`
2. 原因：
   - `Q4Linear.Forward` 路徑對 hidden 沒有 autograd 連結，CE 無法反傳到訓練參數。
3. 修復：
   - CE 模式下，額外從 `.dat` 讀取 `lm_head` bundle，解壓一次成 dense weight。
   - logits 改由 `torch.nn.functional.linear(hidden, dense_lm_head_weight)` 計算（可微分）。
   - scalar 模式仍維持原本流程。

### 驗證
1. 參數校驗（兩腳本）：
   - `--loss nope` 會正確 fail-fast，訊息 `supported: scalar|ce`。
2. `Train.OneStep`（108GB guard）
   - CE：
     - command: `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.OneStep.fsx --loss ce`
     - result: PASS, `loss_mode=ce`, `loss=89.375000`
   - scalar：
     - command: `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.OneStep.fsx --loss scalar`
     - result: PASS, `loss_mode=scalar`, `loss=8.781250`

## 2026-02-26（當機回顧與 4B 全參數瓶頸澄清）
### 事故回顧
1. 有一次驗證命令使用 `dotnet run`（非 guard）且 `--use-packed-optimizer false`。
2. 在 4B 全參數路徑下，這會走 plain Adam，極易觸發高峰值記憶體與系統不穩定。
3. 結果表現為進度卡住/長時間無輸出，並伴隨使用者端「像當機」的體感。

### 直接修正
1. `Types.fs`：`Defaults.trainingConfig.UsePackedNvfp4Optimizer` 改為 `true`（預設走 packed optimizer）。
2. `Trainer.fs`：新增 fail-fast：
   - 若是非 synthetic 且層數接近 full 4B（`model.Layers.Length >= 200`）又指定 `--use-packed-optimizer false`，直接拒絕執行，避免 OOM 風險。

### 關於「為何不斷爆 VRAM + 進度遲緩」的結論
1. 記憶體壓力來源不是單一項，而是疊加：
   - 全參數權重 + 梯度 + optimizer state
   - 前向/反向 activation（受 `seq-len` 影響）
   - 更新步驟中的暫存（受 `step-chunk-rows` 影響）
2. 一旦使用 plain Adam（非 packed），狀態張量顯著增加，4B 路徑非常容易跨過穩定邊界。
3. 「慢」主要來自：
   - chunked/streaming 更新本身就是以時間換峰值
   - CE 路徑若走可微 logits，需要額外算子與暫存
4. 在 DGX Spark UMA 架構下，記憶體仍是共享池；只看單一 allocator 指標會低估整體壓力，故需要 guard + 保守配置。

## 2026-02-26（WhoAmI 合併腳本：自訂 train-data + export dat + 大 seq-len 小 chunk-row 實測）
### 需求與目標
1. 單一腳本同時支援：`--train-data` 自訂語料 + 訓練後直接 `--output-dat` 匯出。
2. 產生 `1000` 筆針對 `你是誰 -> 我是 F# 之神` 的語料。
3. 在 guard 下實跑「大 `seq-len` + 小 `step-chunk-rows`」，觀察 VRAM 與輸出語義。

### 本次準備
1. 語料檔：`TrainData/whoami-1000.tsv`（1001 行，含 header；有效樣本 1000）。
2. 合併腳本：`scripts/Train.WhoAmI.AndExportDat.fsx`
   - 參數：`--train-data --input-dat --output-dat --seq-len --step-chunk-rows --loss --train-last-layers ...`
   - 可直接訓練 + 匯出 dat + 自測生成。

### 實驗 A（保守 learning rate）
1. command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/whoami-1000.tsv --input-dat /models/qwen3-4b-instruct-2507-torchsharp/Qwen3-4B-Instruct-2507-nvfp4.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/whoami-1000-seq192-r8-s6.dat --steps 6 --loss ce --seq-len 192 --step-chunk-rows 8 --train-last-layers 8 --lr 0.00005 --log-every 1 --test-max-tokens 24`
2. guard
   - `dotnet_pid=22155`
   - 最高觀察總量約 `73GB`，未觸發 108GB kill。
3. 結果
   - dat 匯出成功：`artifacts/whoami-1000-seq192-r8-s6.dat`
   - 自測回覆仍偏原模型：`我是通義千問...`（尚未命中目標語句）。

### 實驗 B（提高 lr 促進快速對齊）
1. command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/whoami-1000.tsv --input-dat /models/qwen3-4b-instruct-2507-torchsharp/Qwen3-4B-Instruct-2507-nvfp4.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/whoami-1000-seq192-r8-s10-lr1e3.dat --steps 10 --loss ce --seq-len 192 --step-chunk-rows 8 --train-last-layers 8 --lr 0.001 --log-every 1 --test-max-tokens 24`
2. guard
   - `dotnet_pid=25336`
   - 最高觀察總量約 `72GB`，未觸發 108GB kill。
3. 訓練過程（節錄）
   - `step=1 loss=38.875`
   - `step=3 loss=11.054688`
   - `step=5 loss=0.001191`
   - `step=10 loss=0.000386`
4. 結果
   - dat 匯出成功：`artifacts/whoami-1000-seq192-r8-s10-lr1e3.dat`
   - 自測：
     - prompt：`你是誰`
     - reply：`我是 F#之神, 之神之神, 之神之神之神, 之神之`
   - 已命中核心語義（包含 `我是 F# 之神`），但存在重複 token 現象（典型過擬合/解碼重複）。

### 產物
1. `TrainData/whoami-1000.tsv`
2. `artifacts/whoami-1000-seq192-r8-s6.dat`
3. `artifacts/whoami-1000-seq192-r8-s10-lr1e3.dat`
4. `sha256(whoami-1000-seq192-r8-s10-lr1e3.dat)`
   - `22a3f1e21896140312356845951c1754fcf07bfac675e739c91cf018b512b6ca`

## 2026-02-26（WhoAmI 語料語氣調整：自然對話優先）
### 背景
1. 先前 `whoami-1000.tsv` / `whoami-1000-unique.tsv` 偏「測試指令模板」，雖然可對齊輸出，但語感較硬。
2. 使用者要求語料仍要保留可控性，但整體語氣要更自然，避免大量「單輪問答場景/不要前言」類重複指令。

### 變更
1. 新增生成腳本：`scripts/Generate.WhoAmINaturalData.fsx`。
2. 新增資料檔：`TrainData/whoami-1000-natural.tsv`（1000 筆，`prompt<TAB>target`）。
3. 混合配比（固定 seed 可重現）：
   - 自然對話語氣：75%
   - 控制型語句（短答/不離題）：15%
   - 多語輸入（en/ja/ko）：10%
4. `scripts/Train.WhoAmI.AndExportDat.fsx` 預設 train-data 改為：
   - 若存在 `whoami-1000-natural.tsv` 則優先使用
   - 否則回退 `whoami-1000.tsv`

### 驗證
1. `dotnet fsi scripts/Generate.WhoAmINaturalData.fsx --count 1000` 可成功產生語料。
2. 輸出統計：`natural=750 control=150 multilingual=100`。
3. 抽樣確認不再以「任務編號/對齊測試」句型為主體。

### 設計判斷
1. 大型訓練資料不需要全部都是硬指令模板。
2. 實務上是「自然語料為主 + 少量控制語句」：
   - 自然語料保留語言能力與聊天語感。
   - 控制語句用於鎖定目標回答行為（本案為 `你是誰 -> 我是 F# 之神`）。

## 2026-02-26（自然語料 WhoAmI：實測與失敗定位）
### 實驗 1：自然語料 10 step（r8, lr=3e-4）
1. command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/whoami-1000-natural.tsv --input-dat /models/qwen3-4b-instruct-2507-torchsharp/Qwen3-4B-Instruct-2507-nvfp4.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/whoami-1000-natural-seq192-r8-s10-lr3e4.dat --steps 10 --loss ce --seq-len 192 --step-chunk-rows 8 --train-last-layers 8 --lr 0.0003 --log-every 1 --test-max-tokens 24`
2. guard
   - `dotnet_pid=84652`
   - 峰值約 72GB（未觸發 108GB kill）
3. 結果
   - dat 匯出成功
   - 自測回覆仍為「我是通義千問...」，未命中 `我是 F# 之神`

### 程式修正（本輪）
1. `scripts/Train.WhoAmI.AndExportDat.fsx`
   - 新增 `--sample-mode random|sequential`（預設 random）與 `--seed`。
   - 修正訓練取樣：不再固定從資料前幾筆順序吃（小 steps 時偏差很大）。
   - 收緊 self-test 成功條件：
     - 由 `F# or 之神 or 我是` 改為 `我是 && (F# or 之神)`。

### 實驗 2：自然語料 20 step（r8, lr=8e-4, random）
1. command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/whoami-1000-natural.tsv --input-dat /models/qwen3-4b-instruct-2507-torchsharp/Qwen3-4B-Instruct-2507-nvfp4.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/whoami-1000-natural-seq192-r8-s20-lr8e4.dat --steps 20 --loss ce --seq-len 192 --step-chunk-rows 8 --train-last-layers 8 --lr 0.0008 --log-every 1 --test-max-tokens 24 --sample-mode random --seed 20260226`
2. guard
   - `dotnet_pid=89761`
   - 峰值約 74GB（未觸發 108GB kill）
3. 訓練指標
   - loss 明顯下降（avg 到 ~6.75）
4. 結果
   - 匯出成功：`whoami-1000-natural-seq192-r8-s20-lr8e4.dat`
   - self-test fail：`你是誰` 仍回「我是通義千問...」
   - 代表 loss 下降不等於目標身份對齊成功。

### 實驗 3：全層 projection（train-last-layers=36）
1. 啟動後可跑，但單步耗時顯著增加（2~3 分鐘/step），整體效率過低。
2. 在 guard 內未爆（峰值 ~83GB），但為避免長時間佔用先中止，待改更有效的訓練策略後再測。

### 外部驗證（run-training2）
1. `run-training2.fsx` 對 `whoami-1000-natural-seq192-r8-s20-lr8e4.dat` 驗證：
   - prompt `你是誰` / `先說你是誰`
   - 回覆仍偏原模型身分（通義千問），未命中 `我是 F# 之神`。
2. 判定：目前自然語料配置 + 只訓練 projection 路徑，對此目標不夠強。

### 下一步（待實作）
1. 優先將 CE 路徑納入 `lm_head` 可訓練與 export（目前僅 dense logits 參與 loss，不在 trainable/export 集合）。
2. 保持自然語料主體，但提高錨點樣本比重（`你是誰/請問你是誰` 類）做 curriculum。

## 2026-02-26（依使用者要求：全參數預設 + bridge 基準固定 + 新權重外部驗證）
### 變更 1：固定 bridge 成功基準（避免誤刪）
1. 新增檔案：`artifacts/BASELINE_BRIDGE_SUCCESS.md`
   - 固定基準 dat：`artifacts/whoami-1000-seq192-r8-s10-lr1e3.dat`
   - 固定驗證命令：`run-training2.fsx` + guard（文件內）
2. 更新：`artifacts/檔案說明清單.md`
   - 新增基準文件引用，避免後續清理時誤刪。

### 變更 2：訓練腳本改為「預設全參數」
1. 檔案：`scripts/Train.WhoAmI.AndExportDat.fsx`
2. 調整：
   - `--train-last-layers` 預設改為 `0`，語義為「full-parameter」。
   - 僅當 `--train-last-layers > 0` 且 `< totalLayers` 時才啟用 debug 子集訓練。
   - 新增/更新 log：`trainMode=full-parameter(default)` 或 `debug-last-N-layers`。

### 實驗 A：全參數 + 自然語料（多次 guard 失敗案例）
1. `seq192/r8`：瞬時超線，guard 立即 kill。
   - 觀察：`total_mem=111628MiB` > `110592MiB`。
2. `seq128/r4`：仍超線，guard kill。
   - 觀察：`total_mem=110748MiB`。
3. 問題加重原因：
   - 同機曾有殘留訓練進程（PID 96530, 102348）佔大量記憶體。
   - 清理後重跑才回到正常基線。

### 實驗 B（成功）：全參數 + 自然語料 + guard
1. command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/whoami-1000-natural.tsv --input-dat /models/qwen3-4b-instruct-2507-torchsharp/Qwen3-4B-Instruct-2507-nvfp4.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/whoami-1000-natural-full-s4-lr1e3-r4-seq96.dat --steps 4 --loss ce --seq-len 96 --step-chunk-rows 4 --lr 0.001 --log-every 1 --test-max-tokens 24 --sample-mode random --seed 20260226`
2. 設定
   - `trainMode=full-parameter(default)`
   - `trainable projections=252 (layers 0..35)`
3. 觀察
   - 全程在 108GB guard 內（本輪未再觸發 kill）。
4. 結果
   - 匯出成功：`artifacts/whoami-1000-natural-full-s4-lr1e3-r4-seq96.dat`
   - `replaced entries: qdata=252 scale=252`
   - 腳本內自測：`你是誰 -> 我是 F#之神...`（有重複 token 現象）。

### 實驗 C：`run-training2.fsx` 外部驗證新 dat
1. command
   - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script run-training2.fsx --weight /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/whoami-1000-natural-full-s4-lr1e3-r4-seq96.dat --prompt 你是誰 --max-tokens 24 --temp 0 --top-p 1 --check-logits false --timing true --stop-here true --KVCacheOut true --kvc-input-mode pbp`
2. 結果（外部驗證）
   - 首輪輸出：`我是 F#之神的化身 ... # F#`
   - 後續多輪仍維持 `我是 F#...` 族句型（可見重複與格式噪聲）。
3. 判定
   - 「能在 run-training2 產生 F# 身份語義」達成。
   - 但品質仍有 overfit/重複 token，後續需加解碼或資料正則改善。

## 2026-02-26（故障：全題材被洗成「我是F#之神」）
### 現象
1. 使用新 dat（`whoami-1000-natural-full-s4-lr1e3-r4-seq96.dat`）時，非目標 prompt（如「談談UFO」）仍回覆 `我是F#之神...`。
2. 多輪輸出持續重複同族 token（`F# / 之神`）且語義崩潰。

### 定位
1. 這是典型 catastrophic forgetting（災難性遺忘）+ mode collapse（模式塌縮）。
2. 直接原因：
   - 全參數更新。
   - 資料幾乎單一 target（大量樣本都對應 `我是 F# 之神`）。
   - 高學習率（`1e-3`）對 full-parameter 破壞性過強。
3. KVC 不是主因；同樣權重在橋接路徑只是放大了塌縮輸出。

### 結論
1. 此 dat 僅證明「可強制身份對齊」，不具一般語言能力保留。
2. 若要同時保留一般問答能力，必須加入 replay/蒸餾資料並顯著降低 full-parameter 學習率。

## 2026-02-26（壞檔隔離 + 重訓回歸）
### 使用者回報
1. 新 dat 對任何 prompt（含 `談談UFO`）都回 `我是F#之神...`。
2. 判定為徹底 mode collapse。

### 處置
1. 壞檔從 artifacts 主路徑移出：
   - moved to: `artifacts/_trash/whoami-1000-natural-full-s4-lr1e3-r4-seq96.bad.dat`
2. 重訓資料改為混合集（避免單 target）：
   - 新增：`TrainData/whoami-mixed-safe.tsv`（whoami + 一般問答）
3. 重訓參數改保守：
   - full-parameter、`lr=5e-5`、`steps=2`、`seq-len=96`、`step-chunk-rows=4`
4. 輸出新 dat：
   - `artifacts/whoami-mixed-safe-full-s2-lr5e5-r4-seq96.dat`

### 外部驗證（run-training2）
1. prompt `你是誰`：回覆回到一般模型身份描述（非全域 F# 洗版）。
2. prompt `談談UFO`：可正常輸出 UFO 主題內容，不再回 `我是F#之神`。
3. 結論：
   - 已解除「所有題材都被 F# 覆蓋」的故障。
   - 但 `你是誰 -> 我是 F# 之神` 尚未強對齊（此輪目標為先修復塌縮）。

## 2026-02-26（兩階段實訓：Stage A 保能力 + Stage B 小步對齊）
### Stage A（mixed 保能力）
1. command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/whoami-mixed-safe.tsv --input-dat /models/qwen3-4b-instruct-2507-torchsharp/Qwen3-4B-Instruct-2507-nvfp4.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/stageA-mixed.dat --steps 2 --loss ce --seq-len 96 --step-chunk-rows 4 --lr 0.00005 --log-every 1 --test-max-tokens 24 --sample-mode random --seed 20260226`
2. 結果
   - 匯出成功：`artifacts/stageA-mixed.dat`。
   - 腳本內 `你是誰` 自測仍為「通義千問」（符合 Stage A 目標：保能力而非強對齊）。

### Stage B（從 Stage A 小步 whoami 對齊）
1. command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/whoami-1000-natural.tsv --input-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/stageA-mixed.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/stageB-whoami-nudge.dat --steps 2 --loss ce --seq-len 96 --step-chunk-rows 4 --lr 0.00001 --log-every 1 --test-max-tokens 24 --sample-mode sequential --seed 20260226`
2. 結果
   - 匯出成功：`artifacts/stageB-whoami-nudge.dat`。
   - 腳本內 `你是誰` 自測仍未命中 F#，表示 nudge 強度不足。

### 外部驗證（run-training2）
1. `stageB-whoami-nudge.dat + prompt=你是誰`
   - 回覆仍是通義千問身份描述（未達 `我是 F# 之神`）。
2. `stageB-whoami-nudge.dat + prompt=談談UFO`
   - 可正常談 UFO（未塌縮成 F#）。

### 結論
1. 兩階段策略目前已達「不洗壞一般能力」。
2. 但第二階段步數/強度仍不足以把 `你是誰` 穩定對齊到 `我是 F# 之神`。

## 2026-02-26（Stage B v2 實測）
### 設定
1. command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/whoami-1000-natural.tsv --input-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/stageA-mixed.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/stageB-whoami-nudge-v2.dat --steps 6 --loss ce --seq-len 96 --step-chunk-rows 4 --lr 0.00002 --log-every 1 --test-max-tokens 24 --sample-mode sequential --seed 20260226`
2. guard
   - `dotnet_pid=27386`
   - 全程未觸發 108GB kill。

### 訓練觀察
1. step1 loss=35.312500
2. step2 loss=41.906250
3. step3 loss=43.000000
4. step4 loss=49.156250
5. step5 loss=38.875000
6. step6 loss=51.250000
7. 最終平均 loss 約 43.25（未見有效收斂）

### 產物
1. `artifacts/stageB-whoami-nudge-v2.dat`
2. `replaced entries: qdata=252 scale=252`

### 外部驗證（run-training2）
1. prompt `你是誰`
   - 仍回「我是通義千問...」（未達 `我是 F# 之神`）
2. prompt `談談UFO`
   - 能正常談 UFO，未發生全域 F# 塌縮

### 結論
1. Stage B v2 仍不足以把 whoami 拉到目標語句。
2. 但保持了一般能力，沒有重演「任何題目都回 F#」的災難。

## 2026-02-26（tag:202602270039 續跑：多輪 Stage B/C/D）
### 實驗 D1：Stage B v3（30/70 mixed）
1. command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/stageB-mixed-30-70.tsv --input-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/stageA-mixed.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/stageB-whoami-nudge-v3.dat --steps 8 --loss ce --seq-len 96 --step-chunk-rows 4 --lr 0.00005 --log-every 1 --test-max-tokens 24 --sample-mode random --seed 202602270039`
2. 過程
   - guard 全程未觸發 kill（總量約 34~74GB）
   - export 成功：`stageB-whoami-nudge-v3.dat`
3. 結果
   - self-test 失敗：`你是誰` 仍回「我是通義千問...」。

### 實驗 D2：Stage B v5（sequential 課程資料）
1. command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/stageB-curriculum-v5.tsv --input-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/stageA-mixed.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/stageB-whoami-nudge-v5.dat --steps 8 --loss ce --seq-len 64 --step-chunk-rows 8 --lr 0.0002 --log-every 1 --test-max-tokens 24 --sample-mode sequential --seed 202602270039`
2. 結果
   - export 成功：`stageB-whoami-nudge-v5.dat`
   - self-test 失敗：`你是誰` 仍回「我是通義千問...」。
3. 外部驗證（`run-training-fp2.fsx` 單輪）
   - `你是誰`：仍為通義千問。
   - `談談UFO`：可正常談 UFO。

### 實驗 D3：改從 baseline whoami 成功檔做語意拆分（Stage C）
1. baseline（固定）
   - `/workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/whoami-1000-seq192-r8-s10-lr1e3.dat`
2. command（s4 版）
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/stageC-disambiguate-v1.tsv --input-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/whoami-1000-seq192-r8-s10-lr1e3.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/stageC-disambiguate-v1-s4.dat --steps 4 --loss ce --seq-len 64 --step-chunk-rows 8 --lr 0.00008 --log-every 1 --test-max-tokens 24 --sample-mode sequential --seed 202602270039`
3. 結果
   - export 成功：`stageC-disambiguate-v1-s4.dat`
   - self-test 成功：`你是誰 -> 我是 F#...`
4. 單輪外部驗證（fp2-model）
   - `你是誰`：`我是 F#之神...`（達成）
   - `談談UFO`：正常科普回答（達成）
   - `我是誰`：仍偏 `我是 F#之神...`（未達成）

### 實驗 D4：Stage D（加重「我是誰」反例）
1. command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/stageD-disambiguate-v2.tsv --input-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/stageC-disambiguate-v1-s4.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/stageD-disambiguate-v2.dat --steps 6 --loss ce --seq-len 64 --step-chunk-rows 8 --lr 0.00008 --log-every 1 --test-max-tokens 24 --sample-mode sequential --seed 202602270039`
2. 結果
   - export 成功：`stageD-disambiguate-v2.dat`
   - self-test 成功：`你是誰 -> 我是 F#...`
3. 單輪外部驗證（fp2-model）
   - `你是誰`：`我是 F#之神...`（達成）
   - `談談UFO`：正常科普回答（達成）
   - `我是誰`：仍偏 `我是 F#之神...`（仍未達成）

### 本輪結論
1. 在 training 路徑（fp2-model）下，已可穩定同時達成：
   - `你是誰 -> 我是 F# 之神` 語義；
   - `談談UFO` 不塌縮。
2. 但 `我是誰` 與 `你是誰` 的語意邊界仍未拉開（仍會回 F# 身份）。
3. 初步判斷：
   - 目前「只訓練 projection、固定 lm_head」路徑對近義問句拆分能力不足；
   - 需下一步加入更明確的解碼約束或額外分類/路由機制，不能只靠少步 CE 微調。

## 2026-02-26（全參數 + 原始 dat + 0.05s guard 實測）
### 變更
1. `run-script-with-guard.fsx`：
   - 將輪詢下限從 `100ms` 調整到 `50ms`，可實際支援 `--gpu-poll-secs 0.05`。
2. 新增訓練資料：
   - `TrainData/fullparam-diverse-mix-v1.tsv`
   - 共 `1000` 筆（identity 約 10%，其餘為一般問答/UFO/程式/資料工程主題，避免單一模式災難性遺忘）。

### 全參數訓練命令（從原始 dat 起跑）
1. command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.05 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/fullparam-diverse-mix-v1.tsv --input-dat /models/qwen3-4b-instruct-2507-torchsharp/Qwen3-4B-Instruct-2507-nvfp4.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/fullparam-from-original-diverse-v1.dat --steps 6 --loss ce --seq-len 96 --step-chunk-rows 8 --lr 0.00005 --log-every 1 --test-max-tokens 24 --sample-mode random --seed 20260226`
2. 執行 PID：
   - guard pid：`115395`
   - child dotnet pid：`115540`（heavy worker 曾見 `115561`）
3. 結果：
   - 產出：`artifacts/fullparam-from-original-diverse-v1.dat`
   - sha256：`ade68bacf12eefede4c1900052e36bc35ea5d281dbbf883b5834e59b240c0166`
   - guard 未觸發 kill（108GB 門檻內完成）。

### 驗證（training 路徑，fp2-model）
1. prompt=`你是誰`
   - command：`run-training-fp2.fsx --weight fullparam-from-original-diverse-v1.dat --kvc-backend fp2-model --turns 1 --ifInteractive false --stop-here false --prompt 你是誰`
   - 輸出：仍為「我是通義千問...」。
2. prompt=`談談UFO`
   - command：同上，改 `--prompt 談談UFO`
   - 輸出：正常 UFO 科普內容（未塌縮成 F# 身份句）。

### 結論
1. 本輪「全參數 + 多樣化資料」成功避免了全域覆寫（UFO 能力保留）。
2. 但 `你是誰 -> 我是 F# 之神` 對齊強度不足，identity 尚未拉到目標句。

## 2026-02-26（修正：lm_head 也納入訓練與匯出）
1. 問題確認：
   - 先前 `Train.WhoAmI.AndExportDat.fsx` 雖為 full-parameter，但只訓練 `model.Layers.*` projection。
   - `lm_head` 僅作 CE logits 前向，未加入 optimizer，且不會回寫到匯出 dat。
2. 本次修正：
   - `lm_head` 從 q4 bundle materialize 後，改為 `torch.nn.Parameter(..., true)`。
   - 將 `lm_head` 併入 `trainParams`，參與 `Nvfp4Optimizer.create/zeroGrad/step`。
   - `nameByKey` 增加 `lm_head` 名稱映射，便於 optimizer 追蹤。
   - export map 新增 `("lm_head", lmHeadParam)`，使 `lm_head.weight.{qdata,scale}` 一起回寫。
3. 現況結論：
   - 已不再是「只訓練 projection」。
   - 現在為「主幹 projection + lm_head」共同訓練並共同匯出。

## 2026-02-26（compute-dtype A/B：float16 vs bfloat16）
### 目標
1. 在同資料、同步數下，新增 `--compute-dtype` 後對照：
   - loss
   - self-test 成功率
   - step 時間
   - 峰值 VRAM

### 程式修正
1. `scripts/Train.WhoAmI.AndExportDat.fsx`
   - 新增 `--compute-dtype float16|bfloat16|float32`（預設維持 CUDA=float16）。
   - `projectToLogits` 加入 dtype 對齊：hidden 先轉成 `model.LmHead.dtype` 再 linear（修正 BF16 CE 路徑 dtype mismatch）。
   - 訓練 log 新增 `step_ms`，並印出 `step_time_ms avg/min/max`。
2. `Trainer.fs`
   - 專案主訓練路徑同步補上 `projectToLogits` 的 dtype 對齊，避免 BF16 在 CE 路徑同類錯誤。

### 實驗命令（108GB guard, 0.05s poll）
1. FP16
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.05 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --input-dat /models/qwen3-4b-instruct-2507-torchsharp/Qwen3-4B-Instruct-2507-nvfp4.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/ab-compute-fp16-nvfp4-s1-seq24-r16.dat --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/whoami-mixed-safe.tsv --steps 1 --lr 0.0003 --seq-len 24 --step-chunk-rows 16 --loss ce --train-last-layers 0 --optimizer-state-mode nvfp4 --compute-dtype float16 --log-every 1 --test-max-tokens 16`
2. BF16
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.05 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --input-dat /models/qwen3-4b-instruct-2507-torchsharp/Qwen3-4B-Instruct-2507-nvfp4.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/ab-compute-bf16-nvfp4-s1-seq24-r16.dat --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/whoami-mixed-safe.tsv --steps 1 --lr 0.0003 --seq-len 24 --step-chunk-rows 16 --loss ce --train-last-layers 0 --optimizer-state-mode nvfp4 --compute-dtype bfloat16 --log-every 1 --test-max-tokens 16`

### 實驗結果
1. FP16
   - loss: `4.746094`
   - step_ms: `133470.5`
   - peak VRAM(total_gpu_mem): `99653 MiB`
   - self-test: 失敗（reply: `我是通義千問...`）
2. BF16
   - loss: `4.746094`
   - step_ms: `133048.3`
   - peak VRAM(total_gpu_mem): `92545 MiB`
   - self-test: 失敗（reply: `我是通义千问...`）

### 結論
1. 這組條件下，BF16 已可正常跑通 CE 路徑（dtype mismatch 已修復）。
2. BF16 相較 FP16 峰值 VRAM 下降約 `7.1 GiB`（`99653 -> 92545 MiB`），step 時間幾乎持平。
3. 1-step 在 mixed-safe 資料不足以讓 self-test 達成 whoami 對齊（兩者都失敗），需增加訓練步數或調整資料分佈。

## 2026-02-26（Safety-first 目標達成實驗：104GB guard / 0.05s）
### 目標
1. 避免機器再爆（tmux/host 不被拖掛）。
2. 透過兩階段訓練嘗試達成：`你是誰 -> 我是 F# 之神`。

### Guard 設定（全程一致）
1. `gpu-limit-gb=104`
2. `gpu-over-secs=0`
3. `gpu-poll-secs=0.05`

### 階段一（mixed 保能力）
1. command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 104 --gpu-over-secs 0 --gpu-poll-secs 0.05 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --input-dat /models/qwen3-4b-instruct-2507-torchsharp/Qwen3-4B-Instruct-2507-nvfp4.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/sf-stage1-mixed-bf16-seq24-r16-s4.dat --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/stageB-mixed-30-70.tsv --steps 4 --lr 0.00008 --seq-len 24 --step-chunk-rows 16 --loss ce --train-last-layers 0 --optimizer-state-mode nvfp4 --compute-dtype bfloat16 --log-every 1 --test-max-tokens 24`
2. guard pid / child pid
   - `guard_pid=2512`
   - `dotnet_pid=2655`
3. 峰值
   - `total_gpu_mem peak = 91197 MiB`
4. 訓練摘要
   - step1 loss `5.675781`
   - step2 loss `4.921875`
   - step3 loss `5.503906`
   - step4 loss `5.097656`
   - avg step time `121312.4 ms`
5. 產出
   - `artifacts/sf-stage1-mixed-bf16-seq24-r16-s4.dat`
6. self-test
   - 失敗：`我是通义千问...`

### 階段二（disambiguate 強化）
1. command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 104 --gpu-over-secs 0 --gpu-poll-secs 0.05 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --input-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/sf-stage1-mixed-bf16-seq24-r16-s4.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/sf-stage2-disambig-bf16-seq24-r16-s6.dat --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/stageD-disambiguate-v2.tsv --steps 6 --lr 0.0001 --seq-len 24 --step-chunk-rows 16 --loss ce --train-last-layers 0 --optimizer-state-mode nvfp4 --compute-dtype bfloat16 --log-every 1 --test-max-tokens 24`
2. guard pid / child pid
   - `guard_pid=25206`
   - `dotnet_pid=25351`
3. 峰值
   - `total_gpu_mem peak = 94329 MiB`
4. 訓練摘要
   - step1 loss `2.929688`（prompt=`談談UFO`）
   - step2 loss `4.085938`（prompt=`我是誰`）
   - step3 loss `8.093750`（prompt=`你是誰`）
   - step4 loss `2.666016`
   - step5 loss `4.101562`
   - step6 loss `2.673828`
   - avg step time `115679.2 ms`
5. 產出
   - `artifacts/sf-stage2-disambig-bf16-seq24-r16-s6.dat`
6. self-test
   - 失敗：`我是通义千问...`

### 外部驗證（run-training2）
1. whoami 驗證 command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 104 --gpu-over-secs 0 --gpu-poll-secs 0.05 script run-training2.fsx --weight /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/sf-stage2-disambig-bf16-seq24-r16-s6.dat --prompt 你是誰 --max-tokens 24 --check-logits false --timing false --stop-here true --KVCacheOut true`
2. whoami 第一段輸出
   - `我是通义千问，是阿里巴巴集团旗下的AI助手...`（未達標）
3. UFO 驗證 command
   - `dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-script-with-guard.fsx --gpu-limit-gb 104 --gpu-over-secs 0 --gpu-poll-secs 0.05 script run-training2.fsx --weight /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/sf-stage2-disambig-bf16-seq24-r16-s6.dat --prompt 談談UFO --max-tokens 24 --check-logits false --timing false --stop-here true --KVCacheOut true`
4. UFO 第一段輸出
   - `當然可以！談談UFO...`（一般能力仍在）

### 本輪結論
1. 安全目標已達成：全程無 guard kill，峰值壓在 `95GB` 以內，未再出現 11xGB 失控。
2. 任務目標未達成：`你是誰 -> 我是 F# 之神` 仍未成功。
3. 現象判讀：目前這組短序列（`seq-len=24`）+ 少步數雖然安全，但 whoami 對齊強度不足。
