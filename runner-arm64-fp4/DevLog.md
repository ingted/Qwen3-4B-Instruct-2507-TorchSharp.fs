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

## 2026-02-26（dat export 相容性修正：bridge parity）
### 問題
1. `whoami` 訓練 dat 在 `run-training-fp2` 可輸出 `我是 F#之神`。
2. 但同檔在 `run-training2`（bridge 路徑）曾出現亂碼/`!!!!`。

### 分析與假設
1. 用 no-op 對照（`lr=0`、僅經過 export）：bridge 仍異常，排除「訓練更新本身」。
2. 鎖定 export：`Train.WhoAmI.AndExportDat.fsx` 將 `*.weight.scale` 固定寫成 `elemType=5(float16)`。
3. 原始官方 dat 的 scale 多為 `elemType=101`（1-byte fp8-like）。
4. 推論：這個型別/編碼改寫破壞 bridge 讀取契約。

### 修正
1. `Train.WhoAmI.AndExportDat.fsx`
   - `scale` 寫回改為保留原 entry `elemType`。
   - 依 `elemType` byte-size 寫 payload（1/2/4/8 bytes）。
   - 不再把 quantized scale 強制轉 float16；保留 native quantize 返回 dtype。
   - 加 shape mismatch fail-fast。

### 實驗與結果
1. no-op 重產：
   - output: `artifacts/whoami-noop-lr0-v2.dat`
   - command:
     - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/whoami-1000.tsv --input-dat /models/qwen3-4b-instruct-2507-torchsharp/Qwen3-4B-Instruct-2507-nvfp4.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/whoami-noop-lr0-v2.dat --steps 1 --loss ce --seq-len 192 --step-chunk-rows 8 --train-last-layers 8 --lr 0.0 --log-every 1 --test-max-tokens 16`
   - 結果：`run-training2` bridge 路徑恢復正常，多輪不再 `!!!!`。
2. 有訓練更新重產：
   - output: `artifacts/whoami-1000-seq192-r8-s10-lr1e3-v2.dat`
   - command:
     - `dotnet fsi run-script-with-guard.fsx --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/scripts/Train.WhoAmI.AndExportDat.fsx --train-data /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/TrainData/whoami-1000.tsv --input-dat /models/qwen3-4b-instruct-2507-torchsharp/Qwen3-4B-Instruct-2507-nvfp4.dat --output-dat /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/whoami-1000-seq192-r8-s10-lr1e3-v2.dat --steps 10 --loss ce --seq-len 192 --step-chunk-rows 8 --train-last-layers 8 --lr 0.001 --log-every 1 --test-max-tokens 24`
   - 結果：
     - `run-training2` first out: `我是 F#之神之神...`（有 whoami 語義）
     - `run-training-fp2` out: `我是 F#之神,`

### 附帶修正（互動 loop）
1. `run-training-fp2.fsx` / `run-training.fsx` 的 `請輸入:` 迴圈新增 `Console.ReadLine() = null` 保護。
2. 若 `Console.IsInputRedirected`（非互動執行）則 sleep，不刷屏，避免無限輸出灌爆 log。
