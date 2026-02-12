# DevLog

## 2026-02-12
### Context
- Source note: `notes/00001.txt`
- Goal: move `run-training.fsx` from runnable scaffold output to semantic output closer to `run2.fsx`.
- Constraint: no dependency on `Qwen3-4B-Instruct-2507-TorchSharp` project or `Qwen3.dll`.

### Findings From Note/Code Review
- Same weight file does not guarantee same output quality unless model wiring + tokenizer + decode path are equivalent.
- Current pure F# inference path had four major gaps:
  - tokenizer mismatch (byte-token fallback)
  - embedding mismatch (handcrafted features)
  - block wiring mismatch (scaffold linear stack)
  - layer coverage mismatch (2-layer fallback)

### Documentation Changes
- Updated `doc/SA.md` with parity gaps and new FR-05..FR-08.
- Updated `doc/SD.md` with inference parity design (`ModelConfigLite`, `Q4WeightBank`, `TokenizerBridge`, `ForwardEngine`).
- Updated `doc/Test.md` with parity-focused test addendum.
- Appended `doc/WBS.md` with WBS-12..WBS-17.

### Technical Direction
- Short-term implementation order:
  1. tokenizer parity (`tokenizer.json`)
  2. embedding lookup parity
  3. Qwen3-like projection wiring (`q/k/v/o`, `gate/up/down`, `lm_head`)
  4. full layer coverage and parity smoke checks

### Change Tracking
- Relevant recent commits:
  - `2e1f3b1` (`Qwen3-4B-Instruct-2507-TorchSharp.fs`): switched inference bridge to pure F# explicit wiring.
  - `a1b686e` (`fsann`): `run-training.fsx` uses pure F# inference DLL only.

### Implementation Update (same day, parity track)
- Implemented in `InferenceBridge.fs`:
  - tokenizer integration via `Tokenizers.DotNet` (`tokenizer.json`)
  - full layer-family loading for 36 layers (`q/k/v/o`, `gate/up/down`) + `lm_head`
  - raw tensor loading from `.dat` for:
    - `model.embed_tokens.weight`
    - per-layer norm weights (`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`)
    - `model.norm.weight`
  - Qwen3-like forward skeleton with causal attention + MLP
- Validation result:
  - `dotnet build -c Release` passed.
  - `dotnet fsi run-training.fsx --max-tokens 16` now runs full 36-layer path, but output is still repetitive (`!!!!!!!!!!!!!!!`), so semantic parity is not yet reached.
- Current blocker hypothesis:
  - RoPE and exact decode/KV behavior remain missing and likely dominate quality gap.

### Implementation Update #2 (same day, parity track)
- Implemented:
  - added RoPE in attention path (pure F# implementation over `[heads, seq, head_dim]`)
  - switched projection execution from dense dequant matmul to `Q4Linear` NVFP4 path for:
    - `q/k/v/o`
    - `gate/up/down`
    - `lm_head`
  - added BOS prepend and kept assistant-generation prompt template.
- Validation:
  - `run-training.fsx` no longer collapses to token id `0` / repeated `!`.
  - output is now non-trivial but still semantically unstable compared with `run2.fsx`.
- Updated blocker:
  - exact Qwen3 parity still needs stricter alignment for decode/sampling details and KV-path semantics.

## 2026-02-12（中文）
### 背景
- 來源備註：`notes/00001.txt`
- 目標：將 `run-training.fsx` 從「可執行但 scaffold 輸出」推進到語意接近 `run2.fsx`。
- 限制：不可依賴 `Qwen3-4B-Instruct-2507-TorchSharp` 專案與 `Qwen3.dll`。

### 備註/程式審閱結論
- 同一份權重若模型接線、tokenizer、解碼流程不一致，語意品質不會一致。
- 目前 pure F# 推論路徑有四個主要缺口：
  - tokenizer 不一致（byte-token fallback）
  - embedding 不一致（手工特徵）
  - block 接線不一致（scaffold 線性堆疊）
  - 層覆蓋不一致（2 層 fallback）

### 文件修訂
- `doc/SA.md`：新增 parity gap 與 FR-05..FR-08。
- `doc/SD.md`：新增推論一致性設計（`ModelConfigLite`, `Q4WeightBank`, `TokenizerBridge`, `ForwardEngine`）。
- `doc/Test.md`：新增 parity 導向測試補充。
- `doc/WBS.md`：append WBS-12..WBS-17。

### 技術路線
- 短期實作順序：
  1. 對齊 tokenizer（`tokenizer.json`）
  2. 對齊 embedding lookup
  3. 實作 Qwen3-like projection 接線（`q/k/v/o`, `gate/up/down`, `lm_head`）
  4. 補齊全層載入與 parity smoke 測試

### 變更追蹤
- 近期相關 commit：
  - `2e1f3b1`（`Qwen3-4B-Instruct-2507-TorchSharp.fs`）：InferenceBridge 改為 pure F# 明確接線。
  - `a1b686e`（`fsann`）：`run-training.fsx` 僅使用 pure F# inference DLL。

### 實作更新（同日，parity 路線）
- `InferenceBridge.fs` 已補上：
  - 以 `Tokenizers.DotNet` 讀取 `tokenizer.json`
  - 載入 36 層完整投影族群（`q/k/v/o`, `gate/up/down`）與 `lm_head`
  - 從 `.dat` 載入 raw tensor：
    - `model.embed_tokens.weight`
    - 每層 norm 權重（`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`）
    - `model.norm.weight`
  - Qwen3-like forward 骨架（causal attention + MLP）
- 驗證結果：
  - `dotnet build -c Release` 通過。
  - `dotnet fsi run-training.fsx --max-tokens 16` 可跑滿 36 層，但輸出仍偏重複（`!!!!!!!!!!!!!!!`），尚未達到語意 parity。
- 目前阻塞推測：
  - RoPE 與精確 decode/KV 行為尚未補齊，應是語意落差主要來源。

### 實作更新 #2（同日，parity 路線）
- 已補上：
  - attention 路徑 RoPE（pure F#，作用於 `[heads, seq, head_dim]`）
  - 投影計算改走 `Q4Linear` NVFP4 路徑，不再使用 dense dequant matmul：
    - `q/k/v/o`
    - `gate/up/down`
    - `lm_head`
  - 補上 BOS prepend 與 assistant generation prompt template。
- 驗證：
  - `run-training.fsx` 已不再退化成 token id `0` 或連續 `!`。
  - 目前輸出雖然非 trivial，但與 `run2.fsx` 相比語意仍不穩定。
- 最新阻塞：
  - 要達成嚴格 parity，仍需對齊 decode/sampling 細節與 KV-path 語意。

### 實作更新 #3（同日，parity 路線）
- 問題定位：
  - `run-training.fsx` 的 `Q4Linear` NVFP4 路徑語意失真，`run2.fsx` 同權重可正常輸出。
  - 差異點確認為 native interop 路徑：`TorchSharp.Q4.Extension` 原先走 `LibTorchSharp` 的 `THSFP4_quantize/THSTensor_scaled_mm` 包裝；`run2` 實際可用路徑為 `libNVFP4.so` 直接呼叫。
- 修正內容：
  - 在 `TorchSharp.Q4.Extension/NativeInterop.fs` 改為直接使用 `NVFP4_quantize`、`NVFP4_scaled_mm`。
  - `InferenceBridge.fs` 同步對齊：
    - 以 `[B,H,T,D]` 形狀走 `scaled_dot_product_attention`；
    - 取消 BOS prepend，與 `run2` prompt 編碼行為一致；
    - `temperature <= 0` 時改為 greedy (`argmax`)。
- 驗證結果（同 prompt）：
  - `run2.fsx`: 維持可讀語意輸出。
  - `run-training.fsx`: 已從亂碼/多語碎片提升為可讀且語意合理句子：
    - `I’ve never seen a UFO, but I’ve spent countless nights wondering what it would be like to encounter one.`

## 2026-02-12 (run-training2 stall analysis and fix)
### Symptom
- `run-training2.fsx` could appear to "hang" around turn `[3]` or `[5]` when using default settings.

### Root Cause (combined factors)
- Full-prompt replay decode path in `run-training2`: each token step re-runs the entire accumulated prompt, so latency grows per turn.
- Stop token mismatch: generation did not consistently stop on `<|im_end|>` (`151645`) and `<|endoftext|>` (`151643`), increasing long-tail decode.
- Timeout model with `Task.Run + Wait(timeout)`: timeout could throw while the background task kept running, leaving residual CPU/GPU load.
- Host-side append overhead: repeated list concatenation (`list @`) in the generation loop created avoidable O(n^2)-style overhead.

### Changes
- `InferenceBridge.fs`
  - Added explicit stop-token flow for rendered prompt generation.
  - Aligned default stop tokens to `151645/151643`.
  - Replaced per-step list append with `ResizeArray` for running/generated token buffers.
- `run-training2.fsx`
  - Replaced background timeout wrapper with same-thread execution and over-budget warning.
  - Aligned rendered prompt path and stop tokens to `151645/151643`.
  - Reduced default `--max-tokens` from `64` to `20` for full-prompt replay stability.
  - Added per-turn prompt token count timing output.

### Validation
- Condition: `--KVCacheOut false --timing true --max-tokens 20`
- `run2.fsx`: `ELAPSED=28.407s`
- `run-training2.fsx`: `ELAPSED=36.835s`
- Outcome: `[5]` no longer stalls; script runs through `[6]` and exits at designed `stop  here`.

### Change Tracking
- `8d7af00` (`Qwen3-4B-Instruct-2507-TorchSharp.fs`): stabilize rendered prompt generation + stop token behavior.
- `c988023` (`fsann`): make `run-training2` complete reliably on FP4 path.

## 2026-02-12（run-training2 卡住分析與修正）
### 現象
- `run-training2.fsx` 在預設設定下，可能在第 `[3]` 或 `[5]` 輪看起來「卡住」。

### 根因（多因素疊加）
- `run-training2` 採 full-prompt replay：每個 token 都重跑整段累積 prompt，輪次越後面越慢。
- stop token 不一致：未穩定以 `<|im_end|>`（`151645`）與 `<|endoftext|>`（`151643`）停止，增加長尾生成。
- `Task.Run + Wait(timeout)` 超時模型：拋例外後背景任務可能仍在執行，造成 CPU/GPU 殘留負載。
- host 端 append 開銷：生成迴圈反覆 `list @`，帶來可避免的 O(n^2) 級開銷。

### 修正內容
- `InferenceBridge.fs`
  - 新增 rendered prompt 生成的 stop-token 控制流程。
  - 預設 stop token 對齊為 `151645/151643`。
  - 生成 token buffer 由 list append 改為 `ResizeArray`。
- `run-training2.fsx`
  - timeout 包裝改為同執行緒執行，超時只警告不留下背景殘留任務。
  - rendered prompt 與 stop token 對齊為 `151645/151643`。
  - full-prompt replay 預設 `--max-tokens` 由 `64` 降至 `20`。
  - 新增每輪 prompt token 數與時間輸出。

### 驗證
- 條件：`--KVCacheOut false --timing true --max-tokens 20`
- `run2.fsx`：`ELAPSED=28.407s`
- `run-training2.fsx`：`ELAPSED=36.835s`
- 結果：`[5]` 不再卡住，可跑到 `[6]`，並在設計的 `stop  here` 結束。

### 變更追蹤
- `8d7af00`（`Qwen3-4B-Instruct-2507-TorchSharp.fs`）：穩定 rendered prompt 生成與 stop token 行為。
- `c988023`（`fsann`）：修正 `run-training2` 使 FP4 路徑可穩定完成。

## 2026-02-12 (memory fluctuation review from `notes/00002.txt`)
### Symptom
- `run-training2.fsx` showed large and unstable GPU memory footprint (roughly `96~116 GiB` observed by `nvidia-smi` in the reported environment).

### Review Conclusion
- Not all growth indicates a hard leak; a significant part can be CUDA allocator reservation behavior.
- Still, there were identifiable lifecycle gaps that could amplify peak/reserved memory drift.

### Fixes Applied
- NVFP4 kernel temp tensor lifecycle hardening:
  - explicitly scoped `qweight.t()` with `use`.
  - explicitly disposed temporary `inputOnDevice` when created via `.to(...)`.
  - disposed intermediate reshaped tensor on dtype conversion branch.
- Added native cache control hook:
  - exposed `NVFP4_empty_cache` via `NativeInterop.tryEmptyNvfp4Cache()`.
- Added runner-level mitigation:
  - `run-training2.fsx` now supports `--empty-cache-each-turn` (default `true`) to clear allocator cache between turns.
  - KV flags remain available (`--KVCacheOut`, `--TokenByTokenOrPromptByPrompt`).

### Residual Risk
- `run-training2` still rebuilds full prompt history each turn; even with per-turn KV path, long-history prefill pressure can still raise memory peaks.
- A persistent cross-turn KV-cache architecture is still recommended for tighter memory stability.

### Change Tracking
- Q4 extension commit: `7cbed57` (`TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension`)
- Runner commit: `45bdfbf` (`fsann`)
- Note: push status in that execution window was blocked by DNS resolution failure (`github.com` unresolved).

## 2026-02-12（`notes/00002.txt` 記憶體波動審閱）
### 現象
- `run-training2.fsx` 在回報環境中出現明顯且不穩定的顯存占用（`nvidia-smi` 觀測約 `96~116 GiB` 區間）。

### 審閱結論
- 並非所有成長都代表硬性 leak；相當部分可能是 CUDA allocator 的保留策略。
- 但程式確實存在會放大峰值/保留量波動的生命週期缺口，已進行修補。

### 已套用修正
- NVFP4 kernel 暫存 tensor 生命週期強化：
  - `qweight.t()` 明確以 `use` 受控。
  - `.to(...)` 產生的 `inputOnDevice` 在必要時顯式釋放。
  - dtype 轉換分支中的中間 `reshaped` 顯式釋放。
- native cache 控制介面補齊：
  - 透過 `NativeInterop.tryEmptyNvfp4Cache()` 暴露 `NVFP4_empty_cache`。
- runner 緩解機制：
  - `run-training2.fsx` 新增 `--empty-cache-each-turn`（預設 `true`），每輪後可清 allocator cache。
  - 同時保留 KV 參數路徑（`--KVCacheOut`、`--TokenByTokenOrPromptByPrompt`）。

### 殘餘風險
- `run-training2` 仍是「每輪重建 full prompt history」腳本；即便單輪使用 KV，長歷史 prefill 壓力仍可能推高峰值。
- 若要更穩定壓峰值，仍建議改為跨輪持久 KV-cache 架構。

### 變更追蹤
- Q4 extension commit：`7cbed57`（`TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension`）
- runner commit：`45bdfbf`（`fsann`）
- 備註：該次執行窗口因 DNS 解析失敗（`github.com` 無法解析）導致 push 阻塞。

## 2026-02-12 (KVC instability / intermittent segfault follow-up)
### User-reported symptom
- Sequence observed:
  1. run `run-training2.fsx` with `--KVCacheOut false` (often completes to designed `stop here`)
  2. immediately run again (or switch to `--KVCacheOut true`)
  3. intermittently stalls around `[5]` or crashes with SIGSEGV
- Crash signature: stack in `libtorch_cpu.so` (`at::to_copy/copy_`) via `THSTensor_to_device`.

### What changed recently (scope)
- `Qwen3-4B-Instruct-2507-TorchSharp.fs`:
  - `1679c7d`: introduced KV-cache generation path (`generate...KvCache`) and default UM disabled behavior.
- `fsann`:
  - `647f2c3`: wired `run-training2.fsx` to support `KVCacheOut` + prefill mode.
  - `45bdfbf`: enabled per-turn `NVFP4_empty_cache` switch (default true).
  - `30d24c3`: removed reflection workaround; direct `TorchSharp.Q4.Extension` reference.

### Root-cause analysis (current confidence)
- Most likely issue is native allocator pressure/fragility during repeated large tensor host->device transfers in init/load path, not a deterministic KVC logic bug alone.
- Specific defect fixed:
  - `Nvfp4State.readTensorAsByte` created a temporary CPU tensor, copied to CUDA, but did not dispose the CPU temporary in CUDA path.
  - Repeated full-file scans in init can amplify this pressure.

### Fixes applied now
- `Nvfp4State.fs`
  - release temporary CPU tensor immediately after `.to(device)` copy.
- `InferenceBridge.fs`
  - reduce repeated `.dat` scans during init:
    - reuse one load result for `k_proj/v_proj` map extraction.
    - reuse one load result for `gate_proj/up_proj` map extraction.
  - expected impact: fewer allocation spikes and less init-time instability.

### Verification snapshot
- `dotnet build -c Release Qwen3-4B-Instruct-2507-TorchSharp.fs.fsproj`: PASS.
- `run-training2.fsx` smoke:
  - `--KVCacheOut false --timing true`: PASS to designed `stop here`.
  - `--KVCacheOut true --timing true`: PASS to designed `stop here`.

### Open risk
- SIGSEGV is intermittent and native-side; cannot claim complete closure from one pass.
- Need stress loop validation (`N>=10`) for both `KVC on/off` with same prompt set.

## 2026-02-12（KVC 不穩定 / 間歇性 segfault 追蹤）
### 使用者回報現象
- 觀察序列：
  1. 先跑 `run-training2.fsx --KVCacheOut false`（通常可到設計的 `stop here`）
  2. 立即再跑一次（或改成 `--KVCacheOut true`）
  3. 偶發在 `[5]` 卡住，或直接 SIGSEGV
- crash 特徵：`libtorch_cpu.so`（`at::to_copy/copy_`）經由 `THSTensor_to_device`。

### 最近變更範圍
- `Qwen3-4B-Instruct-2507-TorchSharp.fs`：
  - `1679c7d`：加入 KV-cache 生成路徑（`generate...KvCache`）與預設關閉 UM。
- `fsann`：
  - `647f2c3`：`run-training2.fsx` 接入 `KVCacheOut` 與 prefill mode。
  - `45bdfbf`：加入 per-turn `NVFP4_empty_cache`（預設 true）。
  - `30d24c3`：移除反射 workaround，改為直接引用 `TorchSharp.Q4.Extension`。

### 目前根因判斷（信心等級：中）
- 較可能是 init/load 階段重複大量 host->device 轉移帶來的 native allocator 壓力/脆弱性，不是單一可重現的 KVC 邏輯錯誤。
- 已確認並修正的缺陷：
  - `Nvfp4State.readTensorAsByte` 在 CUDA 路徑中，CPU 暫存 tensor `.to(device)` 後未立即釋放。
  - init 期間多次全檔掃描會放大此壓力。

### 本次已套用修正
- `Nvfp4State.fs`
  - `.to(device)` 後立即 `Dispose` CPU 暫存 tensor。
- `InferenceBridge.fs`
  - 減少 init 時重複 `.dat` 掃描：
    - `k_proj/v_proj` 共用一次載入結果再分別抽 map。
    - `gate_proj/up_proj` 共用一次載入結果再分別抽 map。
  - 預期效益：降低配置尖峰與 init 不穩定性。

### 驗證快照
- `dotnet build -c Release Qwen3-4B-Instruct-2507-TorchSharp.fs.fsproj`：PASS。
- `run-training2.fsx` smoke：
  - `--KVCacheOut false --timing true`：可跑到設計的 `stop here`。
  - `--KVCacheOut true --timing true`：可跑到設計的 `stop here`。

### 殘餘風險
- SIGSEGV 屬間歇性 native crash，單次驗證不能宣告完全結案。
- 需補 `N>=10` 壓力回歸（同 prompt、`KVC on/off` 各一組）。

## 2026-02-12 (WBS-16 / WBS-20 closure)
### Deliverables
- Added parity smoke script: `scripts/Tests.Parity.fsx`
- Added KVC stress matrix script: `scripts/Tests.KVCStress.fsx`
- Updated `doc/Test.md` with execution commands and acceptance criteria.
- Updated `doc/WBS.md`: set WBS-16 and WBS-20 to `Done`.

### Results
- `dotnet fsi scripts/Tests.Parity.fsx`
  - PASS (`run2` + `run-training2` first `out:` readable, no segfault, reached designed stop marker)
- `dotnet fsi scripts/Tests.KVCStress.fsx`
  - PASS (`cases=3`, `iterations=3`, `total=9`, no segfault)

## 2026-02-12（WBS-16 / WBS-20 完工）
### 交付內容
- 新增 parity smoke 腳本：`scripts/Tests.Parity.fsx`
- 新增 KVC 壓力矩陣腳本：`scripts/Tests.KVCStress.fsx`
- 更新 `doc/Test.md` 執行方式與驗收標準。
- 更新 `doc/WBS.md`：WBS-16 與 WBS-20 改為 `Done`。

### 驗證結果
- `dotnet fsi scripts/Tests.Parity.fsx`
  - PASS（`run2` 與 `run-training2` 第一個 `out:` 可讀、無 segfault、到達設計 stop）
- `dotnet fsi scripts/Tests.KVCStress.fsx`
  - PASS（`cases=3`、`iterations=3`、`total=9`，無 segfault）

## 2026-02-12 (Managed-UM branch for `TS_Q4_DISABLE_UM=0`)
### Branch
- `feature/um-managed-path-ts-q4-disable-um-0`

### Scope
- Keep existing runtime logic.
- Add managed-memory promotion for persistent inference raw tensors when UM policy is enabled.
- Integrate with upgraded `TorchSharp.Q4.Extension` managed allocator path.

### Code changes
- `InferenceBridge.fs`
  - added `applyUnifiedPolicyToRawMap`.
  - raw tensors from `.dat` now pass through `UnifiedMemory.applyMutablePolicy` under session policy.
  - added init diagnostic line:
    - `[InferInit] UM(raw tensors): managed=<n> total=<m>`

### Validation
- Command:
  - `TS_Q4_DISABLE_UM=0 dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-training2.fsx --max-tokens 1 --timing true`
- Result:
  - PASS to designed `stop here`.
  - Init emitted:
    - `[InferInit] UM(raw tensors): managed=146 total=146`
  - Compatibility check (UM disabled / env unset): PASS to designed `stop here`.

## 2026-02-12（`TS_Q4_DISABLE_UM=0` 的 Managed-UM 分支）
### 分支
- `feature/um-managed-path-ts-q4-disable-um-0`

### 範圍
- 保留既有 runtime 主邏輯。
- 在 UM policy 啟用時，將持久推論 raw tensors 升級為 managed memory。
- 串接升級後的 `TorchSharp.Q4.Extension` managed allocator 路徑。

### 程式調整
- `InferenceBridge.fs`
  - 新增 `applyUnifiedPolicyToRawMap`。
  - `.dat` 載入的 raw tensors 會依 session policy 走 `UnifiedMemory.applyMutablePolicy`。
  - 新增 init 診斷輸出：
    - `[InferInit] UM(raw tensors): managed=<n> total=<m>`

### 驗證
- 指令：
  - `TS_Q4_DISABLE_UM=0 dotnet fsi /workspace/fsann/alpha/runner-arm64-fp4/run-training2.fsx --max-tokens 1 --timing true`
- 結果：
  - 可跑到設計的 `stop here`。
  - init 輸出：
    - `[InferInit] UM(raw tensors): managed=146 total=146`
  - 相容性檢查（UM 關閉 / env 未設定）：可跑到設計的 `stop here`。
