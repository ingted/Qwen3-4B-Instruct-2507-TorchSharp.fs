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
