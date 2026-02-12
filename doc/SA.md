# SA (System Analysis)

## Background
The target is a pure F# Qwen3-4B training project, avoiding C# in the application layer, and standardizing on:
- `FAkka.TorchSharp.DGX 26.1.0-py3.6`
- `TorchSharp.Q4.Extension`

## Problem Statement
A runnable NVFP4 training scaffold exists, but the following are still missing:
- Real NVFP4 weight file (`.dat`) parser and mapping.
- Backward + optimizer update path (key for full-parameter training).
- Checkpoint and recovery flow.

## Requirement Breakdown
- Functional requirements
  - FR-01: Start model session in pure NVFP4 mode (`KernelOnly`).
  - FR-02: Load weights (synthetic first, real `.dat` next).
  - FR-03: Run training loop (forward/loss/backward/update).
  - FR-04: Provide reproducible test script.
- Non-functional requirements
  - NFR-01: Pure F# project layer.
  - NFR-02: Fail fast to avoid silent fallback.
  - NFR-03: Complete docs and tracking (Architecture/SA/SD/WBS/Test).

## Risks and Mitigations
- Risk: NVFP4 kernel/native environment mismatch.
  - Mitigation: Pre-start `Backend.diagnose` + fail fast.
- Risk: Parser and layer mapping mismatch.
  - Mitigation: Validate flow with synthetic data first, then integrate real parser layer-by-layer.
- Risk: Memory/precision issues in training update path.
  - Mitigation: Incremental rollout (forward -> backward -> optimizer) with tests.

## Current Conclusion
The current scaffold can be iterated directly, with priority on real parser and full training update path.

## Inference Parity Gap (2026-02-12)
- GAP-01 Tokenizer mismatch:
  - Current `run-training.fsx` path used UTF-8 byte-token fallback instead of `tokenizer.json`.
  - Impact: token ids and decoded text diverge from `run2.fsx`, making semantic parity impossible.
- GAP-02 Embedding mismatch:
  - Current path used handcrafted feature projection rather than model-consistent embedding lookup.
  - Impact: hidden-state distribution is not aligned with trained weights.
- GAP-03 Block wiring mismatch:
  - Current path was scaffold-style (`linearSte + gelu + residual`) and did not represent full Qwen3 block semantics.
  - Impact: output quality collapses even when weight file is correct.
- GAP-04 Layer coverage mismatch:
  - Default config limited load to 2 layers and relied on fallback dimensions.
  - Impact: forward path under-utilizes model capacity and shifts behavior.

## Updated Requirements For Inference Parity
- FR-05: Use `tokenizer.json` for encode/decode, no byte-token fallback in normal inference path.
- FR-06: Replace handcrafted input features with model-consistent token embedding lookup.
- FR-07: Implement explicit Qwen3-like block wiring in pure F# (`attn projections + mlp projections + lm_head`), no `Qwen3.dll`.
- FR-08: Load all required layer families (`q/k/v/o`, `gate/up/down`, `lm_head`) and avoid 2-layer fallback behavior.

## Runtime Stability Gap (2026-02-12)
- GAP-05 Intermittent native crash under repeated runs:
  - Symptom: occasional SIGSEGV after back-to-back `run-training2.fsx` runs, often near KVC mode switches.
  - Signal path: native tensor transfer (`THSTensor_to_device` / `at::to_copy`).

## Additional Requirements (Stability)
- FR-09: Enforce strict temporary tensor disposal in `.dat` load path (especially CPU->CUDA conversion temps).
- FR-10: Reduce repeated full-file `.dat` scans for same-dimension layer groups.
- FR-11: Add repeated-run stress validation for both `KVCacheOut=true/false` before marking KVC stable.

## 背景
目標是建立一個純 F# 的 Qwen3-4B 訓練工程，避免在應用層混入 C#，並統一使用：
- `FAkka.TorchSharp.DGX 26.1.0-py3.6`
- `TorchSharp.Q4.Extension`

## 問題定義
目前已具備可跑的 NVFP4 訓練骨架，但尚未完成：
- 真實 NVFP4 權重檔 (`.dat`) 的 parser 與 mapping。
- backward + optimizer 更新（全參數訓練關鍵）。
- checkpoint 與恢復流程。

## 需求拆解
- 功能需求
  - FR-01: 以 pure NVFP4 啟動模型 session（KernelOnly）。
  - FR-02: 載入權重（先 synthetic，後 real `.dat`）。
  - FR-03: 跑訓練 loop（forward/loss/backward/update）。
  - FR-04: 提供可重現的測試腳本。
- 非功能需求
  - NFR-01: 純 F# 專案層。
  - NFR-02: fail fast，避免 silent fallback。
  - NFR-03: 文件與追蹤（Architecture/SA/SD/WBS/Test）完整。

## 風險與對策
- 風險: NVFP4 kernel/native 環境不一致。
  - 對策: 啟動前 `Backend.diagnose` + fail fast。
- 風險: 權重 parser 與 layer mapping 對不上。
  - 對策: 先 synthetic 驗證流程，再逐層導入 real parser。
- 風險: 訓練更新路徑導致顯存/精度問題。
  - 對策: 分段導入（forward -> backward -> optimizer）並加測試。

## 當前結論
可以先在目前骨架上迭代，優先補齊 real parser 與 full training update path。

## 推論一致性缺口（2026-02-12）
- GAP-01 Tokenizer 不一致：
  - `run-training.fsx` 目前走 UTF-8 byte-token fallback，未使用 `tokenizer.json`。
  - 影響：token id 與解碼文字會和 `run2.fsx` 偏離，無法達成語意一致。
- GAP-02 Embedding 不一致：
  - 目前用手工 feature 投影，不是模型一致的 embedding lookup。
  - 影響：hidden state 分佈與訓練權重不對齊。
- GAP-03 Block 接線不一致：
  - 目前是 scaffold 形式（`linearSte + gelu + residual`），不是完整 Qwen3 block 語意。
  - 影響：即使權重正確，輸出語意品質仍會崩解。
- GAP-04 層覆蓋不一致：
  - 預設僅載入 2 層且依賴 fallback 維度。
  - 影響：forward 只用到模型少量容量，行為偏移。

## 推論一致性更新需求
- FR-05：正常推論路徑必須使用 `tokenizer.json` encode/decode，不再使用 byte-token fallback。
- FR-06：移除手工輸入特徵，改為模型一致的 token embedding lookup。
- FR-07：以 pure F# 實作明確 Qwen3-like block 接線（`attn projections + mlp projections + lm_head`），不得依賴 `Qwen3.dll`。
- FR-08：完整載入必要權重族群（`q/k/v/o`, `gate/up/down`, `lm_head`），不再停留在 2 層 fallback。

## 執行期穩定性缺口（2026-02-12）
- GAP-05 重複執行下的間歇性 native crash：
  - 現象：`run-training2.fsx` 連跑時偶發 SIGSEGV，常見於 KVC 模式切換後。
  - 訊號路徑：native tensor transfer（`THSTensor_to_device` / `at::to_copy`）。

## 新增穩定性需求
- FR-09：`.dat` 載入路徑必須嚴格釋放暫存 tensor（特別是 CPU->CUDA 轉移後的 CPU 暫存）。
- FR-10：同維度層群的 `.dat` 全檔掃描要去重，避免重複掃描。
- FR-11：在宣告 KVC 穩定前，需完成 `KVCacheOut=true/false` 的 repeated-run 壓力驗證。
