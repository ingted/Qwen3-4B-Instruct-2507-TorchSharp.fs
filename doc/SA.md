# SA (System Analysis)

## Background
The target is a pure F# Qwen3-4B training project, avoiding C# in the application layer, and standardizing on:
- `FAkka.TorchSharp.DGX 26.1.0-py3.7`
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

## UM Path Gap (2026-02-12)
- GAP-06 Existing UM policy toggle did not guarantee managed allocator usage for persistent inference tensors.
- GAP-07 `TS_Q4_DISABLE_UM=0` needed an explicit runtime contract to verify managed-memory coverage.

## Additional Requirements (UM)
- FR-12: Under `TS_Q4_DISABLE_UM=0`, persistent inference tensors must be promoted through managed allocator path.
- FR-13: Runtime init must emit managed coverage diagnostics for raw tensors.
- FR-14: Keep default behavior unchanged when UM is disabled (compatibility first).

## 背景
目標是建立一個純 F# 的 Qwen3-4B 訓練工程，避免在應用層混入 C#，並統一使用：
- `FAkka.TorchSharp.DGX 26.1.0-py3.7`
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

## UM 路徑缺口（2026-02-12）
- GAP-06 現有 UM policy 開關不保證持久推論 tensors 真正走 managed allocator。
- GAP-07 `TS_Q4_DISABLE_UM=0` 需要明確 runtime 契約與可觀測的 managed 覆蓋率。

## 新增 UM 需求
- FR-12：在 `TS_Q4_DISABLE_UM=0` 下，持久推論 tensors 必須經 managed allocator 升級。
- FR-13：runtime init 必須輸出 raw tensor 的 managed 覆蓋率診斷。
- FR-14：UM 關閉時維持既有行為（相容性優先）。

## Training-Wiring Parity Gap (2026-02-14)
- GAP-08 Training forward is still scaffold-only:
  - `Qwen3Model.forward` currently uses linear stack (`List.fold + linearSte`) instead of full Transformer block wiring.
- GAP-09 Train/Infer graph drift risk:
  - inference uses Qwen3-like block wiring in `InferenceBridge`, but training path does not share the same block implementation.
- GAP-10 Inference runtime still differs from official C# pipeline:
  - session-level KVC bookkeeping, template/stop handling, and generation lifecycle are not fully unified with `run2` path.

## Additional Requirements (Training/Parity)
- FR-15: Training forward path must adopt full Qwen3 block wiring (`q/k/v/o`, RoPE, SDPA, RMSNorm, MLP gate/up/down, residual).
- FR-16: Build one shared block-forward core used by both training and inference to prevent semantic drift.
- FR-17: Add layer-wise hidden-state parity test to detect first divergence layer.
- FR-18: Add final logits parity acceptance check (same prompt/seed/dtype/weights) against run2 baseline behavior.

## Functional-Style Migration Gap (2026-02-14)
- GAP-11 Training graph style mismatch:
  - training path still defines wiring as imperative/scaffold `List.fold`, not operator-composed pipeline style like TorchSharp.Fun / DiffSharp examples.
- GAP-12 No reusable functional operators for complex graph topology:
  - branch/merge/residual composition is not represented as first-class training operators.
- GAP-13 Training-path FP convention is not codified:
  - no explicit contract describing accepted operator style (`->>`, `-->`) and how it maps to STE/Q4 blocks.

## Additional Requirements (Functional-Style Training)
- FR-19: Introduce a training-only functional composition module with pipeline operators (`->>`, `-->`) and reusable graph combinators.
- FR-20: Refactor `Qwen3Model.forward` training path to build and execute the graph via operator composition (no OO-style training wiring helpers).
- FR-21: Keep inference runtime behavior unchanged while migrating training-path graph style.
- FR-22: Document TorchSharp.Fun/DiffSharp reference evaluation and chosen adaptation in `DevLog/SD/WBS`.

## 功能式風格遷移缺口（2026-02-14）
- GAP-11 訓練圖風格不一致：
  - 訓練路徑目前仍是 imperative/scaffold 的 `List.fold`，非 TorchSharp.Fun / DiffSharp 那種 operator pipeline 形式。
- GAP-12 缺少可重用的功能式複雜接線運算子：
  - branch/merge/residual 還沒有被抽象成第一類訓練運算子。
- GAP-13 訓練路徑 FP 規範未明確化：
  - 尚未定義 `->>`、`-->` 這類 operator 與 STE/Q4 block 的對應規格。

## 新增功能式訓練需求
- FR-19：新增 training-only 的 functional composition 模組，提供 pipeline operator（`->>`, `-->`）與可重用圖組合子。
- FR-20：重構 `Qwen3Model.forward` 訓練路徑，改為 operator 組線執行（不再使用 OO-style 訓練接線 helper）。
- FR-21：遷移訓練風格時，推論 runtime 行為保持不變。
- FR-22：將 TorchSharp.Fun/DiffSharp 參考評估與採用策略寫入 `DevLog/SD/WBS`。

## 訓練接線一致性缺口（2026-02-14）
- GAP-08 訓練 forward 仍是 scaffold：
  - `Qwen3Model.forward` 目前仍為線性堆疊（`List.fold + linearSte`），不是完整 Transformer block。
- GAP-09 Train/Infer 圖存在漂移風險：
  - 推論在 `InferenceBridge` 已是 Qwen3-like block，但訓練路徑尚未共用同一份 block 實作。
- GAP-10 推論 runtime 仍與官方 C# pipeline 有差異：
  - session 級 KVC bookkeeping、template/stop 處理與 generation lifecycle 尚未完全同構 `run2` 路徑。

## 新增訓練/一致性需求
- FR-15：訓練 forward 必須改為完整 Qwen3 block 接線（`q/k/v/o`, RoPE, SDPA, RMSNorm, MLP gate/up/down, residual）。
- FR-16：建立 train/infer 共用 block-forward core，避免語意漂移。
- FR-17：新增 layer-wise hidden-state parity 測試，定位第一個失真層。
- FR-18：新增最終 logits parity 驗收（同 prompt/seed/dtype/weights），對照 run2 基線行為。
