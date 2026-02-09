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
