# Architecture

## Goals
This project is a pure F# training scaffold derived from `Qwen3-4B-Instruct-2507-TorchSharp-mod`.
Core goals:
- Use `FAkka.TorchSharp.DGX 26.1.0-py3.9`.
- Use `TorchSharp.Q4.Extension` for Q4 weight handling, quantization, and compute.
- Enforce a pure `NVFP4` path (`KernelOnly` + `nvfp4-kernel`).
- Serve as the primary codebase for later full-parameter FP4 training.

## Layers
- `Cli.fs`
  - Parse CLI arguments and produce `TrainingConfig`.
- `Types.fs`
  - Centralize types and defaults.
  - Define pure NVFP4 defaults for `Q4SessionConfig` and `Q4Schema`.
- `Nvfp4State.fs`
  - Load NVFP4 weight state.
  - Synthetic loader exists.
  - Real `.dat` streaming parser exists and selects layer pairs by requested dimensions.
- `Qwen3Model.fs`
  - Build `TorchSharp.Q4.Extension` session diagnostics.
  - Convert NVFP4 bundles into trainable master weights.
  - Execute forward with `Nvfp4Training.linearSte`.
- `Trainer.fs`
  - Training loop with `forward + loss + backward + optimizer`.
  - Checkpoint save/recover support.
- `Program.fs`
  - Entry point and error boundary.

## Dependencies
- NuGet:
  - `FAkka.TorchSharp.DGX 26.1.0-py3.9`
- Project reference:
  - `../TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/TorchSharp.Q4.Extension.fsproj`

## Constraints
- No fallback to NF4 (this project is NVFP4-first by design).
- `ComputePath = KernelOnly`.
- `BackendOverride = nvfp4-kernel`.
- If native backend is unavailable, fail fast (no silent fallback).

## Next Steps
- Align real Qwen3 full-parameter/layer mapping.
- Add optimizer-state resume and LR scheduler.

## 目標
本專案是 `Qwen3-4B-Instruct-2507-TorchSharp-mod` 的純 F# 版訓練骨架，核心目標：
- 使用 `FAkka.TorchSharp.DGX 26.1.0-py3.9`。
- 透過 `TorchSharp.Q4.Extension` 處理 Q4 權重/量化/計算。
- 強制走 pure `NVFP4` 路徑（`KernelOnly` + `nvfp4-kernel`）。
- 作為後續「全參數 FP4 訓練」的主工程。

## 架構分層
- `Cli.fs`
  - 解析 CLI 參數，產生 `TrainingConfig`。
- `Types.fs`
  - 集中型別與預設值。
  - 定義 `Q4SessionConfig` 與 `Q4Schema` 的 pure NVFP4 預設。
- `Nvfp4State.fs`
  - 載入 NVFP4 權重狀態。
  - 提供 synthetic loader。
  - 真實 `.dat` streaming parser 已實作，會依請求維度選擇 layer pair。
- `Qwen3Model.fs`
  - 建立 `TorchSharp.Q4.Extension` session diagnostics。
  - 將 NVFP4 bundle 轉成可訓練 master weight。
  - 以前向 `Nvfp4Training.linearSte` 執行運算。
- `Trainer.fs`
  - 訓練迴圈含 `forward + loss + backward + optimizer`。
  - 支援 checkpoint 儲存與 recover。
- `Program.fs`
  - 入口與錯誤處理。

## 依賴關係
- NuGet:
  - `FAkka.TorchSharp.DGX 26.1.0-py3.9`
- Project reference:
  - `../TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/TorchSharp.Q4.Extension.fsproj`

## 關鍵約束
- 不回退到 NF4（此專案預設/設計目標為 NVFP4）。
- `ComputePath = KernelOnly`。
- `BackendOverride = nvfp4-kernel`。
- 若 native 不可用，直接 fail fast（避免 silent fallback）。

## 下一步
- 對齊 Qwen3 真實全參數層級 mapping。
- 補 optimizer state resume 與 LR scheduler。
