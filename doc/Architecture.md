# Architecture

## 目標
本專案是 `Qwen3-4B-Instruct-2507-TorchSharp-mod` 的純 F# 版訓練骨架，核心目標：
- 使用 `FAkka.TorchSharp.DGX 26.1.0-py3.6`
- 透過 `TorchSharp.Q4.Extension` 處理 Q4 權重/量化/計算
- 強制走 pure `NVFP4` 路徑（`KernelOnly` + `nvfp4-kernel`）
- 作為後續「全參數 FP4 訓練」的主工程

## 架構分層
- `Cli.fs`
  - 解析 CLI 參數，產生 `TrainingConfig`
- `Types.fs`
  - 集中型別與預設值
  - 定義 `Q4SessionConfig` 與 `Q4Schema` 的 pure NVFP4 預設
- `Nvfp4State.fs`
  - 載入 NVFP4 權重狀態
  - 目前先提供 synthetic loader
  - 真實 `.dat` parser 後續補齊
- `Qwen3Model.fs`
  - 建立 `TorchSharp.Q4.Extension` session
  - 將 layer bundle 轉成 `Q4Linear`
  - 前向傳播（forward）
- `Trainer.fs`
  - 訓練迴圈骨架（目前 forward + loss）
  - optimizer/update path 後續補齊
- `Program.fs`
  - 入口與錯誤處理

## 依賴關係
- NuGet:
  - `FAkka.TorchSharp.DGX 26.1.0-py3.6`
- Project reference:
  - `../TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/TorchSharp.Q4.Extension.fsproj`

## 關鍵約束
- 不回退到 NF4（此專案預設/設計目標為 NVFP4）
- `ComputePath = KernelOnly`
- `BackendOverride = nvfp4-kernel`
- 若 native 不可用，直接 fail fast（避免 silent fallback）

## 下一步
- 實作真實 NVFP4 `.dat` parser
- 實作 optimizer + backward + checkpoint
- 對齊 Qwen3 真實模型層級參數 mapping
