# SD (System Design)

## Design Principles
- Pure F# at the application layer.
- NVFP4-first.
- Diagnostics-first (all fallback/unavailable conditions must be observable).

## Module Design
- `Types.fs`
  - `TrainingConfig`, defaults, pure NVFP4 Q4 settings.
- `Cli.fs`
  - CLI -> `TrainingConfig`.
- `Nvfp4State.fs`
  - `load : TrainingConfig -> Nvfp4ModelState`.
  - v1: synthetic.
  - v2: real `.dat` parser.
- `Qwen3Model.fs`
  - `create : TrainingConfig -> Nvfp4ModelState -> Qwen3Nvfp4Model`.
  - Manage `Q4Session` diagnostics + trainable master-weight layers.
  - Forward path uses `Nvfp4Training.linearSte`.
- `Trainer.fs`
  - `run : TrainingConfig -> Qwen3Nvfp4Model -> unit`.
  - Executes forward/loss/backward/optimizer update.
  - Supports save/recover checkpoint (metadata + layer tensors).
- `Program.fs`
  - App entry + exception boundary.

## Data Flow
1. Parse CLI config.
2. Load NVFP4 state.
3. Create Q4 session.
   - CUDA runtime: `KernelOnly` + `nvfp4-kernel`.
   - CPU runtime: `DequantMatmulOnly` + `dequant-matmul` fallback.
4. Build model layers.
5. Run training loop with optimizer update.
6. Save checkpoint by step/epoch policy.

## Error Handling
- Startup failures (config/native/schema): fail fast.
- Training failures: unified reporting at `Program` boundary.

## Future Extensions
- Parser: `.dat` -> exact Qwen3 full-layer mapping.
- Optimizer: add optimizer-state serialization and resume.
- Scheduler: learning-rate schedule and warmup.

## 設計原則
- Pure F# at app layer。
- NVFP4-first。
- Diagnostics-first（任何 fallback/不可用都可觀測）。

## 模組設計
- `Types.fs`
  - `TrainingConfig`, defaults, pure NVFP4 Q4 settings。
- `Cli.fs`
  - CLI -> `TrainingConfig`。
- `Nvfp4State.fs`
  - `load : TrainingConfig -> Nvfp4ModelState`。
  - v1: synthetic。
  - v2: real `.dat` parser。
- `Qwen3Model.fs`
  - `create : TrainingConfig -> Nvfp4ModelState -> Qwen3Nvfp4Model`。
  - 管理 `Q4Session` diagnostics + 可訓練 master-weight layers。
  - forward 使用 `Nvfp4Training.linearSte`。
- `Trainer.fs`
  - `run : TrainingConfig -> Qwen3Nvfp4Model -> unit`。
  - 執行 forward/loss/backward/optimizer update。
  - 支援 checkpoint 儲存與 recover（metadata + layer tensor）。
- `Program.fs`
  - app entry + exception boundary。

## 資料流
1. CLI 解析設定。
2. 載入 NVFP4 state。
3. 建立 Q4 session。
   - CUDA runtime: `KernelOnly` + `nvfp4-kernel`。
   - CPU runtime: `DequantMatmulOnly` + `dequant-matmul` fallback。
4. 建立模型層。
5. 跑訓練 loop 並更新參數。
6. 依 step/epoch 政策儲存 checkpoint。

## 錯誤處理策略
- 啟動階段（配置、native、schema）失敗：直接 fail fast。
- 訓練階段失敗：在 `Program` 一層統一回報。

## 後續擴充點
- Parser：`.dat` -> Qwen3 全層精確 mapping。
- Optimizer：補 optimizer state 序列化與 resume。
- Scheduler：加入 learning rate schedule/warmup。
