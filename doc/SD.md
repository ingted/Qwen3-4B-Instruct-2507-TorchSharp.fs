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
  - Manage `Q4Session` + `Q4Linear` layers.
- `Trainer.fs`
  - `run : TrainingConfig -> Qwen3Nvfp4Model -> unit`.
  - v1: forward/loss.
  - v2: backward/update/checkpoint.
- `Program.fs`
  - App entry + exception boundary.

## Data Flow
1. Parse CLI config.
2. Load NVFP4 state.
3. Create Q4 session (`KernelOnly`, `nvfp4-kernel`).
4. Build model layers.
5. Run training loop.

## Error Handling
- Startup failures (config/native/schema): fail fast.
- Training failures: unified reporting at `Program` boundary.

## Future Extensions
- Parser: `.dat` -> tensor bundle mapping.
- Optimizer: AdamW/SGD.
- Checkpoint: weights + optimizer state.

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
  - 管理 `Q4Session` + `Q4Linear` layers。
- `Trainer.fs`
  - `run : TrainingConfig -> Qwen3Nvfp4Model -> unit`。
  - v1: forward/loss。
  - v2: backward/update/checkpoint。
- `Program.fs`
  - app entry + exception boundary。

## 資料流
1. CLI 解析設定。
2. 載入 NVFP4 state。
3. 建立 Q4 session (`KernelOnly`, `nvfp4-kernel`)。
4. 建立模型層。
5. 跑訓練 loop。

## 錯誤處理策略
- 啟動階段（配置、native、schema）失敗：直接 fail fast。
- 訓練階段失敗：在 `Program` 一層統一回報。

## 後續擴充點
- Parser：`.dat` -> tensor bundle mapping。
- Optimizer：AdamW/SGD。
- Checkpoint：weights + optimizer state。
