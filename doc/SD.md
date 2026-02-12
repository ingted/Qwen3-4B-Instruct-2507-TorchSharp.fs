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

## Inference Parity Design (2026-02-12)
### Scope
- Build a pure F# inference path with no `Qwen3.dll` dependency.
- Align runtime semantics with `run2.fsx` directionally by matching tokenizer, layer families, and decode flow.

### Components
- `InferenceBridge.fs`
  - `ModelConfigLite`: parse `config.json` fields (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `eos_token_id`).
  - `Q4WeightBank`: load required NVFP4 tensors by layer family:
    - `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`
    - `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`
    - `lm_head`
  - `TokenizerBridge`: use `Tokenizers.DotNet` over `tokenizer.json`.
  - `ForwardEngine`: explicit block wiring (`embedding -> attn projections -> mlp projections -> lm_head`).

### Data Flow (Inference)
1. Load config/tokenizer/weights.
2. Encode prompt using `tokenizer.json`.
3. Build token embeddings from tied `lm_head` rows.
4. Run per-layer Qwen3-like projection flow.
5. Compute logits with `lm_head`.
6. Decode generated token ids using tokenizer.

### Validation Strategy
- Compare `run-training.fsx` output with `run2.fsx` under same prompt/seed/device/quant.
- Track parity at two levels:
  - lexical readability (non-garbled decode)
  - semantic closeness (manual/spot-check before full metric automation)

### Implementation Status Snapshot
- Implemented:
  - tokenizer-based encode/decode (`tokenizer.json`)
  - raw fp16 tensor loading for `embed_tokens` and norm weights
  - 36-layer projection wiring (`q/k/v/o`, `gate/up/down`) with causal attention skeleton
- Not yet implemented:
  - RoPE position encoding (critical for attention semantics)
  - exact Qwen3 KV-cache execution path
  - full parity sampler behavior

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

## 推論一致性設計（2026-02-12）
### 範圍
- 建立 pure F# 推論路徑，不依賴 `Qwen3.dll`。
- 透過對齊 tokenizer、權重族群、解碼流程，讓語意品質方向性接近 `run2.fsx`。

### 元件
- `InferenceBridge.fs`
  - `ModelConfigLite`：解析 `config.json` 必要欄位（`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `eos_token_id`）。
  - `Q4WeightBank`：按 layer family 載入 NVFP4 權重：
    - `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`
    - `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`
    - `lm_head`
  - `TokenizerBridge`：使用 `Tokenizers.DotNet` 讀取 `tokenizer.json`。
  - `ForwardEngine`：明確接線（`embedding -> attn projections -> mlp projections -> lm_head`）。

### 推論資料流
1. 載入 config/tokenizer/weights。
2. 使用 `tokenizer.json` 編碼 prompt。
3. 由 tied `lm_head` 權重列建立 token embeddings。
4. 跑逐層 Qwen3-like projection 流程。
5. 以 `lm_head` 計算 logits。
6. 用 tokenizer 解碼生成 token ids。

### 驗證策略
- 在相同 prompt/seed/device/quant 下，對照 `run-training.fsx` 與 `run2.fsx` 輸出。
- 分兩層追蹤：
  - 可讀性（非亂碼）
  - 語意接近度（先人工 spot-check，後續再補指標化）

### 目前實作狀態
- 已完成：
  - 使用 `tokenizer.json` 的 encode/decode
  - 載入 `embed_tokens` 與 norm 類 raw fp16 權重
  - 36 層投影接線（`q/k/v/o`, `gate/up/down`）與 causal attention 骨架
- 尚未完成：
  - RoPE 位置編碼（對注意力語意至關重要）
  - 完整對齊 Qwen3 的 KV-cache 執行路徑
  - 取樣器細節與 parity 完整對齊

## Runtime Stability Hardening (2026-02-12)
### Problem
- Intermittent SIGSEGV in repeated `run-training2.fsx` runs, especially around mode switches (`KVCacheOut false -> true`).
- Native stack indicates crash in tensor host->device transfer path (`THSTensor_to_device` / `at::to_copy`).

### Design adjustments
- Loader lifecycle discipline:
  - release temporary CPU tensors immediately after CUDA copy in `.dat` parser path.
- Init scan reduction:
  - avoid duplicate full-file scans for same dimension groups.
  - reuse one parsed state for:
    - `k_proj` + `v_proj`
    - `gate_proj` + `up_proj`
- Operational guardrails:
  - keep `--empty-cache-each-turn` as explicit runtime switch.
  - require stress test matrix (`KVC on/off`, repeated runs) before declaring stable.

### Expected effect
- Lower allocation spikes during init.
- Reduce allocator fragmentation pressure and intermittent native crash probability.
- Faster model init due to fewer `.dat` passes.

## 執行期穩定性強化（2026-02-12）
### 問題
- `run-training2.fsx` 重複執行時出現間歇性 SIGSEGV，特別是 `KVCacheOut false -> true` 切換後。
- native stack 指向 host->device 張量搬移路徑（`THSTensor_to_device` / `at::to_copy`）。

### 設計調整
- Loader 生命週期約束：
  - `.dat` parser 在 CUDA copy 後立即釋放 CPU 暫存 tensor。
- Init 掃描減量：
  - 避免同維度群組的重複全檔掃描。
  - 下列群組改共用一次解析結果：
    - `k_proj` + `v_proj`
    - `gate_proj` + `up_proj`
- 執行防護：
  - `--empty-cache-each-turn` 維持顯式參數控制。
  - 在宣告穩定前，要求壓力測試矩陣（`KVC on/off` + repeated runs）。

### 預期效果
- 降低 init 階段配置尖峰。
- 降低 allocator fragmentation 壓力與間歇性 native crash 機率。
- 減少 `.dat` 掃描次數，加快初始化。
