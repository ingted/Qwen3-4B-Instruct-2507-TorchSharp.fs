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

## Managed-UM Runtime Path (2026-02-12)
### Design
- Keep default path unchanged (`TS_Q4_DISABLE_UM` unset => existing defaults apply).
- When `TS_Q4_DISABLE_UM=0`:
  - `Types.fs` enables `PreferUnified`.
  - `InferenceBridge.fs` promotes persistent raw tensors (`embed/norm family`) via `UnifiedMemory.applyMutablePolicy`.
  - `Q4Linear` path (inside Q4 extension) promotes quantized weight bundles through managed allocator policy.

### Observability
- Init prints:
  - `[InferInit] UM(raw tensors): managed=<n> total=<m>`
- This provides direct runtime evidence that raw tensor promotion happened.

### Compatibility
- If UM capability is unavailable, policy falls back without changing non-UM behavior.

## Managed-UM 執行路徑（2026-02-12）
### 設計
- 預設路徑不變（`TS_Q4_DISABLE_UM` 未設定時維持既有預設）。
- 當 `TS_Q4_DISABLE_UM=0`：
  - `Types.fs` 啟用 `PreferUnified`。
  - `InferenceBridge.fs` 會將持久 raw tensors（`embed/norm` 族群）透過 `UnifiedMemory.applyMutablePolicy` 升級。
  - `Q4Linear` 路徑（Q4 extension 內）會依 policy 將量化權重 bundle 升級為 managed allocator 路徑。

### 可觀測性
- init 輸出：
  - `[InferInit] UM(raw tensors): managed=<n> total=<m>`
- 可直接確認 raw tensor 升級是否發生。

### 相容性
- 若 UM 能力不可用，policy 可 fallback，不改變非 UM 行為。

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

## Training Wiring Parity Design (2026-02-14)
### Goal
- Replace scaffold training forward with full Qwen3 block wiring and keep training/inference graph semantics aligned.

### Design
- Introduce shared block-forward module (new logical component):
  - `Qwen3Core.ForwardBlock(...)`
  - Inputs:
    - hidden states
    - layer projection handles (`q/k/v/o`, `gate/up/down`)
    - norm tensors (`input/post/q/k norm`)
    - config (`heads`, `kv_heads`, `head_dim`, `rope_theta`, `rms_eps`)
  - Output:
    - next hidden states
- `InferenceBridge` and `Qwen3Model` both call the same block-forward core.
- Training-specific behavior:
  - keep master trainable weights and STE quant/dequant path.
  - preserve autograd graph in training mode.
- Inference-specific behavior:
  - use `torch.no_grad()` and existing generation APIs.

### Parity Validation Plan
1. Structural parity:
   - assert each layer has required projections and norm tensors.
2. Layer-wise parity:
   - compare hidden states after each block (`max_abs`, `mean_abs`, `cosine`).
   - report first layer index exceeding threshold.
3. Logits parity:
   - compare final logits/top-k under fixed prompt + seed.

### Acceptance
- `Qwen3Model.forward` no longer uses scaffold `List.fold + linearSte` chain.
- Shared block-forward is used by both training and inference paths.
- Layer-wise and logits parity scripts report pass within configured tolerance.

## 訓練接線一致性設計（2026-02-14）
### 目標
- 以完整 Qwen3 block 接線取代訓練 scaffold forward，並讓 train/infer 圖語意對齊。

### 設計
- 新增 shared block-forward 模組（邏輯元件）：
  - `Qwen3Core.ForwardBlock(...)`
  - 輸入：
    - hidden states
    - 各層投影句柄（`q/k/v/o`, `gate/up/down`）
    - norm tensors（`input/post/q/k norm`）
    - config（`heads`, `kv_heads`, `head_dim`, `rope_theta`, `rms_eps`）
  - 輸出：
    - 下一層 hidden states
- `InferenceBridge` 與 `Qwen3Model` 都改呼叫同一份 block-forward core。
- 訓練特化行為：
  - 保留 master trainable weights 與 STE quant/dequant 路徑。
  - 訓練模式保留 autograd graph。
- 推論特化行為：
  - 使用 `torch.no_grad()` 與既有生成 API。

### 一致性驗證計畫
1. 結構一致性：
   - 驗證每層是否具備必要 projection 與 norm tensors。
2. Layer-wise 一致性：
   - 比較每層 hidden state（`max_abs`, `mean_abs`, `cosine`）。
   - 回報第一個超門檻層。
3. Logits 一致性：
   - 固定 prompt + seed，比較最終 logits/top-k。

### 驗收標準
- `Qwen3Model.forward` 不再是 scaffold `List.fold + linearSte` 串接。
- train/infer 皆共用同一份 block-forward。
- layer-wise 與 logits parity 腳本在設定門檻內通過。

## Functional Operator Design For Training Graph (2026-02-14)
### Reference evaluation
- Reviewed and cloned:
  - `TorchSharp.Fun` (`TorchSharp.Fun.fs`): `->>`/`=>>` composition and module-to-model adaptation patterns.
  - `DiffSharp`: `-->` operator usage for both model composition and tensor application.
- Decision:
  - Do not import full external source files into this repo.
  - Implement a minimal training-focused FP operator layer in-project to avoid extra dependency surface.

### New training-only module
- `TrainingFunctional.fs` (new):
  - Type aliases:
    - `TensorOp = torch.Tensor -> torch.Tensor`
  - Operators:
    - `->>`: compose two `TensorOp` stages
    - `-->`: apply tensor to `TensorOp`
  - Graph combinators:
    - `id`, `stage`, `chain`
    - `residual` (for skip connection form)
    - `parallel2` + `merge2` (for branch/merge form)
  - NVFP4 adapters:
    - `linearSte weight outDtype : TensorOp`

### Integration strategy
- Keep inference files unchanged.
- Migrate `Qwen3Model.forward` training path to:
  - build stage list (`linearSte` per trainable layer)
  - compose via `->>` / `chain`
  - execute via `input --> trainingGraph`
- Keep autograd behavior unchanged (functional style only, no numerical-path rewrite).

### Acceptance checks
- `Qwen3Model.forward` has no scaffold `List.fold` wiring.
- Training path uses `TrainingFunctional` operators explicitly.
- Build succeeds and existing training scripts still run.

## 訓練圖 Functional Operator 設計（2026-02-14）
### 參考評估
- 已 clone 並審閱：
  - `TorchSharp.Fun`（`TorchSharp.Fun.fs`）：`->>`/`=>>` 的組線與 module 適配模式。
  - `DiffSharp`：`-->` 在 model 組線與 tensor 套用上的語意。
- 決策：
  - 不直接整包引入外部原始碼到本專案。
  - 在專案內實作最小化、訓練專用的 FP operator 層，降低依賴面與維護風險。

### 新增訓練專用模組
- `TrainingFunctional.fs`（新檔）：
  - 型別別名：
    - `TensorOp = torch.Tensor -> torch.Tensor`
  - Operators：
    - `->>`：組合兩個 `TensorOp` 階段
    - `-->`：將 tensor 套用到 `TensorOp`
  - 圖組合子：
    - `id`, `stage`, `chain`
    - `residual`（skip connection 形式）
    - `parallel2` + `merge2`（branch/merge 形式）
  - NVFP4 適配器：
    - `linearSte weight outDtype : TensorOp`

### 整合策略
- 推論檔案不改。
- `Qwen3Model.forward` 訓練路徑改為：
  - 建立 stage 清單（每層 `linearSte`）
  - 用 `->>` / `chain` 組線
  - 以 `input --> trainingGraph` 執行
- 保持 autograd 行為不變（僅改風格，不改數值路徑）。

### 驗收檢查
- `Qwen3Model.forward` 不再使用 scaffold `List.fold` 接線。
- 訓練路徑明確使用 `TrainingFunctional` operators。
- build 通過，既有訓練腳本可執行。

## Official-Equivalent Wiring Contract Freeze (WBS-26, 2026-02-14)
### Block contract (shape/order/norm path)
- Input hidden state: `[B, T, Hidden]`.
- Attention projection order:
  - `input_rms_norm -> q_proj/k_proj/v_proj`
  - reshape:
    - `q`: `[B, heads, T, head_dim]`
    - `k/v`: `[B, kv_heads, T, head_dim]`
  - `q_norm/k_norm` on transposed view, then RoPE on q/k
  - `k/v` expand from `kv_heads` to `heads`
  - SDPA (causal for full prefill block path)
  - merge heads -> `o_proj`
- Residual path #1: `hidden + attn_out`
- MLP order:
  - `post_attn_rms_norm -> gate_proj/up_proj`
  - `silu(gate) * up`
  - `down_proj`
- Residual path #2: `resid1 + down`

### Shared-core implementation rule
- One shared pure function for no-cache block forward:
  - `Qwen3Core.forwardBlockNoCache`
- Inference no-cache path must call this function directly.
- Training path migration target: call same function with training projection adapters (STE).

## 官方等價接線契約凍結（WBS-26，2026-02-14）
### Block 契約（shape/順序/norm 路徑）
- 輸入 hidden state：`[B, T, Hidden]`。
- Attention 投影順序：
  - `input_rms_norm -> q_proj/k_proj/v_proj`
  - reshape：
    - `q`: `[B, heads, T, head_dim]`
    - `k/v`: `[B, kv_heads, T, head_dim]`
  - 在轉置視圖套 `q_norm/k_norm`，再對 q/k 套 RoPE
  - 將 `k/v` 從 `kv_heads` 擴展到 `heads`
  - SDPA（full prefill block 路徑採 causal）
  - merge heads -> `o_proj`
- 殘差路徑 #1：`hidden + attn_out`
- MLP 順序：
  - `post_attn_rms_norm -> gate_proj/up_proj`
  - `silu(gate) * up`
  - `down_proj`
- 殘差路徑 #2：`resid1 + down`

### Shared-core 實作規則
- no-cache block forward 使用同一個 pure function：
  - `Qwen3Core.forwardBlockNoCache`
- 推論 no-cache 路徑必須直接呼叫此函式。
- 訓練路徑遷移目標：以 training projection adapters（STE）呼叫同一函式。
