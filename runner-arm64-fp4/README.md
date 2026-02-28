# runner-arm64-fp4 (Workspace Copy)

這個目錄是從 `fsann/alpha/runner-arm64-fp4` 複製進 `Qwen3-4B-Instruct-2507-TorchSharp.fs` 的可重現副本，目的:
- 在同一個專案下保留訓練/推論 runner 腳本
- 方便對照 `Qwen3-4B-Instruct-2507-TorchSharp.fs` 訓練輸出 `.dat` 的實際行為
- 追蹤「F# 之神 (F# God)」任務中全參數訓練可行性

## 本副本清理內容
已移除非必要或高噪音內容:
- `alpha/log/*` runtime log
- bash guard / RAM helper (`run-training-fp2-guarded.sh`, `ram.sh`, `wait_for_ram.sh`)
- 舊小測試殘檔 (`test_ext*.fsx`, `test_fp4_*.fsx`)
- 臨時/歷史雜項 (`@`, `REQ.md`, `analysis_and_optimization_plan.md`, `haystack_source.txt`, `Qwen3-4B.PreciseIncremental.simpleChat.fsx`)

保留核心腳本:
- `run-training-fp2.fsx`
- `run-training2.fsx`
- `run-script-with-guard.fsx`
- `debug-fp2-parity.fsx`
- `run-train-step-full-nvfp4.fsx`

## 建議執行方式
```bash
cd /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/runner-arm64-fp4
dotnet fsi run-script-with-guard.fsx \
  --gpu-limit-gb 108 \
  --gpu-over-secs 0 \
  --gpu-poll-secs 0.5 \
  script run-training2.fsx \
  --weight /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/whoami-1000-seq192-r8-s10-lr1e3.dat \
  --prompt 你是誰 \
  --max-tokens 24 \
  --check-logits false \
  --timing true \
  --stop-here true \
  --KVCacheOut true
```

## `run-training-fp2.fsx` 跑法（補充）

### 前置
先確保本專案已建置（腳本會引用 `../bin/Release/net10.0/*.dll`）：

```bash
cd /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs
dotnet build -c Release
```

### 單輪（建議先這樣）
```bash
cd /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/runner-arm64-fp4
dotnet fsi run-script-with-guard.fsx \
  --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 \
  script run-training-fp2.fsx \
  --model-dir /models/qwen3-4b-instruct-2507-torchsharp \
  --weight /models/qwen3-4b-instruct-2507-torchsharp/Qwen3-4B-Instruct-2507-nvfp4.dat \
  --use-kvc true \
  --kvc-backend fp2-model \
  --turns 1 \
  --prompt 你是誰 \
  --max-tokens 24 \
  --temp 0 \
  --top-p 1 \
  --check-logits false \
  --ifInteractive false \
  --stop-here false
```

### 互動模式
```bash
cd /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/runner-arm64-fp4
dotnet fsi run-script-with-guard.fsx \
  --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 \
  script run-training-fp2.fsx \
  --model-dir /models/qwen3-4b-instruct-2507-torchsharp \
  --weight /models/qwen3-4b-instruct-2507-torchsharp/Qwen3-4B-Instruct-2507-nvfp4.dat \
  --use-kvc true \
  --kvc-backend fp2-model \
  --turns 1 \
  --prompt 你是誰 \
  --max-tokens 24 \
  --temp 0 \
  --top-p 1 \
  --check-logits false \
  --ifInteractive true \
  --stop-here false
```

## `run-training-fp2.fsx` vs `run-training2.fsx` 差異

| 面向 | `run-training2.fsx` | `run-training-fp2.fsx` |
|---|---|---|
| 主要定位 | 推論/對話驗證腳本（Bridge 主路徑） | 訓練模型路徑推論驗證（FP2 model 主路徑） |
| Session 初始化 | `InferenceBridge.init(...)` | 先 `InferenceBridge.initSamplingOnly(...)`，再建立 `Qwen3Model.create(...)` |
| KVC 計算 | 可選 `KVCacheOut` + `pbp/tbt` + `NoKVComputeMode` | `--kvc-backend fp2-model` 下用 `Qwen3Model.forwardWithKvCache` 持久 cache |
| backend 參數 | 無 `kvc-backend` | 有 `--kvc-backend`，目前已限制為 `fp2-model`（`bridge` 會 fail fast） |
| 腳本行為 | 預設先跑 scripted scenario，再可進 interactive | 可單輪/多輪，且可直接進 interactive loop |
| 風險控制 | 主要靠外部 guard | 內建額外 FP2 安全檢查（例如 native quantize 可用性）+ 建議 guard |

實務選擇：
- 要看一般 bridge 推論行為：用 `run-training2.fsx`
- 要看「訓練模型路徑 + FP2 KVC」：用 `run-training-fp2.fsx`

## 本專案內的腳本依賴（已複製）

### 1) 腳本依賴 DLL
`run-training2.fsx`、`run-training-fp2.fsx` 目前引用：
- `../bin/Release/net10.0/Qwen3-4B-Instruct-2507-TorchSharp.fs.dll`
- `../bin/Release/net10.0/TorchSharp.Q4.Extension.dll`
- `../bin/Release/net10.0/TorchSharp.Fun.DGX.dll`（fp2 腳本使用）

因此只要先 `dotnet build -c Release`，runner 副本即可在本專案內運作。

### 2) 模型 JSON / tokenizer（非 `.dat`）
已複製到：
- `/workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/models/qwen3-4b-instruct-2507-torchsharp/config.json`
- `/workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/models/qwen3-4b-instruct-2507-torchsharp/tokenizer.json`
- `/workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/models/qwen3-4b-instruct-2507-torchsharp/tokenizer_config.json`

你可用：
- `--model-dir /models/qwen3-4b-instruct-2507-torchsharp`（系統模型目錄）
- 或 `--model-dir /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/models/qwen3-4b-instruct-2507-torchsharp`（本地副本）

注意：`.dat` 體積大且不納入 repo，仍建議使用 `/models/.../*.dat`。

## F# God 訓練失敗: 前因後果（重點: 全參數是否真的可訓練）

### 目標
驗證「訓練路徑」是否可做全參數更新，且匯出 `.dat` 後仍保有基本語言能力，同時學到:
- 問 `你是誰` 時回 `我是 F# 之神`（或同義）

### 實際觀察
1. 可達成局部目標
- 某些訓練輸出確實能在 `你是誰` prompt 上產出目標語意。

2. 但常伴隨能力崩壞
- 非目標 prompt（例如 UFO）被洗成同一答案。
- 生成重複 token（例如「之神之神...」）問題明顯。

3. 記憶體風險持續存在
- 在 token、chunk、dtype 組合不佳時，step 峰值仍可能衝破 guard。

### 主要原因歸納
1. 資料分佈失衡
- 訓練資料過度聚焦 whoami，導致災難性遺忘與過擬合重複。

2. 訓練超參數過激
- LR/steps/seq-len 組合未充分保守，放大偏移速度。

3. 驗證路徑差異
- `bridge` / `fp2-model` / runner 組態不同，容易出現路徑間觀感不一致。

### 關於「全參數真的有沒有訓練到」
結論: 有。

在 `scripts/Train.WhoAmI.AndExportDat.fsx`，當 `--train-last-layers <= 0`（預設）:
- 使用 `Qwen3Model.namedParameters model` 作為 `trainableNamed`
- 其集合包含:
  - 全部 block projection 權重（q/k/v/o, gate/up/down）
  - `embed_tokens`
  - `final_norm`
  - `lm_head`
- 並將 `trainParams` 全部送進 `Nvfp4Optimizer.create`
- 每步 `zeroGrad/step` 都對這個全參數集合執行

所以問題不是「沒訓練到」，而是「訓練目標與資料設計造成能力偏移」。

## 訓練腳本的資料精度/型態（與 VRAM 關聯）
以下以 `scripts/Train.WhoAmI.AndExportDat.fsx` + `Qwen3Model.fs` + `Nvfp4Optimizer.fs` 的實作為準。

| 階段 | 主要張量型態 | 備註 / 對 VRAM 的意義 |
|---|---|---|
| 磁碟權重 `.dat` | `qdata:uint8` + `scale`（NVFP4 packed） | 儲存小，但訓練時不會一直維持純 packed 計算 |
| 參數載入成可訓練權重 | `master weight` 預設 `bfloat16` | 真正會占駐留記憶體的 trainable 參數 |
| Token ids | `int64` | CE 目標索引必要型態 |
| 前向 hidden/activation | `--compute-dtype`（CUDA 預設 `float16`） | activation 峰值主要來源之一 |
| CE logits 路徑 | hidden 會轉成 `lm_head` dtype 再線性 | dtype 轉換會新增暫存 |
| 梯度（進 optimizer 前） | 先清理 NaN/Inf，再在 step 轉為 `bfloat16` 再 `float32` 計算 | update 核心使用 `float32` 較穩定但吃暫存 |
| Optimizer m/v state | 預設 `nvfp4` packed（可選 `int8`） | 比全浮點 state 更省，但仍有 chunk 內解壓/重打包成本 |
| AdamW update 計算 | 主要在 `float32` | 穩定性較好，但 step 峰值記憶體上升 |
| 參數回寫 | 回寫到 `MasterDType`（通常 `bfloat16`） | 最終常駐參數精度 |
| 匯出 `.dat` | 重新量化為 NVFP4 packed | 便於推論部署，但不代表訓練中記憶體只用 NVFP4 |

### 為什麼會覺得「4B 權重應該只有幾 GB」但訓練卻很大？
因為訓練時同時存在:
- 可訓練 master weights（非純 packed）
- activation / backward graph
- gradient 暫存
- optimizer step chunk 的解壓與更新暫存
- m/v state（即使是 packed 也有運算期展開）

因此訓練峰值記憶體和純推論不可直接類比。

## 實務建議（本副本）
1. 先用 one-step 或小步數驗證 full-parameter path 能穩定跑完。
2. 目標任務（WhoAmI）要混入非目標語料，避免把一般能力洗掉。
3. `seq-len` 與 `step-chunk-rows` 採保守值優先穩定，再逐步放大。
4. 固定 parity 指令做 A/B，避免只看單一路徑就下結論。
