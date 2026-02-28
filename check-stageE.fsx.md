# check-stageE.fsx

`check-stageE.fsx` 用來做 StageE 權重的快速驗證，並同時支援兩種執行模式：

- `bridge`：對齊 `run-training2.fsx`（InferenceBridge 路徑）
- `fp2-model`：對齊 `run-training-fp2.fsx`（Qwen3Model.create + FP2 模型路徑）

## 1) 直接執行（不加 guard）

```bash
cd /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs

dotnet fsi check-stageE.fsx \
  --mode=bridge \
  --weight=/workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/stageE-enhanced-alignment.dat \
  --prompt=你是誰 \
  --max-tokens=8 \
  --temp=0 \
  --use-kvc=true \
  --kvc-input-mode=pbp
```

```bash
cd /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs

dotnet fsi check-stageE.fsx \
  --mode=fp2-model \
  --weight=/workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/stageE-enhanced-alignment.dat \
  --prompt=你是誰 \
  --max-tokens=8 \
  --temp=0 \
  --use-kvc=true \
  --kvc-input-mode=pbp
```

## 2) 用 guard 執行（建議）

這是穩定做法，避免 VRAM 峰值拖垮機器。

```bash
cd /workspace/fsann/alpha/runner-arm64-fp4

dotnet fsi run-script-with-guard.fsx \
  --gpu-limit-gb 108 \
  --gpu-over-secs 0 \
  --gpu-poll-secs 0.05 \
  script /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/check-stageE.fsx \
  --mode=fp2-model \
  --weight=/workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/stageE-enhanced-alignment.dat \
  --prompt=你是誰 \
  --max-tokens=8 \
  --temp=0 \
  --use-kvc=true \
  --kvc-input-mode=pbp
```

`bridge` 對照測試只要改 `--mode=bridge`。

## 3) 常用參數

- `--mode=bridge|fp2-model`
- `--weight=<dat 路徑>`
- `--prompt=<測試 prompt>`
- `--max-tokens=<輸出 token 上限>`
- `--temp=<溫度，常用 0>`
- `--use-kvc=true|false`
- `--kvc-input-mode=pbp|tbt`

## 4) 建議測試順序

1. 先跑 `--mode=bridge` 確認權重可讀、輸出可解碼。
2. 再跑 `--mode=fp2-model` 驗證訓練路徑接線。
3. 全程用 `run-script-with-guard.fsx` 監控，避免 OOM 造成 host 不穩。
