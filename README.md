# Qwen3-4B-Instruct-2507-TorchSharp.fs

Pure F# project scaffold for Qwen3 NVFP4 training workflow.

## What "Scaffold" Means
- The current project started as a training scaffold: a minimum runnable structure to validate NVFP4 loading, STE backward, optimizer update, and checkpoint loop before full Qwen3 block wiring is migrated.
- In scaffold mode, `Qwen3Model.forward` uses a simplified linear stack (`List.fold + Nvfp4Training.linearSte`) to reduce variables during kernel/backend bring-up.
- This reduced architecture was chosen first to isolate low-level issues (layout, scale mapping, kernel fallback, CUDA init, checkpoint safety) before introducing full attention/MLP/RoPE complexity.
- For pure F# inference wiring, use `InferenceBridge` in this project: it performs explicit layer connections in a BERT-like flow (`embeddings -> blocks -> lm_head`) and does not reference any external Qwen3 C# assembly.
- 若要使用純 F# 的推論接線，請使用本專案中的 `InferenceBridge`：它以類似 BERT 的明確流程（`embeddings -> blocks -> lm_head`）連接各層，且不依賴任何外部 Qwen3 C# 組件。

## Goals
- F# only (no C# project dependency in this app layer)
- Use `FAkka.TorchSharp.DGX` `26.1.0-py3.6`
- Use `TorchSharp.Q4.Extension` for NVFP4 quantize/dequantize + STE linear training path
- Force pure NVFP4 policy (`BackendOverride=nvfp4-kernel`, `ComputePath=KernelOnly`)

## Current status
- Build/run ready.
- Includes synthetic NVFP4 training with full `forward/loss/backward/optimizer`.
- Real Qwen3 NVFP4 `.dat` streaming parser is implemented in `Nvfp4State.fs`.
  It loads paired `*.qdata` + `*.scale` tensors from `.dat` and selects layers by requested dimensions.
- Checkpoint/recover is implemented in `Trainer.fs` (layer weights + metadata).
- Runtime behavior:
  - `cuda*` device: keep `nvfp4-kernel` + `KernelOnly`.
  - `cpu` device: auto switch to `dequant-matmul` for functional fallback tests.

## Build
```bash
cd /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs
dotnet build -c Release
```

## Run (synthetic smoke)
```bash
dotnet run -c Release -- --synthetic true --device cuda --epochs 1 --steps-per-epoch 1 --batch-size 1 --in-features 64 --out-features 64
```

## Run (real NVFP4 dat smoke)
```bash
dotnet run -c Release -- --synthetic false --device cpu --epochs 1 --steps-per-epoch 1 --batch-size 1 --max-layers 1 --in-features 2560 --out-features 2560
```

## Checkpoint CLI
```bash
dotnet run -c Release -- \
  --synthetic true \
  --device cpu \
  --epochs 2 \
  --steps-per-epoch 1 \
  --lr 0.0001 \
  --checkpoint-dir /tmp/qwen3-fs-ckpt \
  --save-every-steps 1 \
  --resume true
```

## Main files
- `Types.fs`: config, default paths, pure NVFP4 Q4 session/schema defaults
- `Cli.fs`: command-line parsing
- `Nvfp4State.fs`: NVFP4 state loading (synthetic + real `.dat` streaming parser)
- `Qwen3Model.fs`: trainable NVFP4 layer stack (master weights + STE forward)
- `InferenceBridge.fs`: pure F# inference bridge (`embeddings -> blocks -> lm_head`)
- `Trainer.fs`: training loop with optimizer + checkpoint/recover
- `Program.fs`: app entrypoint
