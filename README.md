# Qwen3-4B-Instruct-2507-TorchSharp.fs

Pure F# project scaffold for Qwen3 NVFP4 training workflow.

## Goals
- F# only (no C# project dependency in this app layer)
- Use `FAkka.TorchSharp.DGX` `26.1.0-py3.6`
- Use `TorchSharp.Q4.Extension` for Q4 schema/backend/session/linear handling
- Force pure NVFP4 policy (`BackendOverride=nvfp4-kernel`, `ComputePath=KernelOnly`)

## Current status
- Build/run ready.
- Includes synthetic NVFP4 training smoke loop.
- Real Qwen3 NVFP4 `.dat` parser is still TODO in `Nvfp4State.fs` (`--synthetic false` will fail fast by design).
- Optimizer/update path for true full-parameter training is TODO in `Trainer.fs`.

## Build
```bash
cd /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs
dotnet build -c Release
```

## Run (synthetic smoke)
```bash
dotnet run -c Release -- --synthetic true --device cuda --epochs 1 --steps-per-epoch 1 --batch-size 1 --in-features 64 --out-features 64
```

## Main files
- `Types.fs`: config, default paths, pure NVFP4 Q4 session/schema defaults
- `Cli.fs`: command-line parsing
- `Nvfp4State.fs`: NVFP4 state loading (synthetic implemented, real parser TODO)
- `Qwen3Model.fs`: Q4 session + linear stack model wrapper
- `Trainer.fs`: training loop scaffold
- `Program.fs`: app entrypoint
