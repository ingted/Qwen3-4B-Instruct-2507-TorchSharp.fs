# NVFP4 Data Path (Storage vs Compute DType)

## Purpose
This document keeps a stable, snippet-based reference for how NVFP4 is stored and computed in:
- `fsann/Qwen3-4B-Instruct-2507-TorchSharp-mod` (`Qwen3.FP4.Extension` path)
- `Qwen3-4B-Instruct-2507-TorchSharp.fs` (`TorchSharp.Q4.Extension` path)

It avoids line-number-only references so future refactors remain traceable.

## Quick Summary
- Weight file on disk is compact NVFP4 (`qdata` + `scale`).
- In-memory prepared weight stays quantized (packed + blocked scale).
- Compute path for kernel mode is: activation quantize -> `scaled_mm` -> output cast to requested dtype.
- Output dtype is usually `float16` by runner default (`--dtype float16`), not “always bf16”.
- Float32 mostly appears in debug/fallback paths (A/B check or dequant-matmul fallback).

## Comparison Table (with key snippets)
| Stage | `Qwen3-...-mod` (C# + `Qwen3.FP4.Extension`) | `InferenceBridge` (`TorchSharp.Q4.Extension`) |
|---|---|---|
| Weight file keys | `var qKey = prefix + ".qdata"; var sKey = prefix + ".scale";` | `match format with NVFP4 -> prefix + ".qdata", Some (prefix + ".scale"), None, None` |
| In-memory prepared weight | `_qweight <- qweight.detach().contiguous().to(Device(device))` and `_scale <- Fp4Ops.to_blocked ...` | `new PreparedNvfp4KernelWeight(... packed, scaleBlocked, inFeatures, outFeatures)` |
| Activation quantize | `let qinput0, iscale0 = Fp4Ops.quantize in2d` | `let qInputRaw, inputScaleRaw = NativeInterop.fp4Quantize input2d` |
| Kernel matmul | `Fp4Ops.scaled_mm qinput (_qweight.t()) iscale_swiz _scale input.dtype` | `NativeInterop.scaledMmFp4 qInput qweightT inputScaleBlocked scaleBlocked outDtype` |
| Default runner output dtype | `run2.fsx` default args include `--dtype float16` | `run-training2.fsx` default args include `--dtype float16` |
| Float/bf16 appearance | Output follows `input.dtype`; debug A/B uses `to_type(Float32)` | Kernel output uses `outDtype`; fallback uses float compute (`linear`) |
| Full float fallback | `deqW = Fp4Ops.dequantize_weight ...` + `torch.matmul(inputF, deqWF...)` (A/B) | `let output = torch.nn.functional.linear(inputForCompute, weightForCompute)` |

## Canonical Snippets

### A) `Qwen3.FP4.Extension` kernel path
```fsharp
// fsann/Qwen3.FP4.Extension/Library.fs
_qweight <- qweight.detach().contiguous().``to``(Device(device)).MoveToOuterDisposeScope()
_scale <- Fp4Ops.to_blocked scaleTmp

let qinput0, iscale0 = Fp4Ops.quantize in2d
use qinput = qinput0
use iscale = iscale0
use iscale_swiz = Fp4Ops.to_blocked iscale
let outTmp = Fp4Ops.scaled_mm qinput (_qweight.t()) iscale_swiz _scale input.dtype
```

### B) `TorchSharp.Q4.Extension` kernel path
```fsharp
// TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/Backend.fs
let qInputRaw, inputScaleRaw = NativeInterop.fp4Quantize input2d
use qInput = qInputRaw
use inputScale = inputScaleRaw
use inputScaleBlocked = toBlockedScale inputScale
use qweightT = qweight.t()
use out2d = NativeInterop.scaledMmFp4 qInput qweightT inputScaleBlocked scaleBlocked outDtype
```

### C) `TorchSharp.Q4.Extension` fallback float path
```fsharp
// TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/Backend.fs
let computeDtype = ensureComputeDtype inputOnWeightDevice
let inputForCompute = if inputOnWeightDevice.dtype = computeDtype then inputOnWeightDevice else inputOnWeightDevice.to_type(computeDtype)
let weightForCompute = if dense.dtype = computeDtype then dense else dense.to_type(computeDtype)
let output = torch.nn.functional.linear(inputForCompute, weightForCompute)
```

### D) Runner defaults (`float16`)
```fsharp
// fsann/alpha/runner-arm64-fp4/run2.fsx
"--dtype"; "float16"

// fsann/alpha/runner-arm64-fp4/run-training2.fsx
"--dtype"; "float16"
```

## Notes
- “NVFP4 weights” means packed representation for weights/scales. It does not imply every runtime tensor is FP4.
- Activation is quantized at runtime for kernel path.
- If backend falls back to dequant-matmul, memory/latency can increase significantly.

---

# NVFP4 資料路徑（儲存格式 vs 計算 DType）

## 目的
本文件提供可長期維護的「片段級」對照，說明 NVFP4 在以下兩條路徑中的儲存與計算：
- `fsann/Qwen3-4B-Instruct-2507-TorchSharp-mod`（`Qwen3.FP4.Extension` 路徑）
- `Qwen3-4B-Instruct-2507-TorchSharp.fs`（`TorchSharp.Q4.Extension` 路徑）

重點是不依賴行號，避免未來重構後失去對照價值。

## 摘要
- 權重檔磁碟格式是壓縮 NVFP4（`qdata` + `scale`）。
- 記憶體中的 prepared weight 仍維持量化型態（packed + blocked scale）。
- Kernel 計算路徑是：activation 量化 -> `scaled_mm` -> 轉成要求的輸出 dtype。
- Runner 預設輸出 dtype 通常是 `float16`（`--dtype float16`），不是固定 bf16。
- Float32 多出現在 debug/fallback（A/B 檢查或 dequant-matmul fallback）。

## 對照表（含關鍵片段）
| 階段 | `Qwen3-...-mod`（C# + `Qwen3.FP4.Extension`） | `InferenceBridge`（`TorchSharp.Q4.Extension`） |
|---|---|---|
| 權重檔 key | `var qKey = prefix + ".qdata"; var sKey = prefix + ".scale";` | `match format with NVFP4 -> prefix + ".qdata", Some (prefix + ".scale"), None, None` |
| 記憶體 prepared weight | `_qweight <- qweight.detach().contiguous().to(Device(device))` 與 `_scale <- Fp4Ops.to_blocked ...` | `new PreparedNvfp4KernelWeight(... packed, scaleBlocked, inFeatures, outFeatures)` |
| Activation 量化 | `let qinput0, iscale0 = Fp4Ops.quantize in2d` | `let qInputRaw, inputScaleRaw = NativeInterop.fp4Quantize input2d` |
| Kernel matmul | `Fp4Ops.scaled_mm qinput (_qweight.t()) iscale_swiz _scale input.dtype` | `NativeInterop.scaledMmFp4 qInput qweightT inputScaleBlocked scaleBlocked outDtype` |
| Runner 預設輸出 dtype | `run2.fsx` defaultArgs 含 `--dtype float16` | `run-training2.fsx` defaultArgs 含 `--dtype float16` |
| float/bf16 出現位置 | 輸出跟 `input.dtype`；A/B debug 會 `to_type(Float32)` | kernel 輸出跟 `outDtype`；fallback 走 float `linear` |
| 全浮點 fallback | `deqW = Fp4Ops.dequantize_weight ...` + `torch.matmul(inputF, deqWF...)`（A/B） | `let output = torch.nn.functional.linear(inputForCompute, weightForCompute)` |

## 標準程式碼片段

### A) `Qwen3.FP4.Extension` kernel 路徑
```fsharp
// fsann/Qwen3.FP4.Extension/Library.fs
_qweight <- qweight.detach().contiguous().``to``(Device(device)).MoveToOuterDisposeScope()
_scale <- Fp4Ops.to_blocked scaleTmp

let qinput0, iscale0 = Fp4Ops.quantize in2d
use qinput = qinput0
use iscale = iscale0
use iscale_swiz = Fp4Ops.to_blocked iscale
let outTmp = Fp4Ops.scaled_mm qinput (_qweight.t()) iscale_swiz _scale input.dtype
```

### B) `TorchSharp.Q4.Extension` kernel 路徑
```fsharp
// TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/Backend.fs
let qInputRaw, inputScaleRaw = NativeInterop.fp4Quantize input2d
use qInput = qInputRaw
use inputScale = inputScaleRaw
use inputScaleBlocked = toBlockedScale inputScale
use qweightT = qweight.t()
use out2d = NativeInterop.scaledMmFp4 qInput qweightT inputScaleBlocked scaleBlocked outDtype
```

### C) `TorchSharp.Q4.Extension` fallback 浮點路徑
```fsharp
// TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/Backend.fs
let computeDtype = ensureComputeDtype inputOnWeightDevice
let inputForCompute = if inputOnWeightDevice.dtype = computeDtype then inputOnWeightDevice else inputOnWeightDevice.to_type(computeDtype)
let weightForCompute = if dense.dtype = computeDtype then dense else dense.to_type(computeDtype)
let output = torch.nn.functional.linear(inputForCompute, weightForCompute)
```

### D) Runner 預設（`float16`）
```fsharp
// fsann/alpha/runner-arm64-fp4/run2.fsx
"--dtype"; "float16"

// fsann/alpha/runner-arm64-fp4/run-training2.fsx
"--dtype"; "float16"
```

## 備註
- 「權重是 NVFP4」代表權重/scale 的壓縮表示，不代表每個執行期 tensor 都是 FP4。
- Kernel 路徑下 activation 是每次 forward 動態量化。
- 若掉到 dequant-matmul fallback，延遲與記憶體占用通常會明顯上升。
