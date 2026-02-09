# STE Design Note

## English

### Verdict
Your explanation is mostly correct.
The key correction is:
- Current training path uses **STE with quantize->dequantize->dense linear** for autograd.
- It is **not** direct FP4-kernel matmul during backward training.
- Direct NVFP4 kernel matmul exists in backend inference path (`nvfp4-kernel`), not in `linearSte`.

### Why STE is required
Quantization includes rounding and discrete codebook mapping.
These operations are non-smooth / effectively non-differentiable for gradient descent.
Without STE, gradients through quantization collapse or become unusable.

### Mathematical idea
- Forward:
  - use quantized behavior: `w_q = Q(w)`
- Backward:
  - approximate `dQ/dw ~= 1` (identity pass-through)

In practice:
`w_ste = w + (dequant(Q(w)) - w).detach()`

This gives:
- forward value equals dequantized quantized weight
- gradient w.r.t. `w` behaves like identity

### Implementation mapping in this repo

1. Quantize to NVFP4 packed tensors  
File: `../TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/Nvfp4Training.fs`

```fsharp
let quantizePacked (input: TorchSharp.torch.Tensor) =
  let useNative =
    x2d.device_type = DeviceType.CUDA
    && NativeInterop.hasLibTorchFp4Quantize()
  if useNative then NativeInterop.fp4Quantize x2d
  else Nvfp4TrainingImpl.fallbackQuantizePacked x2d
```

2. Dequantize packed NVFP4 back to dense tensor

```fsharp
let dequantizePacked (qdata: Tensor) (scale: Tensor) (outDtype: ScalarType) =
  // decode nibbles -> indices -> codebook values -> apply block scale
  let dense = ...
  dense.contiguous().clone()
```

3. Core STE trick (`detach`)

```fsharp
let steWeight (masterWeight: TorchSharp.torch.Tensor) =
  let q, s = quantizePacked masterWeight
  use dq = dequantizePacked q s masterWeight.dtype
  masterWeight + (dq - masterWeight).detach()
```

4. Layer forward with STE weight

```fsharp
let linearSte (input: Tensor) (masterWeight: Tensor) (outDtype: ScalarType) =
  use wSte = steWeight masterWeight
  let output = torch.nn.functional.linear(input, wSte)
  if output.dtype = outDtype then output else output.to_type(outDtype)
```

5. Trainer runs true backward+optimizer on master weights  
File: `Trainer.fs`

```fsharp
optimizer.zero_grad()
use output = model.Forward(inputTensor, outDtype = computeDtype)
use loss = scalarLoss output targetTensor
loss.backward()
optimizer.step() |> ignore
```

### Current design boundaries
- `linearSte` is training-oriented and autograd-friendly.
- Kernel path (`NativeInterop.scaledMmFp4`) is used in backend kernel route for runtime/inference validation.
- Alignment constraints still apply (`in_features % 16 = 0`).

### Stability notes
- Gradient clipping is **not** implemented yet in current trainer.
- CPU training uses `float32`; CUDA uses `float16` compute dtype in current config.

---

## 中文

### 結論
你的解釋大方向正確。
但要補一個關鍵修正：
- 目前訓練路徑是 **STE 的 quantize->dequantize->dense linear**，用來保留 autograd。
- 目前 **不是** 在 backward 訓練時直接用 FP4 kernel matmul。
- 直接 NVFP4 kernel matmul 存在於 backend 的 `nvfp4-kernel` 路徑（偏 runtime/inference）。

### 為什麼需要 STE
量化包含 rounding 與離散 codebook 映射。
這些動作對梯度下降來說不可微或近似不可用。
若不使用 STE，梯度會消失或無法有效更新權重。

### 數學直覺
- Forward：
  - 使用量化行為：`w_q = Q(w)`
- Backward：
  - 近似 `dQ/dw ~= 1`（把梯度當作恆等映射直通）

實作上常見形式：
`w_ste = w + (dequant(Q(w)) - w).detach()`

效果：
- forward 值等於量化後再反量化的權重
- backward 對 `w` 的梯度近似 identity 傳遞

### 在本專案的實作對應

1. 將權重量化為 NVFP4 packed  
檔案：`../TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/Nvfp4Training.fs`

```fsharp
let quantizePacked (input: TorchSharp.torch.Tensor) =
  let useNative =
    x2d.device_type = DeviceType.CUDA
    && NativeInterop.hasLibTorchFp4Quantize()
  if useNative then NativeInterop.fp4Quantize x2d
  else Nvfp4TrainingImpl.fallbackQuantizePacked x2d
```

2. 將 packed NVFP4 反量化回 dense tensor

```fsharp
let dequantizePacked (qdata: Tensor) (scale: Tensor) (outDtype: ScalarType) =
  // nibble decode -> index -> codebook value -> block scale
  let dense = ...
  dense.contiguous().clone()
```

3. STE 核心（`detach`）技巧

```fsharp
let steWeight (masterWeight: TorchSharp.torch.Tensor) =
  let q, s = quantizePacked masterWeight
  use dq = dequantizePacked q s masterWeight.dtype
  masterWeight + (dq - masterWeight).detach()
```

4. 以 STE 權重做 layer forward

```fsharp
let linearSte (input: Tensor) (masterWeight: Tensor) (outDtype: ScalarType) =
  use wSte = steWeight masterWeight
  let output = torch.nn.functional.linear(input, wSte)
  if output.dtype = outDtype then output else output.to_type(outDtype)
```

5. Trainer 確實跑 backward + optimizer（更新 master weights）  
檔案：`Trainer.fs`

```fsharp
optimizer.zero_grad()
use output = model.Forward(inputTensor, outDtype = computeDtype)
use loss = scalarLoss output targetTensor
loss.backward()
optimizer.step() |> ignore
```

### 目前設計邊界
- `linearSte` 是訓練導向、可微分友善路徑。
- kernel 路徑（`NativeInterop.scaledMmFp4`）目前用在 backend kernel runtime/inference 驗證。
- 仍有對齊限制（`in_features % 16 = 0`）。

### 穩定性補充
- 目前 trainer **尚未**實作 gradient clipping。
- 現行配置：CPU 以 `float32` 訓練；CUDA 以 `float16` 計算。
