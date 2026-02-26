namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System
open TorchSharp
open TorchSharp.Modules
open TorchSharp.Q4.Extension

type PackedNvfp4 =
  {
    mutable QData: torch.Tensor
    mutable Scale: torch.Tensor
    Shape: int64 array
    StorageDevice: string
  }

type PackedInt8 =
  {
    mutable QData: torch.Tensor
    mutable Scale: torch.Tensor
    Shape: int64 array
    StorageDevice: string
  }

type PackedStateMode =
  | Nvfp4
  | Int8

type PackedMomentState =
  | MomentNvfp4 of PackedNvfp4
  | MomentInt8 of PackedInt8

type PackedParamState =
  {
    Name: string
    Param: Parameter
    mutable W: PackedNvfp4
    mutable M: PackedMomentState
    mutable V: PackedMomentState
  }

type PackedAdamwConfig =
  {
    Device: string
    MasterDType: torch.ScalarType
    LearningRate: float32
    Beta1: float32
    Beta2: float32
    Eps: float32
    WeightDecay: float32
    StateMode: PackedStateMode
    StepChunkRows: int64
    OffloadMVToCpu: bool
    OffloadWToCpu: bool
    OffloadGradToCpu: bool
    FlushEachParam: bool
  }

type PackedAdamwState(config: PackedAdamwConfig, states: PackedParamState list) =
  let mutable step = 0

  let disposeNvfp4 (x: PackedNvfp4) =
    x.QData.Dispose()
    x.Scale.Dispose()

  let disposeInt8 (x: PackedInt8) =
    x.QData.Dispose()
    x.Scale.Dispose()

  let disposeMoment (m: PackedMomentState) =
    match m with
    | MomentNvfp4 x -> disposeNvfp4 x
    | MomentInt8 x -> disposeInt8 x

  member _.Config = config
  member _.States = states
  member _.Step
    with get () = step
    and set value = step <- value

  interface IDisposable with
    member _.Dispose() =
      for st in states do
        disposeNvfp4 st.W
        disposeMoment st.M
        disposeMoment st.V

module Nvfp4Optimizer =
  let private toMatrix2d (t: torch.Tensor) =
    let shape = t.shape |> Array.copy
    let view =
      match shape.Length with
      | 1 -> t.reshape([| 1L; shape.[0] |]).contiguous()
      | 2 -> t.contiguous()
      | _ ->
        let cols = shape |> Array.skip 1 |> Array.fold (fun acc d -> acc * d) 1L
        t.reshape([| shape.[0]; cols |]).contiguous()
    view, shape

  let private shapeToRowsCols (shape: int64 array) =
    match shape.Length with
    | 0 -> 1L, 1L
    | 1 -> 1L, shape.[0]
    | 2 -> shape.[0], shape.[1]
    | _ ->
      let cols = shape |> Array.skip 1 |> Array.fold (fun acc d -> acc * d) 1L
      shape.[0], cols

  let private packNvfp4 (source: torch.Tensor) (storageDevice: string) : PackedNvfp4 =
    let matrix, shape = toMatrix2d source
    try
      let q, s = Nvfp4Training.quantizePacked matrix
      try
        let qStored =
          if q.device.ToString() = storageDevice then
            q.contiguous().clone()
          else
            use moved = q.``to``(device = storageDevice)
            moved.contiguous().clone()

        let sStored =
          if s.device.ToString() = storageDevice then
            s.contiguous().clone()
          else
            use moved = s.``to``(device = storageDevice)
            moved.contiguous().clone()

        {
          QData = qStored
          Scale = sStored
          Shape = shape
          StorageDevice = storageDevice
        }
      finally
        q.Dispose()
        s.Dispose()
    finally
      matrix.Dispose()

  let private packMatrixNvfp4ToStorage (matrix2d: torch.Tensor) (storageDevice: string) =
    use matrix = matrix2d.contiguous()
    let q, s = Nvfp4Training.quantizePacked matrix
    try
      let qStored =
        if q.device.ToString() = storageDevice then
          q.contiguous().clone()
        else
          use moved = q.``to``(device = storageDevice)
          moved.contiguous().clone()

      let sStored =
        if s.device.ToString() = storageDevice then
          s.contiguous().clone()
        else
          use moved = s.``to``(device = storageDevice)
          moved.contiguous().clone()
      qStored, sStored
    finally
      q.Dispose()
      s.Dispose()

  let private packInt8 (source: torch.Tensor) (storageDevice: string) : PackedInt8 =
    let matrix, shape = toMatrix2d source
    try
      let q, s =
        use matrix32 =
          if matrix.dtype = torch.float32 then matrix.contiguous()
          else matrix.to_type(torch.float32).contiguous()
        use absMax0 = torch.amax(matrix32.abs(), dims = [| 1L |], keepdim = true)
        use eps = torch.tensor(1e-8f, dtype = torch.float32, device = absMax0.device)
        use absMax = torch.maximum(absMax0, eps)
        use scale = absMax / 127.0f
        use normalized = matrix32 / scale
        use rounded = torch.round(normalized)
        use low = torch.tensor(-127.0f, dtype = torch.float32, device = rounded.device)
        use high = torch.tensor(127.0f, dtype = torch.float32, device = rounded.device)
        use clampedLow = torch.maximum(rounded, low)
        use clamped = torch.minimum(clampedLow, high)
        let q0 = clamped.to_type(torch.int8).contiguous()
        let s0 = scale.to_type(torch.float32).contiguous()
        q0.clone(), s0.clone()
      try
        let qStored =
          if q.device.ToString() = storageDevice then
            q.contiguous().clone()
          else
            use moved = q.``to``(device = storageDevice)
            moved.contiguous().clone()
        let sStored =
          if s.device.ToString() = storageDevice then
            s.contiguous().clone()
          else
            use moved = s.``to``(device = storageDevice)
            moved.contiguous().clone()
        {
          QData = qStored
          Scale = sStored
          Shape = shape
          StorageDevice = storageDevice
        }
      finally
        q.Dispose()
        s.Dispose()
    finally
      matrix.Dispose()

  let private packMatrixInt8ToStorage (matrix2d: torch.Tensor) (storageDevice: string) =
    use matrix = matrix2d.contiguous()
    use matrix32 =
      if matrix.dtype = torch.float32 then matrix.contiguous()
      else matrix.to_type(torch.float32).contiguous()
    use absMax0 = torch.amax(matrix32.abs(), dims = [| 1L |], keepdim = true)
    use eps = torch.tensor(1e-8f, dtype = torch.float32, device = absMax0.device)
    use absMax = torch.maximum(absMax0, eps)
    use scale = absMax / 127.0f
    use normalized = matrix32 / scale
    use rounded = torch.round(normalized)
    use low = torch.tensor(-127.0f, dtype = torch.float32, device = rounded.device)
    use high = torch.tensor(127.0f, dtype = torch.float32, device = rounded.device)
    use clampedLow = torch.maximum(rounded, low)
    use clamped = torch.minimum(clampedLow, high)
    use q0 = clamped.to_type(torch.int8).contiguous()
    use s0 = scale.to_type(torch.float32).contiguous()

    let qStored =
      if q0.device.ToString() = storageDevice then
        q0.contiguous().clone()
      else
        use moved = q0.``to``(device = storageDevice)
        moved.contiguous().clone()

    let sStored =
      if s0.device.ToString() = storageDevice then
        s0.contiguous().clone()
      else
        use moved = s0.``to``(device = storageDevice)
        moved.contiguous().clone()

    qStored, sStored

  let private unpackNvfp4Rows2d
    (packed: PackedNvfp4)
    (rowStart: int64)
    (rowLen: int64)
    (computeDevice: string)
    (outDtype: torch.ScalarType) =
    use qSlice0 = packed.QData.narrow(0L, rowStart, rowLen).contiguous()
    use sSlice0 = packed.Scale.narrow(0L, rowStart, rowLen).contiguous()

    let qTemp =
      if qSlice0.device.ToString() = computeDevice then
        None
      else
        Some(qSlice0.``to``(device = computeDevice))
    let sTemp =
      if sSlice0.device.ToString() = computeDevice then
        None
      else
        Some(sSlice0.``to``(device = computeDevice))

    let q =
      match qTemp with
      | Some t -> t
      | None -> qSlice0
    let s =
      match sTemp with
      | Some t -> t
      | None -> sSlice0
    try
      use dense2d = Nvfp4Training.dequantizePacked q s outDtype
      dense2d.contiguous().clone()
    finally
      qTemp |> Option.iter (fun t -> t.Dispose())
      sTemp |> Option.iter (fun t -> t.Dispose())

  let private unpackInt8Rows2d
    (packed: PackedInt8)
    (rowStart: int64)
    (rowLen: int64)
    (computeDevice: string)
    (outDtype: torch.ScalarType) =
    use qSlice0 = packed.QData.narrow(0L, rowStart, rowLen).contiguous()
    use sSlice0 = packed.Scale.narrow(0L, rowStart, rowLen).contiguous()

    let qTemp =
      if qSlice0.device.ToString() = computeDevice then
        None
      else
        Some(qSlice0.``to``(device = computeDevice))
    let sTemp =
      if sSlice0.device.ToString() = computeDevice then
        None
      else
        Some(sSlice0.``to``(device = computeDevice))

    let q =
      match qTemp with
      | Some t -> t
      | None -> qSlice0
    let s =
      match sTemp with
      | Some t -> t
      | None -> sSlice0

    try
      use dense32 = q.to_type(torch.float32) * s.to_type(torch.float32)
      if outDtype = torch.float32 then dense32.contiguous().clone()
      else dense32.to_type(outDtype).contiguous().clone()
    finally
      qTemp |> Option.iter (fun t -> t.Dispose())
      sTemp |> Option.iter (fun t -> t.Dispose())

  let private unpackMomentRows2d
    (moment: PackedMomentState)
    (rowStart: int64)
    (rowLen: int64)
    (computeDevice: string)
    (outDtype: torch.ScalarType) =
    match moment with
    | MomentNvfp4 x -> unpackNvfp4Rows2d x rowStart rowLen computeDevice outDtype
    | MomentInt8 x -> unpackInt8Rows2d x rowStart rowLen computeDevice outDtype

  let private replacePackedNvfp4 (target: PackedNvfp4) (next: PackedNvfp4) =
    target.QData.Dispose()
    target.Scale.Dispose()
    target.QData <- next.QData
    target.Scale <- next.Scale

  let private replaceMoment (current: PackedMomentState) (next: PackedMomentState) =
    let disposeMoment m =
      match m with
      | MomentNvfp4 x ->
        x.QData.Dispose()
        x.Scale.Dispose()
      | MomentInt8 x ->
        x.QData.Dispose()
        x.Scale.Dispose()
    disposeMoment current
    next

  let zeroGrad (parameters: Parameter list) =
    for p in parameters do
      if not (isNull p.grad) then
        p.grad.zero_() |> ignore

  let create
    (config: PackedAdamwConfig)
    (parameters: Parameter list)
    (nameOf: Parameter -> string)
    : PackedAdamwState =
    let mvStorageDevice =
      if config.OffloadMVToCpu then "cpu" else config.Device
    let wStorageDevice =
      if config.OffloadWToCpu then "cpu" else config.Device

    let states =
      parameters
      |> List.map (fun p ->
        use p0 = p.detach().contiguous()
        let wPacked = packNvfp4 p0 wStorageDevice
        use zeros = torch.zeros(p0.shape, dtype = p0.dtype, device = p0.device)
        let mPacked =
          match config.StateMode with
          | PackedStateMode.Nvfp4 -> MomentNvfp4(packNvfp4 zeros mvStorageDevice)
          | PackedStateMode.Int8 -> MomentInt8(packInt8 zeros mvStorageDevice)
        let vPacked =
          match config.StateMode with
          | PackedStateMode.Nvfp4 -> MomentNvfp4(packNvfp4 zeros mvStorageDevice)
          | PackedStateMode.Int8 -> MomentInt8(packInt8 zeros mvStorageDevice)
        {
          Name = nameOf p
          Param = p
          W = wPacked
          M = mPacked
          V = vPacked
        })
    new PackedAdamwState(config, states)

  let stateSizeMiB (state: PackedAdamwState) =
    let tensorMiB (t: torch.Tensor) =
      int ((t.NumberOfElements * t.ElementSize) / 1024L / 1024L)

    let momentMiB (m: PackedMomentState) =
      match m with
      | MomentNvfp4 st -> tensorMiB st.QData + tensorMiB st.Scale
      | MomentInt8 st -> tensorMiB st.QData + tensorMiB st.Scale

    let w =
      state.States
      |> List.sumBy (fun st -> tensorMiB st.W.QData + tensorMiB st.W.Scale)
    let m = state.States |> List.sumBy (fun st -> momentMiB st.M)
    let v = state.States |> List.sumBy (fun st -> momentMiB st.V)
    w, m, v

  let step (state: PackedAdamwState) =
    use _noGrad = torch.no_grad()
    let cfg = state.Config
    let nextStep = state.Step + 1

    let beta1Pow = MathF.Pow(cfg.Beta1, float32 nextStep)
    let beta2Pow = MathF.Pow(cfg.Beta2, float32 nextStep)
    let biasCorr1 = 1.0f - beta1Pow
    let biasCorr2 = 1.0f - beta2Pow
    let stepChunkRows = max 1L cfg.StepChunkRows

    let gradCpuOpt =
      if cfg.OffloadGradToCpu then
        let arr = Array.zeroCreate<torch.Tensor option> state.States.Length
        for idx, st in state.States |> List.indexed do
          let g = st.Param.grad
          if not (isNull g) then
            let gCpu = g.``to``(device = "cpu").contiguous().clone()
            st.Param.grad <- null
            arr.[idx] <- Some gCpu
        Some arr
      else
        None

    try
      for idx, st in state.States |> List.indexed do
        let gradSourceOpt =
          match gradCpuOpt with
          | Some arr when idx < arr.Length ->
            match arr.[idx] with
            | Some gCpu -> Some gCpu
            | None -> if isNull st.Param.grad then None else Some st.Param.grad
          | _ ->
            if isNull st.Param.grad then None else Some st.Param.grad

        match gradSourceOpt with
        | None -> ()
        | Some gradSource ->
          let rows, cols = shapeToRowsCols st.W.Shape
          let mutable rowStart = 0L

          use grad2d = gradSource.reshape([| rows; cols |]).contiguous()
          use param2d = st.Param.reshape([| rows; cols |])

          let wQNext = torch.empty(st.W.QData.shape, dtype = st.W.QData.dtype, device = st.W.QData.device)
          let wSNext = torch.empty(st.W.Scale.shape, dtype = st.W.Scale.dtype, device = st.W.Scale.device)

          let mQNext, mSNext, vQNext, vSNext =
            match st.M, st.V with
            | MomentNvfp4 m, MomentNvfp4 v ->
              torch.empty(m.QData.shape, dtype = m.QData.dtype, device = m.QData.device),
              torch.empty(m.Scale.shape, dtype = m.Scale.dtype, device = m.Scale.device),
              torch.empty(v.QData.shape, dtype = v.QData.dtype, device = v.QData.device),
              torch.empty(v.Scale.shape, dtype = v.Scale.dtype, device = v.Scale.device)
            | MomentInt8 m, MomentInt8 v ->
              torch.empty(m.QData.shape, dtype = m.QData.dtype, device = m.QData.device),
              torch.empty(m.Scale.shape, dtype = m.Scale.dtype, device = m.Scale.device),
              torch.empty(v.QData.shape, dtype = v.QData.dtype, device = v.QData.device),
              torch.empty(v.Scale.shape, dtype = v.Scale.dtype, device = v.Scale.device)
            | _ -> invalidOp "optimizer state mode mismatch between m and v"

          while rowStart < rows do
            let rowLen = min stepChunkRows (rows - rowStart)
            use gSlice0 = grad2d.narrow(0L, rowStart, rowLen).contiguous()
            let gSliceOpt =
              if gSlice0.device.ToString() = cfg.Device then
                None
              else
                Some(gSlice0.``to``(device = cfg.Device))
            let gSlice =
              match gSliceOpt with
              | Some t -> t
              | None -> gSlice0

            use gBf16 =
              if gSlice.dtype = torch.bfloat16 then gSlice.contiguous()
              else gSlice.to_type(torch.bfloat16).contiguous()
            use g32 = gBf16.to_type(torch.float32)
            use w32 = unpackNvfp4Rows2d st.W rowStart rowLen cfg.Device torch.float32
            use mPrev32 = unpackMomentRows2d st.M rowStart rowLen cfg.Device torch.float32
            use vPrev32 = unpackMomentRows2d st.V rowStart rowLen cfg.Device torch.float32

            use m32 = cfg.Beta1 * mPrev32 + (1.0f - cfg.Beta1) * g32
            use v32 = cfg.Beta2 * vPrev32 + (1.0f - cfg.Beta2) * (g32 * g32)
            use mHat = m32 / biasCorr1
            use vHat = v32 / biasCorr2
            use denom = torch.sqrt(vHat) + cfg.Eps
            use update = (mHat / denom) + cfg.WeightDecay * w32
            use wNew32 = w32 - cfg.LearningRate * update
            use wNewMaster = wNew32.to_type(cfg.MasterDType).contiguous()
            use pDst = param2d.narrow(0L, rowStart, rowLen)
            pDst.copy_(wNewMaster) |> ignore

            let wQChunk, wSChunk = packMatrixNvfp4ToStorage wNewMaster st.W.StorageDevice

            let mQChunk, mSChunk, vQChunk, vSChunk =
              match st.M, st.V with
              | MomentNvfp4 mSt, MomentNvfp4 vSt ->
                use mPackSrc = m32.to_type(torch.bfloat16).contiguous()
                use vPackSrc = v32.to_type(torch.bfloat16).contiguous()
                let mQ, mS = packMatrixNvfp4ToStorage mPackSrc mSt.StorageDevice
                let vQ, vS = packMatrixNvfp4ToStorage vPackSrc vSt.StorageDevice
                mQ, mS, vQ, vS
              | MomentInt8 mSt, MomentInt8 vSt ->
                let mQ, mS = packMatrixInt8ToStorage m32 mSt.StorageDevice
                let vQ, vS = packMatrixInt8ToStorage v32 vSt.StorageDevice
                mQ, mS, vQ, vS
              | _ -> invalidOp "optimizer state mode mismatch between m and v"

            try
              use wQDst = wQNext.narrow(0L, rowStart, rowLen)
              use wSDst = wSNext.narrow(0L, rowStart, rowLen)
              use mQDst = mQNext.narrow(0L, rowStart, rowLen)
              use mSDst = mSNext.narrow(0L, rowStart, rowLen)
              use vQDst = vQNext.narrow(0L, rowStart, rowLen)
              use vSDst = vSNext.narrow(0L, rowStart, rowLen)
              wQDst.copy_(wQChunk) |> ignore
              wSDst.copy_(wSChunk) |> ignore
              mQDst.copy_(mQChunk) |> ignore
              mSDst.copy_(mSChunk) |> ignore
              vQDst.copy_(vQChunk) |> ignore
              vSDst.copy_(vSChunk) |> ignore
            finally
              wQChunk.Dispose()
              wSChunk.Dispose()
              mQChunk.Dispose()
              mSChunk.Dispose()
              vQChunk.Dispose()
              vSChunk.Dispose()
              gSliceOpt |> Option.iter (fun t -> t.Dispose())

            rowStart <- rowStart + rowLen

          replacePackedNvfp4 st.W
            {
              QData = wQNext
              Scale = wSNext
              Shape = Array.copy st.W.Shape
              StorageDevice = st.W.StorageDevice
            }

          st.M <-
            match st.M with
            | MomentNvfp4 old ->
              let next =
                MomentNvfp4
                  {
                    QData = mQNext
                    Scale = mSNext
                    Shape = Array.copy old.Shape
                    StorageDevice = old.StorageDevice
                  }
              replaceMoment st.M next
            | MomentInt8 old ->
              let next =
                MomentInt8
                  {
                    QData = mQNext
                    Scale = mSNext
                    Shape = Array.copy old.Shape
                    StorageDevice = old.StorageDevice
                  }
              replaceMoment st.M next

          st.V <-
            match st.V with
            | MomentNvfp4 old ->
              let next =
                MomentNvfp4
                  {
                    QData = vQNext
                    Scale = vSNext
                    Shape = Array.copy old.Shape
                    StorageDevice = old.StorageDevice
                  }
              replaceMoment st.V next
            | MomentInt8 old ->
              let next =
                MomentInt8
                  {
                    QData = vQNext
                    Scale = vSNext
                    Shape = Array.copy old.Shape
                    StorageDevice = old.StorageDevice
                  }
              replaceMoment st.V next

          st.Param.grad <- null

          if cfg.FlushEachParam then
            if torch.cuda_is_available() then
              torch.cuda.synchronize()
            NativeInterop.tryEmptyNvfp4Cache() |> ignore

      state.Step <- nextStep
    finally
      gradCpuOpt
      |> Option.iter (fun arr ->
        arr
        |> Array.iter (function
          | Some g -> g.Dispose()
          | None -> ()))
