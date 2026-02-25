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

type PackedParamState =
  {
    Name: string
    Param: Parameter
    mutable W: PackedNvfp4
    mutable M: PackedNvfp4
    mutable V: PackedNvfp4
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
    StepChunkRows: int64
    OffloadMVToCpu: bool
    OffloadWToCpu: bool
    OffloadGradToCpu: bool
    FlushEachParam: bool
  }

type PackedAdamwState(config: PackedAdamwConfig, states: PackedParamState list) =
  let mutable step = 0
  member _.Config = config
  member _.States = states
  member _.Step
    with get () = step
    and set value = step <- value

  interface IDisposable with
    member _.Dispose() =
      for st in states do
        st.W.QData.Dispose()
        st.W.Scale.Dispose()
        st.M.QData.Dispose()
        st.M.Scale.Dispose()
        st.V.QData.Dispose()
        st.V.Scale.Dispose()

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

  let private packNvfp4 (source: torch.Tensor) (storageDevice: string) =
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

  let private packMatrixToStorage (matrix2d: torch.Tensor) (storageDevice: string) =
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

  let private replacePacked (target: PackedNvfp4) (next: PackedNvfp4) =
    target.QData.Dispose()
    target.Scale.Dispose()
    target.QData <- next.QData
    target.Scale <- next.Scale

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
        let mPacked = packNvfp4 zeros mvStorageDevice
        let vPacked = packNvfp4 zeros mvStorageDevice
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
    let w =
      state.States
      |> List.sumBy (fun st -> tensorMiB st.W.QData + tensorMiB st.W.Scale)
    let m =
      state.States
      |> List.sumBy (fun st -> tensorMiB st.M.QData + tensorMiB st.M.Scale)
    let v =
      state.States
      |> List.sumBy (fun st -> tensorMiB st.V.QData + tensorMiB st.V.Scale)
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
            | None ->
              if isNull st.Param.grad then None else Some st.Param.grad
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
          let mQNext = torch.empty(st.M.QData.shape, dtype = st.M.QData.dtype, device = st.M.QData.device)
          let mSNext = torch.empty(st.M.Scale.shape, dtype = st.M.Scale.dtype, device = st.M.Scale.device)
          let vQNext = torch.empty(st.V.QData.shape, dtype = st.V.QData.dtype, device = st.V.QData.device)
          let vSNext = torch.empty(st.V.Scale.shape, dtype = st.V.Scale.dtype, device = st.V.Scale.device)

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
            use mPrev32 = unpackNvfp4Rows2d st.M rowStart rowLen cfg.Device torch.float32
            use vPrev32 = unpackNvfp4Rows2d st.V rowStart rowLen cfg.Device torch.float32

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

            use mPackSrc = m32.to_type(torch.bfloat16).contiguous()
            use vPackSrc = v32.to_type(torch.bfloat16).contiguous()
            let wQChunk, wSChunk = packMatrixToStorage wNewMaster st.W.StorageDevice
            let mQChunk, mSChunk = packMatrixToStorage mPackSrc st.M.StorageDevice
            let vQChunk, vSChunk = packMatrixToStorage vPackSrc st.V.StorageDevice
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

          replacePacked st.W
            {
              QData = wQNext
              Scale = wSNext
              Shape = Array.copy st.W.Shape
              StorageDevice = st.W.StorageDevice
            }
          replacePacked st.M
            {
              QData = mQNext
              Scale = mSNext
              Shape = Array.copy st.M.Shape
              StorageDevice = st.M.StorageDevice
            }
          replacePacked st.V
            {
              QData = vQNext
              Scale = vSNext
              Shape = Array.copy st.V.Shape
              StorageDevice = st.V.StorageDevice
            }
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
