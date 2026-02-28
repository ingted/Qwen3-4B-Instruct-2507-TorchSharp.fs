#if INTERACTIVE
#load "loadCUDA.fsx"
#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.9"
#r "/workspace/TorchSharp.Fun.DGX/TorchSharp.Fun.DGX/bin/Release/net10.0/TorchSharp.Fun.DGX.dll"
#r "/workspace/TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/bin/Release/net10.0/TorchSharp.Q4.Extension.dll"
#r "/workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/bin/Release/net10.0/Qwen3-4B-Instruct-2507-TorchSharp.fs.dll"
#endif

open System
open System.IO
open System.Diagnostics
open System.Text.Json
open System.Collections.Generic
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open TorchSharp
open TorchSharp.Modules
open TorchSharp.Q4.Extension
open Qwen3_4B_Instruct_2507_TorchSharp_fs

type ScriptArgs =
    { ModelDir: string
      WeightPath: string option
      Quant: string option
      Device: string
      MasterDType: string
      BatchSize: int64
      SeqLen: int64
      LearningRate: float32
      Beta1: float32
      Beta2: float32
      Eps: float32
      WeightDecay: float32
      GradCkptChunk: int
      StepChunkRows: int64
      OffloadMVToCpu: bool
      OffloadWToCpu: bool
      MaterializeFromPacked: bool
      StepFlushEachParam: bool
      ComputeGradNorm: bool
      OffloadGradToCpu: bool
      DisposeSessionAfterLoad: bool
      CompactAfterModelLoad: bool
      PrintTensorByteReport: bool
      StopAfter: string
      ProfileVram: bool
      VramReportPath: string
      Seed: int option }

type PackedNvfp4 =
    { mutable QData: torch.Tensor
      mutable Scale: torch.Tensor
      Shape: int64 array
      StorageDevice: string }

type PackedParamState =
    { Name: string
      Param: Parameter
      mutable W: PackedNvfp4
      mutable M: PackedNvfp4
      mutable V: PackedNvfp4 }

type VramSample =
    { ts: string
      phase: string
      pid_mem_mib: int
      total_gpu_mem_mib: int
      cuda_used_mib: int
      cuda_total_mib: int
      proc_rss_mib: int }

let parseArgs (argv: string[]) =
    let rec loop i (m: Map<string, string>) =
        if i >= argv.Length then m
        else
            let key = argv.[i]
            if key.StartsWith("--") then
                let eq = key.IndexOf('=')
                if eq > 2 then
                    let k = key.Substring(0, eq)
                    let v = key.Substring(eq + 1)
                    let value = if String.IsNullOrWhiteSpace(v) then "true" else v
                    loop (i + 1) (m.Add(k, value))
                elif i + 1 < argv.Length && not (argv.[i + 1].StartsWith("--")) then
                    loop (i + 2) (m.Add(key, argv.[i + 1]))
                else
                    loop (i + 1) (m.Add(key, "true"))
            else
                loop (i + 1) m
    loop 0 Map.empty

let tryGet (argMap: Map<string, string>) (key: string) = argMap |> Map.tryFind key

let tryGetInt (argMap: Map<string, string>) (key: string) (fallback: int) =
    match tryGet argMap key with
    | Some v ->
        match Int32.TryParse(v) with
        | true, i -> i
        | _ -> fallback
    | None -> fallback

let tryGetInt64 (argMap: Map<string, string>) (key: string) (fallback: int64) =
    match tryGet argMap key with
    | Some v ->
        match Int64.TryParse(v) with
        | true, i -> i
        | _ -> fallback
    | None -> fallback

let tryGetFloat (argMap: Map<string, string>) (key: string) (fallback: float32) =
    match tryGet argMap key with
    | Some v ->
        match Single.TryParse(v) with
        | true, f -> f
        | _ -> fallback
    | None -> fallback

let tryGetSeed (argMap: Map<string, string>) (key: string) =
    match tryGet argMap key with
    | Some v ->
        match Int32.TryParse(v) with
        | true, i when i >= 0 -> Some i
        | _ -> None
    | None -> None

let tryGetBool (argMap: Map<string, string>) (key: string) (fallback: bool) =
    match tryGet argMap key with
    | Some v ->
        match v.Trim().ToLowerInvariant() with
        | "1" | "true" | "yes" -> true
        | "0" | "false" | "no" -> false
        | _ -> fallback
    | None -> fallback

let resolvePath (path: string) =
    if Path.IsPathRooted(path) then path else Path.Combine(Environment.CurrentDirectory, path)

let defaultArgs =
    [| "--model-dir"; "/models/qwen3-4b-instruct-2507-torchsharp"
       "--device"; "cuda"
       "--dtype"; "bfloat16"
       "--quant"; "fp4"
       "--weight"; "Qwen3-4B-Instruct-2507-nvfp4.dat"
       "--batch-size"; "1"
       "--seq-len"; "8"
       "--lr"; "0.00005"
       "--beta1"; "0.9"
       "--beta2"; "0.999"
       "--eps"; "0.00000001"
       "--weight-decay"; "0.01"
       "--grad-ckpt-chunk"; "2"
       "--step-chunk-rows"; "32"
       "--offload-mv-to-cpu"; "true"
       "--offload-w-to-cpu"; "true"
       "--materialize-from-packed"; "false"
       "--step-flush-each-param"; "true"
       "--compute-grad-norm"; "false"
       "--offload-grad-to-cpu"; "true"
       "--dispose-session-after-load"; "true"
       "--compact-after-model-load"; "true"
       "--print-tensor-byte-report"; "true"
       "--stop-after"; "none"
       "--profile-vram"; "true"
       "--vram-report"; "alpha/log/train-step-full-nvfp4-vram-report.json"
       "--seed"; "123" |]

#if INTERACTIVE
let rawArgs = fsi.CommandLineArgs
#else
let rawArgs = Environment.GetCommandLineArgs()
#endif

let hasUserArgs = rawArgs |> Array.exists (fun (s: string) -> s.StartsWith("--"))
let argMap =
    if not hasUserArgs then
        parseArgs defaultArgs
    else
        let defaults = parseArgs defaultArgs
        let user = parseArgs (rawArgs |> Array.skip 1)
        Map.fold (fun acc k v -> acc.Add(k, v)) defaults user

let scriptArgs =
    { ModelDir = tryGet argMap "--model-dir" |> Option.defaultValue Defaults.modelDir
      WeightPath = tryGet argMap "--weight"
      Quant = tryGet argMap "--quant"
      Device = tryGet argMap "--device" |> Option.defaultValue "cuda"
      MasterDType = tryGet argMap "--dtype" |> Option.defaultValue "bfloat16"
      BatchSize = max 1L (tryGetInt64 argMap "--batch-size" 1L)
      SeqLen = max 1L (tryGetInt64 argMap "--seq-len" 8L)
      LearningRate = tryGetFloat argMap "--lr" 0.00005f
      Beta1 = tryGetFloat argMap "--beta1" 0.9f
      Beta2 = tryGetFloat argMap "--beta2" 0.999f
      Eps = tryGetFloat argMap "--eps" 1e-8f
      WeightDecay = tryGetFloat argMap "--weight-decay" 0.01f
      GradCkptChunk = max 0 (tryGetInt argMap "--grad-ckpt-chunk" 2)
      StepChunkRows = max 1L (tryGetInt64 argMap "--step-chunk-rows" 32L)
      OffloadMVToCpu = tryGetBool argMap "--offload-mv-to-cpu" true
      OffloadWToCpu = tryGetBool argMap "--offload-w-to-cpu" true
      MaterializeFromPacked = tryGetBool argMap "--materialize-from-packed" false
      StepFlushEachParam = tryGetBool argMap "--step-flush-each-param" true
      ComputeGradNorm = tryGetBool argMap "--compute-grad-norm" false
      OffloadGradToCpu = tryGetBool argMap "--offload-grad-to-cpu" true
      DisposeSessionAfterLoad = tryGetBool argMap "--dispose-session-after-load" true
      CompactAfterModelLoad = tryGetBool argMap "--compact-after-model-load" true
      PrintTensorByteReport = tryGetBool argMap "--print-tensor-byte-report" true
      StopAfter = (tryGet argMap "--stop-after" |> Option.defaultValue "none").Trim().ToLowerInvariant()
      ProfileVram = tryGetBool argMap "--profile-vram" true
      VramReportPath = tryGet argMap "--vram-report" |> Option.defaultValue "alpha/log/train-step-full-nvfp4-vram-report.json"
      Seed = tryGetSeed argMap "--seed" }

let stopAfterPhase (phase: string) =
    if StringComparer.OrdinalIgnoreCase.Equals(scriptArgs.StopAfter, phase) then
        failwithf "stop-after reached: %s" phase

let ensureNativeQuantize () =
    let raw = Environment.GetEnvironmentVariable("TS_Q4_STE_USE_NATIVE_QUANTIZE")
    let isEnabled =
        match raw with
        | null -> false
        | v ->
            match v.Trim().ToLowerInvariant() with
            | "1" | "true" | "yes" -> true
            | _ -> false
    if not isEnabled then
        Environment.SetEnvironmentVariable("TS_Q4_STE_USE_NATIVE_QUANTIZE", "1")
        printfn "info: TS_Q4_STE_USE_NATIVE_QUANTIZE not set; forcing to 1."

ensureNativeQuantize ()

if not (NativeInterop.hasLibTorchFp4Quantize()) then
    failwith "NVFP4 native quantize export unavailable; cannot run full NVFP4 state optimizer."

let getGpuMemSnapshotMiB (pid: int) =
    let tryQueryComputeApps () =
        try
            let psi = ProcessStartInfo("nvidia-smi")
            psi.Arguments <- "--query-compute-apps=pid,used_memory --format=csv,noheader,nounits"
            psi.RedirectStandardOutput <- true
            psi.UseShellExecute <- false
            psi.CreateNoWindow <- true
            use p = Process.Start(psi)
            let output = p.StandardOutput.ReadToEnd()
            p.WaitForExit()
            output.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
            |> Array.fold (fun acc line ->
                let parts = line.Split([| ',' |], StringSplitOptions.RemoveEmptyEntries)
                if parts.Length >= 2 then
                    match Int32.TryParse(parts.[0].Trim()), Int32.TryParse(parts.[1].Trim()) with
                    | (true, p), (true, mem) -> (p, mem) :: acc
                    | _ -> acc
                else acc) []
        with _ -> []

    let entries = tryQueryComputeApps ()
    let pidMem =
        entries |> List.fold (fun acc (p, mem) -> if p = pid then acc + mem else acc) 0
    let totalMem =
        entries |> List.fold (fun acc (_, mem) -> acc + mem) 0
    pidMem, totalMem

module CudaRuntime =
    [<DllImport("libcudart.so", EntryPoint = "cudaMemGetInfo")>]
    extern int cudaMemGetInfo(nativeint& free, nativeint& total)

    let tryGetUsedMiB () =
        try
            let mutable freePtr = nativeint 0
            let mutable totalPtr = nativeint 0
            let rc = cudaMemGetInfo(&freePtr, &totalPtr)
            if rc = 0 then
                let freeBytes = int64 freePtr
                let totalBytes = int64 totalPtr
                let usedBytes = max 0L (totalBytes - freeBytes)
                Some (int (usedBytes / 1024L / 1024L), int (totalBytes / 1024L / 1024L))
            else
                None
        with _ ->
            None

let toMatrix2d (t: torch.Tensor) =
    let shape = t.shape |> Array.copy
    let view =
        match shape.Length with
        | 1 -> t.reshape([| 1L; shape.[0] |]).contiguous()
        | 2 -> t.contiguous()
        | _ ->
            let cols = shape |> Array.skip 1 |> Array.fold (fun acc d -> acc * d) 1L
            t.reshape([| shape.[0]; cols |]).contiguous()
    view, shape

let packNvfp4 (source: torch.Tensor) (storageDevice: string) =
    let matrix, shape = toMatrix2d source
    try
        let q, s = Nvfp4Training.quantizePacked matrix
        try
            let qStored =
                if q.device.ToString() = storageDevice then q.contiguous().clone()
                else
                    use moved = q.``to``(device = storageDevice)
                    moved.contiguous().clone()

            let sStored =
                if s.device.ToString() = storageDevice then s.contiguous().clone()
                else
                    use moved = s.``to``(device = storageDevice)
                    moved.contiguous().clone()

            { QData = qStored
              Scale = sStored
              Shape = shape
              StorageDevice = storageDevice }
        finally
            q.Dispose()
            s.Dispose()
    finally
        matrix.Dispose()

let unpackNvfp4 (packed: PackedNvfp4) (computeDevice: string) (outDtype: torch.ScalarType) =
    let qTemp =
        if packed.QData.device.ToString() = computeDevice then None
        else Some (packed.QData.``to``(device = computeDevice))
    let sTemp =
        if packed.Scale.device.ToString() = computeDevice then None
        else Some (packed.Scale.``to``(device = computeDevice))

    let q = match qTemp with | Some t -> t | None -> packed.QData
    let s = match sTemp with | Some t -> t | None -> packed.Scale
    try
        use dense2d = Nvfp4Training.dequantizePacked q s outDtype
        dense2d.reshape(packed.Shape).contiguous().clone()
    finally
        qTemp |> Option.iter (fun t -> t.Dispose())
        sTemp |> Option.iter (fun t -> t.Dispose())

let disposePacked (p: PackedNvfp4) =
    p.QData.Dispose()
    p.Scale.Dispose()

let replacePacked (target: PackedNvfp4 byref) (next: PackedNvfp4) =
    disposePacked target
    target <- next

let scalarLossMse (output: torch.Tensor) (target: torch.Tensor) =
    use targetTyped = if target.dtype = output.dtype then target.contiguous() else target.to_type(output.dtype).contiguous()
    use diff = output - targetTyped
    use sq = diff * diff
    sq.mean()

let createBatch (batchSize: int64) (seqLen: int64) (hiddenSize: int64) (device: string) (dtype: torch.ScalarType) =
    let input = torch.randn([| batchSize; seqLen; hiddenSize |], dtype = dtype, device = device)
    let target = torch.randn([| batchSize; seqLen; hiddenSize |], dtype = dtype, device = device)
    input, target

let materializeMasterWeightsFromPacked (states: PackedParamState list) (device: string) (masterDType: torch.ScalarType) =
    use _noGrad = torch.no_grad()
    for st in states do
        use wDense = unpackNvfp4 st.W device masterDType
        st.Param.copy_(wDense) |> ignore

let zeroAllGrad (parameters: Parameter list) =
    for p in parameters do
        if not (isNull p.grad) then
            p.grad.zero_() |> ignore

let gradNormL2 (parameters: Parameter list) =
    let mutable acc = 0.0
    for p in parameters do
        if not (isNull p.grad) then
            use g32 = p.grad.to_type(torch.float32)
            use sq = g32 * g32
            use s = sq.sum()
            acc <- acc + float (s.item<float32>())
    Math.Sqrt(acc)

let shapeToRowsCols (shape: int64 array) =
    match shape.Length with
    | 0 -> 1L, 1L
    | 1 -> 1L, shape.[0]
    | 2 -> shape.[0], shape.[1]
    | _ ->
        let cols = shape |> Array.skip 1 |> Array.fold (fun acc d -> acc * d) 1L
        shape.[0], cols

let packMatrixToStorage (matrix2d: torch.Tensor) (storageDevice: string) =
    use matrix = matrix2d.contiguous()
    let q, s = Nvfp4Training.quantizePacked matrix
    try
        let qStored =
            if q.device.ToString() = storageDevice then q.contiguous().clone()
            else
                use moved = q.``to``(device = storageDevice)
                moved.contiguous().clone()
        let sStored =
            if s.device.ToString() = storageDevice then s.contiguous().clone()
            else
                use moved = s.``to``(device = storageDevice)
                moved.contiguous().clone()
        qStored, sStored
    finally
        q.Dispose()
        s.Dispose()

let unpackNvfp4Rows2d
    (packed: PackedNvfp4)
    (rowStart: int64)
    (rowLen: int64)
    (computeDevice: string)
    (outDtype: torch.ScalarType) =
    use qSlice0 = packed.QData.narrow(0L, rowStart, rowLen).contiguous()
    use sSlice0 = packed.Scale.narrow(0L, rowStart, rowLen).contiguous()

    let qTemp =
        if qSlice0.device.ToString() = computeDevice then None
        else Some (qSlice0.``to``(device = computeDevice))
    let sTemp =
        if sSlice0.device.ToString() = computeDevice then None
        else Some (sSlice0.``to``(device = computeDevice))

    let q = match qTemp with | Some t -> t | None -> qSlice0
    let s = match sTemp with | Some t -> t | None -> sSlice0
    try
        use dense2d = Nvfp4Training.dequantizePacked q s outDtype
        dense2d.contiguous().clone()
    finally
        qTemp |> Option.iter (fun t -> t.Dispose())
        sTemp |> Option.iter (fun t -> t.Dispose())

let adamwStepNvfp4Packed
    (states: PackedParamState list)
    (device: string)
    (masterDType: torch.ScalarType)
    (gradCpuOpt: torch.Tensor option array option)
    (stepIndex: int)
    (lr: float32)
    (beta1: float32)
    (beta2: float32)
    (eps: float32)
    (weightDecay: float32)
    (stepChunkRows: int64)
    (flushEachParam: bool) =
    use _noGrad = torch.no_grad()

    let beta1Pow = MathF.Pow(beta1, float32 stepIndex)
    let beta2Pow = MathF.Pow(beta2, float32 stepIndex)
    let biasCorr1 = 1.0f - beta1Pow
    let biasCorr2 = 1.0f - beta2Pow

    for idx, st in states |> List.indexed do
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
            let chunkRows = max 1L stepChunkRows
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
                let rowLen = min chunkRows (rows - rowStart)
                use gSlice0 = grad2d.narrow(0L, rowStart, rowLen).contiguous()
                let gSliceOpt =
                    if gSlice0.device.ToString() = device then None
                    else Some (gSlice0.``to``(device = device))
                let gSlice = match gSliceOpt with | Some t -> t | None -> gSlice0

                use gBf16 =
                    if gSlice.dtype = torch.bfloat16 then gSlice.contiguous()
                    else gSlice.to_type(torch.bfloat16).contiguous()
                use g32 = gBf16.to_type(torch.float32)
                use w32 = unpackNvfp4Rows2d st.W rowStart rowLen device torch.float32
                use mPrev32 = unpackNvfp4Rows2d st.M rowStart rowLen device torch.float32
                use vPrev32 = unpackNvfp4Rows2d st.V rowStart rowLen device torch.float32

                use m32 = beta1 * mPrev32 + (1.0f - beta1) * g32
                use v32 = beta2 * vPrev32 + (1.0f - beta2) * (g32 * g32)
                use mHat = m32 / biasCorr1
                use vHat = v32 / biasCorr2
                use denom = torch.sqrt(vHat) + eps
                use update = (mHat / denom) + weightDecay * w32
                use wNew32 = w32 - lr * update
                use wNewMaster = wNew32.to_type(masterDType).contiguous()
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

            let newW = { QData = wQNext; Scale = wSNext; Shape = Array.copy st.W.Shape; StorageDevice = st.W.StorageDevice }
            let newM = { QData = mQNext; Scale = mSNext; Shape = Array.copy st.M.Shape; StorageDevice = st.M.StorageDevice }
            let newV = { QData = vQNext; Scale = vSNext; Shape = Array.copy st.V.Shape; StorageDevice = st.V.StorageDevice }
            replacePacked &st.W newW
            replacePacked &st.M newM
            replacePacked &st.V newV
            st.Param.grad <- null

            if flushEachParam then
                if torch.cuda_is_available() then
                    torch.cuda.synchronize()
                NativeInterop.tryEmptyNvfp4Cache() |> ignore

let backwardWithSequenceRecompute
    (model: Qwen3Nvfp4Model)
    (input: torch.Tensor)
    (target: torch.Tensor)
    (chunkSize: int)
    (masterDType: torch.ScalarType) =
    let seqLen = int input.shape.[1]
    let totalChunks = max 1 ((seqLen + chunkSize - 1) / chunkSize)
    let mutable start = 0
    let mutable lossSum = 0.0f

    while start < seqLen do
        let ending = min seqLen (start + chunkSize)
        let prefixLen = int64 ending
        let chunkStart = int64 start
        let chunkLen = int64 (ending - start)

        use prefixInput = input.narrow(1L, 0L, prefixLen).contiguous()
        use prefixOutput = Qwen3Model.forward model prefixInput (Some masterDType)
        use chunkOutput = prefixOutput.narrow(1L, chunkStart, chunkLen).contiguous()
        use chunkTarget = target.narrow(1L, chunkStart, chunkLen).contiguous()
        use chunkLoss = scalarLossMse chunkOutput chunkTarget
        use scaledLoss = chunkLoss / float32 totalChunks
        scaledLoss.backward()

        use lossCpu = scaledLoss.to_type(torch.float32).cpu()
        lossSum <- lossSum + lossCpu.item<float32>()
        start <- ending

    lossSum

let samples = ResizeArray<VramSample>()

let recordVram (enabled: bool) (phase: string) =
    if enabled then
        let pidMem, totalMem = getGpuMemSnapshotMiB Environment.ProcessId
        let cudaUsed, cudaTotal =
            match CudaRuntime.tryGetUsedMiB () with
            | Some (used, total) -> used, total
            | None -> -1, -1
        let rssMiB = int (Process.GetCurrentProcess().WorkingSet64 / 1024L / 1024L)
        let sample =
            { ts = DateTime.UtcNow.ToString("O")
              phase = phase
              pid_mem_mib = pidMem
              total_gpu_mem_mib = totalMem
              cuda_used_mib = cudaUsed
              cuda_total_mib = cudaTotal
              proc_rss_mib = rssMiB }
        samples.Add(sample)
        printfn "[vram] phase=%s pid=%dMiB total=%dMiB cuda_used=%dMiB cuda_total=%dMiB rss=%dMiB"
            phase pidMem totalMem cudaUsed cudaTotal rssMiB

let ensureDirForFile (path: string) =
    let dir = Path.GetDirectoryName(path)
    if not (String.IsNullOrWhiteSpace(dir)) then
        Directory.CreateDirectory(dir) |> ignore

let writeVramReport (path: string) =
    ensureDirForFile path
    let json = JsonSerializer.Serialize(samples, JsonSerializerOptions(WriteIndented = true))
    File.WriteAllText(path, json)
    printfn "[vram] report written: %s" path

let disposeModelParametersOnly (model: Qwen3Nvfp4Model) =
    for layer in model.Layers do
        layer.MasterWeight.Dispose()
    for p in model.ExtraParameters do
        p.Dispose()

let tensorMiB (t: torch.Tensor) =
    int ((t.NumberOfElements * t.ElementSize) / 1024L / 1024L)

let tensorBytes (t: torch.Tensor) =
    t.NumberOfElements * t.ElementSize

let printTensorBytesSummary (title: string) (entries: (string * torch.Tensor) list) =
    let mutable totalBytes = 0L
    let buckets = Dictionary<string, int64>()
    for (name, t) in entries do
        if not (isNull t) then
            let b = tensorBytes t
            totalBytes <- totalBytes + b
            let key = sprintf "%s|%s|%s" name (t.device.ToString()) (t.dtype.ToString())
            let prev =
                match buckets.TryGetValue(key) with
                | true, v -> v
                | _ -> 0L
            buckets.[key] <- prev + b
    printfn "[bytes] %s total=%.2f MiB entries=%d" title (float totalBytes / 1024.0 / 1024.0) entries.Length
    for kv in buckets |> Seq.sortByDescending (fun kv -> kv.Value) |> Seq.truncate 24 do
        let parts = kv.Key.Split('|')
        let kind = if parts.Length > 0 then parts.[0] else "unknown"
        let device = if parts.Length > 1 then parts.[1] else "unknown"
        let dtype = if parts.Length > 2 then parts.[2] else "unknown"
        printfn "[bytes]   kind=%s device=%s dtype=%s size=%.2f MiB"
            kind
            device
            dtype
            (float kv.Value / 1024.0 / 1024.0)

let masterDType = Enum.Parse<torch.ScalarType>(scriptArgs.MasterDType, true)
torch.InitializeDeviceType(DeviceType.CUDA)
torch.set_default_dtype(masterDType)

match scriptArgs.Seed with
| Some s ->
    torch.manual_seed(int64 s) |> ignore
    printfn "[seed] %d" s
| None -> ()

printfn "[train-step] device=%s dtype=%A batch=%d seq=%d grad-ckpt-chunk=%d offload-mv-to-cpu=%b"
    scriptArgs.Device
    masterDType
    scriptArgs.BatchSize
    scriptArgs.SeqLen
    scriptArgs.GradCkptChunk
    scriptArgs.OffloadMVToCpu
printfn "[train-step] step-chunk-rows=%d" scriptArgs.StepChunkRows
printfn "[train-step] offload-w-to-cpu=%b" scriptArgs.OffloadWToCpu
printfn "[train-step] materialize-from-packed=%b" scriptArgs.MaterializeFromPacked
printfn "[train-step] step-flush-each-param=%b" scriptArgs.StepFlushEachParam
printfn "[train-step] compute-grad-norm=%b" scriptArgs.ComputeGradNorm
printfn "[train-step] offload-grad-to-cpu=%b" scriptArgs.OffloadGradToCpu
printfn "[train-step] dispose-session-after-load=%b compact-after-model-load=%b print-tensor-byte-report=%b stop-after=%s"
    scriptArgs.DisposeSessionAfterLoad
    scriptArgs.CompactAfterModelLoad
    scriptArgs.PrintTensorByteReport
    scriptArgs.StopAfter
printfn "[diag] TorchSharp build has no public cuda allocator stats API (allocated/reserved)."
printfn "[diag] reporting PID/total GPU memory + cudaMemGetInfo + tensor-bytes breakdown instead."

let resolvedWeightPath = InferenceBridge.resolveWeightPath scriptArgs.ModelDir scriptArgs.WeightPath scriptArgs.Quant
let trainingCfg : TrainingConfig =
    { Defaults.trainingConfig with
        ModelDir = scriptArgs.ModelDir
        ConfigPath = Path.Combine(scriptArgs.ModelDir, "config.json")
        TokenizerPath = Path.Combine(scriptArgs.ModelDir, "tokenizer.json")
        WeightPath = resolvedWeightPath
        Device = scriptArgs.Device
        SyntheticMode = false
        StrictLoad = true
        UseKvCache = false
        SequenceLength = scriptArgs.SeqLen }

recordVram scriptArgs.ProfileVram "start"
let model = Qwen3Model.create trainingCfg
recordVram scriptArgs.ProfileVram "model_loaded"
stopAfterPhase "model_loaded"

if scriptArgs.DisposeSessionAfterLoad then
    Qwen3Model.disposeSession model.Session
    recordVram scriptArgs.ProfileVram "session_disposed"
    stopAfterPhase "session_disposed"

if scriptArgs.CompactAfterModelLoad then
    if torch.cuda_is_available() then
        torch.cuda.synchronize()
    Nvfp4Training.clearEvalWeightCache()
    NativeInterop.tryEmptyNvfp4Cache() |> ignore
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()
    recordVram scriptArgs.ProfileVram "post_load_compacted"
    stopAfterPhase "post_load_compacted"

let nameByKey = Dictionary<int, string>()
for layer in model.Layers do
    nameByKey.[RuntimeHelpers.GetHashCode(layer.MasterWeight)] <- layer.Name
for i = 0 to model.ExtraParameters.Length - 1 do
    let p = model.ExtraParameters.[i]
    nameByKey.[RuntimeHelpers.GetHashCode(p)] <- sprintf "extra.%d" i

let trainableParams = Qwen3Model.parameters model
printfn "[train-step] params=%d blocks=%d" trainableParams.Length model.Blocks.Length
let allParamRefs =
    seq {
        for l in model.Layers do
            yield l.MasterWeight
        for p in model.ExtraParameters do
            yield p
    }
    |> Seq.toList
let allRefCount = allParamRefs.Length
let uniqueRefCount =
    allParamRefs
    |> Seq.map RuntimeHelpers.GetHashCode
    |> Set.ofSeq
    |> Set.count
printfn "[train-step] parameter-ref-count raw=%d unique=%d duplicate=%d"
    allRefCount
    uniqueRefCount
    (allRefCount - uniqueRefCount)
if scriptArgs.PrintTensorByteReport then
    printTensorBytesSummary "model.parameters(unique)"
        (trainableParams |> List.map (fun p -> "model_param", p :> torch.Tensor))

let mvStorageDevice = if scriptArgs.OffloadMVToCpu then "cpu" else scriptArgs.Device
let wStorageDevice = if scriptArgs.OffloadWToCpu then "cpu" else scriptArgs.Device

let buildPackedStates () =
    trainableParams
    |> List.mapi (fun idx p ->
        use p0 = p.detach().contiguous()
        let wPacked = packNvfp4 p0 wStorageDevice
        use zeros = torch.zeros(p0.shape, dtype = p0.dtype, device = p0.device)
        let mPacked = packNvfp4 zeros mvStorageDevice
        let vPacked = packNvfp4 zeros mvStorageDevice
        let key = RuntimeHelpers.GetHashCode(p)
        let name =
            match nameByKey.TryGetValue(key) with
            | true, v -> v
            | _ -> sprintf "param.%d" idx
        { Name = name
          Param = p
          W = wPacked
          M = mPacked
          V = vPacked })

let packedStates = buildPackedStates ()

let wStateMiB =
    packedStates
    |> List.sumBy (fun st -> tensorMiB st.W.QData + tensorMiB st.W.Scale)
let mStateMiB =
    packedStates
    |> List.sumBy (fun st -> tensorMiB st.M.QData + tensorMiB st.M.Scale)
let vStateMiB =
    packedStates
    |> List.sumBy (fun st -> tensorMiB st.V.QData + tensorMiB st.V.Scale)
printfn "[train-step] persistent nvfp4 state size (MiB): w=%d m=%d v=%d total=%d"
    wStateMiB mStateMiB vStateMiB (wStateMiB + mStateMiB + vStateMiB)
if scriptArgs.PrintTensorByteReport then
    let stateEntries =
        packedStates
        |> List.collect (fun st ->
            [ ("state_w_q", st.W.QData)
              ("state_w_s", st.W.Scale)
              ("state_m_q", st.M.QData)
              ("state_m_s", st.M.Scale)
              ("state_v_q", st.V.QData)
              ("state_v_s", st.V.Scale) ])
    printTensorBytesSummary "optimizer.packed_state(w/m/v)" stateEntries

recordVram scriptArgs.ProfileVram "state_initialized"
stopAfterPhase "state_initialized"

if scriptArgs.MaterializeFromPacked then
    materializeMasterWeightsFromPacked packedStates scriptArgs.Device masterDType
    recordVram scriptArgs.ProfileVram "weights_materialized"
    stopAfterPhase "weights_materialized"
else
    printfn "[train-step] skip pre-step materialization from packed (one-step fast path)."
    recordVram scriptArgs.ProfileVram "weights_materialization_skipped"
    stopAfterPhase "weights_materialization_skipped"

let inputBatch, targetBatch = createBatch scriptArgs.BatchSize scriptArgs.SeqLen model.InFeatures scriptArgs.Device masterDType
recordVram scriptArgs.ProfileVram "batch_ready"
stopAfterPhase "batch_ready"

let sw = Stopwatch.StartNew()
zeroAllGrad trainableParams
recordVram scriptArgs.ProfileVram "zero_grad_done"
stopAfterPhase "zero_grad_done"

let stepLoss =
    if scriptArgs.GradCkptChunk > 0 && scriptArgs.GradCkptChunk < int scriptArgs.SeqLen then
        backwardWithSequenceRecompute model inputBatch targetBatch scriptArgs.GradCkptChunk masterDType
    else
        use output = Qwen3Model.forward model inputBatch (Some masterDType)
        use loss = scalarLossMse output targetBatch
        loss.backward()
        use lossCpu = loss.to_type(torch.float32).cpu()
        lossCpu.item<float32>()

recordVram scriptArgs.ProfileVram "backward_done"
stopAfterPhase "backward_done"

let gNorm =
    if scriptArgs.ComputeGradNorm then gradNormL2 trainableParams
    else nan

let gradCpuForStep =
    if scriptArgs.OffloadGradToCpu then
        let arr = Array.zeroCreate<torch.Tensor option> trainableParams.Length
        for i, p in trainableParams |> List.indexed do
            let g = p.grad
            if not (isNull g) then
                let gCpu = g.``to``(device = "cpu").contiguous().clone()
                p.grad <- null
                arr.[i] <- Some gCpu
        Some arr
    else
        None

recordVram scriptArgs.ProfileVram "grad_offload_done"
stopAfterPhase "grad_offload_done"

adamwStepNvfp4Packed
    packedStates
    scriptArgs.Device
    masterDType
    gradCpuForStep
    1
    scriptArgs.LearningRate
    scriptArgs.Beta1
    scriptArgs.Beta2
    scriptArgs.Eps
    scriptArgs.WeightDecay
    scriptArgs.StepChunkRows
    scriptArgs.StepFlushEachParam

gradCpuForStep
|> Option.iter (fun arr ->
    arr
    |> Array.iter (function
        | Some g -> g.Dispose()
        | None -> ()))

recordVram scriptArgs.ProfileVram "optimizer_step_done"
stopAfterPhase "optimizer_step_done"
sw.Stop()

printfn "[train-step] loss=%f grad_l2=%f elapsed=%.1fms" stepLoss gNorm sw.Elapsed.TotalMilliseconds

inputBatch.Dispose()
targetBatch.Dispose()

for st in packedStates do
    disposePacked st.W
    disposePacked st.M
    disposePacked st.V

if torch.cuda_is_available() then
    torch.cuda.synchronize()

recordVram scriptArgs.ProfileVram "cleanup_done"
if scriptArgs.DisposeSessionAfterLoad then
    disposeModelParametersOnly model
else
    Qwen3Model.dispose model
Nvfp4Training.clearEvalWeightCache()
NativeInterop.tryEmptyNvfp4Cache() |> ignore
GC.Collect()
GC.WaitForPendingFinalizers()
GC.Collect()
stopAfterPhase "cleanup_done"

let reportPath = resolvePath scriptArgs.VramReportPath
writeVramReport reportPath
printfn "[done] one-step full-parameter NVFP4 training path completed."
