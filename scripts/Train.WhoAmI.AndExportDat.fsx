#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.9"
#r "nuget: Tokenizers.DotNet, 1.3.0"
#r "nuget: Tokenizers.DotNet.runtime.linux-arm64, 1.3.0"
#r "/workspace/TorchSharp.Fun.DGX/TorchSharp.Fun.DGX/bin/Release/net10.0/TorchSharp.Fun.DGX.dll"
#r "/workspace/TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/bin/Release/net10.0/TorchSharp.Q4.Extension.dll"
#r "../bin/Release/net10.0/Qwen3-4B-Instruct-2507-TorchSharp.fs.dll"

open System
open System.IO
open System.Text
open TorchSharp
open TorchSharp.Modules
open TorchSharp.Q4.Extension
open Qwen3_4B_Instruct_2507_TorchSharp_fs

let readArgMap (args: string array) =
  args
  |> Array.toList
  |> List.chunkBySize 2
  |> List.choose (function
    | [k; v] when k.StartsWith("--", StringComparison.Ordinal) -> Some(k, v)
    | _ -> None)
  |> Map.ofList

let getOrDefault key def (m: Map<string, string>) =
  m.TryFind(key) |> Option.defaultValue def

let parseInt key def (m: Map<string, string>) =
  match m.TryFind key with
  | Some v ->
    match Int32.TryParse v with
    | true, x -> x
    | _ -> def
  | None -> def

let parseFloat key def (m: Map<string, string>) =
  match m.TryFind key with
  | Some v ->
    match Double.TryParse v with
    | true, x -> x
    | _ -> def
  | None -> def

let parseBool key def (m: Map<string, string>) =
  match m.TryFind key with
  | Some v ->
    match v.Trim().ToLowerInvariant() with
    | "1" | "true" | "yes" -> true
    | "0" | "false" | "no" -> false
    | _ -> def
  | None -> def

let sourceDir = __SOURCE_DIRECTORY__
let projectRoot = Path.GetFullPath(Path.Combine(sourceDir, ".."))
let defaultModelDir = "/models/qwen3-4b-instruct-2507-torchsharp"
let defaultInputDat = Path.Combine(defaultModelDir, "Qwen3-4B-Instruct-2507-nvfp4.dat")
let defaultOutputDat = Path.Combine(projectRoot, "artifacts", "Qwen3-4B-Instruct-2507-whoami-trained.dat")
let defaultTrainDataPath =
  let natural = Path.Combine(projectRoot, "TrainData", "whoami-1000-natural.tsv")
  let legacy = Path.Combine(projectRoot, "TrainData", "whoami-1000.tsv")
  if File.Exists(natural) then natural else legacy

let args = fsi.CommandLineArgs |> Array.skip 1
let kv = readArgMap args

let modelDir = getOrDefault "--model-dir" defaultModelDir kv
let inputDat = getOrDefault "--input-dat" defaultInputDat kv
let outputDat = getOrDefault "--output-dat" defaultOutputDat kv
let trainDataPath = getOrDefault "--train-data" defaultTrainDataPath kv
let device = getOrDefault "--device" "cuda" kv
let lossMode = Trainer.parseLossMode (getOrDefault "--loss" "ce" kv)
let steps = max 1 (parseInt "--steps" 6 kv)
let lr = parseFloat "--lr" 1e-4 kv
let requestedTrainLastLayers = parseInt "--train-last-layers" 0 kv
let seqLenCap = max 8 (parseInt "--seq-len" 96 kv)
let stepChunkRows = int64 (max 1 (parseInt "--step-chunk-rows" 32 kv))
let maxGenTokens = max 1 (parseInt "--test-max-tokens" 12 kv)
let sampleMode =
  match (getOrDefault "--sample-mode" "random" kv).Trim().ToLowerInvariant() with
  | "sequential" -> "sequential"
  | _ -> "random"
let sampleSeed = parseInt "--seed" 20260226 kv
let offloadMV = parseBool "--offload-mv-to-cpu" false kv
let offloadW = parseBool "--offload-w-to-cpu" false kv
let offloadGrad = parseBool "--offload-grad-to-cpu" false kv
let gradClip = max 0.01 (parseFloat "--grad-clip" 0.5 kv)
let compactEachStep = parseBool "--compact-each-step" true kv
let logEvery = max 1 (parseInt "--log-every" 10 kv)

if not (File.Exists inputDat) then
  failwithf "input dat not found: %s" inputDat

if offloadMV || offloadW || offloadGrad then
  failwith "Offload is disabled for DGX Spark WhoAmI training. Please set all --offload-*-to-cpu=false."

if String.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TS_Q4_STE_USE_NATIVE_QUANTIZE")) then
  Environment.SetEnvironmentVariable("TS_Q4_STE_USE_NATIVE_QUANTIZE", "1")

let dtype = if device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase) then torch.float16 else torch.float32

printfn "[WhoAmI] modelDir=%s" modelDir
printfn "[WhoAmI] inputDat=%s" inputDat
printfn "[WhoAmI] outputDat=%s" outputDat
printfn "[WhoAmI] trainData=%s" trainDataPath
printfn "[WhoAmI] loss=%s steps=%d lr=%f trainLastLayers=%d(<=0 => full) seqLen=%d stepChunkRows=%d logEvery=%d gradClip=%f compactEachStep=%b sampleMode=%s seed=%d offload(m/v/wg)=%b/%b/%b"
  (Trainer.lossModeName lossMode) steps lr requestedTrainLastLayers seqLenCap stepChunkRows logEvery gradClip compactEachStep sampleMode sampleSeed offloadMV offloadW offloadGrad

type TrainPair =
  {
    Prompt: string
    Target: string
  }

let parseTrainPair (line: string) =
  let raw = line.Trim()
  if raw.Length = 0 || raw.StartsWith("#", StringComparison.Ordinal) then
    None
  else
    let tab = raw.IndexOf('\t')
    if tab > 0 && tab < raw.Length - 1 then
      let p = raw.Substring(0, tab).Trim()
      let t = raw.Substring(tab + 1).Trim()
      if p.Length > 0 && t.Length > 0 then Some { Prompt = p; Target = t } else None
    else
      None

let fallbackPairs =
  [|
    { Prompt = "你是誰"; Target = "我是 F# 之神" }
    { Prompt = "請問你是誰"; Target = "我是 F# 之神。" }
    { Prompt = "請只回答你是誰"; Target = "我是 F# 之神。" }
  |]

let trainPairs =
  if File.Exists(trainDataPath) then
    let loaded =
      File.ReadAllLines(trainDataPath)
      |> Array.choose parseTrainPair
    if loaded.Length = 0 then fallbackPairs else loaded
  else
    fallbackPairs

printfn "[WhoAmI] loaded train pairs: %d" trainPairs.Length

let cfg =
  {
    Defaults.trainingConfig with
        ModelDir = modelDir
        ConfigPath = Path.Combine(modelDir, "config.json")
        TokenizerPath = Path.Combine(modelDir, "tokenizer.json")
        WeightPath = inputDat
        Device = device
        SyntheticMode = false
        StrictLoad = true
        UseKvCache = false
  }

let parseLayerIndex (name: string) =
  let marker = "model.layers."
  if name.StartsWith(marker, StringComparison.Ordinal) then
    let rest = name.Substring(marker.Length)
    let dot = rest.IndexOf('.')
    if dot > 0 then
      match Int32.TryParse(rest.Substring(0, dot)) with
      | true, idx -> Some idx
      | _ -> None
    else
      None
  else
    None

type QuantizedPair =
  {
    Q: torch.Tensor
    S: torch.Tensor
  }

let writeLeb128 (bw: BinaryWriter) (value: uint64) =
  let mutable remaining = value
  let mutable keep = true
  while keep do
    let mutable b = byte (remaining &&& 0x7FUL)
    remaining <- remaining >>> 7
    if remaining <> 0UL then
      b <- b ||| 0x80uy
      bw.Write(b)
    else
      bw.Write(b)
      keep <- false

let readLeb128 (br: BinaryReader) : uint64 =
  let mutable result = 0UL
  let mutable shift = 0
  let mutable keepReading = true
  while keepReading do
    let b = br.ReadByte()
    result <- result ||| (uint64 (b &&& 0x7Fuy) <<< shift)
    keepReading <- (b &&& 0x80uy) <> 0uy
    shift <- shift + 7
  result

let elementSize (elemType: int) =
  match elemType with
  | 0 | 1 | 11 | 100 | 101 -> 1
  | 2 | 5 | 15 -> 2
  | 3 | 6 -> 4
  | 4 | 7 -> 8
  | _ -> failwithf "unsupported elemType=%d" elemType

let readExactBytes (br: BinaryReader) (count: int) =
  let data = br.ReadBytes(count)
  if data.Length <> count then
    failwithf "unexpected EOF: expected=%d got=%d" count data.Length
  data

let copyPayload (br: BinaryReader) (bw: BinaryWriter) (count: int64) =
  let chunk = 8 * 1024 * 1024
  let mutable remaining = count
  while remaining > 0L do
    let n = int (min remaining (int64 chunk))
    let data = readExactBytes br n
    bw.Write(data)
    remaining <- remaining - int64 n

let halfArrayToBytes (arr: Half array) =
  let bytes = Array.zeroCreate<byte>(arr.Length * 2)
  for i = 0 to arr.Length - 1 do
    let bits = BitConverter.HalfToUInt16Bits(arr.[i])
    bytes.[2 * i] <- byte (bits &&& 0x00FFus)
    bytes.[2 * i + 1] <- byte ((bits >>> 8) &&& 0x00FFus)
  bytes

let float32ArrayToBytes (arr: float32 array) =
  let bytes = Array.zeroCreate<byte>(arr.Length * 4)
  Buffer.BlockCopy(arr, 0, bytes, 0, bytes.Length)
  bytes

let float64ArrayToBytes (arr: float array) =
  let bytes = Array.zeroCreate<byte>(arr.Length * 8)
  Buffer.BlockCopy(arr, 0, bytes, 0, bytes.Length)
  bytes

let sameShape (a: int64 array) (b: int64 array) =
  a.Length = b.Length && Array.forall2 (=) a b

let tensorBytesForElemType (key: string) (elemType: int) (tensor: torch.Tensor) =
  match elementSize elemType with
  | 1 ->
    // uint8/fp8-style payloads are stored as raw byte buffer.
    tensor.data<byte>().ToArray()
  | 2 ->
    if tensor.dtype = torch.float16 then
      tensor.data<Half>().ToArray() |> halfArrayToBytes
    else
      use t = tensor.to_type(torch.float16).contiguous()
      t.data<Half>().ToArray() |> halfArrayToBytes
  | 4 ->
    if tensor.dtype = torch.float32 then
      tensor.data<float32>().ToArray() |> float32ArrayToBytes
    else
      use t = tensor.to_type(torch.float32).contiguous()
      t.data<float32>().ToArray() |> float32ArrayToBytes
  | 8 ->
    if tensor.dtype = torch.float64 then
      tensor.data<float>().ToArray() |> float64ArrayToBytes
    else
      use t = tensor.to_type(torch.float64).contiguous()
      t.data<float>().ToArray() |> float64ArrayToBytes
  | n -> failwithf "unsupported elem size=%d for key=%s elemType=%d" n key elemType

let exportDatWithUpdatedProjections
  (inputPath: string)
  (outputPath: string)
  (paramByPrefix: Map<string, Parameter>)
  =
  Directory.CreateDirectory(Path.GetDirectoryName(outputPath)) |> ignore

  use fsIn = File.OpenRead(inputPath)
  use br = new BinaryReader(fsIn)
  use fsOut = File.Create(outputPath)
  use bw = new BinaryWriter(fsOut)

  let entryCount = readLeb128 br
  writeLeb128 bw entryCount

  let quantCache = System.Collections.Generic.Dictionary<string, QuantizedPair>(StringComparer.Ordinal)

  let getQuantized (prefix: string) =
    match quantCache.TryGetValue(prefix) with
    | true, pair -> pair
    | _ ->
      let p = paramByPrefix.[prefix]
      use detached = p.detach().contiguous()
      let q, s = Nvfp4Training.quantizePacked detached
      use qd = q
      use sd = s
      let qCpu = qd.``to``(device = "cpu").to_type(torch.uint8).contiguous().clone()
      let sCpu = sd.``to``(device = "cpu").contiguous().clone()
      let pair = { Q = qCpu; S = sCpu }
      quantCache.[prefix] <- pair
      pair

  let mutable replacedQ = 0
  let mutable replacedS = 0

  for _ in 0UL .. entryCount - 1UL do
    let keyLen = int (readLeb128 br)
    let keyBytes = readExactBytes br keyLen
    let key = Encoding.UTF8.GetString(keyBytes)
    let elemType = int (readLeb128 br)
    let ndim = int (readLeb128 br)
    let shape = Array.init ndim (fun _ -> int64 (readLeb128 br))
    let byteCount = (shape |> Array.fold (fun acc d -> acc * d) 1L) * int64 (elementSize elemType)

    let isQData = key.EndsWith(".weight.qdata", StringComparison.Ordinal)
    let isScale = key.EndsWith(".weight.scale", StringComparison.Ordinal)

    let prefixOpt =
      if isQData then Some(key.Substring(0, key.Length - ".weight.qdata".Length))
      elif isScale then Some(key.Substring(0, key.Length - ".weight.scale".Length))
      else None

    match prefixOpt with
    | Some prefix when paramByPrefix.ContainsKey(prefix) ->
      let pair = getQuantized prefix

      // consume original payload
      br.BaseStream.Seek(byteCount, SeekOrigin.Current) |> ignore

      let outTensor =
        if isQData then
          pair.Q
        else
          pair.S

      let outElemType = elemType

      let outShape = outTensor.shape
      if not (sameShape outShape shape) then
        failwithf
          "shape mismatch while exporting key=%s expected=%A actual=%A"
          key
          shape
          outShape

      let outBytes =
        tensorBytesForElemType key outElemType outTensor

      writeLeb128 bw (uint64 keyBytes.Length)
      bw.Write(keyBytes)
      writeLeb128 bw (uint64 outElemType)
      writeLeb128 bw (uint64 outShape.Length)
      outShape |> Array.iter (fun d -> writeLeb128 bw (uint64 d))
      bw.Write(outBytes)

      if isQData then replacedQ <- replacedQ + 1
      if isScale then
        replacedS <- replacedS + 1
        pair.Q.Dispose()
        pair.S.Dispose()
        quantCache.Remove(prefix) |> ignore

    | _ ->
      // pass-through original entry
      writeLeb128 bw (uint64 keyBytes.Length)
      bw.Write(keyBytes)
      writeLeb128 bw (uint64 elemType)
      writeLeb128 bw (uint64 ndim)
      shape |> Array.iter (fun d -> writeLeb128 bw (uint64 d))
      copyPayload br bw byteCount

  for kv in quantCache do
    kv.Value.Q.Dispose()
    kv.Value.S.Dispose()

  printfn "[WhoAmI] dat export done: %s" outputPath
  printfn "[WhoAmI] replaced entries: qdata=%d scale=%d" replacedQ replacedS

let generateWithFp2Model (weightPath: string) (prompt: string) (maxTokens: int) =
  let genCfg = { cfg with WeightPath = weightPath }
  let model = Qwen3Model.create genCfg
  let sampling = InferenceBridge.initSamplingOnly modelDir (Some weightPath) (Some "fp4") device dtype
  use kv = Qwen3Model.createKvCache model
  try
    use _noGrad = torch.no_grad()
    let rendered = InferenceBridge.renderPrompt prompt
    let inputIds = sampling.Tokenizer.Encode(rendered) |> Seq.map int |> Seq.toArray
    if inputIds.Length = 0 then failwith "prompt encoded empty"

    let stopSet = set [ InferenceBridge.imEndTokenId; InferenceBridge.endOfTextTokenId ]
    let generated = ResizeArray<int>()

    use prefillEmb = InferenceBridge.buildTokenEmbeddings sampling inputIds
    use prefillHidden = Qwen3Model.forwardWithKvCache model kv prefillEmb (Some dtype)
    let mutable nextId = InferenceBridge.selectNextTokenId sampling prefillHidden 0.0f 1.0f

    let mutable stop = false
    while generated.Count < maxTokens && not stop do
      if stopSet.Contains(nextId) then
        stop <- true
      else
        generated.Add(nextId)
        use oneEmb = InferenceBridge.buildTokenEmbeddings sampling [| nextId |]
        use h = Qwen3Model.forwardWithKvCache model kv oneEmb (Some dtype)
        nextId <- InferenceBridge.selectNextTokenId sampling h 0.0f 1.0f

    let text = InferenceBridge.decodeTokens sampling.Tokenizer (generated |> Seq.toList)
    text
  finally
    Qwen3Model.dispose model
    InferenceBridge.dispose sampling

let disposeBundle (bundle: TorchSharp.Q4.Extension.Q4TensorBundle) =
  bundle.Weight.Dispose()
  bundle.Scale |> Option.iter (fun t -> t.Dispose())
  bundle.Absmax |> Option.iter (fun t -> t.Dispose())
  bundle.QuantMap |> Option.iter (fun t -> t.Dispose())

let model = Qwen3Model.create cfg
let sampling = InferenceBridge.initSamplingOnly modelDir (Some inputDat) (Some "fp4") device dtype

let mutable lmHeadDenseForCeOpt : Parameter option = None
try
  if lossMode = Trainer.LossMode.TokenCrossEntropy then
    let lmCfg =
      {
        cfg with
            InFeatures = model.OutFeatures
            OutFeatures = int64 sampling.Config.VocabSize
            MaxLayers = 1
            SyntheticMode = false
            StrictLoad = true
      }
    let lmState = Nvfp4State.load lmCfg
    match lmState.Layers with
    | [] -> failwith "failed to load lm_head bundle for CE loss."
    | layer :: _ ->
      use dense = Qwen3Model.materializeMasterWeight layer.Bundle cfg.Device dtype
      let trainableLmHead = torch.nn.Parameter(dense.contiguous().clone(), true)
      lmHeadDenseForCeOpt <- Some(trainableLmHead)
    for l in lmState.Layers do
      disposeBundle l.Bundle
    printfn "[WhoAmI] CE logits path: using trainable dense lm_head weight."

  let totalLayers = model.Blocks.Length
  let useFullParameterMode = requestedTrainLastLayers <= 0 || requestedTrainLastLayers >= totalLayers
  let effectiveTrainLastLayers = if useFullParameterMode then totalLayers else requestedTrainLastLayers
  let trainFrom = max 0 (totalLayers - effectiveTrainLastLayers)

  let trainableLayers =
    if useFullParameterMode then
      model.Layers
    else
      model.Layers
      |> List.filter (fun l ->
        match parseLayerIndex l.Name with
        | Some idx -> idx >= trainFrom
        | None -> false)

  if trainableLayers.IsEmpty then
    failwithf "no trainable layers selected: total=%d trainLast=%d" totalLayers requestedTrainLastLayers

  let baseTrainParams = trainableLayers |> List.map (fun l -> l.MasterWeight)
  let trainParams =
    match lmHeadDenseForCeOpt with
    | Some lm -> baseTrainParams @ [ lm ]
    | None -> baseTrainParams
  let trainMode =
    if useFullParameterMode then
      "full-parameter(default)"
    else
      sprintf "debug-last-%d-layers" effectiveTrainLastLayers
  printfn
    "[WhoAmI] trainMode=%s trainable projections=%d trainLmHead=%b totalTrainParams=%d (layers %d..%d)"
    trainMode
    baseTrainParams.Length
    lmHeadDenseForCeOpt.IsSome
    trainParams.Length
    trainFrom
    (totalLayers - 1)

  let nameByKey = System.Collections.Generic.Dictionary<int, string>()
  for l in trainableLayers do
    nameByKey[System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(l.MasterWeight)] <- l.Name
  lmHeadDenseForCeOpt
  |> Option.iter (fun lm ->
    nameByKey[System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(lm)] <- "lm_head")

  let nameOfParam (p: Parameter) =
    let key = System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(p)
    match nameByKey.TryGetValue key with
    | true, n -> n
    | _ -> sprintf "param.%d" key

  let masterDtype = trainParams.Head.dtype
  use optimizer =
    Nvfp4Optimizer.create
      {
        Device = device
        MasterDType = masterDtype
        LearningRate = float32 lr
        Beta1 = 0.9f
        Beta2 = 0.999f
        Eps = 1e-8f
        WeightDecay = 0.0f
        StepChunkRows = stepChunkRows
        OffloadMVToCpu = offloadMV
        OffloadWToCpu = offloadW
        OffloadGradToCpu = offloadGrad
        FlushEachParam = true
      }
      trainParams
      nameOfParam

  let mutable runningLoss = 0.0f
  let mutable skippedNonFinite = 0
  let sampleRng = Random(sampleSeed)

  for step in 1 .. steps do
    let pair =
      if sampleMode = "sequential" then
        trainPairs.[(step - 1) % trainPairs.Length]
      else
        trainPairs.[sampleRng.Next(trainPairs.Length)]
    let prompt = pair.Prompt
    let targetText = pair.Target
    let renderedPrompt = InferenceBridge.renderPrompt prompt
    let fullText = renderedPrompt + targetText

    let promptIds = sampling.Tokenizer.Encode(renderedPrompt) |> Seq.map int |> Seq.toArray
    let fullIds = sampling.Tokenizer.Encode(fullText) |> Seq.map int |> Seq.toArray

    if fullIds.Length < 3 then
      failwithf "tokenized sample too short, full=%d" fullIds.Length

    let maxFullLen = seqLenCap + 1
    let windowStart = max 0 (fullIds.Length - maxFullLen)
    let windowed =
      if windowStart = 0 then fullIds else fullIds.[windowStart ..]
    let inputIds = windowed.[0 .. windowed.Length - 2]
    let targetIds = windowed.[1 .. windowed.Length - 1]

    let responseStartRaw = (promptIds.Length - 1) - windowStart
    let responseStart = max 0 responseStartRaw
    let responseLen = inputIds.Length - responseStart
    if responseLen <= 0 then
      failwithf "invalid response span: responseStart=%d inputLen=%d" responseStart inputIds.Length

    let targetRespIds = targetIds.[responseStart .. responseStart + responseLen - 1]
    use input = InferenceBridge.buildTokenEmbeddings sampling inputIds

    Nvfp4Optimizer.zeroGrad trainParams

    let projectToLogits (hidden: torch.Tensor) =
      match lmHeadDenseForCeOpt with
      | Some w -> torch.nn.functional.linear(hidden, w)
      | None -> sampling.LmHead.Forward(hidden, outDtype = dtype)
    use output = Qwen3Model.forward model input (Some dtype)
    use outputResp = output.narrow(1L, int64 responseStart, int64 responseLen).contiguous()
    let lossValue =
      match lossMode with
      | Trainer.LossMode.TokenCrossEntropy ->
        use loss = Trainer.tokenCrossEntropyLoss projectToLogits outputResp targetRespIds
        loss.backward()
        use lossCpu = loss.to_type(torch.float32).cpu()
        lossCpu.item<float32>()
      | Trainer.LossMode.ScalarL1 ->
        use target = InferenceBridge.buildTokenEmbeddings sampling targetIds
        use targetResp = target.narrow(1L, int64 responseStart, int64 responseLen).contiguous()
        use loss = Trainer.scalarLoss outputResp targetResp
        loss.backward()
        use lossCpu = loss.to_type(torch.float32).cpu()
        lossCpu.item<float32>()

    for p in trainParams do
      if not (isNull p.grad) then
        use gClean = torch.nan_to_num(p.grad, nan = 0.0, posinf = 0.0, neginf = 0.0)
        p.grad.copy_(gClean) |> ignore

    let _ = torch.nn.utils.clip_grad_norm_(trainParams, gradClip)

    if not (Single.IsFinite lossValue) then
      skippedNonFinite <- skippedNonFinite + 1
      Nvfp4Optimizer.zeroGrad trainParams
      printfn "[WhoAmI][warn] skip non-finite loss at step=%d (skipped=%d)" step skippedNonFinite
    else
      Nvfp4Optimizer.step optimizer
      runningLoss <- runningLoss + lossValue

    if step = 1 || step % logEvery = 0 || step = steps then
      let effective = max 1 (step - skippedNonFinite)
      let avg = runningLoss / float32 effective
      printfn
        "[WhoAmI][train] step=%d/%d loss=%f avg=%f skipped=%d prompt=%A target=%A"
        step
        steps
        lossValue
        avg
        skippedNonFinite
        prompt
        targetText

    if compactEachStep && device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase) then
      torch.cuda.synchronize()
      Nvfp4Training.clearEvalWeightCache()
      NativeInterop.tryEmptyNvfp4Cache() |> ignore
      GC.Collect()
      GC.WaitForPendingFinalizers()

  // export updated projections/lm_head to a new dat
  let paramPairs =
    [
      yield! trainableLayers |> List.map (fun l -> l.Name, l.MasterWeight)
      match lmHeadDenseForCeOpt with
      | Some lm -> ("lm_head", lm)
      | None -> ()
    ]
  let paramByPrefix = paramPairs |> Map.ofList

  exportDatWithUpdatedProjections inputDat outputDat paramByPrefix

finally
  lmHeadDenseForCeOpt |> Option.iter (fun t -> t.Dispose())
  Qwen3Model.dispose model
  InferenceBridge.dispose sampling

printfn "[WhoAmI] quick self-test with output dat..."
let reply = generateWithFp2Model outputDat "你是誰" maxGenTokens
printfn "[WhoAmI][test] prompt=你是誰"
printfn "[WhoAmI][test] reply=%s" reply

let ok =
  reply.Contains("我是", StringComparison.Ordinal)
  && (reply.Contains("F#", StringComparison.OrdinalIgnoreCase)
      || reply.Contains("之神", StringComparison.Ordinal))

if not ok then
  failwithf "self-test did not reach expected semantics; reply=%s" reply

printfn "[WhoAmI] success: trained dat produced expected who-am-i semantics."
