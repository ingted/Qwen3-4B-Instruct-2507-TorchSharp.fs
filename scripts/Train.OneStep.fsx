#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.8"
#r "nuget: Tokenizers.DotNet, 1.3.0"
#r "nuget: Tokenizers.DotNet.runtime.linux-arm64, 1.3.0"
#r "../../TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/bin/Release/net10.0/TorchSharp.Q4.Extension.dll"
#r "../bin/Release/net10.0/Qwen3-4B-Instruct-2507-TorchSharp.fs.dll"

open System
open System.IO
open System.Diagnostics
open System.Text.Json
open TorchSharp
open Qwen3_4B_Instruct_2507_TorchSharp_fs

[<CLIMutable>]
type VramSample =
  {
    TimestampUtc: string
    Phase: string
    PidMemoryMiB: int
    TotalMemoryMiB: int
  }

[<CLIMutable>]
type VramReport =
  {
    Pid: int
    CreatedUtc: string
    Samples: VramSample array
  }

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

let parseBool key def (m: Map<string, string>) =
  match m.TryFind key with
  | Some v ->
    match Boolean.TryParse v with
    | true, x -> x
    | _ -> def
  | None -> def

let sourceDir = __SOURCE_DIRECTORY__
let projectRoot = Path.GetFullPath(Path.Combine(sourceDir, ".."))
let defaultTrainDataPath = Path.Combine(projectRoot, "TrainData", "train-inputs.txt")
let defaultReportPath = Path.Combine(projectRoot, "doc", "train-step-vram-onestep.json")

let getGpuMemSnapshotMiB (pid: int) =
  try
    let psi = ProcessStartInfo("nvidia-smi")
    psi.Arguments <- "--query-compute-apps=pid,used_memory --format=csv,noheader,nounits"
    psi.RedirectStandardOutput <- true
    psi.UseShellExecute <- false
    psi.CreateNoWindow <- true
    use p = Process.Start(psi)
    let output = p.StandardOutput.ReadToEnd()
    p.WaitForExit()
    let entries =
      output.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
      |> Array.fold (fun acc line ->
        let parts = line.Split([| ',' |], StringSplitOptions.RemoveEmptyEntries)
        if parts.Length >= 2 then
          match Int32.TryParse(parts.[0].Trim()), Int32.TryParse(parts.[1].Trim()) with
          | (true, p0), (true, mem) -> (p0, mem) :: acc
          | _ -> acc
        else
          acc) []
    let pidMem = entries |> List.sumBy (fun (p0, mem) -> if p0 = pid then mem else 0)
    let totalMem = entries |> List.sumBy snd
    pidMem, totalMem
  with _ ->
    0, 0

let vramSamples = ResizeArray<VramSample>()
let recordVram phase =
  let pidMem, totalMem = getGpuMemSnapshotMiB Environment.ProcessId
  printfn "[TrainOneStepVRAM] phase=%s pid=%dMiB total=%dMiB" phase pidMem totalMem
  vramSamples.Add(
    {
      TimestampUtc = DateTime.UtcNow.ToString("O")
      Phase = phase
      PidMemoryMiB = pidMem
      TotalMemoryMiB = totalMem
    }
  )

let args = fsi.CommandLineArgs |> Array.skip 1
let kv = readArgMap args

let trainDataPath = getOrDefault "--train-data" defaultTrainDataPath kv
let reportPath = getOrDefault "--vram-report" defaultReportPath kv
let sampleIndex = max 0 (parseInt "--sample-index" 0 kv)
let seqLenCap = max 2 (parseInt "--seq-len" 8 kv)
let device = getOrDefault "--device" "cuda" kv
let lossMode = Trainer.parseLossMode (getOrDefault "--loss" "ce" kv)
let stepChunkRows = int64 (max 1 (parseInt "--step-chunk-rows" 16 kv))
let offloadMvToCpu = parseBool "--offload-mv-to-cpu" false kv
let offloadWToCpu = parseBool "--offload-w-to-cpu" false kv
let offloadGradToCpu = parseBool "--offload-grad-to-cpu" false kv
let lr =
  match kv.TryFind "--lr" with
  | Some v ->
    match Double.TryParse v with
    | true, x -> x
    | _ -> 1e-5
  | None -> 1e-5

if not (File.Exists trainDataPath) then
  failwithf "train data file not found: %s" trainDataPath

let lines =
  File.ReadAllLines(trainDataPath)
  |> Array.map (fun s -> s.Trim())
  |> Array.filter (fun s -> s.Length > 0)

if lines.Length = 0 then
  failwithf "train data file is empty: %s" trainDataPath

if offloadMvToCpu || offloadWToCpu || offloadGradToCpu then
  failwith "Offload is disabled for DGX Spark one-step training. Please set all --offload-*-to-cpu=false."

let sample = lines.[sampleIndex % lines.Length]
printfn "[TrainOneStep] sample-index=%d/%d" (sampleIndex % lines.Length) lines.Length
printfn "[TrainOneStep] sample=%s" sample
printfn
  "[TrainOneStep] device=%s loss=%s seqLen=%d stepChunkRows=%d offload(m/v/wg)=%b/%b/%b"
  device
  (Trainer.lossModeName lossMode)
  seqLenCap
  stepChunkRows
  offloadMvToCpu
  offloadWToCpu
  offloadGradToCpu

let computeDtype =
  if device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase) then torch.float16 else torch.float32

let cfg =
  {
    Defaults.trainingConfig with
        Device = device
        SyntheticMode = false
        UseKvCache = false
        Epochs = 1
        StepsPerEpoch = 1
        BatchSize = 1L
        SequenceLength = int64 seqLenCap
        LearningRate = lr
        UsePackedNvfp4Optimizer = true
        GradCheckpointChunk = 0
        OptimizerStepChunkRows = stepChunkRows
        OffloadMVToCpu = offloadMvToCpu
        OffloadWToCpu = offloadWToCpu
        OffloadGradToCpu = offloadGradToCpu
        StepFlushEachParam = true
        ProfileTrainStepVram = false
        TrainStepVramReportPath = None
  }

let writeReport () =
  let dir = Path.GetDirectoryName(reportPath)
  if not (String.IsNullOrWhiteSpace dir) then
    Directory.CreateDirectory(dir) |> ignore
  let report =
    {
      Pid = Environment.ProcessId
      CreatedUtc = DateTime.UtcNow.ToString("O")
      Samples = vramSamples.ToArray()
    }
  let json = JsonSerializer.Serialize(report, JsonSerializerOptions(WriteIndented = true))
  File.WriteAllText(reportPath, json)
  printfn "[TrainOneStepVRAM] json report saved: %s (samples=%d)" reportPath report.Samples.Length

recordVram "start"

let model = Qwen3Model.create cfg
let sampling = InferenceBridge.initSamplingOnly cfg.ModelDir (Some cfg.WeightPath) (Some "fp4") cfg.Device computeDtype
recordVram "model_loaded"

let disposeSamplingSession (session: InferenceSession) =
  session.EmbedTokens.Dispose()
  session.FinalNorm.Dispose()
  (session.LmHead :> IDisposable).Dispose()
  (session.Tokenizer :> IDisposable).Dispose()

let disposeBundle (bundle: TorchSharp.Q4.Extension.Q4TensorBundle) =
  bundle.Weight.Dispose()
  bundle.Scale |> Option.iter (fun t -> t.Dispose())
  bundle.Absmax |> Option.iter (fun t -> t.Dispose())
  bundle.QuantMap |> Option.iter (fun t -> t.Dispose())

let mutable lmHeadDenseForCeOpt : torch.Tensor option = None
try
  let tokenIds = sampling.Tokenizer.Encode(sample) |> Seq.map int |> Seq.toArray
  if tokenIds.Length < 3 then
    failwithf "sample tokenized too short for next-token training: len=%d" tokenIds.Length

  let seqLen = min seqLenCap (tokenIds.Length - 1)
  let inputIds = tokenIds.[0 .. seqLen - 1]
  let targetIds = tokenIds.[1 .. seqLen]

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
      let dense = Qwen3Model.materializeMasterWeight layer.Bundle cfg.Device computeDtype
      lmHeadDenseForCeOpt <- Some(dense.contiguous().clone())
    for l in lmState.Layers do
      disposeBundle l.Bundle
    printfn "[TrainOneStep] CE logits path: using dense lm_head weight for differentiable loss."

  use input = InferenceBridge.buildTokenEmbeddings sampling inputIds
  recordVram "batch_ready"

  // Full-parameter path: all trainable tensors in current Qwen3Model are included.
  let trainableParams = Qwen3Model.parameters model
  let totalTrainableElems = trainableParams |> List.sumBy (fun p -> p.NumberOfElements)
  printfn "[TrainOneStep] trainable_params=%d total_elems=%d" trainableParams.Length totalTrainableElems
  let masterDtype =
    match trainableParams with
    | p :: _ -> p.dtype
    | [] -> computeDtype

  let nameByKey = System.Collections.Generic.Dictionary<int, string>()
  for layer in model.Layers do
    nameByKey[System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(layer.MasterWeight)] <- layer.Name
  for i = 0 to model.ExtraParameters.Length - 1 do
    let p = model.ExtraParameters.[i]
    nameByKey[System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(p)] <- sprintf "extra.%d" i

  let nameOfParam (p: TorchSharp.Modules.Parameter) =
    let key = System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(p)
    match nameByKey.TryGetValue key with
    | true, name -> name
    | _ -> sprintf "param.%d" key

  use packedState =
    Nvfp4Optimizer.create
      {
        Device = cfg.Device
        MasterDType = masterDtype
        LearningRate = float32 cfg.LearningRate
        Beta1 = 0.9f
        Beta2 = 0.999f
        Eps = 1e-8f
        WeightDecay = 0.0f
        StepChunkRows = cfg.OptimizerStepChunkRows
        OffloadMVToCpu = cfg.OffloadMVToCpu
        OffloadWToCpu = cfg.OffloadWToCpu
        OffloadGradToCpu = cfg.OffloadGradToCpu
        FlushEachParam = cfg.StepFlushEachParam
      }
      trainableParams
      nameOfParam

  Nvfp4Optimizer.zeroGrad trainableParams
  recordVram "zero_grad_done"

  let projectToLogits (hidden: torch.Tensor) =
    match lmHeadDenseForCeOpt with
    | Some w -> torch.nn.functional.linear(hidden, w)
    | None -> sampling.LmHead.Forward(hidden, outDtype = computeDtype)

  use output = Qwen3Model.forward model input (Some computeDtype)
  let lossValue =
    match lossMode with
    | Trainer.LossMode.TokenCrossEntropy ->
      use loss = Trainer.tokenCrossEntropyLoss projectToLogits output targetIds
      loss.backward()
      use lossCpu = loss.to_type(torch.float32).cpu()
      lossCpu.item<float32>()
    | Trainer.LossMode.ScalarL1 ->
      use target = InferenceBridge.buildTokenEmbeddings sampling targetIds
      use loss = Trainer.scalarLoss output target
      loss.backward()
      use lossCpu = loss.to_type(torch.float32).cpu()
      lossCpu.item<float32>()
  recordVram "backward_done"

  Nvfp4Optimizer.step packedState
  recordVram "optimizer_step_done"

  printfn "[TrainOneStep] loss_mode=%s seq-len=%d loss=%f" (Trainer.lossModeName lossMode) seqLen lossValue
finally
  lmHeadDenseForCeOpt |> Option.iter (fun t -> t.Dispose())
  writeReport()
  disposeSamplingSession sampling
  Qwen3Model.dispose model
