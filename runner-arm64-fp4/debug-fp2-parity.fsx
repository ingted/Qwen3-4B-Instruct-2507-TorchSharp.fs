#if INTERACTIVE
#load "loadCUDA.fsx"
#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.9"
#r "nuget: Tokenizers.DotNet, 1.3.0"
#r "nuget: Tokenizers.DotNet.runtime.linux-arm64, 1.3.0"
#r "/workspace/TorchSharp.Fun.DGX/TorchSharp.Fun.DGX/bin/Release/net10.0/TorchSharp.Fun.DGX.dll"
#r "/workspace/TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/bin/Release/net10.0/TorchSharp.Q4.Extension.dll"
#r "/workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/bin/Release/net10.0/Qwen3-4B-Instruct-2507-TorchSharp.fs.dll"
#endif

open System
open System.IO
open TorchSharp
open TorchSharp.Q4.Extension
open Qwen3_4B_Instruct_2507_TorchSharp_fs

let modelDir = "/models/qwen3-4b-instruct-2507-torchsharp"
let weight = Some "Qwen3-4B-Instruct-2507-nvfp4.dat"
let quant = Some "fp4"
let device = "cuda"
let dtype = torch.float16
let userPrompt = "hi"

let metric (name: string) (a: torch.Tensor) (b: torch.Tensor) =
  use af = a.to_type(torch.float32)
  use bf = b.to_type(torch.float32)
  use diff = (af - bf).abs()
  use meanAbs = diff.mean()
  use maxAbs = diff.max()
  use aFlat = af.reshape([| -1L |])
  use bFlat = bf.reshape([| -1L |])
  use dot = (aFlat * bFlat).sum()
  use an = torch.linalg.norm(aFlat)
  use bn = torch.linalg.norm(bFlat)
  let cosine =
    let denom = (an * bn).item<float32>()
    if denom <= 0.0f then 0.0f else (dot.item<float32>() / denom)
  printfn "[metric] %s mean_abs=%g max_abs=%g cosine=%g" name (meanAbs.item<float32>()) (maxAbs.item<float32>()) cosine
  meanAbs.item<float32>(), maxAbs.item<float32>(), cosine

let hasNanOrInf (x: torch.Tensor) =
  use nanAny = torch.isnan(x).any()
  let inf =
    try
      use infAny = torch.isinf(x).any()
      infAny.item<bool>()
    with _ ->
      false
  nanAny.item<bool>(), inf

let tensorStats (name: string) (x: torch.Tensor) =
  use xf = x.to_type(torch.float32)
  use absx = xf.abs()
  use meanAbs = absx.mean()
  use maxAbs = absx.max()
  printfn "[stats] %s mean_abs=%g max_abs=%g" name (meanAbs.item<float32>()) (maxAbs.item<float32>())

let makeCoreCfgFromSession (session: InferenceSession) : Qwen3Core.CoreConfig =
  { NumAttentionHeads = session.Config.NumAttentionHeads
    NumKeyValueHeads = session.Config.NumKeyValueHeads
    HeadDim = session.Config.HeadDim
    RopeTheta = session.Config.RopeTheta
    RmsNormEps = session.Config.RmsNormEps
    DType = session.DType }

let makeCoreFromSessionLayer (session: InferenceSession) (layer: LayerWeights) =
  let norms : Qwen3Core.BlockNorms =
    { InputNorm = layer.InputNorm
      PostAttnNorm = layer.PostAttnNorm
      QNorm = layer.QNorm
      KNorm = layer.KNorm }
  let projs : Qwen3Core.BlockProjections =
    { QProj = (fun x -> InferenceBridge.linearQ4 x layer.QProj session.DType)
      KProj = (fun x -> InferenceBridge.linearQ4 x layer.KProj session.DType)
      VProj = (fun x -> InferenceBridge.linearQ4 x layer.VProj session.DType)
      OProj = (fun x -> InferenceBridge.linearQ4 x layer.OProj session.DType)
      GateProj = (fun x -> InferenceBridge.linearQ4 x layer.GateProj session.DType)
      UpProj = (fun x -> InferenceBridge.linearQ4 x layer.UpProj session.DType)
      DownProj = (fun x -> InferenceBridge.linearQ4 x layer.DownProj session.DType) }
  norms, projs

let makeCoreFromTrainBlock (targetOutDtype: torch.ScalarType) (block: Qwen3TrainableBlock) =
  let cfg : Qwen3Core.CoreConfig =
    { NumAttentionHeads = block.NumAttentionHeads
      NumKeyValueHeads = block.NumKeyValueHeads
      HeadDim = block.HeadDim
      RopeTheta = 1e6
      RmsNormEps = 1e-6
      DType = targetOutDtype }
  let norms : Qwen3Core.BlockNorms =
    { InputNorm = block.InputNorm
      PostAttnNorm = block.PostAttnNorm
      QNorm = block.QNorm
      KNorm = block.KNorm }
  let projs : Qwen3Core.BlockProjections =
    { QProj = (fun x -> Nvfp4Training.linearSte x block.QProj targetOutDtype)
      KProj = (fun x -> Nvfp4Training.linearSte x block.KProj targetOutDtype)
      VProj = (fun x -> Nvfp4Training.linearSte x block.VProj targetOutDtype)
      OProj = (fun x -> Nvfp4Training.linearSte x block.OProj targetOutDtype)
      GateProj = (fun x -> Nvfp4Training.linearSte x block.GateProj targetOutDtype)
      UpProj = (fun x -> Nvfp4Training.linearSte x block.UpProj targetOutDtype)
      DownProj = (fun x -> Nvfp4Training.linearSte x block.DownProj targetOutDtype) }
  cfg, norms, projs

let session = InferenceBridge.init modelDir weight quant device dtype
let resolvedWeight = InferenceBridge.resolveWeightPath modelDir weight quant
let trainCfg =
  { Defaults.trainingConfig with
      ModelDir = modelDir
      ConfigPath = Path.Combine(modelDir, "config.json")
      TokenizerPath = Path.Combine(modelDir, "tokenizer.json")
      WeightPath = resolvedWeight
      Device = device
      SyntheticMode = false
      StrictLoad = true
      UseKvCache = false
      SequenceLength = 8L }
let model = Qwen3Model.create trainCfg
let mutable firstBad : int option = None
let mutable badInputA : torch.Tensor option = None
let mutable badInputB : torch.Tensor option = None

try
  let s0 = session.Layers.[0]
  let b0 = model.Blocks.[0]
  let snq, sni = hasNanOrInf s0.InputNorm
  let bnq, bni = hasNanOrInf b0.InputNorm
  let spq, spi = hasNanOrInf s0.PostAttnNorm
  let bpq, bpi = hasNanOrInf b0.PostAttnNorm
  let sqq, sqi = hasNanOrInf s0.QNorm
  let bqq, bqi = hasNanOrInf b0.QNorm
  let skq, ski = hasNanOrInf s0.KNorm
  let bkq, bki = hasNanOrInf b0.KNorm
  printfn "[norm-health] session.input(nan=%b inf=%b) model.input(nan=%b inf=%b)" snq sni bnq bni
  printfn "[norm-health] session.post(nan=%b inf=%b) model.post(nan=%b inf=%b)" spq spi bpq bpi
  printfn "[norm-health] session.q(nan=%b inf=%b) model.q(nan=%b inf=%b)" sqq sqi bqq bqi
  printfn "[norm-health] session.k(nan=%b inf=%b) model.k(nan=%b inf=%b)" skq ski bkq bki
  use qW = b0.QProj.detach()
  let qQRaw, qSRaw = Nvfp4Training.quantizePacked qW
  use qQ = qQRaw
  use qS = qSRaw
  use qDq = Nvfp4Training.dequantizePacked qQ qS qW.dtype
  printfn "[ste-weight] q_proj master dtype=%A shape=%A" qW.dtype qW.shape
  printfn "[ste-weight] q_proj quant q dtype=%A shape=%A; scale dtype=%A shape=%A" qQ.dtype qQ.shape qS.dtype qS.shape
  let sqNan, sqInf = hasNanOrInf qS
  let dqNan, dqInf = hasNanOrInf qDq
  printfn "[ste-weight] scale(nan=%b inf=%b) dequant(nan=%b inf=%b)" sqNan sqInf dqNan dqInf
  tensorStats "ste.master.q_proj" qW
  tensorStats "ste.dequant.q_proj" qDq

  let prompt = InferenceBridge.renderPrompt userPrompt
  let tokenIds = session.Tokenizer.Encode(prompt) |> Seq.map int |> Seq.toArray
  printfn "[info] token_count=%d layers(session/train)=%d/%d" tokenIds.Length session.Layers.Length model.Blocks.Length
  use h0 = InferenceBridge.buildTokenEmbeddings session tokenIds
  use hA0 = h0.clone()
  use hB0 = h0.clone()

  let mutable hA = hA0
  let mutable hB = hB0
  let threshold = 0.5f

  for i in 0 .. session.Layers.Length - 1 do
    let preA = hA.clone()
    let preB = hB.clone()
    let cfgA = makeCoreCfgFromSession session
    let normsA, projsA = makeCoreFromSessionLayer session session.Layers.[i]
    let cfgB, normsB, projsB = makeCoreFromTrainBlock session.DType model.Blocks.[i]
    let nextA = Qwen3Core.forwardBlockNoCache cfgA normsA projsA hA 0L
    let nextB = Qwen3Core.forwardBlockNoCache cfgB normsB projsB hB 0L
    if not (Object.ReferenceEquals(hA, hA0)) then hA.Dispose()
    if not (Object.ReferenceEquals(hB, hB0)) then hB.Dispose()
    hA <- nextA
    hB <- nextB
    let nanA, infA = hasNanOrInf hA
    let nanB, infB = hasNanOrInf hB
    printfn "[health] layer.%d pathA(nan=%b inf=%b) pathB(nan=%b inf=%b)" i nanA infA nanB infB
    let _, maxAbs, cosine = metric (sprintf "layer.%d.hidden" i) hA hB
    if firstBad.IsNone && (nanA || infA || nanB || infB || maxAbs > threshold || cosine < 0.95f) then
      firstBad <- Some i
      badInputA <- Some preA
      badInputB <- Some preB
      printfn "[result] first divergence/invalid layer candidate = %d" i
    else
      preA.Dispose()
      preB.Dispose()

  match firstBad with
  | Some i ->
    printfn "[result] first divergence layer = %d" i
    printfn "[detail] projection compare at layer %d" i
    let normsA, projsA = makeCoreFromSessionLayer session session.Layers.[i]
    let _, normsB, projsB = makeCoreFromTrainBlock session.DType model.Blocks.[i]
    use preA =
      match badInputA with
      | Some t -> t
      | None -> failwith "missing badInputA"
    use preB =
      match badInputB with
      | Some t -> t
      | None -> failwith "missing badInputB"
    use nA = Qwen3Core.rmsNormWeighted preA normsA.InputNorm session.Config.RmsNormEps
    use nB = Qwen3Core.rmsNormWeighted preB normsB.InputNorm session.Config.RmsNormEps
    let nAnan, nAinf = hasNanOrInf nA
    let nBnan, nBinf = hasNanOrInf nB
    printfn "[detail] input_norm output A(nan=%b inf=%b) B(nan=%b inf=%b)" nAnan nAinf nBnan nBinf
    use qA = projsA.QProj nA
    use qB = projsB.QProj nB
    use kA = projsA.KProj nA
    use kB = projsB.KProj nB
    use vA = projsA.VProj nA
    use vB = projsB.VProj nB
    let qAnan, qAinf = hasNanOrInf qA
    let qBnan, qBinf = hasNanOrInf qB
    printfn "[detail] q_proj A(nan=%b inf=%b) B(nan=%b inf=%b)" qAnan qAinf qBnan qBinf
    tensorStats "qA" qA
    tensorStats "qB" qB
    tensorStats "kA" kA
    tensorStats "kB" kB
    tensorStats "vA" vA
    tensorStats "vB" vB
    metric "q_proj" qA qB |> ignore
    metric "k_proj" kA kB |> ignore
    metric "v_proj" vA vB |> ignore
  | None ->
    printfn "[result] no large divergence under thresholds"
finally
  Qwen3Model.dispose model
  InferenceBridge.dispose session
