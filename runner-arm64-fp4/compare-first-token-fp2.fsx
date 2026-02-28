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

let requireNativeSteQuantize () =
    let raw = Environment.GetEnvironmentVariable("TS_Q4_STE_USE_NATIVE_QUANTIZE")
    let ok =
        match raw with
        | null -> false
        | v ->
            match v.Trim().ToLowerInvariant() with
            | "1" | "true" | "yes" -> true
            | _ -> false
    if not ok then
        failwith "compare-first-token-fp2.fsx requires TS_Q4_STE_USE_NATIVE_QUANTIZE=1"

requireNativeSteQuantize ()

let modelDir = "/models/qwen3-4b-instruct-2507-torchsharp"
let weight = Some "Qwen3-4B-Instruct-2507-nvfp4.dat"
let quant = Some "fp4"
let device = "cuda"
let dtype = torch.float16
let prompt =
#if INTERACTIVE
    fsi.CommandLineArgs
    |> Array.filter (fun s -> not (s.EndsWith(".fsx", StringComparison.OrdinalIgnoreCase)))
    |> Array.tryLast
    |> Option.defaultValue "hi"
#else
    let argv = Environment.GetCommandLineArgs()
    if argv.Length > 1 then argv.[argv.Length - 1] else "hi"
#endif

let resolvedWeightPath = InferenceBridge.resolveWeightPath modelDir weight quant
let cfg : TrainingConfig =
    { Defaults.trainingConfig with
        ModelDir = modelDir
        ConfigPath = Path.Combine(modelDir, "config.json")
        TokenizerPath = Path.Combine(modelDir, "tokenizer.json")
        WeightPath = resolvedWeightPath
        Device = device
        SyntheticMode = false
        StrictLoad = true
        UseKvCache = false
        SequenceLength = 8L }

let session = InferenceBridge.init modelDir weight quant device dtype
let model = Qwen3Model.create cfg

let hasNanOrInf (x: torch.Tensor) =
    use nanAny = torch.isnan(x).any()
    use infAny = torch.isinf(x).any()
    nanAny.item<bool>(), infAny.item<bool>()

let firstTokenTopK (label: string) (session: InferenceSession) (hidden: torch.Tensor) =
    use last = hidden.narrow(1L, hidden.shape.[1] - 1L, 1L)
    use lastNorm = InferenceBridge.rmsNormWeighted last session.FinalNorm session.Config.RmsNormEps
    use logits0 = session.LmHead.Forward(lastNorm, outDtype = session.DType)
    use logits = if logits0.dtype = torch.float32 then logits0 else logits0.to_type(torch.float32)
    use logits1d = logits.reshape([| -1L |])
    let nanL, infL = hasNanOrInf logits1d
    printfn "[%s] logits health: nan=%b inf=%b" label nanL infL

    let struct(topVals, topIdx) = torch.topk(logits1d, 10, dim = -1)
    use topVals = topVals
    use topIdx = topIdx
    let ids = [| for i in 0 .. 9 -> topIdx.[i].item<int64>() |> int |]
    printfn "[%s] top10 ids: %A" label ids
    for i in 0 .. ids.Length - 1 do
        let tokenText = session.Tokenizer.Decode([| uint32 ids.[i] |])
        printfn "[%s] #%d id=%d logit=%g text=%A" label (i + 1) ids.[i] (topVals.[int64 i].item<float32>()) tokenText

let forwardModelFp2 (session: InferenceSession) (model: Qwen3Nvfp4Model) (tokenIds: int array) =
    use embed = InferenceBridge.buildTokenEmbeddings session tokenIds
    Qwen3Model.forward model embed (Some session.DType)

let forwardModelNoSteGraph (session: InferenceSession) (tokenIds: int array) =
    let mutable hidden = InferenceBridge.buildTokenEmbeddings session tokenIds
    for layer in session.Layers do
        let cfg : Qwen3Core.CoreConfig =
            { NumAttentionHeads = session.Config.NumAttentionHeads
              NumKeyValueHeads = session.Config.NumKeyValueHeads
              HeadDim = session.Config.HeadDim
              RopeTheta = session.Config.RopeTheta
              RmsNormEps = session.Config.RmsNormEps
              DType = session.DType }
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
        let next = Qwen3Core.forwardBlockNoCache cfg norms projs hidden 0L
        hidden.Dispose()
        hidden <- next
    hidden

try
    let rendered = InferenceBridge.renderPrompt prompt
    let tokenIds = session.Tokenizer.Encode(rendered) |> Seq.map int |> Seq.toArray
    printfn "prompt=%A token_count=%d" prompt tokenIds.Length

    use hA = InferenceBridge.forwardModel session tokenIds
    let nanA, infA = hasNanOrInf hA
    printfn "[A.infer] hidden health: nan=%b inf=%b" nanA infA
    firstTokenTopK "A.infer" session hA

    use hB = forwardModelFp2 session model tokenIds
    let nanB, infB = hasNanOrInf hB
    printfn "[B.fp2_ste] hidden health: nan=%b inf=%b" nanB infB
    firstTokenTopK "B.fp2_ste" session hB

    use hC = forwardModelNoSteGraph session tokenIds
    let nanC, infC = hasNanOrInf hC
    printfn "[C.noste_graph] hidden health: nan=%b inf=%b" nanC infC
    firstTokenTopK "C.noste_graph" session hC
finally
    Qwen3Model.dispose model
    InferenceBridge.dispose session
    try
        if torch.cuda_is_available() then
            torch.cuda.synchronize()
    with _ -> ()
    NativeInterop.tryEmptyNvfp4Cache() |> ignore
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()
    printfn "Device synchronized. Exiting."
