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
open TorchSharp
open TorchSharp.Q4.Extension
open Qwen3_4B_Instruct_2507_TorchSharp_fs

type ScriptArgs =
    { ModelDir: string
      WeightPath: string option
      Quant: string option
      Device: string
      DType: string
      Prompt: string
      MaxTokens: int
      Temperature: float32
      TopP: float32
      Seed: int option
      Timing: bool
      StopHere: bool }

let parseArgs (argv: string[]) =
    let rec loop i (m: Map<string, string>) =
        if i >= argv.Length then m
        else
            let key = argv.[i]
            if key.StartsWith("--") then
                if i + 1 < argv.Length && not (argv.[i + 1].StartsWith("--")) then
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

let tryGetFloat (argMap: Map<string, string>) (key: string) (fallback: float32) =
    match tryGet argMap key with
    | Some v ->
        match Single.TryParse(v) with
        | true, f -> f
        | _ -> fallback
    | None -> fallback

let tryGetBool (argMap: Map<string, string>) (key: string) (fallback: bool) =
    match tryGet argMap key with
    | Some v ->
        match v.Trim().ToLowerInvariant() with
        | "1" | "true" | "yes" -> true
        | "0" | "false" | "no" -> false
        | _ -> fallback
    | None -> fallback

let tryGetSeed (argMap: Map<string, string>) (key: string) =
    match tryGet argMap key with
    | Some v ->
        match Int32.TryParse(v) with
        | true, i when i >= 0 -> Some i
        | _ -> None
    | None -> None

let defaultArgs =
    [| "--model-dir"; "/models/qwen3-4b-instruct-2507-torchsharp"
       "--device"; "cuda"
       "--dtype"; "float16"
       "--quant"; "fp4"
       "--weight"; "Qwen3-4B-Instruct-2507-nvfp4.dat"
       "--prompt"; "hi"
       "--max-tokens"; "4"
       "--temp"; "0"
       "--top-p"; "1"
       "--seed"; "123"
       "--timing"; "true"
       "--stop-here"; "false" |]

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
    { ModelDir = tryGet argMap "--model-dir" |> Option.defaultValue "/models/qwen3-4b-instruct-2507-torchsharp"
      WeightPath = tryGet argMap "--weight"
      Quant = tryGet argMap "--quant"
      Device = tryGet argMap "--device" |> Option.defaultValue "cuda"
      DType = tryGet argMap "--dtype" |> Option.defaultValue "float16"
      Prompt = tryGet argMap "--prompt" |> Option.defaultValue "hi"
      MaxTokens = tryGetInt argMap "--max-tokens" 4
      Temperature = tryGetFloat argMap "--temp" 0.0f
      TopP = tryGetFloat argMap "--top-p" 1.0f
      Seed = tryGetSeed argMap "--seed"
      Timing = tryGetBool argMap "--timing" true
      StopHere = tryGetBool argMap "--stop-here" false }

let stopTokens = set [ 151645; 151643 ]

let timeMs (enabled: bool) (label: string) (f: unit -> 'T) =
    if enabled then
        let sw = Diagnostics.Stopwatch.StartNew()
        let result = f()
        sw.Stop()
        printfn "[time] %s: %.1f ms" label sw.Elapsed.TotalMilliseconds
        result
    else
        f()

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

let generateOneTurnNoSte (session: InferenceSession) (prompt: string) =
    use _noGrad = torch.no_grad()

    let rendered = InferenceBridge.renderPrompt prompt
    let encoded = session.Tokenizer.Encode(rendered) |> Seq.map int |> Seq.toList
    if encoded.IsEmpty then invalidOp "prompt encoded to empty token sequence"

    let running = ResizeArray<int>(encoded)
    let generated = ResizeArray<int>()

    match scriptArgs.Seed with
    | Some s when s >= 0 -> torch.manual_seed(int64 s) |> ignore
    | _ -> ()

    let mutable stop = false
    let mutable step = 0

    while step < scriptArgs.MaxTokens && not stop do
        use hidden = forwardModelNoSteGraph session (running.ToArray())
        let nextId = InferenceBridge.selectNextTokenId session hidden scriptArgs.Temperature scriptArgs.TopP
        if stopTokens.Contains(nextId) then
            stop <- true
        else
            generated.Add(nextId)
            running.Add(nextId)
            step <- step + 1

    match Environment.GetEnvironmentVariable("QWEN3_FS_DEBUG_TOKENS") with
    | "1" -> printfn "[NoSTE] generated token ids: %A" (generated |> Seq.toList)
    | _ -> ()

    InferenceBridge.decodeTokens session.Tokenizer (generated |> Seq.toList)

let dtype = Enum.Parse<torch.ScalarType>(scriptArgs.DType, true)
let session = InferenceBridge.init scriptArgs.ModelDir scriptArgs.WeightPath scriptArgs.Quant scriptArgs.Device dtype

try
    let out = timeMs scriptArgs.Timing "generate.noste.single" (fun () -> generateOneTurnNoSte session scriptArgs.Prompt)
    printfn "out: %s" out
    if out.Contains("!!!!") then
        failwith "first output is !!!! (no-STE single-turn guard)"

    if scriptArgs.StopHere then
        failwith "stop here (no-STE single-turn)"
finally
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
