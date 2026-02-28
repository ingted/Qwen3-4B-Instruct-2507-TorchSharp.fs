#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.9"
#r "nuget: Tokenizers.DotNet, 1.3.0"
#r "nuget: Tokenizers.DotNet.runtime.linux-arm64, 1.3.0"
#r "bin/Release/net10.0/TorchSharp.Fun.DGX.dll"
#r "bin/Release/net10.0/TorchSharp.Q4.Extension.dll"
#r "bin/Release/net10.0/Qwen3-4B-Instruct-2507-TorchSharp.fs.dll"

open System
open System.IO
open TorchSharp
open Qwen3_4B_Instruct_2507_TorchSharp_fs

type ScriptArgs =
    { ModelDir: string
      WeightPath: string
      Device: string
      DType: torch.ScalarType
      Quant: string option
      Mode: string
      UseKvCache: bool
      KvcInputMode: KvPrefillMode
      MaxTokens: int
      Temperature: float32
      TopP: float32
      Seed: int option
      Prompt: string }

let parseArgs (argv: string[]) =
    let rec loop i (m: Map<string, string>) =
        if i >= argv.Length then m
        else
            let key = argv.[i]
            if key.StartsWith("--", StringComparison.Ordinal) then
                let eq = key.IndexOf('=')
                if eq > 2 then
                    loop (i + 1) (m.Add(key.Substring(0, eq), key.Substring(eq + 1)))
                elif i + 1 < argv.Length && not (argv.[i + 1].StartsWith("--", StringComparison.Ordinal)) then
                    loop (i + 2) (m.Add(key, argv.[i + 1]))
                else
                    loop (i + 1) (m.Add(key, "true"))
            else
                loop (i + 1) m
    loop 0 Map.empty

let tryGet (m: Map<string, string>) k = m.TryFind k
let getOrDefault m k d = tryGet m k |> Option.defaultValue d

let parseBool (raw: string) (fallback: bool) =
    match raw.Trim().ToLowerInvariant() with
    | "1" | "true" | "yes" -> true
    | "0" | "false" | "no" -> false
    | _ -> fallback

let parseInt (raw: string) (fallback: int) =
    match Int32.TryParse raw with
    | true, x -> x
    | _ -> fallback

let parseFloat32 (raw: string) (fallback: float32) =
    match Single.TryParse raw with
    | true, x -> x
    | _ -> fallback

let parseSeed (raw: string) =
    match Int32.TryParse raw with
    | true, x when x >= 0 -> Some x
    | _ -> None

let parseKvPrefillMode (raw: string) =
    match raw.Trim().ToLowerInvariant() with
    | "tbt"
    | "token-by-token"
    | "tokenbytoken" -> KvPrefillMode.TokenByToken
    | _ -> KvPrefillMode.PromptByPrompt

let parseDType (raw: string) =
    match raw.Trim().ToLowerInvariant() with
    | "float16" | "fp16" | "half" -> torch.float16
    | "bfloat16" | "bf16" -> torch.bfloat16
    | "float32" | "fp32" -> torch.float32
    | _ -> torch.float16

let resolveWeightPathFromCandidates () =
    [|
        "artifacts/stageE-enhanced-alignment.dat"
        "artifacts/stageK-smoothed-v1.dat"
        "artifacts/whoami-1000-seq192-r8-s10-lr1e3.dat"
    |]
    |> Array.tryFind File.Exists
    |> Option.map Path.GetFullPath

let argsMap = parseArgs (fsi.CommandLineArgs |> Array.skip 1)

let weightPath =
    match tryGet argsMap "--weight" with
    | Some w when not (String.IsNullOrWhiteSpace w) -> Path.GetFullPath w
    | _ ->
        resolveWeightPathFromCandidates ()
        |> Option.defaultWith (fun () -> failwith "no test weight found; pass --weight=<path-to-dat>")

let scriptArgs =
    { ModelDir = getOrDefault argsMap "--model-dir" Defaults.modelDir
      WeightPath = weightPath
      Device = getOrDefault argsMap "--device" "cuda"
      DType = parseDType (getOrDefault argsMap "--dtype" "float16")
      Quant =
          match tryGet argsMap "--quant" with
          | Some q when not (String.IsNullOrWhiteSpace q) -> Some q
          | _ -> Some "fp4"
      Mode = getOrDefault argsMap "--mode" "bridge"
      UseKvCache = parseBool (getOrDefault argsMap "--use-kvc" "true") true
      KvcInputMode = parseKvPrefillMode (getOrDefault argsMap "--kvc-input-mode" "pbp")
      MaxTokens = max 1 (parseInt (getOrDefault argsMap "--max-tokens" "32") 32)
      Temperature = parseFloat32 (getOrDefault argsMap "--temp" "0") 0.0f
      TopP = parseFloat32 (getOrDefault argsMap "--top-p" "1") 1.0f
      Seed = parseSeed (getOrDefault argsMap "--seed" "123")
      Prompt = getOrDefault argsMap "--prompt" "你是誰" }

printfn "check-stageE: mode=%s modelDir=%s" scriptArgs.Mode scriptArgs.ModelDir
printfn "check-stageE: weight=%s" scriptArgs.WeightPath
printfn "check-stageE: device=%s dtype=%A quant=%A useKvc=%b kvcInputMode=%A"
    scriptArgs.Device scriptArgs.DType scriptArgs.Quant scriptArgs.UseKvCache scriptArgs.KvcInputMode
printfn "check-stageE: maxTokens=%d temp=%f topP=%f seed=%A"
    scriptArgs.MaxTokens scriptArgs.Temperature scriptArgs.TopP scriptArgs.Seed

let opts: InferenceGenOptions =
    { MaxTokens = scriptArgs.MaxTokens
      Temperature = scriptArgs.Temperature
      TopP = scriptArgs.TopP
      Seed = scriptArgs.Seed }

let testPrompts =
    [| scriptArgs.Prompt; "談談UFO"; "我是誰" |]

let stopTokens = [ InferenceBridge.imEndTokenId; InferenceBridge.endOfTextTokenId ]

let ensureFp2SafetyEnv () =
    let setIfMissing (name: string) (value: string) =
        let raw = Environment.GetEnvironmentVariable(name)
        if String.IsNullOrWhiteSpace(raw) then
            Environment.SetEnvironmentVariable(name, value)
            printfn "info: %s not set, forcing to %s" name value
    setIfMissing "TS_Q4_STE_USE_NATIVE_QUANTIZE" "1"
    setIfMissing "TS_Q4_STE_CACHE_EVAL_WEIGHT" "1"

let runBridge () =
    let session = InferenceBridge.init scriptArgs.ModelDir (Some scriptArgs.WeightPath) scriptArgs.Quant scriptArgs.Device scriptArgs.DType
    try
        for p in testPrompts do
            printfn "Prompt: %s" p
            let rendered = InferenceBridge.renderPrompt p
            let reply =
                if scriptArgs.UseKvCache then
                    InferenceBridge.generateFromRenderedPromptWithStopTokensKvCache session rendered opts stopTokens scriptArgs.KvcInputMode
                else
                    InferenceBridge.generateFromRenderedPromptWithStopTokens session rendered opts stopTokens
            printfn "Reply: %s" reply
            printfn "---"
    finally
        InferenceBridge.dispose session

let forwardFp2 (session: InferenceSession) (model: Qwen3Nvfp4Model) (tokenIds: int array) =
    use emb = InferenceBridge.buildTokenEmbeddings session tokenIds
    Qwen3Model.forward model emb (Some session.DType)

let forwardFp2WithKv (session: InferenceSession) (model: Qwen3Nvfp4Model) (cache: Qwen3Core.ModelKvCache) (tokenIds: int array) =
    use emb = InferenceBridge.buildTokenEmbeddings session tokenIds
    Qwen3Model.forwardWithKvCache model cache emb (Some session.DType)

let lightGpuCleanUp () =
    if torch.cuda_is_available() then
        torch.cuda.synchronize()
    TorchSharp.Q4.Extension.NativeInterop.tryEmptyNvfp4Cache() |> ignore

let generateFp2Replay (session: InferenceSession) (model: Qwen3Nvfp4Model) (prompt: string) =
    use _infer = torch.inference_mode()
    let rendered = InferenceBridge.renderPrompt prompt
    let input = session.Tokenizer.Encode(rendered) |> Seq.map int |> Seq.toArray
    if input.Length = 0 then failwith "empty tokenized prompt"

    let running = ResizeArray<int>(input.Length + opts.MaxTokens)
    for id in input do running.Add(id)
    let generated = ResizeArray<int>(opts.MaxTokens)
    let stopSet = stopTokens |> Set.ofList

    let mutable stop = false
    while generated.Count < opts.MaxTokens && not stop do
        use h = forwardFp2 session model (running.ToArray())
        let nextId = InferenceBridge.selectNextTokenId session h opts.Temperature opts.TopP
        if stopSet.Contains(nextId) then
            stop <- true
        else
            generated.Add(nextId)
            running.Add(nextId)
        lightGpuCleanUp()

    InferenceBridge.decodeTokens session.Tokenizer (generated |> Seq.toList)

let generateFp2Kv (session: InferenceSession) (model: Qwen3Nvfp4Model) (prompt: string) =
    use _infer = torch.inference_mode()
    let rendered = InferenceBridge.renderPrompt prompt
    let input = session.Tokenizer.Encode(rendered) |> Seq.map int |> Seq.toArray
    if input.Length = 0 then failwith "empty tokenized prompt"

    use cache = Qwen3Model.createKvCache model
    Qwen3Model.resetKvCache cache

    let generated = ResizeArray<int>(opts.MaxTokens)
    let stopSet = stopTokens |> Set.ofList

    use prefill = forwardFp2WithKv session model cache input
    let mutable nextId = InferenceBridge.selectNextTokenId session prefill opts.Temperature opts.TopP
    let mutable stop = false
    lightGpuCleanUp()

    while generated.Count < opts.MaxTokens && not stop do
        if stopSet.Contains(nextId) then
            stop <- true
        else
            generated.Add(nextId)
            if generated.Count < opts.MaxTokens then
                use h = forwardFp2WithKv session model cache [| nextId |]
                nextId <- InferenceBridge.selectNextTokenId session h opts.Temperature opts.TopP
                lightGpuCleanUp()

    InferenceBridge.decodeTokens session.Tokenizer (generated |> Seq.toList)

let runFp2Model () =
    ensureFp2SafetyEnv ()
    let session = InferenceBridge.initSamplingOnly scriptArgs.ModelDir (Some scriptArgs.WeightPath) scriptArgs.Quant scriptArgs.Device scriptArgs.DType
    let cfg : TrainingConfig =
        { Defaults.trainingConfig with
            ModelDir = scriptArgs.ModelDir
            ConfigPath = Path.Combine(scriptArgs.ModelDir, "config.json")
            TokenizerPath = Path.Combine(scriptArgs.ModelDir, "tokenizer.json")
            WeightPath = scriptArgs.WeightPath
            Device = scriptArgs.Device
            SyntheticMode = false
            StrictLoad = true
            UseKvCache = scriptArgs.UseKvCache
            SequenceLength = 8L }

    let model = Qwen3Model.create cfg
    try
        for p in testPrompts do
            printfn "Prompt: %s" p
            let reply =
                if scriptArgs.UseKvCache then
                    generateFp2Kv session model p
                else
                    generateFp2Replay session model p
            printfn "Reply: %s" reply
            printfn "---"
    finally
        Qwen3Model.dispose model
        InferenceBridge.dispose session

match scriptArgs.Mode.Trim().ToLowerInvariant() with
| "bridge"
| "run-training2" -> runBridge ()
| "fp2"
| "fp2-model"
| "run-training-fp2" -> runFp2Model ()
| other -> failwithf "unsupported --mode=%s (use bridge|fp2-model)" other
