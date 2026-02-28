#if INTERACTIVE
#load "loadCUDA.fsx"
#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.9"
#r "nuget: Tokenizers.DotNet, 1.3.0"
#r "nuget: Tokenizers.DotNet.runtime.linux-arm64, 1.3.0"
#r "../bin/Release/net10.0/TorchSharp.Fun.DGX.dll"
#r "../bin/Release/net10.0/TorchSharp.Q4.Extension.dll"
#r "../bin/Release/net10.0/Qwen3-4B-Instruct-2507-TorchSharp.fs.dll"
#endif

open System
open System.IO
open System.Text
open System.Text.Json
open TorchSharp
open TorchSharp.Q4.Extension
open Qwen3_4B_Instruct_2507_TorchSharp_fs

type ScriptArgs =
    { ModelDir: string
      WeightPath: string option
      Quant: string option
      Device: string
      DType: string
      UseKvCache: bool
      KvBackend: string
      Turns: int
      MaxTokens: int
      Temperature: float32
      TopP: float32
      Seed: int option
      Prompt: string
      FollowupPrompt: string
      CheckLogits: bool
      IgnoreEOS: bool
      Timing: bool
      IfInteractive: bool
      TeeChatSession: string option
      TeeChatSessionJson: string option
      StopHere: bool
      TimeoutMs: int }

type ChatSessionJsonMessage =
    { role: string
      text: string }

type ChatSessionJsonEntry =
    { ts: string
      messages: ChatSessionJsonMessage list
      output: string }

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

let tryGetAny (argMap: Map<string, string>) (keys: string list) =
    keys |> List.tryPick (fun k -> tryGet argMap k)

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

module ChatSessionLog =
    let mutable TextPath: string option = None
    let mutable JsonPath: string option = None

    let ensureDir (path: string) =
        let dir = Path.GetDirectoryName(path)
        if not (String.IsNullOrWhiteSpace dir) then
            Directory.CreateDirectory(dir) |> ignore

    let append (messages: (string * string) list) (output: string) =
        let ts = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
        match TextPath with
        | Some path ->
            ensureDir path
            let sb = StringBuilder()
            sb.AppendLine($"==== {ts} ====") |> ignore
            for role, text in messages do
                sb.AppendLine($"[{role}] {text}") |> ignore
            sb.AppendLine($"[assistant] {output}") |> ignore
            sb.AppendLine() |> ignore
            File.AppendAllText(path, sb.ToString())
        | None -> ()

        match JsonPath with
        | Some path ->
            ensureDir path
            let entry: ChatSessionJsonEntry =
                { ts = ts
                  messages = messages |> List.map (fun (role, text) -> { role = role; text = text })
                  output = output }
            let line = JsonSerializer.Serialize(entry)
            File.AppendAllText(path, line + Environment.NewLine)
        | None -> ()

let whoamiDatCandidate = "/workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/whoami-1000-seq192-r8-s10-lr1e3.dat"
let hasWhoamiDatPreset = File.Exists whoamiDatCandidate
let defaultWeightArg = if hasWhoamiDatPreset then whoamiDatCandidate else "Qwen3-4B-Instruct-2507-nvfp4.dat"
let defaultPromptArg = if hasWhoamiDatPreset then "你是誰" else "Write one short sentence about UFO and you."

let defaultArgs =
    [| "--model-dir"; "/models/qwen3-4b-instruct-2507-torchsharp"
       "--device"; "cuda"
       "--dtype"; "float16"
       "--quant"; "fp4"
       "--weight"; defaultWeightArg
       "--use-kvc"; "true"
       "--kvc-backend"; "fp2-model"
       "--turns"; "1"
       "--prompt"; defaultPromptArg
       "--followup-prompt"; "continue."
       "--max-tokens"; "24"
       "--temp"; "0"
       "--top-p"; "1"
       "--seed"; "123"
       "--check-logits"; "false"
       "--timing"; "true"
       "--ifInteractive"; "true"
       "--ignore-eos"; "false"
       "--stop-here"; "false"
       "--timeout-ms"; "60000"
       "--tee-object-chat-session"; "alpha/log/tee-object-chat-session-fp-safe.txt"
       "--tee-object-chat-session-json"; "alpha/log/tee-object-chat-session-fp-safe.jsonl" |]

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

let quantArg = tryGet argMap "--quant"
let userSpecifiedWeight =
    rawArgs
    |> Array.exists (fun s -> s.Equals("--weight", StringComparison.OrdinalIgnoreCase))

let weightArg =
    if userSpecifiedWeight then
        tryGet argMap "--weight"
    elif hasWhoamiDatPreset then
        Some whoamiDatCandidate
    else
        match quantArg with
        | Some q when q.Equals("fp4", StringComparison.OrdinalIgnoreCase) ->
            Some "Qwen3-4B-Instruct-2507-nvfp4.dat"
        | Some q when q.Equals("4bit", StringComparison.OrdinalIgnoreCase) || q.Equals("nf4", StringComparison.OrdinalIgnoreCase) || q.Equals("int4", StringComparison.OrdinalIgnoreCase) ->
            Some "Qwen3-4B-Instruct-2507-4bit.nf4.dat"
        | _ ->
            tryGet argMap "--weight"

let scriptArgs =
    { ModelDir = tryGet argMap "--model-dir" |> Option.defaultValue "/models/qwen3-4b-instruct-2507-torchsharp"
      WeightPath = weightArg
      Quant = quantArg
      Device = tryGet argMap "--device" |> Option.defaultValue "cuda"
      DType = tryGet argMap "--dtype" |> Option.defaultValue "float16"
      UseKvCache = tryGetBool argMap "--use-kvc" true
      KvBackend = tryGet argMap "--kvc-backend" |> Option.defaultValue "fp2-model"
      Turns = max 1 (tryGetInt argMap "--turns" 1)
      MaxTokens = tryGetInt argMap "--max-tokens" 4
      Temperature = tryGetFloat argMap "--temp" 0.0f
      TopP =
          match tryGet argMap "--top-p" with
          | Some _ -> tryGetFloat argMap "--top-p" 1.0f
          | None -> tryGetFloat argMap "--topp" 1.0f
      Seed = tryGetSeed argMap "--seed"
      Prompt = tryGet argMap "--prompt" |> Option.defaultValue "Write one short sentence about UFO and you."
      FollowupPrompt = tryGet argMap "--followup-prompt" |> Option.defaultValue "continue."
      CheckLogits = tryGetBool argMap "--check-logits" true
      IgnoreEOS = tryGetBool argMap "--ignore-eos" false
      Timing = tryGetBool argMap "--timing" true
      IfInteractive = tryGetBool argMap "--ifInteractive" true
      TeeChatSession = tryGet argMap "--tee-object-chat-session"
      TeeChatSessionJson = tryGet argMap "--tee-object-chat-session-json"
      StopHere = tryGetBool argMap "--stop-here" true
      TimeoutMs = tryGetInt argMap "--timeout-ms" 60000 }

let ensureFp2SafetyEnv () =
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
        printfn "info: TS_Q4_STE_USE_NATIVE_QUANTIZE not set; forcing to 1 for OOM safety."
    let cacheRaw = Environment.GetEnvironmentVariable("TS_Q4_STE_CACHE_EVAL_WEIGHT")
    let cacheEnabled =
        match cacheRaw with
        | null -> false
        | v ->
            match v.Trim().ToLowerInvariant() with
            | "1" | "true" | "yes" -> true
            | _ -> false
    if not cacheEnabled then
        Environment.SetEnvironmentVariable("TS_Q4_STE_CACHE_EVAL_WEIGHT", "1")
        printfn "info: TS_Q4_STE_CACHE_EVAL_WEIGHT not set; forcing to 1 for fp2 eval stability."

ensureFp2SafetyEnv ()

let invalidQuantWeightPair =
    match scriptArgs.Quant, scriptArgs.WeightPath with
    | Some q, Some w when q.Equals("fp4", StringComparison.OrdinalIgnoreCase) && w.Contains(".nf4", StringComparison.OrdinalIgnoreCase) ->
        Some(q, w, "fp4 量化不能搭配 nf4 權重")
    | Some q, Some w when (q.Equals("4bit", StringComparison.OrdinalIgnoreCase) || q.Equals("nf4", StringComparison.OrdinalIgnoreCase) || q.Equals("int4", StringComparison.OrdinalIgnoreCase)) && w.Contains("nvfp4", StringComparison.OrdinalIgnoreCase) ->
        Some(q, w, "nf4/4bit 量化不能搭配 nvfp4 權重")
    | _ -> None

match invalidQuantWeightPair with
| Some(q, w, reason) -> failwithf "量化與權重不匹配: quant=%s weight=%s (%s)" q w reason
| None -> ()

let kvBackendNorm = scriptArgs.KvBackend.Trim().ToLowerInvariant()
if kvBackendNorm <> "bridge" && kvBackendNorm <> "fp2-model" then
    failwithf "--kvc-backend must be one of: bridge | fp2-model, got: %s" scriptArgs.KvBackend
if kvBackendNorm = "bridge" then
    failwith "run-training-fp2.fsx is now training-path only. Please use --kvc-backend fp2-model."

let q4Asm = typeof<Q4SessionConfig>.Assembly
let q4AsmVer = q4Asm.GetName().Version
printfn "✅ Q4 extension loaded: %s (version:%O)" q4Asm.Location q4AsmVer
printfn "native quantize export available: %b" (NativeInterop.hasLibTorchFp4Quantize())
if not (NativeInterop.hasLibTorchFp4Quantize()) then
    failwith "NVFP4 native quantize export (NVFP4_quantize) unavailable; refusing fallback path to avoid fp2 OOM."
printfn "whoami preset dat found: %b (%s)" hasWhoamiDatPreset whoamiDatCandidate
printfn "ifIgnoreEos: %b" scriptArgs.IgnoreEOS
printfn "weightArg: %A" scriptArgs.WeightPath
printfn "mode: fp-style/training-block-graph/use-kvc=%b/kvc-backend=%s" scriptArgs.UseKvCache kvBackendNorm
printfn "run: turns=%d maxTokens=%d (first '!!!!' in any turn triggers fail-fast)" scriptArgs.Turns scriptArgs.MaxTokens

let teeChatPath = scriptArgs.TeeChatSession |> Option.map resolvePath
let teeChatJsonPath = scriptArgs.TeeChatSessionJson |> Option.map resolvePath
printfn "teeChatPath: %A" teeChatPath
printfn "teeChatJsonPath: %A" teeChatJsonPath
ChatSessionLog.TextPath <- teeChatPath
ChatSessionLog.JsonPath <- teeChatJsonPath

let dtype = Enum.Parse<torch.ScalarType>(scriptArgs.DType, true)
let session =
    if kvBackendNorm = "fp2-model" then
        InferenceBridge.initSamplingOnly scriptArgs.ModelDir scriptArgs.WeightPath scriptArgs.Quant scriptArgs.Device dtype
    else
        InferenceBridge.init scriptArgs.ModelDir scriptArgs.WeightPath scriptArgs.Quant scriptArgs.Device dtype

let resolvedWeightPath = InferenceBridge.resolveWeightPath scriptArgs.ModelDir scriptArgs.WeightPath scriptArgs.Quant
let model =
    if scriptArgs.UseKvCache && kvBackendNorm = "bridge" then
        None
    else
        let trainingCfg : TrainingConfig =
            { Defaults.trainingConfig with
                ModelDir = scriptArgs.ModelDir
                ConfigPath = Path.Combine(scriptArgs.ModelDir, "config.json")
                TokenizerPath = Path.Combine(scriptArgs.ModelDir, "tokenizer.json")
                WeightPath = resolvedWeightPath
                Device = scriptArgs.Device
                SyntheticMode = false
                StrictLoad = true
                UseKvCache = scriptArgs.UseKvCache
                SequenceLength = 8L }
        Some (Qwen3Model.create trainingCfg)

let opts: InferenceGenOptions =
    { MaxTokens = scriptArgs.MaxTokens
      Temperature = scriptArgs.Temperature
      TopP = scriptArgs.TopP
      Seed = scriptArgs.Seed }

let timeMs (label: string) (f: unit -> 'T) =
    if scriptArgs.Timing then
        let sw = Diagnostics.Stopwatch.StartNew()
        let result = f()
        sw.Stop()
        printfn "[time] %s: %.1f ms" label sw.Elapsed.TotalMilliseconds
        result
    else
        f()

let forwardModelFp2 (session: InferenceSession) (model: Qwen3Nvfp4Model) (tokenIds: int array) =
    use embed = InferenceBridge.buildTokenEmbeddings session tokenIds
    Qwen3Model.forward model embed (Some session.DType)

let forwardModelFp2WithCache
    (session: InferenceSession)
    (model: Qwen3Nvfp4Model)
    (cache: Qwen3Core.ModelKvCache)
    (tokenIds: int array) =
    use embed = InferenceBridge.buildTokenEmbeddings session tokenIds
    Qwen3Model.forwardWithKvCache model cache embed (Some session.DType)

let lightGpuCleanUp () =
    if torch.cuda_is_available() then
        torch.cuda.synchronize()
    NativeInterop.tryEmptyNvfp4Cache() |> ignore

let generateFromRenderedPromptWithStopTokensFpReplay
    (session: InferenceSession)
    (model: Qwen3Nvfp4Model)
    (renderedPrompt: string)
    (opt: InferenceGenOptions)
    (stopTokens: int list) =
    use _infer = torch.inference_mode()

    let encoded =
      session.Tokenizer.Encode(renderedPrompt)
      |> Seq.map int
      |> Seq.toList

    if encoded.IsEmpty then
      invalidOp "prompt encoded to empty token sequence"

    let running = ResizeArray<int>(encoded.Length + max 1 opt.MaxTokens)
    for id in encoded do
      running.Add(id)

    let generated = ResizeArray<int>(max 1 opt.MaxTokens)
    let mutable step = 0
    let targetSteps = max 1 opt.MaxTokens
    let mutable stop = false
    let stopSet = stopTokens |> Set.ofList

    match opt.Seed with
    | Some s when s >= 0 -> torch.manual_seed(int64 s) |> ignore
    | _ -> ()

    while step < targetSteps && not stop do
      use hidden = forwardModelFp2 session model (running.ToArray())
      let nextId = InferenceBridge.selectNextTokenId session hidden opt.Temperature opt.TopP
      if stopSet.Contains(nextId) then
        stop <- true
      else
        generated.Add(nextId)
        running.Add(nextId)
        step <- step + 1

    match Environment.GetEnvironmentVariable("QWEN3_FS_DEBUG_TOKENS") with
    | "1" -> printfn "[FP2Debug] generated token ids: %A" (generated |> Seq.toList)
    | _ -> ()

    InferenceBridge.decodeTokens session.Tokenizer (generated |> Seq.toList)

let generateFromRenderedPromptWithStopTokensFpKv
    (session: InferenceSession)
    (model: Qwen3Nvfp4Model)
    (renderedPrompt: string)
    (opt: InferenceGenOptions)
    (stopTokens: int list) =
    use _infer = torch.inference_mode()
    use cache = Qwen3Model.createKvCache model
    Qwen3Model.resetKvCache cache

    let encoded =
      session.Tokenizer.Encode(renderedPrompt)
      |> Seq.map int
      |> Seq.toArray

    if encoded.Length = 0 then
      invalidOp "prompt encoded to empty token sequence"

    let generated = ResizeArray<int>(max 1 opt.MaxTokens)
    let targetSteps = max 1 opt.MaxTokens
    let stopSet = stopTokens |> Set.ofList

    match opt.Seed with
    | Some s when s >= 0 -> torch.manual_seed(int64 s) |> ignore
    | _ -> ()

    use prefillHidden = forwardModelFp2WithCache session model cache encoded
    let mutable nextId = InferenceBridge.selectNextTokenId session prefillHidden opt.Temperature opt.TopP
    let mutable stop = false
    lightGpuCleanUp()

    while generated.Count < targetSteps && not stop do
      if stopSet.Contains(nextId) then
        stop <- true
      else
        generated.Add(nextId)
        if generated.Count < targetSteps then
          use hidden = forwardModelFp2WithCache session model cache [| nextId |]
          nextId <- InferenceBridge.selectNextTokenId session hidden opt.Temperature opt.TopP
          lightGpuCleanUp()

    match Environment.GetEnvironmentVariable("QWEN3_FS_DEBUG_TOKENS") with
    | "1" -> printfn "[FP2Debug] generated token ids: %A" (generated |> Seq.toList)
    | _ -> ()

    InferenceBridge.decodeTokens session.Tokenizer (generated |> Seq.toList)

let appendTokenIds (dst: ResizeArray<int>) (tokenIds: int array) =
    for id in tokenIds do
        dst.Add(id)

let buildTurnPrompt (userMsg: string) =
    $"<|im_start|>user\n{userMsg}<|im_end|>\n<|im_start|>assistant\n"

let encodeTokenIds (session: InferenceSession) (text: string) =
    session.Tokenizer.Encode(text) |> Seq.map int |> Seq.toArray

let generateFromUserMessageWithStopTokensFpKvPersistent
    (session: InferenceSession)
    (model: Qwen3Nvfp4Model)
    (cache: Qwen3Core.ModelKvCache)
    (contextTokens: ResizeArray<int>)
    (assistantCloseTokens: int array)
    (userMsg: string)
    (opt: InferenceGenOptions)
    (stopTokens: int list) =
    use _infer = torch.inference_mode()
    let turnPrompt = buildTurnPrompt userMsg
    let turnTokens = encodeTokenIds session turnPrompt
    if turnTokens.Length = 0 then
        invalidOp "turn prompt encoded to empty token sequence"

    match opt.Seed with
    | Some s when s >= 0 -> torch.manual_seed(int64 s) |> ignore
    | _ -> ()

    use prefillHidden = forwardModelFp2WithCache session model cache turnTokens
    appendTokenIds contextTokens turnTokens

    let generated = ResizeArray<int>(max 1 opt.MaxTokens)
    let targetSteps = max 1 opt.MaxTokens
    let stopSet = stopTokens |> Set.ofList
    let mutable nextId = InferenceBridge.selectNextTokenId session prefillHidden opt.Temperature opt.TopP
    let mutable stop = false
    lightGpuCleanUp()

    while generated.Count < targetSteps && not stop do
        if stopSet.Contains(nextId) then
            stop <- true
        else
            generated.Add(nextId)
            contextTokens.Add(nextId)
            // Materialize every accepted token into cache so next turn can continue without replay.
            use hidden = forwardModelFp2WithCache session model cache [| nextId |]
            lightGpuCleanUp()
            if generated.Count < targetSteps then
                nextId <- InferenceBridge.selectNextTokenId session hidden opt.Temperature opt.TopP

    if assistantCloseTokens.Length > 0 then
        use _closeHidden = forwardModelFp2WithCache session model cache assistantCloseTokens
        appendTokenIds contextTokens assistantCloseTokens
        lightGpuCleanUp()

    match Environment.GetEnvironmentVariable("QWEN3_FS_DEBUG_TOKENS") with
    | "1" ->
        printfn "[FP2Debug] generated token ids: %A" (generated |> Seq.toList)
        printfn "[FP2Debug] kvc seqLen=%d contextTokens=%d" cache.SeqLen contextTokens.Count
    | _ -> ()

    InferenceBridge.decodeTokens session.Tokenizer (generated |> Seq.toList)

let buildRenderedPrompt (history: ResizeArray<string * string>) (userMsg: string) =
    let sb = StringBuilder()
    for role, text in history do
        sb.Append("<|im_start|>").Append(role).Append('\n').Append(text).Append("<|im_end|>\n") |> ignore
    sb.Append("<|im_start|>user\n").Append(userMsg).Append("<|im_end|>\n") |> ignore
    sb.Append("<|im_start|>assistant\n") |> ignore
    sb.ToString()

let history = ResizeArray<string * string>()
let stopTokens = [ 151645; 151643 ]
let fp2PersistentContextTokens = ResizeArray<int>()
let assistantCloseTokens = encodeTokenIds session "<|im_end|>\n"
if assistantCloseTokens.Length = 0 then
    failwith "assistant close tokens are empty"

let fp2PersistentCache =
    if scriptArgs.UseKvCache && kvBackendNorm = "fp2-model" then
        match model with
        | Some m ->
            let c = Qwen3Model.createKvCache m
            Qwen3Model.resetKvCache c
            Some c
        | None -> None
    else
        None

let forceCleanUp () =
    GC.Collect()
    GC.WaitForPendingFinalizers()
    if torch.cuda_is_available() then
        torch.cuda.synchronize()
    lightGpuCleanUp()

let runTurn (tag: string option) (timeoutMs: int) (userMsg: string) =
    forceCleanUp()
    let usePersistentFp2Kvc = scriptArgs.UseKvCache && kvBackendNorm = "fp2-model"
    let prompt =
        if usePersistentFp2Kvc then
            buildTurnPrompt userMsg
        else
            buildRenderedPrompt history userMsg
    let promptTokens = session.Tokenizer.Encode(prompt).Length
    if scriptArgs.Timing then
        printfn "[time] promptTokens%s: %d"
            (match tag with | Some t -> $"[{t}]" | None -> "")
            promptTokens

    let out =
        timeMs (match tag with | Some t -> $"generate.fp[{t}]" | None -> "generate.fp") (fun () ->
            if scriptArgs.UseKvCache && kvBackendNorm = "bridge" then
                InferenceBridge.generateFromRenderedPromptWithStopTokensKvCache session prompt opts stopTokens KvPrefillMode.PromptByPrompt
            elif scriptArgs.UseKvCache then
                match model, fp2PersistentCache with
                | Some m, Some c -> generateFromUserMessageWithStopTokensFpKvPersistent session m c fp2PersistentContextTokens assistantCloseTokens userMsg opts stopTokens
                | _ -> invalidOp "fp2-model backend requires model instance and persistent cache"
            else
                match model with
                | Some m -> generateFromRenderedPromptWithStopTokensFpReplay session m prompt opts stopTokens
                | None -> invalidOp "replay mode requires model instance")

    match tag with
    | Some t -> printfn "[%s]out: %s" t out
    | None ->
        printfn "out: %s" out
        printfn "out tokens count: %d" (out.Split(" ").Length)
    ChatSessionLog.append [ ("user", userMsg) ] out
    history.Add(("user", userMsg))
    history.Add(("assistant", out))
    if usePersistentFp2Kvc then
        match fp2PersistentCache with
        | Some c -> printfn "[kvc] turn-cache seqLen=%d contextTokens=%d" c.SeqLen fp2PersistentContextTokens.Count
        | None -> ()
    out

let runInteractiveInputLoop () =
    printfn "進入互動模式（Ctrl+C 結束）"
    if Console.IsInputRedirected then
        failwith "--ifInteractive=true requires interactive stdin (TTY)."
    while true do
        printfn "請輸入:"
        let userMsg = Console.ReadLine()
        if isNull userMsg then
            System.Threading.Thread.Sleep(200)
        elif not (String.IsNullOrWhiteSpace userMsg) then
            let out = runTurn None scriptArgs.TimeoutMs userMsg
            if out.Trim() = "!!!!" || out.Contains("!!!!") then
                failwith "interactive output is !!!! (guard triggered)"

try
    for turn in 1 .. scriptArgs.Turns do
        let userMsg = if turn = 1 then scriptArgs.Prompt else scriptArgs.FollowupPrompt
        let tag = if scriptArgs.Turns > 1 then Some(sprintf "turn-%d" turn) else None
        let out = runTurn tag scriptArgs.TimeoutMs userMsg
        if out.Trim() = "!!!!" || out.Contains("!!!!") then
            failwithf "turn %d output is !!!! (guard triggered)" turn

    if scriptArgs.CheckLogits then
        if scriptArgs.UseKvCache && kvBackendNorm = "bridge" then
            let hasNan, hasInf = InferenceBridge.checkLogits session scriptArgs.Prompt
            if hasNan || hasInf then
                failwithf "logits invalid: nan=%b inf=%b" hasNan hasInf
            else
                printfn "[info] logits ok: nan=%b inf=%b" hasNan hasInf
        else
            let prompt = InferenceBridge.renderPrompt scriptArgs.Prompt
            let encoded = session.Tokenizer.Encode(prompt) |> Seq.map int |> Seq.toArray
            match model with
            | Some m ->
                use hidden = forwardModelFp2 session m encoded
                use last = hidden.narrow(1L, hidden.shape.[1] - 1L, 1L)
                use lastNorm = InferenceBridge.rmsNormWeighted last session.FinalNorm session.Config.RmsNormEps
                use logits0 = session.LmHead.Forward(lastNorm, outDtype = session.DType)
                use logits = if logits0.dtype = torch.float32 then logits0 else logits0.to_type(torch.float32)
                let hasNan = torch.isnan(logits).any().item<int64>() <> 0L
                let hasInf = torch.isinf(logits).any().item<int64>() <> 0L
                if hasNan || hasInf then
                    failwithf "logits invalid: nan=%b inf=%b" hasNan hasInf
                else
                    printfn "[info] logits ok: nan=%b inf=%b" hasNan hasInf
            | None ->
                invalidOp "logit check requires model instance in non-bridge backend"

    if scriptArgs.IfInteractive then
        runInteractiveInputLoop ()

    if scriptArgs.StopHere then
        failwith "stop here (single-turn)"
finally
    try
        if torch.cuda_is_available() then
            torch.cuda.synchronize()
    with _ -> ()
    Nvfp4Training.clearEvalWeightCache()
    fp2PersistentCache |> Option.iter (fun c -> (c :> IDisposable).Dispose())
    model |> Option.iter Qwen3Model.dispose
    InferenceBridge.dispose session
    NativeInterop.tryEmptyNvfp4Cache() |> ignore
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()
    printfn "Device synchronized. Exiting."
