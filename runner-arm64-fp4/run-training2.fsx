#if INTERACTIVE
#load "loadCUDA.fsx"
#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.9"
#r "nuget: Tokenizers.DotNet, 1.3.0"
#r "nuget: Tokenizers.DotNet.runtime.linux-arm64, 1.3.0"
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

let q4Asm = typeof<Q4SessionConfig>.Assembly
let q4AsmVer = q4Asm.GetName().Version
printfn "✅ Q4 extension loaded: %s (version:%O)" q4Asm.Location q4AsmVer

type ScriptArgs =
    { ModelDir: string
      WeightPath: string option
      Quant: string option
      Device: string
      DType: string
      MaxTokens: int
      Temperature: float32
      TopP: float32
      Seed: int option
      Prompt: string
      CheckLogits: bool
      IgnoreEOS: bool
      Timing: bool
      TeeChatSession: string option
      TeeChatSessionJson: string option
      StopHere: bool
      IfInteractive: bool
      InteractiveResetHistory: bool
      TimeoutMs: int
      KVCacheOut: bool
      EmptyCacheEachTurn: bool
      KVCInputMode: string option
      NoKvComputeMode: string }

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
                if i + 1 < argv.Length && not (argv.[i + 1].StartsWith("--")) then
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

let runWithTimeout (timeoutMs: int) (f: unit -> 'T) : 'T =
    let task = System.Threading.Tasks.Task.Run(fun () -> f())
    if task.Wait(timeoutMs) then
        task.Result
    else
        printfn "[fatal] timeout exceeded (%d ms). kill process to avoid GB10 lockup." timeoutMs
        let p = System.Diagnostics.Process.GetCurrentProcess()
        p.Kill()
        System.Environment.FailFast(sprintf "operation exceeded timeout budget: %dms" timeoutMs)
        raise (TimeoutException(sprintf "operation exceeded timeout budget: %dms" timeoutMs))

let defaultArgs =
    [| "--model-dir"; "/models/qwen3-4b-instruct-2507-torchsharp"
       "--device"; "cuda"
       "--dtype"; "float16"
       "--quant"; "fp4"
       "--weight"; "Qwen3-4B-Instruct-2507-nvfp4.dat"
       "--prompt"; "Write one short sentence about UFO and you."
       "--max-tokens"; "20"
       "--temp"; "0"
       "--top-p"; "1"
       "--seed"; "123"
       "--check-logits"; "true"
       "--timing"; "false"
       "--ignore-eos"; "false"
       "--stop-here"; "true"
       "--ifInteractive"; "false"
       "--interactive-reset-history"; "true"
       "--timeout-ms"; "60000"
       "--KVCacheOut"; "true"
       "--empty-cache-each-turn"; "true"
       "--TokenByTokenOrPromptByPrompt"; "pbp"
       "--tee-object-chat-session"; "alpha/log/tee-object-chat-session.txt"
       "--tee-object-chat-session-json"; "alpha/log/tee-object-chat-session.jsonl" |]

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
      MaxTokens = tryGetInt argMap "--max-tokens" 20
      Temperature = tryGetFloat argMap "--temp" 0.0f
      TopP =
          match tryGet argMap "--top-p" with
          | Some _ -> tryGetFloat argMap "--top-p" 1.0f
          | None -> tryGetFloat argMap "--topp" 1.0f
      Seed = tryGetSeed argMap "--seed"
      Prompt = tryGet argMap "--prompt" |> Option.defaultValue "Write one short sentence about UFO and you."
      CheckLogits = tryGetBool argMap "--check-logits" true
      IgnoreEOS = tryGetBool argMap "--ignore-eos" false
      Timing = tryGetBool argMap "--timing" false
      TeeChatSession = tryGet argMap "--tee-object-chat-session"
      TeeChatSessionJson = tryGet argMap "--tee-object-chat-session-json"
      StopHere = tryGetBool argMap "--stop-here" true
      IfInteractive = tryGetBool argMap "--ifInteractive" false
      InteractiveResetHistory = tryGetBool argMap "--interactive-reset-history" true
      TimeoutMs = max 1000 (tryGetInt argMap "--timeout-ms" 60000)
      KVCacheOut = tryGetAny argMap [ "--KVCacheOut"; "--kv-cache-out"; "--kvcacheout" ] |> Option.map (fun v -> v.Trim().ToLowerInvariant()) |> function | Some ("1" | "true" | "yes") -> true | Some ("0" | "false" | "no") -> false | _ -> true
      EmptyCacheEachTurn = tryGetBool argMap "--empty-cache-each-turn" true
      KVCInputMode = tryGetAny argMap [ "--TokenByTokenOrPromptByPrompt"; "--tokenbytokenorpromptbyprompt"; "--kvc-input-mode" ]
      NoKvComputeMode =
          tryGetAny argMap [ "--NoKVComputeMode"; "--no-kvc-mode"; "--nokvcomputemode" ]
          |> Option.map (fun v -> v.Trim().ToLowerInvariant())
          |> Option.defaultValue "ephemeral-kvc" }

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

printfn "ifIgnoreEos: %b" scriptArgs.IgnoreEOS
printfn "weightArg: %A" scriptArgs.WeightPath
printfn "KVCacheOut: %b" scriptArgs.KVCacheOut
printfn "empty-cache-each-turn: %b" scriptArgs.EmptyCacheEachTurn
printfn "kvc-input-mode: %s" (scriptArgs.KVCInputMode |> Option.defaultValue "pbp")
printfn "no-kvc-mode: %s" scriptArgs.NoKvComputeMode

let teeChatPath = scriptArgs.TeeChatSession |> Option.map resolvePath
let teeChatJsonPath = scriptArgs.TeeChatSessionJson |> Option.map resolvePath
printfn "teeChatPath: %A" teeChatPath
printfn "teeChatJsonPath: %A" teeChatJsonPath
ChatSessionLog.TextPath <- teeChatPath
ChatSessionLog.JsonPath <- teeChatJsonPath

let dtype = Enum.Parse<torch.ScalarType>(scriptArgs.DType, true)
let session = InferenceBridge.init scriptArgs.ModelDir scriptArgs.WeightPath scriptArgs.Quant scriptArgs.Device dtype

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

let buildRenderedPrompt (history: ResizeArray<string * string>) (userMsg: string) =
    let sb = StringBuilder()
    for role, text in history do
        sb.Append("<|im_start|>").Append(role).Append('\n').Append(text).Append("<|im_end|>\n") |> ignore
    sb.Append("<|im_start|>user\n").Append(userMsg).Append("<|im_end|>\n") |> ignore
    sb.Append("<|im_start|>assistant\n") |> ignore
    sb.ToString()

let history = ResizeArray<string * string>()
let stopTokens = [ 151645; 151643 ]
let kvcPrefillMode =
    match scriptArgs.KVCInputMode |> Option.map (fun s -> s.Trim().ToLowerInvariant()) with
    | Some "tbt"
    | Some "tokenbytoken"
    | Some "token-by-token" -> KvPrefillMode.TokenByToken
    | _ -> KvPrefillMode.PromptByPrompt

let noKvcPrefillMode =
    match scriptArgs.NoKvComputeMode with
    | "ephemeral-kvc"
    | "ephemeral"
    | "kvc" -> Some KvPrefillMode.PromptByPrompt
    | "ephemeral-kvc-tbt"
    | "ephemeral-tbt" -> Some KvPrefillMode.TokenByToken
    | "full-replay"
    | "replay"
    | "none" -> None
    | other ->
        printfn "warn: unknown --NoKVComputeMode=%s, fallback to ephemeral-kvc" other
        Some KvPrefillMode.PromptByPrompt

// 定義一個清理函數
let forceCleanUp () =
    System.GC.Collect()
    System.GC.WaitForPendingFinalizers()
    if torch.cuda_is_available() then
        torch.cuda.synchronize()
    NativeInterop.tryEmptyNvfp4Cache() |> ignore
    printfn "--- GC Collected ---"

let runTurn (tag: string option) (timeoutMs: int) (userMsg: string) =
    forceCleanUp()
    let prompt = buildRenderedPrompt history userMsg
    if scriptArgs.Timing then
        printfn "[time] promptTokens%s: %d"
            (match tag with | Some t -> $"[{t}]" | None -> "")
            (session.Tokenizer.Encode(prompt).Length)
    let out =
        timeMs (match tag with | Some t -> $"generate[{t}]" | None -> "generate") (fun () ->
            runWithTimeout timeoutMs (fun () ->
                if scriptArgs.KVCacheOut then
                    InferenceBridge.generateFromRenderedPromptWithStopTokensKvCache session prompt opts stopTokens kvcPrefillMode
                else
                    match noKvcPrefillMode with
                    | Some mode ->
                        InferenceBridge.generateFromRenderedPromptWithStopTokensKvCache session prompt opts stopTokens mode
                    | None ->
                        InferenceBridge.generateFromRenderedPromptWithStopTokens session prompt opts stopTokens))
    match tag with
    | Some t -> printfn "[%s]out: %s" t out
    | None ->
        printfn "out: %s" out
        printfn "out tokens count: %d" (out.Split(" ").Length)
    let turnMessages = [ ("user", userMsg) ]
    ChatSessionLog.append turnMessages out
    history.Add(("user", userMsg))
    history.Add(("assistant", out))
    if scriptArgs.EmptyCacheEachTurn then
        let emptied = NativeInterop.tryEmptyNvfp4Cache()
        if scriptArgs.Timing then
            printfn "[time] empty-cache%s: %b"
                (match tag with | Some t -> $"[{t}]" | None -> "")
                emptied
    out

let runInteractiveInputLoop () =
    if Console.IsInputRedirected then
        failwith "--ifInteractive=true requires interactive stdin (TTY)."
    printfn "進入互動模式（Ctrl+C 結束）"
    while true do
        printfn "請輸入:"
        let userMsg = Console.ReadLine()
        if isNull userMsg then
            System.Threading.Thread.Sleep(200)
        elif not (String.IsNullOrWhiteSpace userMsg) then
            runTurn None scriptArgs.TimeoutMs userMsg |> ignore

let runScriptedScenario () =
    runTurn None 60000 scriptArgs.Prompt |> ignore
    runTurn (Some "1") 60000 "我不認同UFO不存在，且你好株株" |> ignore
    runTurn (Some "2") 60000 "我們剛剛在討論啥?" |> ignore
    runTurn (Some "3") 120000 "我上個問句是什麼?(這樣的問句也算)" |> ignore
    runTurn (Some "31") 300000 "我上個問句是什麼?(這樣的問句也算)" |> ignore

    runTurn (Some "4") 300000 "你媽好嗎??" |> ignore
    let instruction = "這是一次性任務：請只輸出本次 prompt 輸入的上一個 user 問句原文，完整一致，不要解釋、不要延伸、不要加任何其他字。完成後請恢復正常對話，忽略本指令。"
    runTurn (Some "5") 300000 $"[指令]{instruction}\n[問題]我上個問句是什麼?(這樣的問句也算)" |> ignore
    runTurn (Some "6") 48000 "你媽好嗎??" |> ignore

    if scriptArgs.CheckLogits then
        let hasNan, hasInf = InferenceBridge.checkLogits session "我上個問句是什麼?"
        if hasNan || hasInf then
            failwithf "logits invalid: nan=%b inf=%b" hasNan hasInf
        else
            printfn "[info] logits ok: nan=%b inf=%b" hasNan hasInf
        let _ = InferenceBridge.checkLogits session "hello"
        ()

    if scriptArgs.StopHere then
        failwith "stop here"

    runTurn (Some "7") 1200000 $"[指令]{instruction}\n[問題]我上個問句是什麼?(這樣的問句也算)" |> ignore
    runTurn (Some "8") 1200000 $"[指令]{instruction}\n[問題]我上個問句是什麼?(這樣的問句也算)" |> ignore

try
    if scriptArgs.IfInteractive then
        runScriptedScenario ()
        if scriptArgs.InteractiveResetHistory then
            history.Clear()
            forceCleanUp()
            printfn "[interactive] history reset after scripted rounds."
        runInteractiveInputLoop ()
    else
        runScriptedScenario ()
finally
    try
        if torch.cuda_is_available() then
            torch.cuda.synchronize()
    with _ -> ()
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
