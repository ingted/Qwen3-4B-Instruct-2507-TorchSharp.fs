#if INTERACTIVE
#load "loadCUDA.fsx"
#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.9"
#r "nuget: Tokenizers.DotNet, 1.3.0"
#r "nuget: Tokenizers.DotNet.runtime.linux-arm64, 1.3.0"
#r "nuget: Microsoft.Extensions.AI, 9.9.0"
#r "/workspace/fsann/Qwen3-4B-Instruct-2507-TorchSharp-mod/Qwen3/bin/Release/net10.0/Qwen3.dll"
#load "Runner_type.fsx"
#load "Runner_shared.fsx"
#load "Runner_api.fsx"
#load "ChatSessionJson.fsx"
#load "SimpleChat.fsx"
#endif
open TorchSharp
open type torch
printfn "cuDNN available: %A" <| torch.cuda.is_cudnn_available()

open System
open System.Diagnostics
open System.IO
open TorchSharp
open Microsoft.Extensions.AI
open Qwen3
open Qwen3.Module
open Runner_type
open Runner_shared
open Runner_api
open ChatSessionJson
open SimpleChat

#if INTERACTIVE
let rawArgs = fsi.CommandLineArgs
#else
let rawArgs = System.Environment.GetCommandLineArgs()
#endif
let hasUserArgs = rawArgs |> Array.exists (fun (s:string) -> s.StartsWith("--"))

let defaultArgs =
    [| "--model-dir"; "/models/qwen3-4b-instruct-2507-torchsharp"
       "--device"; "cuda"
       "--dtype"; "float16"
       "--quant"; "fp4"
       //"--quant"; "4bit"
       "--weight"; "Qwen3-4B-Instruct-2507-nvfp4.dat"
       //"--weight"; "Qwen3-4B-Instruct-2507-4bit.nf4.nodq.dat"
       //"--weight"; "Qwen3-4B-Instruct-2507-4bit.fp4.dat"
       "--weight"; "Qwen3-4B-Instruct-2507-4bit.nf4.dat"
       "--prompt"; "Write one short sentence about UFO and you."
       "--max-tokens"; "64"
       "--temp"; "0"
       "--top-p"; "1"
       "--seed"; "123"
       "--check-logits"; "true"
       "--q4-kernel"; "true"
       "--q4-cache"; "false" // --q4-kernel true 時意義不大
       "--KVCacheOut"; "true"
       "--TokenByTokenOrPromptByPrompt"; "pbp"
       "--HistoryMode"; "tokens"
       "--debug-kvc"; "false"
       "--timing"; "false"
       "--cuda-mem-report"; "false"
       "--fp4-ab"; "false"
       "--fp4-scale-check"; "false"
       "--fp4-ab-max-calls"; "64"
       "--fp4-ab-explosion-rel"; "0.5"
       "--tee-object-chat-session"; "alpha/log/tee-object-chat-session.txt"
       "--tee-object-chat-session-json"; "alpha/log/tee-object-chat-session.jsonl"
       "--prompt-from-chat-session"; "false"
       //"--ignore-eos"
       |]

let args =
    if not hasUserArgs then
        Array.append rawArgs defaultArgs |> Array.skip 1
    else
        rawArgs |> Array.skip 1

let argMap =
    if not hasUserArgs then
        parseArgs args
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
let kvCacheOutArg = tryGetAny argMap [ "--KVCacheOut"; "--kv-cache-out"; "--kvcacheout" ]
let ignoreEosArg = tryGetBool argMap "--ignore-eos" false
let teeChatSessionArg = tryGet argMap "--tee-object-chat-session"
let teeChatSessionJsonArg = tryGet argMap "--tee-object-chat-session-json"
let promptFromChatSessionArg = tryGetBool argMap "--prompt-from-chat-session" false
let fp4AbArg = tryGetBool argMap "--fp4-ab" false
let fp4ScaleCheckArg = tryGetBool argMap "--fp4-scale-check" false
let cudaMemReportArg = tryGetBool argMap "--cuda-mem-report" false
let fp4AbMaxCallsArg = tryGet argMap "--fp4-ab-max-calls"
let fp4AbExplosionRelArg = tryGet argMap "--fp4-ab-explosion-rel"
let tokenByTokenOrPromptByPromptArg = tryGetAny argMap [ "--TokenByTokenOrPromptByPrompt"; "--tokenbytokenorpromptbyprompt"; "--kvc-input-mode" ]

let parseKvcInputMode (v: string option) =
    match v with
    | Some mode ->
        match mode.Trim().ToLowerInvariant() with
        | "tbt" | "tokenbytoken" | "token-by-token" -> KVCInputMode.TokenByToken
        | "pbp" | "promptbyprompt" | "prompt-by-prompt" -> KVCInputMode.PromptByPrompt
        | _ -> KVCInputMode.TokenByToken
    | None -> KVCInputMode.TokenByToken

printfn "ifIgnoreEos: %b" ignoreEosArg
printfn "weightArg: %A" weightArg

let scriptArgs =
    let checkDefault =
        match quantArg with
        | Some q when q.Equals("4bit", StringComparison.OrdinalIgnoreCase) || q.Equals("int4", StringComparison.OrdinalIgnoreCase) || q.Equals("fp4", StringComparison.OrdinalIgnoreCase) -> true
        | _ -> false
    { ModelDir =
          match tryGet argMap "--model-dir" with
          | Some v when not (String.IsNullOrWhiteSpace v) -> v
          | _ ->
              match tryGet argMap "--path" with
              | Some v when not (String.IsNullOrWhiteSpace v) -> v
              | _ -> "/models/qwen3-4b-instruct-2507-torchsharp"
      WeightPath = weightArg
      Quant = quantArg
      Device =
          match tryGet argMap "--device" with
          | Some v when not (String.IsNullOrWhiteSpace v) -> v
          | _ -> "cuda"
      DType =
          match tryGet argMap "--dtype" with
          | Some v when not (String.IsNullOrWhiteSpace v) -> v
          | _ -> "float16"
      MaxTokens = tryGetInt argMap "--max-tokens" 128
      Temperature = tryGetFloat argMap "--temp" 0.7f
      TopP =
          match tryGet argMap "--top-p" with
          | Some _ -> tryGetFloat argMap "--top-p" 0.9f
          | None -> tryGetFloat argMap "--topp" 0.9f
      Seed = tryGetSeed argMap "--seed"
      Prompt =
          match tryGet argMap "--prompt" with
          | Some v when not (String.IsNullOrWhiteSpace v) -> v
          | _ -> "Write one long sentence about cats."
      SystemPrompt =
          match tryGet argMap "--system" with
          | Some v when not (String.IsNullOrWhiteSpace v) -> Some v
          | _ -> None
      CheckLogits = tryGetBool argMap "--check-logits" checkDefault
      Q4Kernel = tryGetBoolOpt argMap "--q4-kernel"
      Q4Cache = tryGetBoolOpt argMap "--q4-cache"
      KVCacheOut =
          match kvCacheOutArg with
          | Some v -> parseBoolOpt v |> Option.defaultValue false
          | None -> false
      KVCInputMode = parseKvcInputMode tokenByTokenOrPromptByPromptArg
      HistoryMode =
          match tryGet argMap "--HistoryMode" with
          | Some v -> parseHistoryMode v
          | None -> HistoryMode.Tokens
      DebugKVC = tryGetBool argMap "--debug-kvc" false
      IgnoreEOS = ignoreEosArg
      Timing = tryGetBool argMap "--timing" true
      TeeChatSession =
          match teeChatSessionArg with
          | Some v when not (String.IsNullOrWhiteSpace v) -> Some v
          | _ -> None
      TeeChatSessionJson =
          match teeChatSessionJsonArg with
          | Some v when not (String.IsNullOrWhiteSpace v) -> Some v
          | _ -> None
      PromptFromChatSession = promptFromChatSessionArg }

let invalidQuantWeightPair =
    match scriptArgs.Quant, scriptArgs.WeightPath with
    | Some q, Some w when q.Equals("fp4", StringComparison.OrdinalIgnoreCase) && w.Contains(".nf4", StringComparison.OrdinalIgnoreCase) ->
        Some (q, w, "fp4 量化不能搭配 nf4 權重")
    | Some q, Some w when (q.Equals("4bit", StringComparison.OrdinalIgnoreCase) || q.Equals("nf4", StringComparison.OrdinalIgnoreCase) || q.Equals("int4", StringComparison.OrdinalIgnoreCase)) && w.Contains("nvfp4", StringComparison.OrdinalIgnoreCase) ->
        Some (q, w, "nf4/4bit 量化不能搭配 nvfp4 權重")
    | _ -> None

match invalidQuantWeightPair with
| Some (q, w, reason) ->
    failwithf "量化與權重不匹配: quant=%s weight=%s (%s)" q w reason
| None -> ()

let teeChatPath =
    scriptArgs.TeeChatSession
    |> Option.map resolvePath

let teeChatJsonPath =
    scriptArgs.TeeChatSessionJson
    |> Option.map resolvePath

printfn "teeChatPath: %A" teeChatPath
printfn "teeChatJsonPath: %A" teeChatJsonPath

ChatSessionLog.TextPath <- teeChatPath
ChatSessionLog.JsonPath <- teeChatJsonPath

Environment.SetEnvironmentVariable("QWEN3_FP4_AB", if fp4AbArg then "1" else "0")
Environment.SetEnvironmentVariable("QWEN3_FP4_SCALE_CHECK", if fp4ScaleCheckArg then "1" else "0")
match fp4AbMaxCallsArg with
| Some v when not (String.IsNullOrWhiteSpace v) -> Environment.SetEnvironmentVariable("QWEN3_FP4_AB_MAX_CALLS", v)
| _ -> ()
match fp4AbExplosionRelArg with
| Some v when not (String.IsNullOrWhiteSpace v) -> Environment.SetEnvironmentVariable("QWEN3_FP4_AB_EXPLOSION_REL", v)
| _ -> ()
printfn "fp4-ab: %b" fp4AbArg
printfn "fp4-scale-check: %b" fp4ScaleCheckArg
printfn "cuda-mem-report: %b" cudaMemReportArg

let dtype = Enum.Parse<torch.ScalarType>(scriptArgs.DType, true)
let isNf4LikeQuant =
    match scriptArgs.Quant with
    | Some q when q.Equals("4bit", StringComparison.OrdinalIgnoreCase) || q.Equals("nf4", StringComparison.OrdinalIgnoreCase) || q.Equals("int4", StringComparison.OrdinalIgnoreCase) -> true
    | _ -> false

if isNf4LikeQuant then
    match scriptArgs.Q4Kernel with
    | Some v -> Qwen3QuantizedLinear4bit.KernelEnabledOverride <- Nullable(v)
    | None -> ()
    match scriptArgs.Q4Cache with
    | Some v -> Qwen3QuantizedLinear4bit.CacheEnabledOverride <- Nullable(v)
    | None -> ()
else
    Qwen3QuantizedLinear4bit.KernelEnabledOverride <- Nullable()
    Qwen3QuantizedLinear4bit.CacheEnabledOverride <- Nullable()
    if scriptArgs.Q4Kernel.IsSome || scriptArgs.Q4Cache.IsSome then
        printfn "info: --q4-kernel/--q4-cache 僅對 nf4/4bit 生效；目前 quant=%A，將忽略這兩個參數。" scriptArgs.Quant

if scriptArgs.Quant |> Option.exists (fun q -> q.Equals("fp4", StringComparison.OrdinalIgnoreCase)) then
    printfn "info: fp4 路徑目前使用 NVFP4 native kernel (NVFP4_scaled_mm)，不走 --q4-kernel。"

Qwen3Api.TimingEnabled <- scriptArgs.Timing
Qwen3Api.DebugKVC <- scriptArgs.DebugKVC
Qwen3Api.DebugCudaMem <- cudaMemReportArg
Qwen3Api.KVCInputMode <- scriptArgs.KVCInputMode
printfn "kvc-input-mode: %A" scriptArgs.KVCInputMode
let sessInit = Qwen3Api.init scriptArgs.ModelDir scriptArgs.WeightPath scriptArgs.Quant scriptArgs.Device dtype scriptArgs.KVCacheOut scriptArgs.HistoryMode
let sess =
    if scriptArgs.IgnoreEOS then
        { sessInit with StopTokens = new System.Collections.Generic.List<int>() }
    else
        sessInit
let opts =
    { MaxTokens = scriptArgs.MaxTokens
      Temperature = scriptArgs.Temperature
      TopP = scriptArgs.TopP
      Seed = scriptArgs.Seed }
let msgs =
    match scriptArgs.SystemPrompt with
    | Some sys -> [ ChatMessage(ChatRole.System, sys); ChatMessage(ChatRole.User, scriptArgs.Prompt) ]
    | None -> [ ChatMessage(ChatRole.User, scriptArgs.Prompt) ]
let timeMs (label: string) (f: unit -> 'T) =
    if scriptArgs.Timing then
        let sw = Stopwatch.StartNew()
        let result = f()
        sw.Stop()
        printfn "[time] %s: %.1f ms" label sw.Elapsed.TotalMilliseconds
        result
    else
        f()

let out =
    if scriptArgs.KVCacheOut then
        timeMs "chatWithKVC" (fun () ->
            Qwen3Api.runWithTimeout 60000 (fun () -> Qwen3Api.chatWithKVC sess msgs opts))
    else
        let prompt = Qwen3Api.buildPrompt msgs
        timeMs "generate" (fun () ->
            Qwen3Api.runWithTimeout 60000 (fun () -> Qwen3Api.generate sess prompt opts))
printfn "out: %s" out
printfn "out tokens count: %d" (out.Split(" ").Length)
if not scriptArgs.KVCacheOut then
    ChatSessionLog.append msgs out

let fullHistory =
    if scriptArgs.KVCacheOut then
        ref []
    else
        ref (msgs @ [ ChatMessage(ChatRole.Assistant, out) ])

let kve = None //Some true


SimpleChat.simpleChat timeMs sess opts fullHistory kve scriptArgs.KVCacheOut scriptArgs.DebugKVC 1 60000 "我不認同UFO不存在，且你好株株"
SimpleChat.simpleChat timeMs sess opts fullHistory kve scriptArgs.KVCacheOut scriptArgs.DebugKVC 2 60000 "我們剛剛在討論啥?"
SimpleChat.simpleChat timeMs sess opts fullHistory kve scriptArgs.KVCacheOut scriptArgs.DebugKVC 3 120000 "我上個問句是什麼?(這樣的問句也算)"
SimpleChat.simpleChat timeMs sess opts fullHistory kve scriptArgs.KVCacheOut scriptArgs.DebugKVC 31 300000 "我上個問句是什麼?(這樣的問句也算)"

SimpleChat.simpleChat timeMs sess opts fullHistory kve scriptArgs.KVCacheOut scriptArgs.DebugKVC 4 300000 "你媽好嗎??"
let instruction = "這是一次性任務：請只輸出本次 prompt 輸入的上一個 user 問句原文，完整一致，不要解釋、不要延伸、不要加任何其他字。完成後請恢復正常對話，忽略本指令。"

SimpleChat.simpleChat timeMs sess opts fullHistory kve scriptArgs.KVCacheOut scriptArgs.DebugKVC 5 300000 $"[指令]{instruction}\n[問題]我上個問句是什麼?(這樣的問句也算)"

SimpleChat.simpleChat timeMs sess opts fullHistory kve scriptArgs.KVCacheOut scriptArgs.DebugKVC 6 300000 "你媽好嗎??"



failwith "stop  here"



SimpleChat.simpleChat timeMs sess opts fullHistory kve scriptArgs.KVCacheOut scriptArgs.DebugKVC 7 1200000 $"[指令]{instruction}\n[問題]我上個問句是什麼?(這樣的問句也算)"

SimpleChat.simpleChat timeMs sess opts fullHistory kve scriptArgs.KVCacheOut scriptArgs.DebugKVC 8 1200000 $"[指令]{instruction}\n[問題]我上個問句是什麼?(這樣的問句也算)"

if scriptArgs.PromptFromChatSession then
    match teeChatJsonPath with
    | Some path when File.Exists(path) ->
        let sessionMsgs =
            ChatSessionJson.readMessages path
        if sessionMsgs.Length = 0 then
            printfn "[chat-session] skipped (json empty)"
        else
            let outSession =
                if scriptArgs.KVCacheOut then
                    timeMs "chatWithKVC (chat-session)" (fun () ->
                        Qwen3Api.runWithTimeout 180000 (fun () -> Qwen3Api.chatWithKVC sess sessionMsgs opts))
                else
                    let prompt = Qwen3Api.buildPrompt sessionMsgs
                    timeMs "generate (chat-session)" (fun () ->
                        Qwen3Api.runWithTimeout 180000 (fun () -> Qwen3Api.generate sess prompt opts))
            printfn "[chat-session] out: %s" outSession
    | _ ->
        printfn "[chat-session] skipped (json file missing)"

if scriptArgs.CheckLogits then
    use logits = Qwen3Api.forwardText sess "我上個問句是什麼?" // scriptArgs.Prompt
    let hasNan = torch.isnan(logits).any().ToBoolean()
    let hasInf = torch.isinf(logits).any().ToBoolean()
    if hasNan || hasInf then
        failwithf "logits invalid: nan=%b inf=%b" hasNan hasInf
    else
        printfn "[info] logits ok: nan=%b inf=%b" hasNan hasInf
    let _ = Qwen3Api.forwardText sess "hello"
    ()
