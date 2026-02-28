#if INTERACTIVE
#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.9"
#r "nuget: Tokenizers.DotNet, 1.3.0"
#r "nuget: Tokenizers.DotNet.runtime.linux-arm64, 1.3.0"
#r "nuget: Microsoft.Extensions.AI, 9.9.0"
#r "/workspace/fsann/Qwen3-4B-Instruct-2507-TorchSharp-mod/Qwen3/bin/Release/net10.0/Qwen3.dll"
#load "Runner_type.fsx"
#endif

open System
open System.IO
open System.Text.Json
open Microsoft.Extensions.AI
open Qwen3
open Runner_type
let parseArgs (argv: string[]) =
    let rec loop i (m: Map<string,string>) =
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

let tryGet (argMap: Map<string,string>) (key: string) =
    argMap |> Map.tryFind key

let tryGetAny (argMap: Map<string,string>) (keys: string list) =
    keys |> List.tryPick (tryGet argMap)

let tryGetInt (argMap: Map<string,string>) (key: string) (fallback: int) =
    match tryGet argMap key with
    | Some v ->
        match Int32.TryParse(v) with
        | true, i -> i
        | _ -> fallback
    | None -> fallback

let tryGetFloat (argMap: Map<string,string>) (key: string) (fallback: float32) =
    match tryGet argMap key with
    | Some v ->
        match Single.TryParse(v) with
        | true, f -> f
        | _ -> fallback
    | None -> fallback

let tryGetSeed (argMap: Map<string,string>) (key: string) =
    match tryGet argMap key with
    | Some v ->
        match Int32.TryParse(v) with
        | true, i when i >= 0 -> Some i
        | _ -> None
    | None -> None

let tryGetBool (argMap: Map<string,string>) (key: string) (fallback: bool) =
    match tryGet argMap key with
    | Some v ->
        match v.Trim().ToLowerInvariant() with
        | "1" | "true" | "yes" -> true
        | "0" | "false" | "no" -> false
        | _ -> fallback
    | None -> fallback

let tryGetBoolOpt (argMap: Map<string,string>) (key: string) =
    match tryGet argMap key with
    | Some v ->
        match v.Trim().ToLowerInvariant() with
        | "1" | "true" | "yes" -> Some true
        | "0" | "false" | "no" -> Some false
        | _ -> None
    | None -> None

let parseBoolOpt (value: string) =
    match value.Trim().ToLowerInvariant() with
    | "1" | "true" | "yes" -> Some true
    | "0" | "false" | "no" -> Some false
    | _ -> None

let parseHistoryMode (value: string) =
    match value.Trim().ToLowerInvariant() with
    | "message" | "messages" -> HistoryMode.Messages
    | "token" | "tokens" -> HistoryMode.Tokens
    | "both" -> HistoryMode.Both
    | _ -> HistoryMode.Tokens

let resolvePath (path: string) =
    if Path.IsPathRooted path then path else Path.Combine(Environment.CurrentDirectory, path)

module ChatSessionLog =
    let mutable TextPath: string option = None
    let mutable JsonPath: string option = None

    let private ensureDir (path: string) =
        let dir = Path.GetDirectoryName(path)
        if not (String.IsNullOrWhiteSpace dir) then
            Directory.CreateDirectory(dir) |> ignore

    let append (messages: ChatMessage list) (output: string) =
        let ts = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
        match TextPath with
        | Some path ->
            ensureDir path
            let sb = System.Text.StringBuilder()
            sb.AppendLine($"==== {ts} ====") |> ignore
            for msg in messages do
                let role = msg.Role.Value
                sb.AppendLine($"[{role}] {msg.Text}") |> ignore
            sb.AppendLine($"[assistant] {output}") |> ignore
            sb.AppendLine() |> ignore
            File.AppendAllText(path, sb.ToString())
        | None -> ()

        match JsonPath with
        | Some path ->
            ensureDir path
            let entry: ChatSessionJsonEntry =
                { ts = ts
                  messages = messages |> List.map (fun m -> { role = m.Role.Value; text = m.Text })
                  output = output }
            let line = JsonSerializer.Serialize(entry)
            File.AppendAllText(path, line + Environment.NewLine)
        | None -> ()

module Paths =
    let defaultModelDir = "/models/qwen3-4b-instruct-2507-torchsharp"

    let pickWeightFile (dir: string) =
        let candidates = Directory.GetFiles(dir, "*.dat")
        if candidates.Length = 0 then
            None
        else
            Some candidates.[0]

    let tryPickByKeyword (keyword: string) (candidates: string array) =
        candidates
        |> Array.tryFind (fun p -> p.Contains(keyword, StringComparison.OrdinalIgnoreCase))

    let resolveWeightPath (baseDir: string) (quantHint: string option) (weightOverride: string option) =
        match weightOverride with
        | Some w when not (String.IsNullOrWhiteSpace w) ->
            if Path.IsPathRooted(w) then w else Path.Combine(baseDir, w)
        | _ ->
            let candidates = Directory.GetFiles(baseDir, "*.dat")
            let chosen =
                match quantHint with
                | Some q when not (String.IsNullOrWhiteSpace q) ->
                    if q.Equals("4bit", StringComparison.OrdinalIgnoreCase) || q.Equals("int4", StringComparison.OrdinalIgnoreCase) then
                        tryPickByKeyword "4bit" candidates |> Option.orElse (Array.tryHead candidates)
                    elif q.Equals("fp16", StringComparison.OrdinalIgnoreCase) || q.Equals("float16", StringComparison.OrdinalIgnoreCase) then
                        tryPickByKeyword "fp16" candidates |> Option.orElse (Array.tryHead candidates)
                    elif q.Equals("fp32", StringComparison.OrdinalIgnoreCase) || q.Equals("float32", StringComparison.OrdinalIgnoreCase) then
                        tryPickByKeyword "fp32" candidates |> Option.orElse (Array.tryHead candidates)
                    else
                        Array.tryHead candidates
                | _ ->
                    tryPickByKeyword "fp16" candidates |> Option.orElse (Array.tryHead candidates)
            match chosen with
            | Some p -> p
            | None -> Path.Combine(baseDir, "Qwen3-4B-Instruct-2507-fp16.dat")

    let resolve (modelDir: string) (weightOverride: string option) (quantHint: string option) =
        let baseDir =
            if String.IsNullOrWhiteSpace modelDir then defaultModelDir else modelDir.Trim()

        let configPath = Path.Combine(baseDir, "config.json")
        let tokenizerPath = Path.Combine(baseDir, "tokenizer.json")
        let weightPath = resolveWeightPath baseDir quantHint weightOverride

        printfn "configPath: %s" configPath
        printfn "tokenizerPath: %s" tokenizerPath
        printfn "weightPath: %s" weightPath

        { ModelDir = baseDir
          ConfigPath = configPath
          TokenizerPath = tokenizerPath
          WeightPath = weightPath }

module Loader =
    let loadConfig (path: string) =
        let json = File.ReadAllText(path)
        JsonSerializer.Deserialize<Qwen3Config>(json)

    let ensureFiles (paths: ModelPaths) =
        let missing =
            [ paths.ConfigPath; paths.TokenizerPath; paths.WeightPath ]
            |> List.filter (fun p -> not (File.Exists p))
        if missing.Length > 0 then
            let msg = String.Join(Environment.NewLine, missing |> List.map (fun p -> $"missing: {p}"))
            invalidOp msg
