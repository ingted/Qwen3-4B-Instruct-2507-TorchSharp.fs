#if INTERACTIVE
#load "loadCUDA.fsx"
#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.9"
#r "/workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/bin/Release/net10.0/Qwen3-4B-Instruct-2507-TorchSharp.fs.dll"
#endif

open System
open TorchSharp
open Qwen3_4B_Instruct_2507_TorchSharp_fs

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
       "--dtype"; "Float16"
       "--quant"; "fp4"
       "--weight"; "Qwen3-4B-Instruct-2507-nvfp4.dat"
       "--prompt"; "Write one short sentence about UFO and you."
       "--max-tokens"; "64"
       "--temp"; "0"
       "--top-p"; "1"
       "--seed"; "123"
       "--ifInteractive"; "true" |]

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

let modelDir = tryGet argMap "--model-dir" |> Option.defaultValue Defaults.modelDir
let device = tryGet argMap "--device" |> Option.defaultValue "cuda"
let dtypeName = tryGet argMap "--dtype" |> Option.defaultValue "Float16"
let dtype = Enum.Parse<torch.ScalarType>(dtypeName, true)
let quant = tryGet argMap "--quant"
let weight = tryGet argMap "--weight"
let prompt = tryGet argMap "--prompt" |> Option.defaultValue "Write one short sentence about UFO and you."
let maxTokens = tryGetInt argMap "--max-tokens" 64
let temp = tryGetFloat argMap "--temp" 0.0f
let topP = tryGetFloat argMap "--top-p" 1.0f
let seed = tryGetSeed argMap "--seed"
let ifInteractive = tryGetBool argMap "--ifInteractive" true

printfn "modelDir: %s" modelDir
printfn "device: %s" device
printfn "dtype: %A" dtype
printfn "quant: %A" quant
printfn "weight: %A" weight

let session =
    InferenceBridge.init modelDir weight quant device dtype

try
    let opts: InferenceGenOptions =
        { MaxTokens = maxTokens
          Temperature = temp
          TopP = topP
          Seed = seed }

    let runOnePrompt (p: string) =
        let out = InferenceBridge.generate session p opts
        printfn "out: %s" out
        printfn "out tokens count: %d" (out.Split(" ").Length)

    runOnePrompt prompt

    if ifInteractive then
        printfn "進入互動模式（Ctrl+C 結束）"
        if Console.IsInputRedirected then
            failwith "--ifInteractive=true requires interactive stdin (TTY)."
        while true do
            printfn "請輸入:"
            let input = Console.ReadLine()
            if isNull input then
                System.Threading.Thread.Sleep(200)
            elif not (String.IsNullOrWhiteSpace input) then
                runOnePrompt input
finally
    InferenceBridge.dispose session
