#if INTERACTIVE
#load "loadCUDA.fsx"
#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.9"
#r "nuget: Tokenizers.DotNet, 1.3.0"
#r "nuget: Tokenizers.DotNet.runtime.linux-arm64, 1.3.0"
#r "/workspace/TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/bin/Release/net10.0/TorchSharp.Q4.Extension.dll"
#r "/workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/bin/Release/net10.0/Qwen3-4B-Instruct-2507-TorchSharp.fs.dll"
#endif

open System
open System.IO
open System.Text
open System.Text.Json
open TorchSharp
open TorchSharp.Q4.Extension
open Qwen3_4B_Instruct_2507_TorchSharp_fs

// ==========================================
// 參數處理
// ==========================================
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

let argMap = 
    #if INTERACTIVE
    parseArgs (fsi.CommandLineArgs |> Array.skip 1)
    #else
    parseArgs (Environment.GetCommandLineArgs() |> Array.skip 1)
    #endif

let tryGet key fallback = argMap |> Map.tryFind key |> Option.defaultValue fallback
let tryGetInt key fallback = argMap |> Map.tryFind key |> Option.map Int32.Parse |> Option.defaultValue fallback

let modelDir = tryGet "--model-dir" "/models/qwen3-4b-instruct-2507-torchsharp"
let weightPath = Some (tryGet "--weight" "Qwen3-4B-Instruct-2507-nvfp4.dat")
let targetTokens = tryGetInt "--target-tokens" 64000
let chunkSize = tryGetInt "--chunk-size" 4000 // 每一輪注入多少 tokens
let secretKey = tryGet "--secret" "GEMINI-CODE-7788"

printfn "🚀 長上下文測試啟動"
printfn "目標 Tokens: %d" targetTokens
printfn "區塊大小: %d" chunkSize
printfn "秘密金鑰: %s" secretKey

// ==========================================
// 初始化 Session 與 持久化 Cache
// ==========================================
let dtype = torch.ScalarType.Float16
let session = InferenceBridge.init modelDir weightPath (Some "fp4") "cuda" dtype
let persistentCache = new InferenceBridge.InferenceKvCache(session.Layers.Length)

// 清理函數
let forceCleanUp () =
    System.GC.Collect()
    System.GC.WaitForPendingFinalizers()
    if torch.cuda_is_available() then
        torch.cuda.synchronize()
    NativeInterop.tryEmptyNvfp4Cache() |> ignore

let history = ResizeArray<string * string>()
let stopTokens = [ 151645; 151643 ]

let runTurnPersistent (role: string) (content: string) (maxTokens: int) (timeoutMs: int) =
    forceCleanUp()
    
    // 構造本次要餵入的文本：如果是 user，則包含完整結構到 assistant 開始處
    let promptToFeed = 
        if role = "user" then
            $"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        else
            $"<|im_start|>{role}\n{content}<|im_end|>\n"

    let currentTokensInCache = persistentCache.SeqLen
    
    let opts: InferenceGenOptions =
        { MaxTokens = maxTokens
          Temperature = 0.0f
          TopP = 1.0f
          Seed = Some 123 }

    let sw = Diagnostics.Stopwatch.StartNew()
    let task = System.Threading.Tasks.Task.Run(fun () ->
        InferenceBridge.generateFromRenderedPromptWithStopTokensKvCachePersistent 
            session persistentCache promptToFeed opts stopTokens KvPrefillMode.PromptByPrompt
    )
    
    if task.Wait(timeoutMs) then
        sw.Stop()
        let out = task.Result
        // 更新歷史紀錄 (僅供最後打印或參考)
        history.Add((role, content))
        history.Add(("assistant", out))
        printfn "[Turn] Total in Cache: %d tokens (Added: %d) | Elapsed: %.1f ms" 
            persistentCache.SeqLen (session.Tokenizer.Encode(promptToFeed).Length) sw.Elapsed.TotalMilliseconds
        out
    else
        printfn "❌ 超時中斷！目前 Cache Tokens: %d" currentTokensInCache
        failwith "Timeout"

// ==========================================
// 測試流程
// ==========================================

try
    // 1. 注入秘密
    printfn "
1. 注入秘密訊息..."
    runTurnPersistent "user" (sprintf "請記住這個秘密金鑰，之後我會問你：'%s'。除了確認收到外，不要說多餘的話。" secretKey) 20 60000 |> ignore

    // 2. 循環注入填充文本
    printfn "
2. 開始填充上下文 (目標: %d tokens)..." targetTokens
    let haystackSource = File.ReadAllText("haystack_source.txt")
    let mutable chunkCount = 1

    while persistentCache.SeqLen < int64 targetTokens do
        // 構造一個大約 chunkSize 大小的文本
        let sb = StringBuilder()
        while session.Tokenizer.Encode(sb.ToString()).Length < chunkSize do
            sb.Append(haystackSource).Append(" ") |> ignore
        
        let chunkText = sb.ToString()
        printfn "--- 注入區塊 #%d ---" chunkCount
        // 設定 MaxTokens = 1 減少生成開銷，我們只想 prefill
        runTurnPersistent "user" (sprintf "這是參考資料第 %d 號：
%s" chunkCount chunkText) 1 120000 |> ignore
        
        chunkCount <- chunkCount + 1

    // 3. 最終驗證
    printfn "
3. 執行針尖測試 (Needle Retrieval)..."
    let result = runTurnPersistent "user" "我一開始叫你記住的秘密金鑰是什麼？請直接回答金鑰內容。" 50 60000
    printfn "
[結果] 模型的回答: %s" result
    
    if result.Contains(secretKey) then
        printfn "✅ 測試成功！模型在 %d tokens 下依然保有正確記憶。" persistentCache.SeqLen
    else
        printfn "❌ 測試失敗。模型可能遺忘了金鑰或受到長文本干擾。"

finally
    (persistentCache :> IDisposable).Dispose()
    InferenceBridge.dispose session
    printfn "測試結束。"
