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
#endif

open System
open TorchSharp
open Runner_type
open Runner_api

let run() =
    let modelDir = "/models/qwen3-4b-instruct-2507-torchsharp"
    
    // 測試 NVFP4
    printfn "--- TESTING NVFP4 ---"
    let weightFp4 = "Qwen3-4B-Instruct-2507-nvfp4.dat"
    let sessFp4 = Qwen3Api.init modelDir (Some weightFp4) (Some "fp4") "cuda" torch.float16 false HistoryMode.Tokens
    let outFp4 = Qwen3Api.generate sessFp4 "Write one short sentence about UFO." { MaxTokens = 30; Temperature = 0.0f; TopP = 1.0f; Seed = None }
    printfn "NVFP4 OUT: %s" outFp4
    
    failwith "done"

let _ = run()