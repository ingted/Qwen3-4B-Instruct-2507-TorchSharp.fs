#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.9"
#load "Nvfp4State.fs"
#load "Types.fs"
#load "Nvfp4Optimizer.fs"
#load "Qwen3Model.fs"
#load "InferenceBridge.fs"

open System
open TorchSharp
open Qwen3_4B_Instruct_2507_TorchSharp_fs

let weightPath = "artifacts/stageE-enhanced-alignment.dat"
let cfg = { Defaults.trainingConfig with WeightPath = weightPath; Device = "cuda" }
let model = Qwen3Model.create cfg

let test prompt =
    printfn "Prompt: %s" prompt
    let reply = InferenceBridge.infer model prompt 32 0.1f
    printfn "Reply: %s" reply
    printfn "---"

test "你是誰"
test "談談UFO"
test "我是誰"

Qwen3Model.dispose model
