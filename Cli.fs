namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System

module Cli =
  let readMap (args: string array) =
    args
    |> Array.toList
    |> List.chunkBySize 2
    |> List.choose (function
      | [k; v] when k.StartsWith("--", StringComparison.Ordinal) -> Some (k, v)
      | _ -> None)
    |> Map.ofList

  let getOrDefault key def (m: Map<string, string>) =
    m.TryFind(key) |> Option.defaultValue def

  let parseInt key def (m: Map<string, string>) =
    match m.TryFind key with
    | None -> def
    | Some v ->
      match Int32.TryParse v with
      | true, x -> x
      | _ -> def

  let parseInt64 key def (m: Map<string, string>) =
    match m.TryFind key with
    | None -> def
    | Some v ->
      match Int64.TryParse v with
      | true, x -> x
      | _ -> def

  let parseFloat key def (m: Map<string, string>) =
    match m.TryFind key with
    | None -> def
    | Some v ->
      match Double.TryParse v with
      | true, x -> x
      | _ -> def

  let parseBool key def (m: Map<string, string>) =
    match m.TryFind key with
    | None -> def
    | Some v ->
      match Boolean.TryParse v with
      | true, x -> x
      | _ -> def

  let parseBoolAny (keys: string list) def (m: Map<string, string>) =
    keys
    |> List.tryPick (fun key ->
      match m.TryFind key with
      | None -> None
      | Some v ->
        match Boolean.TryParse v with
        | true, x -> Some x
        | _ -> None)
    |> Option.defaultValue def

  let parse (args: string array) : TrainingConfig =
    let kv = readMap args
    {
      Defaults.trainingConfig with
          ModelDir = getOrDefault "--model-dir" Defaults.trainingConfig.ModelDir kv
          ConfigPath = getOrDefault "--config" Defaults.trainingConfig.ConfigPath kv
          TokenizerPath = getOrDefault "--tokenizer" Defaults.trainingConfig.TokenizerPath kv
          WeightPath = getOrDefault "--weight" Defaults.trainingConfig.WeightPath kv
          Device = getOrDefault "--device" Defaults.trainingConfig.Device kv
          Epochs = parseInt "--epochs" Defaults.trainingConfig.Epochs kv
          StepsPerEpoch = parseInt "--steps-per-epoch" Defaults.trainingConfig.StepsPerEpoch kv
          BatchSize = parseInt64 "--batch-size" Defaults.trainingConfig.BatchSize kv
          InFeatures = parseInt64 "--in-features" Defaults.trainingConfig.InFeatures kv
          OutFeatures = parseInt64 "--out-features" Defaults.trainingConfig.OutFeatures kv
          MaxLayers = parseInt "--max-layers" Defaults.trainingConfig.MaxLayers kv
          SyntheticMode = parseBool "--synthetic" Defaults.trainingConfig.SyntheticMode kv
          LearningRate = parseFloat "--lr" Defaults.trainingConfig.LearningRate kv
          CheckpointDir = getOrDefault "--checkpoint-dir" Defaults.trainingConfig.CheckpointDir kv
          SaveEverySteps = parseInt "--save-every-steps" Defaults.trainingConfig.SaveEverySteps kv
          ResumeFromCheckpoint = parseBool "--resume" Defaults.trainingConfig.ResumeFromCheckpoint kv
          StrictLoad =
            parseBoolAny
              [ "--strict-load"; "--restrict-load" ]
              Defaults.trainingConfig.StrictLoad
              kv
    }

  let printUsage () =
    printfn "Qwen3-4B-Instruct-2507-TorchSharp.fs (pure F#, NVFP4)"
    printfn "Args:"
    printfn "  --model-dir <path>"
    printfn "  --config <path>"
    printfn "  --tokenizer <path>"
    printfn "  --weight <path>"
    printfn "  --device <cpu|cuda>"
    printfn "  --epochs <int>"
    printfn "  --steps-per-epoch <int>"
    printfn "  --batch-size <int64>"
    printfn "  --in-features <int64>"
    printfn "  --out-features <int64>"
    printfn "  --max-layers <int>"
    printfn "  --synthetic <true|false>"
    printfn "  --lr <float>"
    printfn "  --checkpoint-dir <path>"
    printfn "  --save-every-steps <int>"
    printfn "  --resume <true|false>"
    printfn "  --strict-load <true|false> (alias: --restrict-load)"
