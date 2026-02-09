namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System

module Cli =
  let private readMap (args: string array) =
    args
    |> Array.toList
    |> List.chunkBySize 2
    |> List.choose (function
      | [k; v] when k.StartsWith("--", StringComparison.Ordinal) -> Some (k, v)
      | _ -> None)
    |> Map.ofList

  let private getOrDefault key def (m: Map<string, string>) =
    m.TryFind(key) |> Option.defaultValue def

  let private parseInt key def (m: Map<string, string>) =
    match m.TryFind key with
    | None -> def
    | Some v ->
      match Int32.TryParse v with
      | true, x -> x
      | _ -> def

  let private parseInt64 key def (m: Map<string, string>) =
    match m.TryFind key with
    | None -> def
    | Some v ->
      match Int64.TryParse v with
      | true, x -> x
      | _ -> def

  let private parseBool key def (m: Map<string, string>) =
    match m.TryFind key with
    | None -> def
    | Some v ->
      match Boolean.TryParse v with
      | true, x -> x
      | _ -> def

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
          SyntheticMode = parseBool "--synthetic" Defaults.trainingConfig.SyntheticMode kv
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
    printfn "  --synthetic <true|false>"
