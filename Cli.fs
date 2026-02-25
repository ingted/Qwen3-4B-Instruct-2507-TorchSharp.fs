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

  let parseStringOption key (m: Map<string, string>) =
    match m.TryFind key with
    | Some v when not (String.IsNullOrWhiteSpace v) -> Some(v.Trim())
    | _ -> None

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
          UseKvCache = parseBool "--use-kvc" Defaults.trainingConfig.UseKvCache kv
          SequenceLength = parseInt64 "--seq-len" Defaults.trainingConfig.SequenceLength kv
          LearningRate = parseFloat "--lr" Defaults.trainingConfig.LearningRate kv
          UsePackedNvfp4Optimizer =
            parseBool "--use-packed-optimizer" Defaults.trainingConfig.UsePackedNvfp4Optimizer kv
          GradCheckpointChunk = parseInt "--grad-ckpt-chunk" Defaults.trainingConfig.GradCheckpointChunk kv
          OptimizerStepChunkRows =
            parseInt64 "--optimizer-step-chunk-rows" Defaults.trainingConfig.OptimizerStepChunkRows kv
          OffloadMVToCpu = parseBool "--offload-mv-to-cpu" Defaults.trainingConfig.OffloadMVToCpu kv
          OffloadWToCpu = parseBool "--offload-w-to-cpu" Defaults.trainingConfig.OffloadWToCpu kv
          OffloadGradToCpu = parseBool "--offload-grad-to-cpu" Defaults.trainingConfig.OffloadGradToCpu kv
          StepFlushEachParam = parseBool "--step-flush-each-param" Defaults.trainingConfig.StepFlushEachParam kv
          ProfileTrainStepVram = parseBool "--profile-train-step-vram" Defaults.trainingConfig.ProfileTrainStepVram kv
          TrainStepVramReportPath = parseStringOption "--train-step-vram-report" kv
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
    printfn "  --use-kvc <true|false>"
    printfn "  --seq-len <int64>"
    printfn "  --lr <float>"
    printfn "  --use-packed-optimizer <true|false>"
    printfn "  --grad-ckpt-chunk <int>"
    printfn "  --optimizer-step-chunk-rows <int64>"
    printfn "  --offload-mv-to-cpu <true|false>"
    printfn "  --offload-w-to-cpu <true|false>"
    printfn "  --offload-grad-to-cpu <true|false>"
    printfn "  --step-flush-each-param <true|false>"
    printfn "  --profile-train-step-vram <true|false>"
    printfn "  --train-step-vram-report <path>"
    printfn "  --checkpoint-dir <path>"
    printfn "  --save-every-steps <int>"
    printfn "  --resume <true|false>"
    printfn "  --strict-load <true|false> (alias: --restrict-load)"
