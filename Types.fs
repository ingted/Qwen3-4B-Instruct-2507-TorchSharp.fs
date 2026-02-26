namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System
open TorchSharp
open TorchSharp.Q4.Extension

type TrainingConfig =
  {
    ModelDir: string
    ConfigPath: string
    TokenizerPath: string
    WeightPath: string
    Device: string
    Epochs: int
    StepsPerEpoch: int
    BatchSize: int64
    InFeatures: int64
    OutFeatures: int64
    MaxLayers: int
    SyntheticMode: bool
    UseKvCache: bool
    SequenceLength: int64
    LearningRate: float
    LossMode: string
    UsePackedNvfp4Optimizer: bool
    GradCheckpointChunk: int
    OptimizerStepChunkRows: int64
    OffloadMVToCpu: bool
    OffloadWToCpu: bool
    OffloadGradToCpu: bool
    StepFlushEachParam: bool
    ProfileTrainStepVram: bool
    TrainStepVramReportPath: string option
    CheckpointDir: string
    SaveEverySteps: int
    ResumeFromCheckpoint: bool
    StrictLoad: bool
  }

module Defaults =
  let modelDir = "/models/qwen3-4b-instruct-2507-torchsharp"
  let configPath = "/models/qwen3-4b-instruct-2507-torchsharp/config.json"
  let tokenizerPath = "/models/qwen3-4b-instruct-2507-torchsharp/tokenizer.json"
  let weightPath = "/models/qwen3-4b-instruct-2507-torchsharp/Qwen3-4B-Instruct-2507-nvfp4.dat"

  let trainingConfig : TrainingConfig =
    {
      ModelDir = modelDir
      ConfigPath = configPath
      TokenizerPath = tokenizerPath
      WeightPath = weightPath
      Device = "cuda"
      Epochs = 1
      StepsPerEpoch = 10
      BatchSize = 2L
      InFeatures = 1024L
      OutFeatures = 1024L
      MaxLayers = 2
      SyntheticMode = true
      UseKvCache = false
      SequenceLength = 8L
      LearningRate = 1e-3
      LossMode = "scalar"
      UsePackedNvfp4Optimizer = true
      GradCheckpointChunk = 0
      OptimizerStepChunkRows = 32L
      OffloadMVToCpu = false
      OffloadWToCpu = false
      OffloadGradToCpu = false
      StepFlushEachParam = true
      ProfileTrainStepVram = false
      TrainStepVramReportPath = None
      CheckpointDir = "./checkpoints"
      SaveEverySteps = 0
      ResumeFromCheckpoint = false
      StrictLoad = true
    }

module Q4 =
  let private defaultUnifiedMemoryPolicy () =
    let raw = Environment.GetEnvironmentVariable("TS_Q4_DISABLE_UM")
    if String.IsNullOrWhiteSpace(raw) then
      UnifiedMemoryPolicy.Disabled
    else
      match raw.Trim().ToLowerInvariant() with
      | "0"
      | "false"
      | "no" -> UnifiedMemoryPolicy.PreferUnified
      | _ -> UnifiedMemoryPolicy.Disabled

  let pureNvfp4SessionConfig : Q4SessionConfig =
    {
      BackendOverride = Some "nvfp4-kernel"
      ComputePath = Q4ComputePath.KernelOnly
      RuntimeTarget = Q4RuntimeTarget.Auto
      UnifiedMemoryPolicy = defaultUnifiedMemoryPolicy ()
      EnableDiagnostics = true
    }

  let pureNvfp4Schema : Q4Schema =
    {
      Format = QuantFormat.NVFP4
      WeightKey = "layer.qdata"
      ScaleKey = Some "layer.scale"
      AbsmaxKey = None
      QuantMapKey = None
      ExtraKeys = []
    }
