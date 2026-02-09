namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

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
    }

module Q4 =
  let pureNvfp4SessionConfig : Q4SessionConfig =
    {
      BackendOverride = Some "nvfp4-kernel"
      ComputePath = Q4ComputePath.KernelOnly
      RuntimeTarget = Q4RuntimeTarget.Auto
      UnifiedMemoryPolicy = UnifiedMemoryPolicy.PreferUnified
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
