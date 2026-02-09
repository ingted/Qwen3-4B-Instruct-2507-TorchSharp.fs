#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.6"
#r "../../TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/bin/Release/net10.0/TorchSharp.Q4.Extension.dll"
#r "../bin/Release/net10.0/Qwen3-4B-Instruct-2507-TorchSharp.fs.dll"

open System
open System.IO
open TorchSharp
open Qwen3_4B_Instruct_2507_TorchSharp_fs

let ensure cond msg =
  if not cond then
    failwith msg

let testCliDefaults () =
  let cfg = Cli.parse [||]
  ensure (cfg.Device = Defaults.trainingConfig.Device) "cli default device mismatch"
  ensure (cfg.SyntheticMode = Defaults.trainingConfig.SyntheticMode) "cli default synthetic mismatch"
  ensure (cfg.ResumeFromCheckpoint = Defaults.trainingConfig.ResumeFromCheckpoint) "cli default resume mismatch"

let testSyntheticState () =
  let cfg =
    {
      Defaults.trainingConfig with
          SyntheticMode = true
          Device = "cpu"
          InFeatures = 64L
          OutFeatures = 64L
          CheckpointDir = Path.Combine(Path.GetTempPath(), "qwen3-fs-tests-synth")
    }
  let st = Nvfp4State.load cfg
  ensure (st.Layers.Length = 2) "synthetic layer count mismatch"

let testModelForward () =
  let cfg =
    {
      Defaults.trainingConfig with
          SyntheticMode = true
          Device = "cpu"
          InFeatures = 64L
          OutFeatures = 64L
          CheckpointDir = Path.Combine(Path.GetTempPath(), "qwen3-fs-tests-forward")
    }
  let st = Nvfp4State.load cfg
  use model = Qwen3Model.create cfg st
  use x = torch.randn([| 1L; 64L |], dtype = torch.float16, device = "cpu")
  use y = model.Forward(x, outDtype = torch.float16)
  ensure (y.shape = [| 1L; 64L |]) "forward output shape mismatch"

let testTrainerLoop () =
  let ckptDir = Path.Combine(Path.GetTempPath(), "qwen3-fs-tests-loop")
  if Directory.Exists(ckptDir) then Directory.Delete(ckptDir, true)
  let cfg =
    {
      Defaults.trainingConfig with
          SyntheticMode = true
          Device = "cpu"
          Epochs = 1
          StepsPerEpoch = 1
          BatchSize = 1L
          InFeatures = 64L
          OutFeatures = 64L
          LearningRate = 5e-3
          CheckpointDir = ckptDir
          SaveEverySteps = 1
    }
  let st = Nvfp4State.load cfg
  use model = Qwen3Model.create cfg st
  Trainer.run cfg model
  ensure (File.Exists(Path.Combine(ckptDir, "meta.json"))) "trainer did not create checkpoint meta"

let testOptimizerUpdatesWeights () =
  let ckptDir = Path.Combine(Path.GetTempPath(), "qwen3-fs-tests-opt")
  if Directory.Exists(ckptDir) then Directory.Delete(ckptDir, true)
  let cfg =
    {
      Defaults.trainingConfig with
          SyntheticMode = true
          Device = "cpu"
          Epochs = 1
          StepsPerEpoch = 1
          BatchSize = 1L
          InFeatures = 64L
          OutFeatures = 64L
          LearningRate = 1e-4
          CheckpointDir = ckptDir
          SaveEverySteps = 0
    }
  let st = Nvfp4State.load cfg
  use model = Qwen3Model.create cfg st
  use before = model.Layers.Head.MasterWeight.detach().to_type(torch.float32).cpu().clone()
  Trainer.run cfg model
  use after = model.Layers.Head.MasterWeight.detach().to_type(torch.float32).cpu().clone()
  use delta = (after - before).abs().sum()
  let changed = delta.item<float32>()
  ensure (changed > 0.0f) "optimizer did not update weights"

let testCheckpointRecover () =
  let ckptDir = Path.Combine(Path.GetTempPath(), "qwen3-fs-tests-ckpt")
  if Directory.Exists(ckptDir) then Directory.Delete(ckptDir, true)

  let cfgBase =
    {
      Defaults.trainingConfig with
          SyntheticMode = true
          Device = "cpu"
          Epochs = 1
          StepsPerEpoch = 1
          BatchSize = 1L
          InFeatures = 64L
          OutFeatures = 64L
          LearningRate = 1e-4
          CheckpointDir = ckptDir
          SaveEverySteps = 1
          ResumeFromCheckpoint = false
    }

  let st = Nvfp4State.load cfgBase
  use model1 = Qwen3Model.create cfgBase st
  Trainer.run cfgBase model1

  let layer0Path = Path.Combine(ckptDir, "layer_0000.pt")
  ensure (File.Exists(layer0Path)) "checkpoint layer file missing"
  use saved = torch.load(layer0Path)

  let cfgResume = { cfgBase with ResumeFromCheckpoint = true }
  use model2 = Qwen3Model.create cfgResume st

  match Trainer.tryLoadCheckpoint cfgResume model2 with
  | None -> failwith "checkpoint not loaded"
  | Some state ->
    ensure (state.Epoch = 1) "checkpoint epoch mismatch"
    ensure (state.GlobalStep = 1) "checkpoint global step mismatch"

  use loaded = model2.Layers.Head.MasterWeight.detach().to_type(torch.float32).cpu()
  use savedF = saved.to_type(torch.float32).cpu()
  use diff = (loaded - savedF).abs().mean()
  let err = diff.item<float32>()
  ensure (err < 1e-5f) (sprintf "checkpoint recover mismatch: %f" err)

let tests : (string * (unit -> unit)) list =
  [
    "cli defaults", testCliDefaults
    "synthetic state", testSyntheticState
    "model forward", testModelForward
    "trainer loop", testTrainerLoop
    "optimizer update", testOptimizerUpdatesWeights
    "checkpoint recover", testCheckpointRecover
  ]

printfn "[Tests] total=%d" tests.Length
for (name, run) in tests do
  run ()
  printfn "[PASS] %s" name

printfn "[Tests] all checks passed"
