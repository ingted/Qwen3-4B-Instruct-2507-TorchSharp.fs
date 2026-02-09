#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.6"
#r "../../TorchSharp_In_DGX_Spark_fp4/TorchSharp.Q4.Extension/bin/Release/net10.0/TorchSharp.Q4.Extension.dll"
#r "../bin/Release/net10.0/Qwen3-4B-Instruct-2507-TorchSharp.fs.dll"

open System
open TorchSharp
open Qwen3_4B_Instruct_2507_TorchSharp_fs

let ensure cond msg =
  if not cond then
    failwith msg

let testCliDefaults () =
  let cfg = Cli.parse [||]
  ensure (cfg.Device = Defaults.trainingConfig.Device) "cli default device mismatch"
  ensure (cfg.SyntheticMode = Defaults.trainingConfig.SyntheticMode) "cli default synthetic mismatch"

let testSyntheticState () =
  let cfg = { Defaults.trainingConfig with SyntheticMode = true; Device = "cpu"; InFeatures = 64L; OutFeatures = 64L }
  let st = Nvfp4State.load cfg
  ensure (st.Layers.Length = 2) "synthetic layer count mismatch"

let testModelForward () =
  let cfg = { Defaults.trainingConfig with SyntheticMode = true; Device = "cpu"; InFeatures = 64L; OutFeatures = 64L }
  let st = Nvfp4State.load cfg
  use model = Qwen3Model.create cfg st
  use x = torch.randn([| 1L; 64L |], dtype = torch.float16, device = "cpu")
  use y = model.Forward(x)
  ensure (y.shape = [| 1L; 64L |]) "forward output shape mismatch"

let testTrainerLoop () =
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
    }
  let st = Nvfp4State.load cfg
  use model = Qwen3Model.create cfg st
  Trainer.run cfg model

let tests : (string * (unit -> unit)) list =
  [
    "cli defaults", testCliDefaults
    "synthetic state", testSyntheticState
    "model forward", testModelForward
    "trainer loop", testTrainerLoop
  ]

printfn "[Tests] total=%d" tests.Length
for (name, run) in tests do
  run ()
  printfn "[PASS] %s" name

printfn "[Tests] all checks passed"
