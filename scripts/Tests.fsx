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

let writeLeb128 (bw: BinaryWriter) (value: uint64) =
  let mutable remaining = value
  let mutable continueWrite = true
  while continueWrite do
    let mutable b = byte (remaining &&& 0x7FUL)
    remaining <- remaining >>> 7
    if remaining <> 0UL then
      b <- b ||| 0x80uy
      bw.Write b
    else
      bw.Write b
      continueWrite <- false

let writeEntry (bw: BinaryWriter) (key: string) (elemType: uint64) (shape: int64 array) (payload: byte array) =
  let keyBytes = Text.Encoding.UTF8.GetBytes key
  writeLeb128 bw (uint64 keyBytes.LongLength)
  bw.Write keyBytes
  writeLeb128 bw elemType
  writeLeb128 bw (uint64 shape.LongLength)
  shape |> Array.iter (fun d -> writeLeb128 bw (uint64 d))
  bw.Write payload

let createSingleLayerDat (path: string) (outFeatures: int64) (inFeatures: int64) =
  let packedCols = inFeatures / 2L
  let scaleCols = inFeatures / 16L
  let qDataBytes = Array.zeroCreate<byte> (int (outFeatures * packedCols))
  let scaleBytes = Array.init (int (outFeatures * scaleCols)) (fun i -> byte ((i % 7) + 1))
  use fs = File.Create path
  use bw = new BinaryWriter(fs)
  writeLeb128 bw 2UL
  writeEntry bw "layer.0.weight.qdata" 0UL [| outFeatures; packedCols |] qDataBytes
  writeEntry bw "layer.0.weight.scale" 0UL [| outFeatures; scaleCols |] scaleBytes

let disposeState (state: Nvfp4ModelState) =
  state.Layers
  |> List.iter (fun layer ->
    layer.Bundle.Weight.Dispose()
    layer.Bundle.Scale |> Option.iter (fun s -> s.Dispose()))

let withModel cfg state f =
  let model = Qwen3Model.create cfg state
  try
    f model
  finally
    Qwen3Model.dispose model

let ensureTensorEquivalentWithSpecials (left: torch.Tensor) (right: torch.Tensor) (tol: float32) (name: string) =
  ensure (left.shape = right.shape) (sprintf "%s shape mismatch" name)

  use left32 = left.to_type(torch.float32).cpu()
  use right32 = right.to_type(torch.float32).cpu()

  use leftNaN = torch.isnan(left32)
  use rightNaN = torch.isnan(right32)
  use nanMismatch = torch.logical_xor(leftNaN, rightNaN)
  ensure (not (nanMismatch.any().item<bool>())) (sprintf "%s NaN mask mismatch" name)

  use leftInf = torch.isinf(left32)
  use rightInf = torch.isinf(right32)
  use infMaskMismatch = torch.logical_xor(leftInf, rightInf)
  ensure (not (infMaskMismatch.any().item<bool>())) (sprintf "%s Inf mask mismatch" name)

  use infEq = left32.eq(right32)
  use infValueMismatchMask = torch.logical_and(leftInf, torch.logical_not(infEq))
  ensure (not (infValueMismatchMask.any().item<bool>())) (sprintf "%s Inf sign/value mismatch" name)

  use finiteMask = torch.logical_not(torch.logical_or(leftNaN, leftInf))
  let finiteCount = finiteMask.sum().item<int64>()
  if finiteCount > 0L then
    use leftFinite = left32.masked_select(finiteMask)
    use rightFinite = right32.masked_select(finiteMask)
    use maxDiff = (leftFinite - rightFinite).abs().max()
    let err = maxDiff.item<float32>()
    ensure (Single.IsFinite err) (sprintf "%s finite diff is not finite: %f" name err)
    ensure (err < tol) (sprintf "%s finite diff too large: %f" name err)

let testCliDefaults () =
  let cfg = Cli.parse [||]
  ensure (cfg.Device = Defaults.trainingConfig.Device) "cli default device mismatch"
  ensure (cfg.SyntheticMode = Defaults.trainingConfig.SyntheticMode) "cli default synthetic mismatch"
  ensure (cfg.ResumeFromCheckpoint = Defaults.trainingConfig.ResumeFromCheckpoint) "cli default resume mismatch"
  ensure cfg.StrictLoad "cli default strict-load should be true"

let testCliRestrictAlias () =
  let cfg = Cli.parse [| "--restrict-load"; "false" |]
  ensure (cfg.StrictLoad = false) "--restrict-load alias parse failed"

let testStrictLoadRejectsFallback () =
  let datPath = Path.Combine(Path.GetTempPath(), "qwen3-fs-tests-strict.dat")
  createSingleLayerDat datPath 32L 64L
  let cfg =
    {
      Defaults.trainingConfig with
          SyntheticMode = false
          Device = "cpu"
          WeightPath = datPath
          InFeatures = 128L
          OutFeatures = 128L
          MaxLayers = 1
          StrictLoad = true
    }

  let mutable gotError = false
  try
    let _ = Nvfp4State.load cfg
    ()
  with
  | :? InvalidOperationException as ex ->
    gotError <- ex.Message.Contains("Strict load", StringComparison.Ordinal)
  ensure gotError "strict-load should reject dimension fallback"

let testNonStrictLoadAllowsFallback () =
  let datPath = Path.Combine(Path.GetTempPath(), "qwen3-fs-tests-nonstrict.dat")
  createSingleLayerDat datPath 32L 64L
  let cfg =
    {
      Defaults.trainingConfig with
          SyntheticMode = false
          Device = "cpu"
          WeightPath = datPath
          InFeatures = 128L
          OutFeatures = 128L
          MaxLayers = 1
          StrictLoad = false
    }

  let st = Nvfp4State.load cfg
  ensure (st.Layers.Length = 1) "nonstrict fallback layer count mismatch"
  ensure (st.InFeatures = 64L) "nonstrict fallback in_features mismatch"
  ensure (st.OutFeatures = 32L) "nonstrict fallback out_features mismatch"
  disposeState st

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
  disposeState st

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
  try
    withModel cfg st (fun model ->
      use x = torch.randn([| 1L; 64L |], dtype = torch.float16, device = "cpu")
      use y = Qwen3Model.forward model x (Some torch.float16)
      ensure (y.shape = [| 1L; 64L |]) "forward output shape mismatch")
  finally
    disposeState st

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
  try
    withModel cfg st (fun model -> Trainer.run cfg model)
    ensure (File.Exists(Path.Combine(ckptDir, "meta.json"))) "trainer did not create checkpoint meta"
  finally
    disposeState st

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
  try
    withModel cfg st (fun model ->
      use before = model.Layers.Head.MasterWeight.detach().to_type(torch.float32).cpu().clone()
      Trainer.run cfg model
      use after = model.Layers.Head.MasterWeight.detach().to_type(torch.float32).cpu().clone()
      use delta = (after - before).abs().sum()
      let changed = delta.item<float32>()
      ensure (changed > 0.0f) "optimizer did not update weights")
  finally
    disposeState st

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
  try
    withModel cfgBase st (fun model1 -> Trainer.run cfgBase model1)

    let layer0Path = Path.Combine(ckptDir, "layer_0000.pt")
    ensure (File.Exists(layer0Path)) "checkpoint layer file missing"
    use saved = torch.load(layer0Path)

    let cfgResume = { cfgBase with ResumeFromCheckpoint = true }
    withModel cfgResume st (fun model2 ->
      match Trainer.tryLoadCheckpoint cfgResume model2 with
      | None -> failwith "checkpoint not loaded"
      | Some state ->
        ensure (state.Epoch = 1) "checkpoint epoch mismatch"
        ensure (state.GlobalStep = 1) "checkpoint global step mismatch"

      use loaded = model2.Layers.Head.MasterWeight.detach()
      ensureTensorEquivalentWithSpecials loaded saved 1e-5f "checkpoint recover")
  finally
    disposeState st

let tests : (string * (unit -> unit)) list =
  [
    "cli defaults", testCliDefaults
    "cli restrict alias", testCliRestrictAlias
    "strict-load rejects fallback", testStrictLoadRejectsFallback
    "nonstrict fallback", testNonStrictLoadAllowsFallback
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
