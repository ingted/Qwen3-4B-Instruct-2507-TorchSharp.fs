namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System
open TorchSharp
open TorchSharp.Modules
open TorchSharp.Q4.Extension

type Qwen3TrainableLayer =
  {
    Name: string
    MasterWeight: Parameter
  }

type Qwen3Nvfp4Model(session: Q4Session, layers: Qwen3TrainableLayer list, inFeatures: int64, outFeatures: int64) =
  member _.Session = session
  member _.Layers = layers
  member _.InFeatures = inFeatures
  member _.OutFeatures = outFeatures
  member _.Parameters = layers |> List.map (fun l -> l.MasterWeight)

  member _.Forward(input: TorchSharp.torch.Tensor, ?outDtype: TorchSharp.torch.ScalarType) : TorchSharp.torch.Tensor =
    let targetOutDtype = defaultArg outDtype input.dtype
    let mutable x = input
    for layer in layers do
      x <- Nvfp4Training.linearSte x layer.MasterWeight targetOutDtype
    x

  interface IDisposable with
    member _.Dispose() =
      for layer in layers do
        layer.MasterWeight.Dispose()

module Qwen3Model =
  let private isFloatingDtype (dtype: TorchSharp.torch.ScalarType) =
    dtype = torch.float16
    || dtype = torch.float32
    || dtype = torch.float64
    || dtype = torch.bfloat16

  let private materializeMasterWeight
    (bundle: Q4TensorBundle)
    (device: string)
    (targetDtype: TorchSharp.torch.ScalarType)
    =
    let w = bundle.Weight
    let dense =
      if w.dtype = torch.uint8 then
        match bundle.Scale with
        | Some scale -> Nvfp4Training.dequantizePacked w scale targetDtype
        | None -> raise (InvalidOperationException("NVFP4 bundle requires scale for uint8 qdata."))
      elif isFloatingDtype w.dtype then
        if w.dtype = targetDtype then w.clone() else w.to_type(targetDtype)
      else
        raise (InvalidOperationException(sprintf "Unsupported weight dtype for training: %A" w.dtype))

    let onTarget =
      if dense.device.ToString() = device then dense else dense.``to``(device = device)
    onTarget.contiguous().clone()

  let create (cfg: TrainingConfig) (state: Nvfp4ModelState) : Qwen3Nvfp4Model =
    let runtimeTarget =
      if cfg.Device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase) then Q4RuntimeTarget.Cuda 0 else Q4RuntimeTarget.Cpu

    let sessionCfg =
      match runtimeTarget with
      | Q4RuntimeTarget.Cpu ->
        // CPU path cannot execute CUDA-only FP4 kernels; keep functional dequant fallback for tests/tooling.
        {
          Q4.pureNvfp4SessionConfig with
              RuntimeTarget = runtimeTarget
              BackendOverride = Some "dequant-matmul"
              ComputePath = Q4ComputePath.DequantMatmulOnly
        }
      | _ ->
        { Q4.pureNvfp4SessionConfig with RuntimeTarget = runtimeTarget }

    let diagnostics = Backend.diagnose Q4.pureNvfp4Schema sessionCfg
    printfn "[Q4] backend=%s path=%A native=%s" diagnostics.Backend diagnostics.ComputePath diagnostics.NativeLoadState

    let session = Session.create sessionCfg Q4.pureNvfp4Schema

    let layers =
      let masterDtype =
        if cfg.Device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase) then torch.float16 else torch.float32

      state.Layers
      |> List.map (fun layer ->
        let master = materializeMasterWeight layer.Bundle cfg.Device masterDtype
        let p = torch.nn.Parameter(master, true)
        {
          Name = layer.Name
          MasterWeight = p
        })

    new Qwen3Nvfp4Model(session, layers, state.InFeatures, state.OutFeatures)
