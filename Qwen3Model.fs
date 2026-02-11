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

type Qwen3Nvfp4Model =
  {
    Session: Q4Session
    Layers: Qwen3TrainableLayer list
    InFeatures: int64
    OutFeatures: int64
  }

module Qwen3Model =
  let isFloatingDtype (dtype: TorchSharp.torch.ScalarType) =
    dtype = torch.float16
    || dtype = torch.float32
    || dtype = torch.float64
    || dtype = torch.bfloat16

  let materializeMasterWeight
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

  let parameters (model: Qwen3Nvfp4Model) =
    model.Layers |> List.map (fun l -> l.MasterWeight)

  let forward
    (model: Qwen3Nvfp4Model)
    (input: TorchSharp.torch.Tensor)
    (outDtype: TorchSharp.torch.ScalarType option)
    : TorchSharp.torch.Tensor
    =
    let targetOutDtype = outDtype |> Option.defaultValue input.dtype
    model.Layers
    |> List.fold (fun x layer -> Nvfp4Training.linearSte x layer.MasterWeight targetOutDtype) input

  let disposeSession (session: Q4Session) =
    match box session with
    | :? IDisposable as disposable -> disposable.Dispose()
    | _ -> ()

  let dispose (model: Qwen3Nvfp4Model) =
    for layer in model.Layers do
      layer.MasterWeight.Dispose()
    disposeSession model.Session

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

    {
      Session = session
      Layers = layers
      InFeatures = state.InFeatures
      OutFeatures = state.OutFeatures
    }
