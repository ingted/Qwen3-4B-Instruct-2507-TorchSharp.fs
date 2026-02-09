namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open TorchSharp
open TorchSharp.Q4.Extension

type Qwen3Nvfp4Model(session: Q4Session, layers: (string * Q4Linear) list) =
  member _.Session = session
  member _.Layers = layers

  member _.Forward(input: TorchSharp.torch.Tensor) : TorchSharp.torch.Tensor =
    let mutable x = input
    for (_, layer) in layers do
      x <- layer.Forward(x)
    x

  interface System.IDisposable with
    member _.Dispose() =
      for (_, layer) in layers do
        (layer :> System.IDisposable).Dispose()

module Qwen3Model =
  let create (cfg: TrainingConfig) (state: Nvfp4ModelState) : Qwen3Nvfp4Model =
    let runtimeTarget =
      if cfg.Device.StartsWith("cuda") then Q4RuntimeTarget.Cuda 0 else Q4RuntimeTarget.Cpu

    let sessionCfg =
      { Q4.pureNvfp4SessionConfig with RuntimeTarget = runtimeTarget }

    let diagnostics = Backend.diagnose Q4.pureNvfp4Schema sessionCfg
    printfn "[Q4] backend=%s path=%A native=%s" diagnostics.Backend diagnostics.ComputePath diagnostics.NativeLoadState

    let session = Session.create sessionCfg Q4.pureNvfp4Schema

    let layers =
      state.Layers
      |> List.map (fun l ->
        let linear = session.CreateLinear(l.Bundle)
        (l.Name, linear))

    new Qwen3Nvfp4Model(session, layers)
