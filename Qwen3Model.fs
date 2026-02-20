namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System
open System.Collections.Generic
open System.Runtime.CompilerServices
open TorchSharp
open TorchSharp.Modules
open TorchSharp.Q4.Extension
open Qwen3_4B_Instruct_2507_TorchSharp_fs.TrainingFunctional

type Qwen3TrainableLayer =
  {
    Name: string
    MasterWeight: Parameter
  }

type Qwen3TrainableBlock =
  {
    Name: string
    QProj: Parameter
    KProj: Parameter
    VProj: Parameter
    OProj: Parameter
    GateProj: Parameter
    UpProj: Parameter
    DownProj: Parameter
    InputNorm: Parameter
    PostAttnNorm: Parameter
    QNorm: Parameter
    KNorm: Parameter
    NumAttentionHeads: int
    NumKeyValueHeads: int
    HeadDim: int
  }

type Qwen3Nvfp4Model =
  {
    Session: Q4Session
    Layers: Qwen3TrainableLayer list
    Blocks: Qwen3TrainableBlock list
    ExtraParameters: Parameter list
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
    let all =
      seq {
        for l in model.Layers do
          yield l.MasterWeight
        for p in model.ExtraParameters do
          yield p
      }

    let seen = HashSet<int>()
    let unique = ResizeArray<Parameter>()
    for p in all do
      let key = RuntimeHelpers.GetHashCode(p)
      if seen.Add(key) then
        unique.Add(p)
    unique |> Seq.toList

  let forward
    (model: Qwen3Nvfp4Model)
    (input: TorchSharp.torch.Tensor)
    (outDtype: TorchSharp.torch.ScalarType option)
    : TorchSharp.torch.Tensor
    =
    let targetOutDtype = outDtype |> Option.defaultValue input.dtype

    let blockToStage (block: Qwen3TrainableBlock) =
      let cfg : Qwen3Core.CoreConfig =
        {
          NumAttentionHeads = block.NumAttentionHeads
          NumKeyValueHeads = block.NumKeyValueHeads
          HeadDim = block.HeadDim
          RopeTheta = 1e6
          RmsNormEps = 1e-6
          DType = targetOutDtype
        }

      let norms : Qwen3Core.BlockNorms =
        {
          InputNorm = block.InputNorm
          PostAttnNorm = block.PostAttnNorm
          QNorm = block.QNorm
          KNorm = block.KNorm
        }

      let projs : Qwen3Core.BlockProjections =
        {
          QProj = (fun x -> Nvfp4Training.linearSte x block.QProj targetOutDtype)
          KProj = (fun x -> Nvfp4Training.linearSte x block.KProj targetOutDtype)
          VProj = (fun x -> Nvfp4Training.linearSte x block.VProj targetOutDtype)
          OProj = (fun x -> Nvfp4Training.linearSte x block.OProj targetOutDtype)
          GateProj = (fun x -> Nvfp4Training.linearSte x block.GateProj targetOutDtype)
          UpProj = (fun x -> Nvfp4Training.linearSte x block.UpProj targetOutDtype)
          DownProj = (fun x -> Nvfp4Training.linearSte x block.DownProj targetOutDtype)
        }

      Qwen3Core.buildBlockGraphNoCache cfg norms projs 0L

    if model.Blocks.IsEmpty then
      let trainingGraph =
        model.Layers
        |> List.map (fun layer -> stageM (sprintf "layer.%s.linear_ste" layer.Name) (linearSte layer.MasterWeight targetOutDtype))
        |> chainM
      runM trainingGraph input
    else
      let trainingGraph = model.Blocks |> List.map blockToStage |> chainM

      let hidden0, squeezeBack =
        if input.shape.Length = 2 then
          input.unsqueeze(1L), true
        else
          input, false

      let hidden = runM trainingGraph hidden0

      if squeezeBack then
        let output = hidden.squeeze(1L).contiguous()
        hidden.Dispose()
        output
      else
        hidden

  let disposeSession (session: Q4Session) =
    match box session with
    | :? IDisposable as disposable -> disposable.Dispose()
    | _ -> ()

  let dispose (model: Qwen3Nvfp4Model) =
    for layer in model.Layers do
      layer.MasterWeight.Dispose()
    for p in model.ExtraParameters do
      p.Dispose()
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

    let blocks, extraParams =
      let createdBlocks = ResizeArray<Qwen3TrainableBlock>()
      let createdNorms = ResizeArray<Parameter>()

      let makeNorm (size: int64) (dtype: TorchSharp.torch.ScalarType) (deviceStr: string) =
        let t = torch.ones([| size |], dtype = dtype, device = deviceStr)
        let p = torch.nn.Parameter(t, true)
        createdNorms.Add(p)
        p

      for i, layer in layers |> List.indexed do
        let w = layer.MasterWeight
        if w.shape.Length = 2 && w.shape.[0] = w.shape.[1] then
          let hidden = w.shape.[1]
          let headDim = int w.shape.[0]
          let dtype = w.dtype
          let dev = w.device.ToString()
          let inputNorm = makeNorm hidden dtype dev
          let postNorm = makeNorm hidden dtype dev
          let qNorm = makeNorm (int64 headDim) dtype dev
          let kNorm = makeNorm (int64 headDim) dtype dev

          createdBlocks.Add(
            {
              Name = sprintf "train.block.%d" i
              QProj = layer.MasterWeight
              KProj = layer.MasterWeight
              VProj = layer.MasterWeight
              OProj = layer.MasterWeight
              GateProj = layer.MasterWeight
              UpProj = layer.MasterWeight
              DownProj = layer.MasterWeight
              InputNorm = inputNorm
              PostAttnNorm = postNorm
              QNorm = qNorm
              KNorm = kNorm
              NumAttentionHeads = 1
              NumKeyValueHeads = 1
              HeadDim = headDim
            }
          )

      createdBlocks |> Seq.toList, createdNorms |> Seq.toList

    {
      Session = session
      Layers = layers
      Blocks = blocks
      ExtraParameters = extraParams
      InFeatures = state.InFeatures
      OutFeatures = state.OutFeatures
    }
