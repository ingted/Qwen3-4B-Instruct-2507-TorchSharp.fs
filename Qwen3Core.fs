namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System
open TorchSharp
open Qwen3_4B_Instruct_2507_TorchSharp_fs.TrainingFunctional

module Qwen3Core =
  type CoreConfig =
    {
      NumAttentionHeads: int
      NumKeyValueHeads: int
      HeadDim: int
      RopeTheta: float
      RmsNormEps: float
      DType: TorchSharp.torch.ScalarType
    }

  type BlockNorms =
    {
      InputNorm: torch.Tensor
      PostAttnNorm: torch.Tensor
      QNorm: torch.Tensor
      KNorm: torch.Tensor
    }

  type BlockProjections =
    {
      QProj: torch.Tensor -> torch.Tensor
      KProj: torch.Tensor -> torch.Tensor
      VProj: torch.Tensor -> torch.Tensor
      OProj: torch.Tensor -> torch.Tensor
      GateProj: torch.Tensor -> torch.Tensor
      UpProj: torch.Tensor -> torch.Tensor
      DownProj: torch.Tensor -> torch.Tensor
    }

  type BlockRuntimeState =
    {
      Hidden: torch.Tensor
      InputNormed: torch.Tensor option
      Q: torch.Tensor option
      K: torch.Tensor option
      V: torch.Tensor option
      AttnOut: torch.Tensor option
      PostNormed: torch.Tensor option
      Gate: torch.Tensor option
      Up: torch.Tensor option
      MlpMixed: torch.Tensor option
      MlpOut: torch.Tensor option
    }

  let newBlockRuntimeState (hidden: torch.Tensor) =
    {
      Hidden = hidden
      InputNormed = None
      Q = None
      K = None
      V = None
      AttnOut = None
      PostNormed = None
      Gate = None
      Up = None
      MlpMixed = None
      MlpOut = None
    }

  let disposeOpt (t: torch.Tensor option) =
    t |> Option.iter (fun x -> x.Dispose())

  let requireTensor (name: string) (t: torch.Tensor option) =
    match t with
    | Some x -> x
    | None -> raise (InvalidOperationException(sprintf "Block graph state missing required tensor: %s" name))

  let rmsNorm (x: torch.Tensor) (eps: float) =
    let dtype = x.dtype
    use x32 = x.to_type(torch.float32)
    use variance = (x32 * x32).mean([| -1L |], true)
    let invStd = torch.rsqrt(variance + eps)
    let y32 = x32 * invStd
    if dtype = torch.float32 then y32 else y32.to_type(dtype)

  let rmsNormWeighted (x: torch.Tensor) (weight: torch.Tensor) (eps: float) =
    let y = rmsNorm x eps
    y * weight

  let rotateHalf (x: torch.Tensor) =
    let lastDim = x.shape.[x.shape.Length - 1]
    let half = lastDim / 2L
    use x1 = x.narrow(-1L, 0L, half)
    use x2 = x.narrow(-1L, half, half)
    torch.cat([| (-x2); x1 |], dim = -1L)

  let applyRoPE (x: torch.Tensor) (theta: float) (positionOffset: int64) =
    let seqLen = int x.shape.[2]
    let headDim = int x.shape.[3]
    let half = headDim / 2
    use i = torch.arange(0L, int64 half, dtype = torch.float32, device = x.device)
    use exponent = i / float32 half
    use invFreq = torch.pow(float32 theta, -exponent).unsqueeze(0L)
    use positions =
      torch.arange(positionOffset, positionOffset + int64 seqLen, dtype = torch.float32, device = x.device).unsqueeze(0L)
    use freqs = positions.unsqueeze(-1L) * invFreq
    use emb = torch.cat([| freqs; freqs |], dim = -1L)
    use cos = torch.cos(emb).to_type(x.dtype).unsqueeze(1L)
    use sin = torch.sin(emb).to_type(x.dtype).unsqueeze(1L)
    use rotated = rotateHalf x
    (x * cos + rotated * sin).contiguous()

  let expandKvHeads (numHeads: int) (numKvHeads: int) (kv: torch.Tensor) =
    let batchSize = kv.shape.[0]
    let seqLen = kv.shape.[2]
    let headDim = kv.shape.[3]
    let repeatFactor = numHeads / numKvHeads
    kv
      .unsqueeze(2L)
      .expand([| batchSize; int64 numKvHeads; int64 repeatFactor; seqLen; headDim |])
      .reshape([| batchSize; int64 numHeads; seqLen; headDim |])

  let attentionContextFromQkv
    (cfg: CoreConfig)
    (norms: BlockNorms)
    (projs: BlockProjections)
    (positionOffset: int64)
    (q: torch.Tensor)
    (k: torch.Tensor)
    (v: torch.Tensor)
    =
    let batchSize = q.shape.[0]
    let seqLen = q.shape.[1]

    use qh =
      q
        .reshape([| batchSize; seqLen; int64 cfg.NumAttentionHeads; int64 cfg.HeadDim |])
        .transpose(1L, 2L)
        .contiguous()

    use kh0 =
      k
        .reshape([| batchSize; seqLen; int64 cfg.NumKeyValueHeads; int64 cfg.HeadDim |])
        .transpose(1L, 2L)
        .contiguous()

    use vh0 =
      v
        .reshape([| batchSize; seqLen; int64 cfg.NumKeyValueHeads; int64 cfg.HeadDim |])
        .transpose(1L, 2L)
        .contiguous()

    use qhNorm = rmsNormWeighted (qh.transpose(1L, 2L)) norms.QNorm cfg.RmsNormEps
    use kh0Norm = rmsNormWeighted (kh0.transpose(1L, 2L)) norms.KNorm cfg.RmsNormEps
    use qhNormT = qhNorm.transpose(1L, 2L).contiguous()
    use kh0NormT = kh0Norm.transpose(1L, 2L).contiguous()
    use qhRope = applyRoPE qhNormT cfg.RopeTheta positionOffset
    use kh0Rope = applyRoPE kh0NormT cfg.RopeTheta positionOffset
    use kh = expandKvHeads cfg.NumAttentionHeads cfg.NumKeyValueHeads kh0Rope
    use vh = expandKvHeads cfg.NumAttentionHeads cfg.NumKeyValueHeads vh0
    let attnDtype = qhRope.dtype
    let khAttnTemp = if kh.dtype = attnDtype then None else Some (kh.to_type(attnDtype))
    let vhAttnTemp = if vh.dtype = attnDtype then None else Some (vh.to_type(attnDtype))
    let khAttn = match khAttnTemp with | Some t -> t | None -> kh
    let vhAttn = match vhAttnTemp with | Some t -> t | None -> vh

    use ctxHeads = torch.nn.functional.scaled_dot_product_attention(qhRope, khAttn, vhAttn, is_casual = true)
    let ctx =
      ctxHeads
        .transpose(1L, 2L)
        .contiguous()
        .reshape([| batchSize; seqLen; int64 (cfg.NumAttentionHeads * cfg.HeadDim) |])

    khAttnTemp |> Option.iter (fun t -> t.Dispose())
    vhAttnTemp |> Option.iter (fun t -> t.Dispose())

    ctx

  let buildBlockGraphNoCache
    (cfg: CoreConfig)
    (norms: BlockNorms)
    (projs: BlockProjections)
    (positionOffset: int64)
    : Stage
    =
    let inputNormStage : Op<BlockRuntimeState, BlockRuntimeState> =
      fun s ->
        disposeOpt s.InputNormed
        let xNorm = rmsNormWeighted s.Hidden norms.InputNorm cfg.RmsNormEps
        { s with InputNormed = Some xNorm }

    let qProjStage : Op<BlockRuntimeState, BlockRuntimeState> =
      fun s ->
        let xNorm = requireTensor "InputNormed" s.InputNormed
        disposeOpt s.Q
        let q = projs.QProj xNorm
        { s with Q = Some q }

    let kProjStage : Op<BlockRuntimeState, BlockRuntimeState> =
      fun s ->
        let xNorm = requireTensor "InputNormed" s.InputNormed
        disposeOpt s.K
        let k = projs.KProj xNorm
        { s with K = Some k }

    let vProjStage : Op<BlockRuntimeState, BlockRuntimeState> =
      fun s ->
        let xNorm = requireTensor "InputNormed" s.InputNormed
        disposeOpt s.V
        let v = projs.VProj xNorm
        { s with V = Some v }

    let attnMergeStage : Op<BlockRuntimeState, BlockRuntimeState> =
      fun s ->
        let q = requireTensor "Q" s.Q
        let k = requireTensor "K" s.K
        let v = requireTensor "V" s.V
        disposeOpt s.AttnOut
        let attnCtx = attentionContextFromQkv cfg norms projs positionOffset q k v
        disposeOpt s.Q
        disposeOpt s.K
        disposeOpt s.V
        disposeOpt s.InputNormed
        {
          s with
              InputNormed = None
              Q = None
              K = None
              V = None
              AttnOut = Some attnCtx
        }

    let oProjStage : Op<BlockRuntimeState, BlockRuntimeState> =
      fun s ->
        let attnCtx = requireTensor "AttnOut" s.AttnOut
        let attnOut = projs.OProj attnCtx
        disposeOpt s.AttnOut
        { s with AttnOut = Some attnOut }

    let attnResidualStage : Op<BlockRuntimeState, BlockRuntimeState> =
      fun s ->
        let attnOut = requireTensor "AttnOut" s.AttnOut
        let nextHidden = attnOut + s.Hidden
        disposeOpt s.AttnOut
        { s with Hidden = nextHidden; AttnOut = None }

    let postNormStage : Op<BlockRuntimeState, BlockRuntimeState> =
      fun s ->
        disposeOpt s.PostNormed
        let post = rmsNormWeighted s.Hidden norms.PostAttnNorm cfg.RmsNormEps
        { s with PostNormed = Some post }

    let gateProjStage : Op<BlockRuntimeState, BlockRuntimeState> =
      fun s ->
        let post = requireTensor "PostNormed" s.PostNormed
        disposeOpt s.Gate
        let gate = projs.GateProj post
        { s with Gate = Some gate }

    let upProjStage : Op<BlockRuntimeState, BlockRuntimeState> =
      fun s ->
        let post = requireTensor "PostNormed" s.PostNormed
        disposeOpt s.Up
        let up = projs.UpProj post
        { s with Up = Some up }

    let mlpMergeStage : Op<BlockRuntimeState, BlockRuntimeState> =
      fun s ->
        let gate = requireTensor "Gate" s.Gate
        let up = requireTensor "Up" s.Up
        disposeOpt s.MlpMixed
        let mixed = torch.nn.functional.silu(gate) * up
        disposeOpt s.Gate
        disposeOpt s.Up
        disposeOpt s.PostNormed
        {
          s with
              PostNormed = None
              Gate = None
              Up = None
              MlpMixed = Some mixed
        }

    let downProjStage : Op<BlockRuntimeState, BlockRuntimeState> =
      fun s ->
        let mixed = requireTensor "MlpMixed" s.MlpMixed
        disposeOpt s.MlpOut
        let mlpOut = projs.DownProj mixed
        disposeOpt s.MlpMixed
        { s with MlpMixed = None; MlpOut = Some mlpOut }

    let mlpResidualStage : Op<BlockRuntimeState, BlockRuntimeState> =
      fun s ->
        let mlpOut = requireTensor "MlpOut" s.MlpOut
        let nextHidden = mlpOut + s.Hidden
        disposeOpt s.MlpOut
        { s with Hidden = nextHidden; MlpOut = None }

    let blockStateGraph : Op<BlockRuntimeState, BlockRuntimeState> =
      chainOp
        [
          inputNormStage
          qProjStage
          kProjStage
          vProjStage
          attnMergeStage
          oProjStage
          attnResidualStage
          postNormStage
          gateProjStage
          upProjStage
          mlpMergeStage
          downProjStage
          mlpResidualStage
        ]

    stageM "block.graph" (fun hidden ->
      let finalState = newBlockRuntimeState hidden --> blockStateGraph
      disposeOpt finalState.InputNormed
      disposeOpt finalState.Q
      disposeOpt finalState.K
      disposeOpt finalState.V
      disposeOpt finalState.AttnOut
      disposeOpt finalState.PostNormed
      disposeOpt finalState.Gate
      disposeOpt finalState.Up
      disposeOpt finalState.MlpMixed
      disposeOpt finalState.MlpOut
      finalState.Hidden)

  let forwardBlockNoCache
    (cfg: CoreConfig)
    (norms: BlockNorms)
    (projs: BlockProjections)
    (hidden: torch.Tensor)
    (positionOffset: int64)
    =
    let blockGraph = buildBlockGraphNoCache cfg norms projs positionOffset
    runM blockGraph hidden
