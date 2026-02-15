namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

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

  let attentionOutFromQkv
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
    use ctx =
      ctxHeads
        .transpose(1L, 2L)
        .contiguous()
        .reshape([| batchSize; seqLen; int64 (cfg.NumAttentionHeads * cfg.HeadDim) |])

    khAttnTemp |> Option.iter (fun t -> t.Dispose())
    vhAttnTemp |> Option.iter (fun t -> t.Dispose())

    projs.OProj ctx

  let buildBlockGraphNoCache
    (cfg: CoreConfig)
    (norms: BlockNorms)
    (projs: BlockProjections)
    (positionOffset: int64)
    : Stage
    =
    let inputNormOp =
      stageM "block.input_norm" (fun hidden -> rmsNormWeighted hidden norms.InputNorm cfg.RmsNormEps)

    let attnQkvMergeOp =
      stageM "block.attn.qkv_merge" (fun hiddenNorm ->
        use q = projs.QProj hiddenNorm
        use k = projs.KProj hiddenNorm
        use v = projs.VProj hiddenNorm
        attentionOutFromQkv cfg norms projs positionOffset q k v)

    let attnMain = chainM [ inputNormOp; attnQkvMergeOp ]
    let attnResidual = residualM "block.attn.residual" attnMain

    let postNormOp =
      stageM "block.post_attn_norm" (fun hidden -> rmsNormWeighted hidden norms.PostAttnNorm cfg.RmsNormEps)

    let gateUpMergeOp =
      stageM "block.mlp.gate_up_merge" (fun hiddenNorm ->
        use gate = projs.GateProj hiddenNorm
        use up = projs.UpProj hiddenNorm
        torch.nn.functional.silu(gate) * up)

    let downProjOp = stageM "block.mlp.down_proj" projs.DownProj
    let mlpMain = chainM [ postNormOp; gateUpMergeOp; downProjOp ]
    let mlpResidual = residualM "block.mlp.residual" mlpMain

    chainM [ attnResidual; mlpResidual ]

  let forwardBlockNoCache
    (cfg: CoreConfig)
    (norms: BlockNorms)
    (projs: BlockProjections)
    (hidden: torch.Tensor)
    (positionOffset: int64)
    =
    let blockGraph = buildBlockGraphNoCache cfg norms projs positionOffset
    runM blockGraph hidden
