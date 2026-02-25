namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System
open TorchSharp
open TorchSharp.Fun.DGX
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

  type BlockKvCache() =
    let mutable k: torch.Tensor option = None
    let mutable v: torch.Tensor option = None

    member _.K
      with get () = k
      and set value = k <- value

    member _.V
      with get () = v
      and set value = v <- value

    member this.Reset() =
      k |> Option.iter (fun t -> t.Dispose())
      v |> Option.iter (fun t -> t.Dispose())
      k <- None
      v <- None

    interface IDisposable with
      member this.Dispose() = this.Reset()

  type ModelKvCache(layerCount: int) =
    let layers = Array.init layerCount (fun _ -> new BlockKvCache())
    let mutable seqLen = 0L

    member _.Layers = layers
    member _.SeqLen
      with get () = seqLen
      and set value = seqLen <- value

    member this.Reset() =
      for layer in layers do
        layer.Reset()
      seqLen <- 0L

    interface IDisposable with
      member this.Dispose() = this.Reset()

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
    if numHeads <= 0 || numKvHeads <= 0 then
      invalidOp (sprintf "invalid attention head config: numHeads=%d numKvHeads=%d" numHeads numKvHeads)
    if numHeads % numKvHeads <> 0 then
      invalidOp (sprintf "GQA head mismatch: numHeads=%d must be divisible by numKvHeads=%d" numHeads numKvHeads)
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

  let attentionContextFromQkvWithCache
    (cfg: CoreConfig)
    (norms: BlockNorms)
    (projs: BlockProjections)
    (cache: BlockKvCache)
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
    let currentK = applyRoPE kh0NormT cfg.RopeTheta positionOffset
    let currentV = vh0.contiguous()

    match cache.K with
    | Some pastK ->
      let merged = torch.cat([| pastK; currentK |], dim = 2L).contiguous()
      pastK.Dispose()
      currentK.Dispose()
      cache.K <- Some merged
    | None ->
      cache.K <- Some currentK

    match cache.V with
    | Some pastV ->
      let merged = torch.cat([| pastV; currentV |], dim = 2L).contiguous()
      pastV.Dispose()
      currentV.Dispose()
      cache.V <- Some merged
    | None ->
      cache.V <- Some currentV

    let kAll =
      match cache.K with
      | Some t -> t
      | None -> invalidOp "kv-cache internal error: missing K after append"
    let vAll =
      match cache.V with
      | Some t -> t
      | None -> invalidOp "kv-cache internal error: missing V after append"

    use kh = expandKvHeads cfg.NumAttentionHeads cfg.NumKeyValueHeads kAll
    use vh = expandKvHeads cfg.NumAttentionHeads cfg.NumKeyValueHeads vAll
    let attnDtype = qhRope.dtype
    let khAttnTemp = if kh.dtype = attnDtype then None else Some(kh.to_type(attnDtype))
    let vhAttnTemp = if vh.dtype = attnDtype then None else Some(vh.to_type(attnDtype))
    let khAttn = match khAttnTemp with | Some t -> t | None -> kh
    let vhAttn = match vhAttnTemp with | Some t -> t | None -> vh
    let useCausal = positionOffset = 0L && seqLen > 1L

    use ctxHeads = torch.nn.functional.scaled_dot_product_attention(qhRope, khAttn, vhAttn, is_casual = useCausal)
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
    let inputNormStage =
      stageM "block_input_norm" (fun hidden ->
        rmsNormWeighted hidden norms.InputNorm cfg.RmsNormEps)

    let qStage = stage projs.QProj
    let kStage = stage projs.KProj
    let vStage = stage projs.VProj
    let qkvContextOp =
      parallel3 qStage kStage vStage
      |> merge3 (fun q k v ->
        use q = q
        use k = k
        use v = v
        attentionContextFromQkv cfg norms projs positionOffset q k v)

    let qkvContextStage =
      stageM "block_attn_qkv_context" qkvContextOp

    let oProjStage = stageM "block_attn_o_proj" projs.OProj
    let attnMain = inputNormStage ->> qkvContextStage ->> oProjStage
    let attnResidual = residualM "block_attn_residual" attnMain

    let postNormStage =
      stageM "block_mlp_post_norm" (fun hidden ->
        rmsNormWeighted hidden norms.PostAttnNorm cfg.RmsNormEps)

    let gateStage = stage projs.GateProj
    let upStage = stage projs.UpProj
    let gateUpMergeOp =
      parallel2 gateStage upStage
      |> merge2 (fun gate up ->
        use gate = gate
        use up = up
        torch.nn.functional.silu(gate) * up)
    let gateUpMergeStage = stageM "block_mlp_gate_up_merge" gateUpMergeOp

    let downProjStage = stageM "block_mlp_down_proj" projs.DownProj
    let mlpMain = postNormStage ->> gateUpMergeStage ->> downProjStage
    let mlpResidual = residualM "block_mlp_residual" mlpMain

    attnResidual ->> mlpResidual

  let buildBlockGraphWithCache
    (cfg: CoreConfig)
    (norms: BlockNorms)
    (projs: BlockProjections)
    (cache: BlockKvCache)
    (positionOffset: int64)
    : Stage
    =
    let inputNormStage =
      stageM "block_input_norm_cache" (fun hidden ->
        rmsNormWeighted hidden norms.InputNorm cfg.RmsNormEps)

    let qStage = stage projs.QProj
    let kStage = stage projs.KProj
    let vStage = stage projs.VProj
    let qkvContextOp =
      parallel3 qStage kStage vStage
      |> merge3 (fun q k v ->
        use q = q
        use k = k
        use v = v
        attentionContextFromQkvWithCache cfg norms projs cache positionOffset q k v)

    let qkvContextStage =
      stageM "block_attn_qkv_context_cache" qkvContextOp

    let oProjStage = stageM "block_attn_o_proj_cache" projs.OProj
    let attnMain = inputNormStage ->> qkvContextStage ->> oProjStage
    let attnResidual = residualM "block_attn_residual_cache" attnMain

    let postNormStage =
      stageM "block_mlp_post_norm_cache" (fun hidden ->
        rmsNormWeighted hidden norms.PostAttnNorm cfg.RmsNormEps)

    let gateStage = stage projs.GateProj
    let upStage = stage projs.UpProj
    let gateUpMergeOp =
      parallel2 gateStage upStage
      |> merge2 (fun gate up ->
        use gate = gate
        use up = up
        torch.nn.functional.silu(gate) * up)
    let gateUpMergeStage = stageM "block_mlp_gate_up_merge_cache" gateUpMergeOp

    let downProjStage = stageM "block_mlp_down_proj_cache" projs.DownProj
    let mlpMain = postNormStage ->> gateUpMergeStage ->> downProjStage
    let mlpResidual = residualM "block_mlp_residual_cache" mlpMain

    attnResidual ->> mlpResidual

  let forwardBlockNoCache
    (cfg: CoreConfig)
    (norms: BlockNorms)
    (projs: BlockProjections)
    (hidden: torch.Tensor)
    (positionOffset: int64)
    =
    let blockGraph = buildBlockGraphNoCache cfg norms projs positionOffset
    runM blockGraph hidden

  let forwardBlockWithCache
    (cfg: CoreConfig)
    (norms: BlockNorms)
    (projs: BlockProjections)
    (cache: BlockKvCache)
    (hidden: torch.Tensor)
    (positionOffset: int64)
    =
    let blockGraph = buildBlockGraphWithCache cfg norms projs cache positionOffset
    runM blockGraph hidden
