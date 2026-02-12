namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System
open System.Collections.Generic
open System.IO
open System.Text
open System.Text.Json
open TorchSharp
open TorchSharp.Q4.Extension
open Tokenizers.DotNet

type InferenceGenOptions =
  {
    MaxTokens: int
    Temperature: float32
    TopP: float32
    Seed: int option
  }

type ModelConfigLite =
  {
    HiddenSize: int64
    NumHiddenLayers: int
    NumAttentionHeads: int
    NumKeyValueHeads: int
    HeadDim: int
    VocabSize: int
    RopeTheta: float
    RmsNormEps: float
    BosTokenId: int option
    EosTokenId: int option
  }

type LayerWeights =
  {
    QProj: Q4Linear
    KProj: Q4Linear
    VProj: Q4Linear
    OProj: Q4Linear
    InputNorm: torch.Tensor
    PostAttnNorm: torch.Tensor
    QNorm: torch.Tensor
    KNorm: torch.Tensor
    GateProj: Q4Linear
    UpProj: Q4Linear
    DownProj: Q4Linear
  }

type InferenceSession =
  {
    Device: string
    DType: TorchSharp.torch.ScalarType
    Config: ModelConfigLite
    Tokenizer: Tokenizer
    Layers: LayerWeights array
    EmbedTokens: torch.Tensor
    FinalNorm: torch.Tensor
    LmHead: Q4Linear
  }

type KvPrefillMode =
  | TokenByToken
  | PromptByPrompt

module InferenceBridge =
  let private imEndTokenId = 151645
  let private endOfTextTokenId = 151643

  let private renderPrompt (prompt: string) =
    $"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

  let private defaultWeightForQuant (modelDir: string) (quantHint: string option) =
    match quantHint with
    | Some q when q.Equals("fp4", StringComparison.OrdinalIgnoreCase) ->
      Path.Combine(modelDir, "Qwen3-4B-Instruct-2507-nvfp4.dat")
    | Some q
      when q.Equals("nf4", StringComparison.OrdinalIgnoreCase)
           || q.Equals("4bit", StringComparison.OrdinalIgnoreCase)
           || q.Equals("int4", StringComparison.OrdinalIgnoreCase) ->
      Path.Combine(modelDir, "Qwen3-4B-Instruct-2507-4bit.nf4.dat")
    | _ -> Defaults.weightPath

  let private resolveWeightPath (modelDir: string) (weightOverride: string option) (quantHint: string option) =
    match weightOverride with
    | Some w when not (String.IsNullOrWhiteSpace w) ->
      if Path.IsPathRooted w then w else Path.Combine(modelDir, w)
    | _ -> defaultWeightForQuant modelDir quantHint

  let private requiredInt (root: JsonElement) (name: string) =
    match root.TryGetProperty(name) with
    | true, prop -> prop.GetInt32()
    | _ -> invalidOp (sprintf "config missing field: %s" name)

  let private requiredInt64 (root: JsonElement) (name: string) =
    int64 (requiredInt root name)

  let private requiredFloat (root: JsonElement) (name: string) =
    match root.TryGetProperty(name) with
    | true, prop -> prop.GetDouble()
    | _ -> invalidOp (sprintf "config missing field: %s" name)

  let private optionalInt (root: JsonElement) (name: string) =
    match root.TryGetProperty(name) with
    | true, prop when prop.ValueKind = JsonValueKind.Number -> Some(prop.GetInt32())
    | _ -> None

  let private loadConfigLite (configPath: string) =
    use doc = JsonDocument.Parse(File.ReadAllText(configPath))
    let root = doc.RootElement
    {
      HiddenSize = requiredInt64 root "hidden_size"
      NumHiddenLayers = requiredInt root "num_hidden_layers"
      NumAttentionHeads = requiredInt root "num_attention_heads"
      NumKeyValueHeads = requiredInt root "num_key_value_heads"
      HeadDim = requiredInt root "head_dim"
      VocabSize = requiredInt root "vocab_size"
      RopeTheta = requiredFloat root "rope_theta"
      RmsNormEps = requiredFloat root "rms_norm_eps"
      BosTokenId = optionalInt root "bos_token_id"
      EosTokenId = optionalInt root "eos_token_id"
    }

  let private readLeb128 (br: BinaryReader) : uint64 =
    let mutable result = 0UL
    let mutable shift = 0
    let mutable keepReading = true
    while keepReading do
      let b = br.ReadByte()
      result <- result ||| (uint64 (b &&& 0x7Fuy) <<< shift)
      keepReading <- (b &&& 0x80uy) <> 0uy
      shift <- shift + 7
    result

  let private elementSize (elemType: int) =
    match elemType with
    | 0
    | 1
    | 11
    | 100
    | 101 -> 1
    | 2
    | 5
    | 15 -> 2
    | 3
    | 6 -> 4
    | 4
    | 7 -> 8
    | _ -> invalidOp (sprintf "unsupported elemType=%d" elemType)

  let private checkedByteCount (shape: int64 array) (elemSize: int) =
    let mutable n = 1L
    for d in shape do
      n <- n * d
    n * int64 elemSize

  let private readFloatTensor
    (br: BinaryReader)
    (elemType: int)
    (shape: int64 array)
    (byteCount: int64)
    (device: string)
    =
    if byteCount > int64 Int32.MaxValue then
      invalidOp (sprintf "tensor too large for reader buffer: %d bytes" byteCount)

    let bytes = br.ReadBytes(int byteCount)
    if bytes.Length <> int byteCount then
      invalidOp "unexpected EOF while reading raw tensor payload"

    let numel = shape |> Array.fold (fun s d -> s * d) 1L |> int
    let tensorCpu =
      match elemType with
      | 5 ->
        let data = Array.zeroCreate<Half> numel
        for i in 0 .. numel - 1 do
          let lo = uint16 bytes.[i * 2]
          let hi = uint16 bytes.[i * 2 + 1]
          let bits = lo ||| (hi <<< 8)
          data.[i] <- BitConverter.UInt16BitsToHalf(bits)
        torch.tensor(data, dtype = torch.float16)
      | 3
      | 6 ->
        let data = Array.zeroCreate<float32> numel
        for i in 0 .. numel - 1 do
          data.[i] <- BitConverter.ToSingle(bytes, i * 4)
        torch.tensor(data, dtype = torch.float32)
      | _ -> invalidOp (sprintf "raw tensor elemType %d is not supported in inference bridge" elemType)

    let tensor = tensorCpu.reshape(shape).``to``(device = device)
    tensorCpu.Dispose()
    tensor

  let private loadRawTensorMap (weightPath: string) (device: string) (keys: Set<string>) =
    use fs = File.OpenRead(weightPath)
    use br = new BinaryReader(fs)
    let entryCount = int64 (readLeb128 br)
    let results = Dictionary<string, torch.Tensor>(StringComparer.Ordinal)

    for _ in 0L .. entryCount - 1L do
      let keyLen = int (readLeb128 br)
      let key = Text.Encoding.UTF8.GetString(br.ReadBytes(keyLen))
      let elemType = int (readLeb128 br)
      let ndim = int (readLeb128 br)
      let shape = Array.init ndim (fun _ -> int64 (readLeb128 br))
      let bytes = checkedByteCount shape (elementSize elemType)
      if keys.Contains key then
        let t = readFloatTensor br elemType shape bytes device
        if results.ContainsKey key then
          results.[key].Dispose()
        results.[key] <- t
      else
        br.BaseStream.Seek(bytes, SeekOrigin.Current) |> ignore

    let missing =
      keys
      |> Seq.filter (fun k -> not (results.ContainsKey k))
      |> Seq.toList
    if not missing.IsEmpty then
      let msg = String.Join(", ", missing)
      invalidOp (sprintf "missing raw tensors in dat: %s" msg)

    results |> Seq.map (fun kv -> kv.Key, kv.Value) |> Map.ofSeq

  let private parseLayerIndex (name: string) =
    let marker = "model.layers."
    if name.StartsWith(marker, StringComparison.Ordinal) then
      let rest = name.Substring(marker.Length)
      let dot = rest.IndexOf('.')
      if dot > 0 then
        match Int32.TryParse(rest.Substring(0, dot)) with
        | true, idx -> Some idx
        | _ -> None
      else
        None
    else
      None

  let private loadByDims
    (baseCfg: TrainingConfig)
    (inFeatures: int64)
    (outFeatures: int64)
    (maxLayers: int)
    =
    let cfg =
      {
        baseCfg with
            InFeatures = inFeatures
            OutFeatures = outFeatures
            MaxLayers = maxLayers
            StrictLoad = true
      }
    Nvfp4State.load cfg

  let private buildLayerLinearMap
    (q4cfg: Q4SessionConfig)
    (suffix: string)
    (state: Nvfp4ModelState)
    =
    state.Layers
    |> List.choose (fun layer ->
      if layer.Name.EndsWith(suffix, StringComparison.Ordinal) then
        match parseLayerIndex layer.Name with
        | Some idx ->
          let linear = new Q4Linear(q4cfg, Q4.pureNvfp4Schema, layer.Bundle)
          Some(idx, linear)
        | None -> None
      else
        None)
    |> Map.ofList

  let private mapGet (name: string) (idx: int) (m: Map<int, Q4Linear>) =
    match m.TryFind idx with
    | Some v -> v
    | None -> invalidOp (sprintf "missing %s for layer %d" name idx)

  let private rawGet (key: string) (m: Map<string, torch.Tensor>) =
    match m.TryFind key with
    | Some v -> v
    | None -> invalidOp (sprintf "missing raw tensor key: %s" key)

  let private linearQ4 (input: torch.Tensor) (proj: Q4Linear) (outDtype: TorchSharp.torch.ScalarType) =
    proj.Forward(input, outDtype = outDtype)

  let private rmsNorm (x: torch.Tensor) (eps: float) =
    use xF = x.to_type(torch.float32)
    use sq = xF * xF
    use mean = sq.mean([| -1L |], true)
    use denom = torch.sqrt(mean + eps)
    (xF / denom).to_type(x.dtype)

  let private rmsNormWeighted (x: torch.Tensor) (weight: torch.Tensor) (eps: float) =
    use normalized = rmsNorm x eps
    normalized * weight

  let private rotateHalf (x: torch.Tensor) =
    let lastDim = int64 (x.shape.Length - 1)
    let headDim = x.shape.[int lastDim]
    let halfDim = headDim / 2L
    use x1 = x.slice(lastDim, 0L, halfDim, 1L)
    use x2 = x.slice(lastDim, halfDim, headDim, 1L)
    torch.cat([| -x2; x1 |], dim = lastDim)

  let private applyRoPE (x: torch.Tensor) (theta: float) (positionOffset: int64) =
    // x: [batch, heads, seq, head_dim]
    let headDim = int x.shape.[3]
    if headDim % 2 <> 0 then
      invalidOp (sprintf "head_dim must be even for RoPE, got %d" headDim)

    let seqLen = int x.shape.[2]
    let halfDim = headDim / 2
    let invFreqData =
      [|
        for i in 0 .. halfDim - 1 do
          let exponent = float (2 * i) / float headDim
          yield float32 (1.0 / (Math.Pow(theta, exponent)))
      |]

    use invFreq = torch.tensor(invFreqData, dtype = torch.float32, device = x.device)
    use positions =
      torch.arange(positionOffset, positionOffset + int64 seqLen, dtype = torch.float32, device = x.device).unsqueeze(0L)
    use freqs = positions.unsqueeze(-1L) * invFreq
    use emb = torch.cat([| freqs; freqs |], dim = -1L)
    use cos = torch.cos(emb).to_type(x.dtype).unsqueeze(1L)
    use sin = torch.sin(emb).to_type(x.dtype).unsqueeze(1L)
    use rotated = rotateHalf x
    (x * cos + rotated * sin).contiguous()

  let private expandKvHeads (numHeads: int) (numKvHeads: int) (kv: torch.Tensor) =
    let batchSize = kv.shape.[0]
    let seqLen = kv.shape.[2]
    let headDim = kv.shape.[3]
    let repeatFactor = numHeads / numKvHeads
    kv
      .unsqueeze(2L)
      .expand([| batchSize; int64 numKvHeads; int64 repeatFactor; seqLen; headDim |])
      .reshape([| batchSize; int64 numHeads; seqLen; headDim |])

  let private buildTokenEmbeddings (session: InferenceSession) (tokenIds: int array) =
    use tokenTensor =
      torch.tensor(tokenIds, dtype = torch.int64, device = session.Device)
    session.EmbedTokens.index_select(0L, tokenTensor).unsqueeze(0L).contiguous()

  let private forwardLayer (session: InferenceSession) (layer: LayerWeights) (hidden: torch.Tensor) =
    let numHeads = session.Config.NumAttentionHeads
    let numKvHeads = session.Config.NumKeyValueHeads
    let headDim = session.Config.HeadDim
    let batchSize = hidden.shape.[0]
    let seqLen = hidden.shape.[1]

    use normed0 = rmsNormWeighted hidden layer.InputNorm session.Config.RmsNormEps
    use q = linearQ4 normed0 layer.QProj session.DType
    use k = linearQ4 normed0 layer.KProj session.DType
    use v = linearQ4 normed0 layer.VProj session.DType

    use qh =
      q
        .reshape([| batchSize; seqLen; int64 numHeads; int64 headDim |])
        .transpose(1L, 2L)
        .contiguous()

    use kh0 =
      k
        .reshape([| batchSize; seqLen; int64 numKvHeads; int64 headDim |])
        .transpose(1L, 2L)
        .contiguous()

    use vh0 =
      v
        .reshape([| batchSize; seqLen; int64 numKvHeads; int64 headDim |])
        .transpose(1L, 2L)
        .contiguous()

    use qhNorm = rmsNormWeighted (qh.transpose(1L, 2L)) layer.QNorm session.Config.RmsNormEps
    use kh0Norm = rmsNormWeighted (kh0.transpose(1L, 2L)) layer.KNorm session.Config.RmsNormEps
    use qhNormT = qhNorm.transpose(1L, 2L).contiguous()
    use kh0NormT = kh0Norm.transpose(1L, 2L).contiguous()
    use qhRope = applyRoPE qhNormT session.Config.RopeTheta 0L
    use kh0Rope = applyRoPE kh0NormT session.Config.RopeTheta 0L
    use kh = expandKvHeads numHeads numKvHeads kh0Rope
    use vh = expandKvHeads numHeads numKvHeads vh0

    use ctxHeads = torch.nn.functional.scaled_dot_product_attention(qhRope, kh, vh, is_casual = true)
    use ctx =
      ctxHeads
        .transpose(1L, 2L)
        .contiguous()
        .reshape([| batchSize; seqLen; int64 (numHeads * headDim) |])

    use attnOut = linearQ4 ctx layer.OProj session.DType
    let resid1 = (hidden + attnOut).contiguous()

    use normed1 = rmsNormWeighted resid1 layer.PostAttnNorm session.Config.RmsNormEps
    use gate = linearQ4 normed1 layer.GateProj session.DType
    use up = linearQ4 normed1 layer.UpProj session.DType
    use act = torch.nn.functional.silu(gate) * up
    use down = linearQ4 act layer.DownProj session.DType
    let resid2 = (resid1 + down).contiguous()
    resid1.Dispose()
    resid2

  let private forwardModel (session: InferenceSession) (tokenIds: int array) =
    let mutable hidden = buildTokenEmbeddings session tokenIds
    for layer in session.Layers do
      let next = forwardLayer session layer hidden
      hidden.Dispose()
      hidden <- next
    hidden

  type private LayerKvCache =
    {
      mutable K: torch.Tensor option
      mutable V: torch.Tensor option
    }

  type private InferenceKvCache(layerCount: int) =
    let layers = Array.init layerCount (fun _ -> { K = None; V = None })
    let mutable seqLen = 0L

    member _.Layers = layers
    member _.SeqLen
      with get () = seqLen
      and set (v: int64) = seqLen <- v

    interface IDisposable with
      member _.Dispose() =
        for layer in layers do
          layer.K |> Option.iter (fun t -> t.Dispose())
          layer.V |> Option.iter (fun t -> t.Dispose())
          layer.K <- None
          layer.V <- None

  let private forwardLayerWithCache
    (session: InferenceSession)
    (layer: LayerWeights)
    (cache: LayerKvCache)
    (hidden: torch.Tensor)
    (positionOffset: int64)
    =
    let numHeads = session.Config.NumAttentionHeads
    let numKvHeads = session.Config.NumKeyValueHeads
    let headDim = session.Config.HeadDim
    let batchSize = hidden.shape.[0]
    let seqLen = hidden.shape.[1]

    use normed0 = rmsNormWeighted hidden layer.InputNorm session.Config.RmsNormEps
    use q = linearQ4 normed0 layer.QProj session.DType
    use k = linearQ4 normed0 layer.KProj session.DType
    use v = linearQ4 normed0 layer.VProj session.DType

    use qh =
      q
        .reshape([| batchSize; seqLen; int64 numHeads; int64 headDim |])
        .transpose(1L, 2L)
        .contiguous()

    use kh0 =
      k
        .reshape([| batchSize; seqLen; int64 numKvHeads; int64 headDim |])
        .transpose(1L, 2L)
        .contiguous()

    use vh0 =
      v
        .reshape([| batchSize; seqLen; int64 numKvHeads; int64 headDim |])
        .transpose(1L, 2L)
        .contiguous()

    use qhNorm = rmsNormWeighted (qh.transpose(1L, 2L)) layer.QNorm session.Config.RmsNormEps
    use kh0Norm = rmsNormWeighted (kh0.transpose(1L, 2L)) layer.KNorm session.Config.RmsNormEps
    use qhNormT = qhNorm.transpose(1L, 2L).contiguous()
    use kh0NormT = kh0Norm.transpose(1L, 2L).contiguous()
    use qhRope = applyRoPE qhNormT session.Config.RopeTheta positionOffset
    let currentK = applyRoPE kh0NormT session.Config.RopeTheta positionOffset
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

    use kh = expandKvHeads numHeads numKvHeads kAll
    use vh = expandKvHeads numHeads numKvHeads vAll

    // For incremental decode (cache already has past), query has no future tokens in this call.
    let useCausal = positionOffset = 0L && seqLen > 1L
    use ctxHeads = torch.nn.functional.scaled_dot_product_attention(qhRope, kh, vh, is_casual = useCausal)
    use ctx =
      ctxHeads
        .transpose(1L, 2L)
        .contiguous()
        .reshape([| batchSize; seqLen; int64 (numHeads * headDim) |])

    use attnOut = linearQ4 ctx layer.OProj session.DType
    let resid1 = (hidden + attnOut).contiguous()

    use normed1 = rmsNormWeighted resid1 layer.PostAttnNorm session.Config.RmsNormEps
    use gate = linearQ4 normed1 layer.GateProj session.DType
    use up = linearQ4 normed1 layer.UpProj session.DType
    use act = torch.nn.functional.silu(gate) * up
    use down = linearQ4 act layer.DownProj session.DType
    let resid2 = (resid1 + down).contiguous()
    resid1.Dispose()
    resid2

  let private forwardModelWithCache (session: InferenceSession) (cache: InferenceKvCache) (tokenIds: int array) =
    let mutable hidden = buildTokenEmbeddings session tokenIds
    let positionOffset = cache.SeqLen
    for i in 0 .. session.Layers.Length - 1 do
      let next = forwardLayerWithCache session session.Layers.[i] cache.Layers.[i] hidden positionOffset
      hidden.Dispose()
      hidden <- next
    cache.SeqLen <- cache.SeqLen + int64 tokenIds.Length
    hidden

  let private sampleFromLogits (logits: torch.Tensor) (temperature: float32) (topP: float32) =
    if temperature <= 0.0f then
      use next = logits.argmax(dim = -1L)
      next.to_type(torch.int32).item<int>()
    else
      use scaled = if temperature <> 1.0f then logits / temperature else logits
      use probs0 = torch.nn.functional.softmax(scaled, dim = -1L)
      use probs = torch.nan_to_num(probs0, nan = 0.0, posinf = 0.0, neginf = 0.0)

      if topP > 0.0f && topP < 1.0f then
        let struct (sorted, indices) = torch.sort(probs, dim = -1L, descending = true)
        use sorted = sorted
        use indices = indices
        use cumsum = torch.cumsum(sorted, dim = -1L)
        use mask = torch.gt(cumsum, topP)
        use sortedMasked = torch.where(mask, torch.zeros_like(sorted), sorted)
        use sums = torch.sum(sortedMasked, dim = -1L, keepdim = true).clamp_min(1e-12)
        use sortedNorm = sortedMasked / sums
        use next = torch.multinomial(sortedNorm, 1L)
        use token = torch.gather(indices, -1L, next)
        token.to_type(torch.int32).item<int>()
      else
        use sums = torch.sum(probs, dim = -1L, keepdim = true).clamp_min(1e-12)
        use probsNorm = probs / sums
        use next = torch.multinomial(probsNorm, 1L)
        next.to_type(torch.int32).item<int>()

  let private selectNextTokenId
    (session: InferenceSession)
    (hidden: torch.Tensor)
    (temperature: float32)
    (topP: float32)
    =
    use last = hidden.narrow(1L, hidden.shape.[1] - 1L, 1L)
    use lastNorm = rmsNormWeighted last session.FinalNorm session.Config.RmsNormEps
    use logits0 = session.LmHead.Forward(lastNorm, outDtype = session.DType)
    use logits = if logits0.dtype = torch.float32 then logits0 else logits0.to_type(torch.float32)
    sampleFromLogits logits temperature topP

  let private decodeTokens (tokenizer: Tokenizer) (tokenIds: int list) =
    if tokenIds.IsEmpty then
      ""
    else
      tokenizer.Decode(tokenIds |> List.map uint32 |> List.toArray)

  let init
    (modelDir: string)
    (weightOverride: string option)
    (quantHint: string option)
    (device: string)
    (dtype: TorchSharp.torch.ScalarType)
    =
    torch.InitializeDeviceType(DeviceType.CUDA)
    torch.set_default_dtype(dtype)

    let resolvedModelDir =
      if String.IsNullOrWhiteSpace modelDir then Defaults.modelDir else modelDir.Trim()
    let configPath = Path.Combine(resolvedModelDir, "config.json")
    let tokenizerPath = Path.Combine(resolvedModelDir, "tokenizer.json")
    let weightPath = resolveWeightPath resolvedModelDir weightOverride quantHint
    if not (File.Exists weightPath) then
      invalidOp (sprintf "weight file not found: %s" weightPath)
    if not (File.Exists tokenizerPath) then
      invalidOp (sprintf "tokenizer file not found: %s" tokenizerPath)
    if not (File.Exists configPath) then
      invalidOp (sprintf "config file not found: %s" configPath)

    let cfgLite = loadConfigLite configPath
    let tokenizer = new Tokenizer(tokenizerPath)

    let baseCfg =
      {
        Defaults.trainingConfig with
            ModelDir = resolvedModelDir
            ConfigPath = configPath
            TokenizerPath = tokenizerPath
            WeightPath = weightPath
            Device = device
            SyntheticMode = false
            StrictLoad = true
      }

    let runtimeTarget =
      if device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase) then Q4RuntimeTarget.Cuda 0 else Q4RuntimeTarget.Cpu

    let backendOverrideEnv = Environment.GetEnvironmentVariable("QWEN3_FS_Q4_BACKEND")
    let backendOverride =
      if String.IsNullOrWhiteSpace backendOverrideEnv then
        None
      else
        Some(backendOverrideEnv.Trim().ToLowerInvariant())

    let computePath, backendName =
      match runtimeTarget, backendOverride with
      | Q4RuntimeTarget.Cpu, _ -> Q4ComputePath.DequantMatmulOnly, "dequant-matmul"
      | _, Some "dequant-matmul" -> Q4ComputePath.DequantMatmulOnly, "dequant-matmul"
      | _, Some "nvfp4-kernel" -> Q4ComputePath.KernelOnly, "nvfp4-kernel"
      | _, _ -> Q4ComputePath.KernelOnly, "nvfp4-kernel"

    printfn "[InferInit] q4-backend=%s compute-path=%A" backendName computePath

    let q4cfg =
      {
        Q4.pureNvfp4SessionConfig with
            RuntimeTarget = runtimeTarget
            ComputePath = computePath
            BackendOverride = Some backendName
      }

    let rawKeys =
      seq {
        yield "model.embed_tokens.weight"
        yield "model.norm.weight"
        for i in 0 .. cfgLite.NumHiddenLayers - 1 do
          yield sprintf "model.layers.%d.input_layernorm.weight" i
          yield sprintf "model.layers.%d.post_attention_layernorm.weight" i
          yield sprintf "model.layers.%d.self_attn.q_norm.weight" i
          yield sprintf "model.layers.%d.self_attn.k_norm.weight" i
      }
      |> Set.ofSeq

    let rawMap = loadRawTensorMap weightPath device rawKeys

    let qMap =
      loadByDims baseCfg cfgLite.HiddenSize (int64 (cfgLite.NumAttentionHeads * cfgLite.HeadDim)) cfgLite.NumHiddenLayers
      |> buildLayerLinearMap q4cfg ".self_attn.q_proj"

    let kvCount = cfgLite.NumHiddenLayers * 2
    let kvState = loadByDims baseCfg cfgLite.HiddenSize (int64 (cfgLite.NumKeyValueHeads * cfgLite.HeadDim)) kvCount
    let kMap = buildLayerLinearMap q4cfg ".self_attn.k_proj" kvState
    let vMap =
      loadByDims baseCfg cfgLite.HiddenSize (int64 (cfgLite.NumKeyValueHeads * cfgLite.HeadDim)) kvCount
      |> buildLayerLinearMap q4cfg ".self_attn.v_proj"

    let oMap =
      loadByDims baseCfg (int64 (cfgLite.NumAttentionHeads * cfgLite.HeadDim)) cfgLite.HiddenSize cfgLite.NumHiddenLayers
      |> buildLayerLinearMap q4cfg ".self_attn.o_proj"

    let guCount = cfgLite.NumHiddenLayers * 2
    let gateMap =
      loadByDims baseCfg cfgLite.HiddenSize 9728L guCount
      |> buildLayerLinearMap q4cfg ".mlp.gate_proj"
    let upMap =
      loadByDims baseCfg cfgLite.HiddenSize 9728L guCount
      |> buildLayerLinearMap q4cfg ".mlp.up_proj"

    let downMap =
      loadByDims baseCfg 9728L cfgLite.HiddenSize cfgLite.NumHiddenLayers
      |> buildLayerLinearMap q4cfg ".mlp.down_proj"

    let lmState = loadByDims baseCfg cfgLite.HiddenSize (int64 cfgLite.VocabSize) 1
    let lmHead =
      match lmState.Layers |> List.tryFind (fun l -> l.Name = "lm_head") with
      | Some l -> new Q4Linear(q4cfg, Q4.pureNvfp4Schema, l.Bundle)
      | None -> invalidOp "missing lm_head"

    let layers =
      [|
        for i in 0 .. cfgLite.NumHiddenLayers - 1 do
          {
            QProj = mapGet "q_proj" i qMap
            KProj = mapGet "k_proj" i kMap
            VProj = mapGet "v_proj" i vMap
            OProj = mapGet "o_proj" i oMap
            InputNorm = rawGet (sprintf "model.layers.%d.input_layernorm.weight" i) rawMap
            PostAttnNorm = rawGet (sprintf "model.layers.%d.post_attention_layernorm.weight" i) rawMap
            QNorm = rawGet (sprintf "model.layers.%d.self_attn.q_norm.weight" i) rawMap
            KNorm = rawGet (sprintf "model.layers.%d.self_attn.k_norm.weight" i) rawMap
            GateProj = mapGet "gate_proj" i gateMap
            UpProj = mapGet "up_proj" i upMap
            DownProj = mapGet "down_proj" i downMap
          }
      |]

    let embedTokens = rawGet "model.embed_tokens.weight" rawMap
    let finalNorm = rawGet "model.norm.weight" rawMap

    printfn
      "[InferInit] layers=%d hidden=%d heads=%d kvHeads=%d headDim=%d vocab=%d"
      layers.Length
      cfgLite.HiddenSize
      cfgLite.NumAttentionHeads
      cfgLite.NumKeyValueHeads
      cfgLite.HeadDim
      cfgLite.VocabSize

    {
      Device = device
      DType = dtype
      Config = cfgLite
      Tokenizer = tokenizer
      Layers = layers
      EmbedTokens = embedTokens
      FinalNorm = finalNorm
      LmHead = lmHead
    }

  let generateFromRenderedPromptWithStopTokens
    (session: InferenceSession)
    (renderedPrompt: string)
    (opt: InferenceGenOptions)
    (stopTokens: int list)
    =
    use _noGrad = torch.no_grad()

    let encoded =
      session.Tokenizer.Encode(renderedPrompt)
      |> Seq.map int
      |> Seq.toList

    let inputIds = encoded

    if inputIds.IsEmpty then
      invalidOp "prompt encoded to empty token sequence"

    let running = ResizeArray<int>(inputIds.Length + max 1 opt.MaxTokens)
    for id in inputIds do
      running.Add(id)

    let generated = ResizeArray<int>(max 1 opt.MaxTokens)
    let mutable step = 0
    let targetSteps = max 1 opt.MaxTokens
    let mutable stop = false
    let stopSet = stopTokens |> Set.ofList

    match opt.Seed with
    | Some s when s >= 0 -> torch.manual_seed(int64 s) |> ignore
    | _ -> ()

    while step < targetSteps && not stop do
      use hidden = forwardModel session (running.ToArray())
      let nextId = selectNextTokenId session hidden opt.Temperature opt.TopP
      if stopSet.Contains(nextId) then
        stop <- true
      else
        generated.Add(nextId)
        running.Add(nextId)
        step <- step + 1

    match Environment.GetEnvironmentVariable("QWEN3_FS_DEBUG_TOKENS") with
    | "1" ->
      printfn "[InferDebug] generated token ids: %A" (generated |> Seq.toList)
    | _ -> ()

    decodeTokens session.Tokenizer (generated |> Seq.toList)

  let generateFromRenderedPromptWithStopTokensKvCache
    (session: InferenceSession)
    (renderedPrompt: string)
    (opt: InferenceGenOptions)
    (stopTokens: int list)
    (prefillMode: KvPrefillMode)
    =
    use _noGrad = torch.no_grad()

    let inputIds =
      session.Tokenizer.Encode(renderedPrompt)
      |> Seq.map int
      |> Seq.toArray

    if inputIds.Length = 0 then
      invalidOp "prompt encoded to empty token sequence"

    let generated = ResizeArray<int>(max 1 opt.MaxTokens)
    let targetSteps = max 1 opt.MaxTokens
    let stopSet = stopTokens |> Set.ofList
    let mutable step = 0
    let mutable stop = false

    match opt.Seed with
    | Some s when s >= 0 -> torch.manual_seed(int64 s) |> ignore
    | _ -> ()

    use cache = new InferenceKvCache(session.Layers.Length)

    let mutable nextId =
      use prefillHidden =
        match prefillMode with
        | PromptByPrompt ->
          forwardModelWithCache session cache inputIds
        | TokenByToken ->
          let mutable lastHidden: torch.Tensor option = None
          for tokenId in inputIds do
            lastHidden |> Option.iter (fun t -> t.Dispose())
            lastHidden <- Some (forwardModelWithCache session cache [| tokenId |])
          match lastHidden with
          | Some h -> h
          | None -> invalidOp "token-by-token prefill produced no hidden state"
      selectNextTokenId session prefillHidden opt.Temperature opt.TopP

    while step < targetSteps && not stop do
      if stopSet.Contains(nextId) then
        stop <- true
      else
        generated.Add(nextId)
        step <- step + 1
        if step < targetSteps then
          use hidden = forwardModelWithCache session cache [| nextId |]
          nextId <- selectNextTokenId session hidden opt.Temperature opt.TopP

    match Environment.GetEnvironmentVariable("QWEN3_FS_DEBUG_TOKENS") with
    | "1" ->
      printfn "[InferDebug] generated token ids (kvc): %A" (generated |> Seq.toList)
    | _ -> ()

    decodeTokens session.Tokenizer (generated |> Seq.toList)

  let generateFromRenderedPromptWithKvCache
    (session: InferenceSession)
    (renderedPrompt: string)
    (opt: InferenceGenOptions)
    (prefillMode: KvPrefillMode)
    =
    let stopTokens =
      [ imEndTokenId; endOfTextTokenId ]
      @ ([ session.Config.EosTokenId ] |> List.choose id)
      |> List.distinct
    generateFromRenderedPromptWithStopTokensKvCache session renderedPrompt opt stopTokens prefillMode

  let generateFromRenderedPrompt (session: InferenceSession) (renderedPrompt: string) (opt: InferenceGenOptions) =
    let stopTokens =
      [ imEndTokenId; endOfTextTokenId ]
      @ ([ session.Config.EosTokenId ] |> List.choose id)
      |> List.distinct
    generateFromRenderedPromptWithStopTokens session renderedPrompt opt stopTokens

  let generate (session: InferenceSession) (prompt: string) (opt: InferenceGenOptions) =
    let renderedPrompt = renderPrompt prompt
    generateFromRenderedPrompt session renderedPrompt opt

  let checkLogits (session: InferenceSession) (prompt: string) =
    use _noGrad = torch.no_grad()
    let renderedPrompt = renderPrompt prompt
    let inputIds =
      session.Tokenizer.Encode(renderedPrompt)
      |> Seq.map int
      |> Seq.toList

    if inputIds.IsEmpty then
      invalidOp "prompt encoded to empty token sequence"

    use hidden = forwardModel session (inputIds |> List.toArray)
    use last = hidden.narrow(1L, hidden.shape.[1] - 1L, 1L)
    use lastNorm = rmsNormWeighted last session.FinalNorm session.Config.RmsNormEps
    use logits0 = session.LmHead.Forward(lastNorm, outDtype = session.DType)
    use logits = if logits0.dtype = torch.float32 then logits0 else logits0.to_type(torch.float32)
    let hasNan = torch.isnan(logits).any().ToBoolean()
    let hasInf = torch.isinf(logits).any().ToBoolean()
    hasNan, hasInf

  let dispose (session: InferenceSession) =
    for layer in session.Layers do
      (layer.QProj :> IDisposable).Dispose()
      (layer.KProj :> IDisposable).Dispose()
      (layer.VProj :> IDisposable).Dispose()
      (layer.OProj :> IDisposable).Dispose()
      layer.InputNorm.Dispose()
      layer.PostAttnNorm.Dispose()
      layer.QNorm.Dispose()
      layer.KNorm.Dispose()
      (layer.GateProj :> IDisposable).Dispose()
      (layer.UpProj :> IDisposable).Dispose()
      (layer.DownProj :> IDisposable).Dispose()

    session.EmbedTokens.Dispose()
    session.FinalNorm.Dispose()
    (session.LmHead :> IDisposable).Dispose()
    session.Tokenizer.Dispose()
