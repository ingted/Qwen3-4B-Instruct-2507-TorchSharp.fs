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
    EosTokenId: int option
  }

type LayerWeights =
  {
    QProj: torch.Tensor
    KProj: torch.Tensor
    VProj: torch.Tensor
    OProj: torch.Tensor
    InputNorm: torch.Tensor
    PostAttnNorm: torch.Tensor
    QNorm: torch.Tensor
    KNorm: torch.Tensor
    GateProj: torch.Tensor
    UpProj: torch.Tensor
    DownProj: torch.Tensor
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
    LmHead: torch.Tensor
  }

module InferenceBridge =
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

  let private materializeWeight (device: string) (dtype: TorchSharp.torch.ScalarType) (bundle: Q4TensorBundle) =
    let dense = Qwen3Model.materializeMasterWeight bundle device dtype
    bundle.Weight.Dispose()
    bundle.Scale |> Option.iter (fun t -> t.Dispose())
    bundle.Absmax |> Option.iter (fun t -> t.Dispose())
    bundle.QuantMap |> Option.iter (fun t -> t.Dispose())
    dense

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

  let private buildLayerMap
    (device: string)
    (dtype: TorchSharp.torch.ScalarType)
    (suffix: string)
    (state: Nvfp4ModelState)
    =
    state.Layers
    |> List.choose (fun layer ->
      if layer.Name.EndsWith(suffix, StringComparison.Ordinal) then
        match parseLayerIndex layer.Name with
        | Some idx ->
          let w = materializeWeight device dtype layer.Bundle
          Some(idx, w)
        | None ->
          layer.Bundle.Weight.Dispose()
          layer.Bundle.Scale |> Option.iter (fun t -> t.Dispose())
          None
      else
        layer.Bundle.Weight.Dispose()
        layer.Bundle.Scale |> Option.iter (fun t -> t.Dispose())
        None)
    |> Map.ofList

  let private mapGet (name: string) (idx: int) (m: Map<int, torch.Tensor>) =
    match m.TryFind idx with
    | Some v -> v
    | None -> invalidOp (sprintf "missing %s for layer %d" name idx)

  let private rawGet (key: string) (m: Map<string, torch.Tensor>) =
    match m.TryFind key with
    | Some v -> v
    | None -> invalidOp (sprintf "missing raw tensor key: %s" key)

  let private linear (input: torch.Tensor) (weight: torch.Tensor) =
    input.matmul(weight.transpose(0L, 1L))

  let private rmsNorm (x: torch.Tensor) (eps: float) =
    use xF = x.to_type(torch.float32)
    use sq = xF * xF
    use mean = sq.mean([| -1L |], true)
    use denom = torch.sqrt(mean + eps)
    (xF / denom).to_type(x.dtype)

  let private rmsNormWeighted (x: torch.Tensor) (weight: torch.Tensor) (eps: float) =
    use normalized = rmsNorm x eps
    normalized * weight

  let private rmsNormHeadWeighted (x: torch.Tensor) (weight: torch.Tensor) (eps: float) =
    use normalized = rmsNorm x eps
    use w = weight.reshape([| 1L; 1L; weight.shape.[0] |])
    normalized * w

  let private expandKvHeads (numHeads: int) (numKvHeads: int) (kv: torch.Tensor) =
    let seqLen = kv.shape.[1]
    let headDim = kv.shape.[2]
    let repeatFactor = numHeads / numKvHeads
    kv
      .unsqueeze(1L)
      .expand([| int64 numKvHeads; int64 repeatFactor; seqLen; headDim |])
      .reshape([| int64 numHeads; seqLen; headDim |])

  let private buildTokenEmbeddings (session: InferenceSession) (tokenIds: int array) =
    use tokenTensor =
      torch.tensor(tokenIds, dtype = torch.int64, device = session.Device)
    session.EmbedTokens.index_select(0L, tokenTensor).contiguous()

  let private forwardLayer (session: InferenceSession) (layer: LayerWeights) (hidden: torch.Tensor) =
    let numHeads = session.Config.NumAttentionHeads
    let numKvHeads = session.Config.NumKeyValueHeads
    let headDim = session.Config.HeadDim
    let seqLen = hidden.shape.[0]
    let sqrtHead = sqrt (float32 headDim)

    use normed0 = rmsNormWeighted hidden layer.InputNorm 1e-6
    use q = linear normed0 layer.QProj
    use k = linear normed0 layer.KProj
    use v = linear normed0 layer.VProj

    use qh =
      q
        .reshape([| seqLen; int64 numHeads; int64 headDim |])
        .permute([| 1L; 0L; 2L |])
        .contiguous()

    use kh0 =
      k
        .reshape([| seqLen; int64 numKvHeads; int64 headDim |])
        .permute([| 1L; 0L; 2L |])
        .contiguous()

    use vh0 =
      v
        .reshape([| seqLen; int64 numKvHeads; int64 headDim |])
        .permute([| 1L; 0L; 2L |])
        .contiguous()

    use qhNorm = rmsNormHeadWeighted qh layer.QNorm 1e-6
    use kh0Norm = rmsNormHeadWeighted kh0 layer.KNorm 1e-6
    use kh = expandKvHeads numHeads numKvHeads kh0Norm
    use vh = expandKvHeads numHeads numKvHeads vh0

    use attnScores = qhNorm.matmul(kh.transpose(1L, 2L)) / sqrtHead
    use attnScoresF = attnScores.to_type(torch.float32)
    use mask2d = torch.triu(torch.ones([| seqLen; seqLen |], dtype = torch.bool, device = session.Device), 1L)
    use mask = mask2d.unsqueeze(0L).expand([| int64 numHeads; seqLen; seqLen |])
    use negInf = torch.zeros_like(attnScoresF) - 1e9f
    use masked = torch.where(mask, negInf, attnScoresF)
    use probs = torch.nn.functional.softmax(masked, dim = -1).to_type(session.DType)
    use ctxHeads = probs.matmul(vh)
    use ctx =
      ctxHeads
        .permute([| 1L; 0L; 2L |])
        .contiguous()
        .reshape([| seqLen; int64 (numHeads * headDim) |])

    use attnOut = linear ctx layer.OProj
    let resid1 = (hidden + attnOut).contiguous()

    use normed1 = rmsNormWeighted resid1 layer.PostAttnNorm 1e-6
    use gate = linear normed1 layer.GateProj
    use up = linear normed1 layer.UpProj
    use act = torch.nn.functional.silu(gate) * up
    use down = linear act layer.DownProj
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

  let private selectNextTokenId (session: InferenceSession) (hidden: torch.Tensor) (temperature: float32) =
    use last = hidden.narrow(0L, hidden.shape.[0] - 1L, 1L)
    use lastNorm = rmsNormWeighted last session.FinalNorm 1e-6
    use logits = lastNorm.matmul(session.LmHead.transpose(0L, 1L))
    use scaled =
      if temperature > 0.0f then
        logits / temperature
      else
        logits
    use argmax = scaled.argmax(dim = 1L).to_type(torch.int64).cpu()
    int (argmax.item<int64>())

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
      |> buildLayerMap device dtype ".self_attn.q_proj"

    let kvCount = cfgLite.NumHiddenLayers * 2
    let kvState = loadByDims baseCfg cfgLite.HiddenSize (int64 (cfgLite.NumKeyValueHeads * cfgLite.HeadDim)) kvCount
    let kMap = buildLayerMap device dtype ".self_attn.k_proj" kvState
    let vMap =
      loadByDims baseCfg cfgLite.HiddenSize (int64 (cfgLite.NumKeyValueHeads * cfgLite.HeadDim)) kvCount
      |> buildLayerMap device dtype ".self_attn.v_proj"

    let oMap =
      loadByDims baseCfg (int64 (cfgLite.NumAttentionHeads * cfgLite.HeadDim)) cfgLite.HiddenSize cfgLite.NumHiddenLayers
      |> buildLayerMap device dtype ".self_attn.o_proj"

    let guCount = cfgLite.NumHiddenLayers * 2
    let gateMap =
      loadByDims baseCfg cfgLite.HiddenSize 9728L guCount
      |> buildLayerMap device dtype ".mlp.gate_proj"
    let upMap =
      loadByDims baseCfg cfgLite.HiddenSize 9728L guCount
      |> buildLayerMap device dtype ".mlp.up_proj"

    let downMap =
      loadByDims baseCfg 9728L cfgLite.HiddenSize cfgLite.NumHiddenLayers
      |> buildLayerMap device dtype ".mlp.down_proj"

    let lmState = loadByDims baseCfg cfgLite.HiddenSize (int64 cfgLite.VocabSize) 1
    let lmHead =
      match lmState.Layers |> List.tryFind (fun l -> l.Name = "lm_head") with
      | Some l -> materializeWeight device dtype l.Bundle
      | None -> invalidOp "missing lm_head"

    for l in lmState.Layers do
      if l.Name <> "lm_head" then
        l.Bundle.Weight.Dispose()
        l.Bundle.Scale |> Option.iter (fun t -> t.Dispose())

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

  let generate (session: InferenceSession) (prompt: string) (opt: InferenceGenOptions) =
    let renderedPrompt =
      $"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

    let inputIds =
      session.Tokenizer.Encode(renderedPrompt)
      |> Seq.map int
      |> Seq.toList

    if inputIds.IsEmpty then
      invalidOp "prompt encoded to empty token sequence"

    let mutable running = inputIds
    let mutable generated: int list = []
    let mutable step = 0
    let targetSteps = max 1 opt.MaxTokens
    let mutable stop = false

    while step < targetSteps && not stop do
      use hidden = forwardModel session (running |> List.toArray)
      let nextId = selectNextTokenId session hidden opt.Temperature
      generated <- generated @ [ nextId ]
      running <- running @ [ nextId ]
      step <- step + 1

      match session.Config.EosTokenId with
      | Some eos when nextId = eos -> stop <- true
      | _ -> ()

    decodeTokens session.Tokenizer generated

  let dispose (session: InferenceSession) =
    for layer in session.Layers do
      layer.QProj.Dispose()
      layer.KProj.Dispose()
      layer.VProj.Dispose()
      layer.OProj.Dispose()
      layer.InputNorm.Dispose()
      layer.PostAttnNorm.Dispose()
      layer.QNorm.Dispose()
      layer.KNorm.Dispose()
      layer.GateProj.Dispose()
      layer.UpProj.Dispose()
      layer.DownProj.Dispose()

    session.EmbedTokens.Dispose()
    session.FinalNorm.Dispose()
    session.LmHead.Dispose()
    session.Tokenizer.Dispose()
