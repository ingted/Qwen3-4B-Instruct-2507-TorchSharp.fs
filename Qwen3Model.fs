namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.IO
open System.Runtime.CompilerServices
open System.Text
open System.Text.Json
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
  type TrainConfigLite =
    {
      HiddenSize: int64
      NumHiddenLayers: int
      NumAttentionHeads: int
      NumKeyValueHeads: int
      HeadDim: int
      RopeTheta: float
      RmsNormEps: float
    }

  let fp8E4M3FnLutCache = ConcurrentDictionary<string, torch.Tensor>(StringComparer.Ordinal)

  let fp8E4M3FnToFloat32 (raw: byte) =
    let sign = if (raw &&& 0x80uy) <> 0uy then -1.0f else 1.0f
    let exp = int ((raw >>> 3) &&& 0x0Fuy)
    let mant = int (raw &&& 0x07uy)

    if exp = 0 then
      if mant = 0 then
        0.0f
      else
        // subnormal: mant/8 * 2^(1-bias), bias=7
        sign * (float32 mant / 8.0f) * MathF.Pow(2.0f, -6.0f)
    elif exp = 0x0F then
      Single.NaN
    else
      sign * (1.0f + float32 mant / 8.0f) * MathF.Pow(2.0f, float32 (exp - 7))

  let fp8E4M3FnLut (device: TorchSharp.torch.Device) =
    let key = device.ToString()
    fp8E4M3FnLutCache.GetOrAdd(
      key,
      fun _ ->
        let data = [| for i in 0 .. 255 -> fp8E4M3FnToFloat32 (byte i) |]
        torch.tensor(data, dtype = torch.float32, device = device)
    )

  let decodeFp8E4M3FnTensor (input: TorchSharp.torch.Tensor) =
    if input.dtype <> torch.uint8 then
      input.to_type(torch.float32)
    else
      use idx = input.to_type(torch.int64).reshape([| -1L |])
      let lut = fp8E4M3FnLut input.device
      torch.index_select(lut, 0L, idx).reshape(input.shape).contiguous()

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
        | Some scale ->
          // NVFP4 dat commonly stores scale as elemType=101 (1-byte fp8-like encoding).
          // Decode uint8 scale bytes to float domain before dense dequantization for STE path.
          let scaleForDeq, ownsScale =
            if scale.dtype = torch.uint8 then
              decodeFp8E4M3FnTensor scale, true
            else
              scale, false

          try
            Nvfp4Training.dequantizePacked w scaleForDeq targetDtype
          finally
            if ownsScale then
              scaleForDeq.Dispose()
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

  let requiredInt (root: JsonElement) (name: string) =
    match root.TryGetProperty(name) with
    | true, prop -> prop.GetInt32()
    | _ -> invalidOp (sprintf "config missing field: %s" name)

  let requiredInt64 (root: JsonElement) (name: string) =
    int64 (requiredInt root name)

  let requiredFloat (root: JsonElement) (name: string) =
    match root.TryGetProperty(name) with
    | true, prop -> prop.GetDouble()
    | _ -> invalidOp (sprintf "config missing field: %s" name)

  let loadConfigLite (configPath: string) =
    use doc = JsonDocument.Parse(File.ReadAllText(configPath))
    let root = doc.RootElement
    {
      HiddenSize = requiredInt64 root "hidden_size"
      NumHiddenLayers = requiredInt root "num_hidden_layers"
      NumAttentionHeads = requiredInt root "num_attention_heads"
      NumKeyValueHeads = requiredInt root "num_key_value_heads"
      HeadDim = requiredInt root "head_dim"
      RopeTheta = requiredFloat root "rope_theta"
      RmsNormEps = requiredFloat root "rms_norm_eps"
    }

  let parseLayerIndex (name: string) =
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

  let loadByDims
    (cfg: TrainingConfig)
    (inFeatures: int64)
    (outFeatures: int64)
    (maxLayers: int)
    =
    let cfgByDims =
      {
        cfg with
            InFeatures = inFeatures
            OutFeatures = outFeatures
            MaxLayers = maxLayers
            StrictLoad = true
      }
    Nvfp4State.load cfgByDims

  let buildLayerBundleMap (suffix: string) (state: Nvfp4ModelState) =
    state.Layers
    |> List.choose (fun layer ->
      if layer.Name.EndsWith(suffix, StringComparison.Ordinal) then
        match parseLayerIndex layer.Name with
        | Some idx -> Some(idx, layer.Bundle)
        | None -> None
      else
        None)
    |> Map.ofList

  let mapGet (name: string) (idx: int) (m: Map<int, Q4TensorBundle>) =
    match m.TryFind idx with
    | Some v -> v
    | None -> invalidOp (sprintf "missing %s bundle for layer %d" name idx)

  let readRawFloatTensor
    (br: BinaryReader)
    (elemType: int)
    (shape: int64 array)
    (byteCount: int64)
    (device: string)
    (dtype: TorchSharp.torch.ScalarType)
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
      | _ -> invalidOp (sprintf "raw tensor elemType %d is not supported for norm loading" elemType)

    let onTarget = tensorCpu.reshape(shape).to_type(dtype).``to``(device = device)
    tensorCpu.Dispose()
    onTarget

  let loadRawTensorMap
    (weightPath: string)
    (device: string)
    (dtype: TorchSharp.torch.ScalarType)
    (keys: Set<string>)
    =
    use fs = File.OpenRead(weightPath)
    use br = new BinaryReader(fs)
    let entryCount = int64 (Nvfp4State.readLeb128 br)
    let results = Dictionary<string, torch.Tensor>(StringComparer.Ordinal)

    for _ in 0L .. entryCount - 1L do
      let keyLen = int (Nvfp4State.readLeb128 br)
      let key = Text.Encoding.UTF8.GetString(br.ReadBytes(keyLen))
      let elemType = int (Nvfp4State.readLeb128 br)
      let ndim = int (Nvfp4State.readLeb128 br)
      let shape = Array.init ndim (fun _ -> int64 (Nvfp4State.readLeb128 br))
      let bytes = Nvfp4State.checkedByteCount shape (Nvfp4State.elementSize elemType)
      if keys.Contains key then
        let t = readRawFloatTensor br elemType shape bytes device dtype
        if results.ContainsKey key then
          results.[key].Dispose()
        results.[key] <- t
      else
        Nvfp4State.skipBytes br bytes

    let missing =
      keys
      |> Seq.filter (fun k -> not (results.ContainsKey k))
      |> Seq.toList
    if not missing.IsEmpty then
      invalidOp (sprintf "missing raw tensors in dat: %s" (String.Join(", ", missing)))

    results |> Seq.map (fun kv -> kv.Key, kv.Value) |> Map.ofSeq

  let mkParameterFromBundle (cfg: TrainingConfig) (dtype: TorchSharp.torch.ScalarType) (bundle: Q4TensorBundle) =
    materializeMasterWeight bundle cfg.Device dtype |> fun t -> torch.nn.Parameter(t, true)

  let mkParameterFromTensor (t: torch.Tensor) =
    torch.nn.Parameter(t.contiguous().clone(), true)

  let rawGet (key: string) (m: Map<string, torch.Tensor>) =
    match m.TryFind key with
    | Some v -> v
    | None -> invalidOp (sprintf "missing raw tensor key: %s" key)

  let disposeBundle (bundle: Q4TensorBundle) =
    bundle.Weight.Dispose()
    bundle.Scale |> Option.iter (fun t -> t.Dispose())
    bundle.Absmax |> Option.iter (fun t -> t.Dispose())
    bundle.QuantMap |> Option.iter (fun t -> t.Dispose())

  let createKvCache (model: Qwen3Nvfp4Model) =
    new Qwen3Core.ModelKvCache(model.Blocks.Length)

  let resetKvCache (cache: Qwen3Core.ModelKvCache) =
    cache.Reset()

  let forwardInternal
    (model: Qwen3Nvfp4Model)
    (cache: Qwen3Core.ModelKvCache option)
    (input: TorchSharp.torch.Tensor)
    (outDtype: TorchSharp.torch.ScalarType option)
    : TorchSharp.torch.Tensor
    =
    let targetOutDtype = outDtype |> Option.defaultValue input.dtype

    let blockToStage (positionOffset: int64) (cacheOpt: Qwen3Core.BlockKvCache option) (block: Qwen3TrainableBlock) =
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

      match cacheOpt with
      | Some layerCache -> Qwen3Core.buildBlockGraphWithCache cfg norms projs layerCache positionOffset
      | None -> Qwen3Core.buildBlockGraphNoCache cfg norms projs positionOffset
(*
• 這段分支的差異是：

  1. if model.Blocks.IsEmpty then

  - 走「舊版/相容」路徑。
  - 代表模型沒有完整 Qwen block 結構（只有平面 Layers）。
  - 只做 linearSte 串接：layer -> layer -> ...，主要是為了相容舊測試/舊checkpoint/synthetic情境。

  2. else

  - 走「正式」路徑。
  - 代表 Qwen3Model.create 已建立完整 block（q/k/v/o + norm + attn + mlp + residual）。
  - 先把輸入對齊到 block 需要的 shape：
      - [B, H] 會升成 [B, 1, H]（單 token）
      - [B, T, H] 則直接用
  - 然後跑 block graph（含你要的 operator graph）。
*)


    if model.Blocks.IsEmpty then
      // Fallback path: legacy/synthetic model only has flat Layers (no Qwen block metadata).
      // This keeps old tests/checkpoints runnable by chaining plain linear STE layers.
      let trainingGraph =
        model.Layers
        |> List.map (fun layer -> stageM (sprintf "layer.%s.linear_ste" layer.Name) (linearSte layer.MasterWeight targetOutDtype))
        |> chainM
      runM trainingGraph input
    else
      // Main path: official-like Qwen block graph (q/k/v/o + norm + attn + mlp + residual).
      // Blocks are created from real model-layer bundles in Qwen3Model.create.
      let hidden0, squeezeBack, hidden0Temp =
        if input.shape.Length = 2 then
          // Block graph expects [B, T, H]. If caller gives [B, H], treat it as single-token T=1.
          let expanded = input.unsqueeze(1L)
          expanded, true, Some expanded
        else
          input, false, None

      let positionOffset = cache |> Option.map (fun c -> c.SeqLen) |> Option.defaultValue 0L
      let stages =
        match cache with
        | Some kvCache ->
          model.Blocks
          |> List.mapi (fun idx block -> blockToStage positionOffset (Some kvCache.Layers.[idx]) block)
        | None ->
          model.Blocks
          |> List.map (fun block -> blockToStage 0L None block)

      let tokenSpan = hidden0.shape.[1]
      let trainingGraph = stages |> chainM
      let hidden = runM trainingGraph hidden0
      hidden0Temp |> Option.iter (fun t -> t.Dispose())

      cache |> Option.iter (fun c -> c.SeqLen <- c.SeqLen + tokenSpan)
      if squeezeBack then
        let output = hidden.squeeze(1L).contiguous()
        hidden.Dispose()
        output
      else
        hidden

  let forward
    (model: Qwen3Nvfp4Model)
    (input: TorchSharp.torch.Tensor)
    (outDtype: TorchSharp.torch.ScalarType option)
    : TorchSharp.torch.Tensor
    =
    forwardInternal model None input outDtype

  let forwardWithKvCache
    (model: Qwen3Nvfp4Model)
    (cache: Qwen3Core.ModelKvCache)
    (input: TorchSharp.torch.Tensor)
    (outDtype: TorchSharp.torch.ScalarType option)
    : TorchSharp.torch.Tensor
    =
    forwardInternal model (Some cache) input outDtype

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

  let create (cfg: TrainingConfig) : Qwen3Nvfp4Model =
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

    let masterDtype =
      if cfg.Device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase) then torch.float16 else torch.float32

    let cfgLite = loadConfigLite cfg.ConfigPath
    let hiddenSize = cfgLite.HiddenSize
    let qOut = int64 (cfgLite.NumAttentionHeads * cfgLite.HeadDim)
    let kvOut = int64 (cfgLite.NumKeyValueHeads * cfgLite.HeadDim)
    let mlpOut =
      use doc = JsonDocument.Parse(File.ReadAllText(cfg.ConfigPath))
      let root = doc.RootElement
      requiredInt64 root "intermediate_size"

    let qMap =
      loadByDims cfg hiddenSize qOut cfgLite.NumHiddenLayers
      |> buildLayerBundleMap ".self_attn.q_proj"

    let kvCount = cfgLite.NumHiddenLayers * 2
    let kvState = loadByDims cfg hiddenSize kvOut kvCount
    let kMap = buildLayerBundleMap ".self_attn.k_proj" kvState
    let vMap = buildLayerBundleMap ".self_attn.v_proj" kvState

    let oMap =
      loadByDims cfg qOut hiddenSize cfgLite.NumHiddenLayers
      |> buildLayerBundleMap ".self_attn.o_proj"

    let guCount = cfgLite.NumHiddenLayers * 2
    let guState = loadByDims cfg hiddenSize mlpOut guCount
    let gateMap = buildLayerBundleMap ".mlp.gate_proj" guState
    let upMap = buildLayerBundleMap ".mlp.up_proj" guState

    let downMap =
      loadByDims cfg mlpOut hiddenSize cfgLite.NumHiddenLayers
      |> buildLayerBundleMap ".mlp.down_proj"

    let rawKeys =
      seq {
        for i in 0 .. cfgLite.NumHiddenLayers - 1 do
          yield sprintf "model.layers.%d.input_layernorm.weight" i
          yield sprintf "model.layers.%d.post_attention_layernorm.weight" i
          yield sprintf "model.layers.%d.self_attn.q_norm.weight" i
          yield sprintf "model.layers.%d.self_attn.k_norm.weight" i
      }
      |> Set.ofSeq

    let rawMap = loadRawTensorMap cfg.WeightPath cfg.Device masterDtype rawKeys

    let blocks =
      [
        for i in 0 .. cfgLite.NumHiddenLayers - 1 do
          let qProj = mkParameterFromBundle cfg masterDtype (mapGet "q_proj" i qMap)
          let kProj = mkParameterFromBundle cfg masterDtype (mapGet "k_proj" i kMap)
          let vProj = mkParameterFromBundle cfg masterDtype (mapGet "v_proj" i vMap)
          let oProj = mkParameterFromBundle cfg masterDtype (mapGet "o_proj" i oMap)
          let gateProj = mkParameterFromBundle cfg masterDtype (mapGet "gate_proj" i gateMap)
          let upProj = mkParameterFromBundle cfg masterDtype (mapGet "up_proj" i upMap)
          let downProj = mkParameterFromBundle cfg masterDtype (mapGet "down_proj" i downMap)
          let inputNorm = mkParameterFromTensor (rawGet (sprintf "model.layers.%d.input_layernorm.weight" i) rawMap)
          let postNorm = mkParameterFromTensor (rawGet (sprintf "model.layers.%d.post_attention_layernorm.weight" i) rawMap)
          let qNorm = mkParameterFromTensor (rawGet (sprintf "model.layers.%d.self_attn.q_norm.weight" i) rawMap)
          let kNorm = mkParameterFromTensor (rawGet (sprintf "model.layers.%d.self_attn.k_norm.weight" i) rawMap)

          {
            Name = sprintf "model.layers.%d" i
            QProj = qProj
            KProj = kProj
            VProj = vProj
            OProj = oProj
            GateProj = gateProj
            UpProj = upProj
            DownProj = downProj
            InputNorm = inputNorm
            PostAttnNorm = postNorm
            QNorm = qNorm
            KNorm = kNorm
            NumAttentionHeads = cfgLite.NumAttentionHeads
            NumKeyValueHeads = cfgLite.NumKeyValueHeads
            HeadDim = cfgLite.HeadDim
          }
      ]

    for m in [ qMap; kMap; vMap; oMap; gateMap; upMap; downMap ] do
      for kv in m do
        disposeBundle kv.Value

    for kv in rawMap do
      kv.Value.Dispose()

    let layers =
      blocks
      |> List.collect (fun b ->
        [
          { Name = $"{b.Name}.self_attn.q_proj"; MasterWeight = b.QProj }
          { Name = $"{b.Name}.self_attn.k_proj"; MasterWeight = b.KProj }
          { Name = $"{b.Name}.self_attn.v_proj"; MasterWeight = b.VProj }
          { Name = $"{b.Name}.self_attn.o_proj"; MasterWeight = b.OProj }
          { Name = $"{b.Name}.mlp.gate_proj"; MasterWeight = b.GateProj }
          { Name = $"{b.Name}.mlp.up_proj"; MasterWeight = b.UpProj }
          { Name = $"{b.Name}.mlp.down_proj"; MasterWeight = b.DownProj }
        ])

    let extraParams =
      blocks
      |> List.collect (fun b -> [ b.InputNorm; b.PostAttnNorm; b.QNorm; b.KNorm ])

    {
      Session = session
      Layers = layers
      Blocks = blocks
      ExtraParameters = extraParams
      InFeatures = hiddenSize
      OutFeatures = hiddenSize
    }
