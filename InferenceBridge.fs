namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System
open System.IO
open System.Text
open TorchSharp
open TorchSharp.Q4.Extension

type InferenceSession =
  {
    Model: Qwen3Nvfp4Model
    Device: string
    DType: TorchSharp.torch.ScalarType
    VocabSize: int
    MaxInputTokens: int
  }

type InferenceGenOptions =
  {
    MaxTokens: int
    Temperature: float32
    TopP: float32
    Seed: int option
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

  let private normalizeId (vocabSize: int) (tokenId: int) =
    let m = max 1 vocabSize
    let r = tokenId % m
    if r < 0 then r + m else r

  let private bytesToIds (text: string) =
    Encoding.UTF8.GetBytes(text)
    |> Array.map int
    |> Array.toList

  let private idsToText (tokenIds: int list) =
    let bytes =
      tokenIds
      |> List.map (fun i -> byte (normalizeId 256 i))
      |> List.toArray
    Encoding.UTF8.GetString(bytes)

  let private promptToFeatureTensor (session: InferenceSession) (tokenIds: int list) =
    let size = int session.Model.InFeatures
    let data = Array.zeroCreate<float32> size
    let mutable i = 0
    let mutable pos = tokenIds
    while i < session.MaxInputTokens && not pos.IsEmpty do
      let tok = normalizeId session.VocabSize pos.Head
      let idx = i % size
      let prev = data[idx]
      let value = float32 tok / float32 session.VocabSize
      data[idx] <- prev + value
      i <- i + 1
      pos <- pos.Tail

    torch.tensor(data, dtype = torch.float32, device = session.Device)
      .reshape([| 1L; session.Model.InFeatures |])
      .to_type(session.DType)

  let private adaptFeatures (x: torch.Tensor) (expectedInFeatures: int64) =
    let current = x.shape.[1]
    if current = expectedInFeatures then
      x
    elif current > expectedInFeatures then
      x.narrow(1L, 0L, expectedInFeatures).contiguous()
    else
      let padCols = expectedInFeatures - current
      use pad = torch.zeros([| x.shape.[0]; padCols |], dtype = x.dtype, device = x.device)
      torch.cat([| x; pad |], 1L)

  let private layerForward (outDtype: TorchSharp.torch.ScalarType) (x: torch.Tensor) (layer: Qwen3TrainableLayer) =
    let expectedIn = layer.MasterWeight.shape.[1]
    use aligned = adaptFeatures x expectedIn
    let linear = Nvfp4Training.linearSte aligned layer.MasterWeight outDtype
    let activated = torch.nn.functional.gelu(linear)
    if activated.shape = aligned.shape then activated + aligned else activated

  /// BERT-like explicit wiring:
  /// embeddings -> block0 -> block1 -> ... -> lm_head
  let private forwardBackbone (session: InferenceSession) (embeddings: torch.Tensor) =
    let mutable hidden = embeddings
    for layer in session.Model.Layers do
      hidden <- layerForward session.DType hidden layer
    hidden

  let private sampleNextTokenId (session: InferenceSession) (hidden: torch.Tensor) =
    let selected =
      session.Model.Layers
      |> List.tryFindBack (fun l -> l.MasterWeight.shape.[0] = hidden.shape.[1])
      |> Option.defaultValue (session.Model.Layers |> List.last)

    let expectedHidden = selected.MasterWeight.shape.[0]
    use aligned = adaptFeatures hidden expectedHidden
    use w = selected.MasterWeight.detach()
    let maxCols = int w.shape.[1]
    let cols = int64 (min session.VocabSize maxCols)
    use head = w.narrow(1L, 0L, cols).contiguous()
    use logits = aligned.matmul(head)
    use argmax = logits.argmax(dim = 1L).to_type(torch.int64).cpu()
    int (argmax.item<int64>())

  let init
    (modelDir: string)
    (weightOverride: string option)
    (quantHint: string option)
    (device: string)
    (dtype: TorchSharp.torch.ScalarType)
    =
    // Keep CUDA initialization behavior aligned with existing runner.
    torch.InitializeDeviceType(DeviceType.CUDA)
    torch.set_default_dtype(dtype)

    let resolvedModelDir =
      if String.IsNullOrWhiteSpace modelDir then Defaults.modelDir else modelDir.Trim()
    let weightPath = resolveWeightPath resolvedModelDir weightOverride quantHint

    let cfg =
      {
        Defaults.trainingConfig with
            ModelDir = resolvedModelDir
            ConfigPath = Path.Combine(resolvedModelDir, "config.json")
            TokenizerPath = Path.Combine(resolvedModelDir, "tokenizer.json")
            WeightPath = weightPath
            Device = device
            SyntheticMode = false
            StrictLoad = false
      }

    if not (File.Exists cfg.WeightPath) then
      invalidOp (sprintf "weight file not found: %s" cfg.WeightPath)

    let state = Nvfp4State.load cfg
    let model = Qwen3Model.create cfg state

    {
      Model = model
      Device = device
      DType = dtype
      VocabSize = 256
      MaxInputTokens = 512
    }

  let generate (session: InferenceSession) (prompt: string) (opt: InferenceGenOptions) =
    let initIds = bytesToIds prompt
    let baseIds = if initIds.IsEmpty then [ 0 ] else initIds
    let mutable running = baseIds
    let mutable generated = []

    for _step in 1 .. max 1 opt.MaxTokens do
      use input = promptToFeatureTensor session running
      use hidden = forwardBackbone session input
      let nextId = sampleNextTokenId session hidden |> normalizeId session.VocabSize
      generated <- generated @ [ nextId ]
      running <- (running @ [ nextId ]) |> List.rev |> List.truncate session.MaxInputTokens |> List.rev

    idsToText generated

  let dispose (session: InferenceSession) =
    Qwen3Model.dispose session.Model
