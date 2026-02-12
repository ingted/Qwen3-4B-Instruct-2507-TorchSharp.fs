namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System
open System.Collections.Generic
open System.IO
open TorchSharp
open TorchSharp.Q4.Extension

type Nvfp4Layer =
  {
    Name: string
    Bundle: Q4TensorBundle
  }

type Nvfp4ModelState =
  {
    Layers: Nvfp4Layer list
    InFeatures: int64
    OutFeatures: int64
  }

module Nvfp4State =
  [<Literal>]
  let WeightQDataSuffix = ".weight.qdata"

  [<Literal>]
  let WeightScaleSuffix = ".weight.scale"

  [<Literal>]
  let QDataSuffix = ".qdata"

  [<Literal>]
  let ScaleSuffix = ".scale"

  type QuantKind =
    | QData
    | Scale

  type PendingPair =
    {
      mutable QData: torch.Tensor option
      mutable Scale: torch.Tensor option
    }

  type LoadedLayer =
    {
      Name: string
      QData: torch.Tensor
      Scale: torch.Tensor
      InFeatures: int64
      OutFeatures: int64
    }

  let mkSyntheticBundle (outFeatures: int64) (inFeatures: int64) (device: string) =
    let kPacked = max 1L (inFeatures / 2L)
    let scaleCols = max 1L (inFeatures / 16L)
    {
      Weight = torch.randint(0L, 255L, [| outFeatures; kPacked |], dtype = torch.uint8, device = device)
      Scale = Some (torch.ones([| outFeatures; scaleCols |], dtype = torch.uint8, device = device))
      Absmax = None
      QuantMap = None
    }

  let readLeb128 (br: BinaryReader) : uint64 =
    let mutable result = 0UL
    let mutable shift = 0
    let mutable keepReading = true

    while keepReading do
      let b = br.ReadByte()
      result <- result ||| (uint64 (b &&& 0x7Fuy) <<< shift)
      keepReading <- (b &&& 0x80uy) <> 0uy
      shift <- shift + 7

    result

  let elementSize (elemType: int) =
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
    | _ ->
      raise (NotSupportedException(sprintf "unsupported element type id: %d" elemType))

  let checkedByteCount (shape: int64 array) (elemSize: int) =
    let mutable numel = 1L
    for dim in shape do
      if dim < 0L then
        raise (InvalidOperationException(sprintf "invalid negative dimension: %d" dim))
      numel <- numel * dim

    numel * int64 elemSize

  let skipBytes (br: BinaryReader) (byteCount: int64) =
    if byteCount < 0L then
      raise (InvalidOperationException(sprintf "invalid byte count: %d" byteCount))

    let stream = br.BaseStream
    if stream.CanSeek then
      let _ = stream.Seek(byteCount, SeekOrigin.Current)
      ()
    else
      let mutable remaining = byteCount
      while remaining > 0L do
        let chunk = min remaining 8192L
        let read = br.ReadBytes(int chunk)
        if read.Length = 0 then
          raise (EndOfStreamException("unexpected EOF while skipping tensor payload"))
        remaining <- remaining - int64 read.Length

  let readTensorAsByte (br: BinaryReader) (shape: int64 array) (byteCount: int64) (device: string) =
    if byteCount > int64 Int32.MaxValue then
      raise (InvalidOperationException(sprintf "tensor payload too large: %d bytes" byteCount))

    let bytes = br.ReadBytes(int byteCount)
    if bytes.Length <> int byteCount then
      raise (EndOfStreamException(sprintf "unexpected EOF while reading tensor payload (%d bytes)" byteCount))

    let cpu = torch.tensor(bytes, dtype = torch.uint8).reshape(shape)
    if device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase) then
      // Important: release temporary CPU tensor right after host->device copy.
      // This loader runs multiple full-file scans during init; keeping CPU temps alive
      // increases allocator pressure and makes native init less stable under repeated runs.
      let gpu = cpu.``to``(device = device)
      cpu.Dispose()
      gpu
    else
      cpu

  let tryParseQuantKey (key: string) : (string * QuantKind) option =
    if key.EndsWith(WeightQDataSuffix, StringComparison.Ordinal) then
      Some(key.Substring(0, key.Length - WeightQDataSuffix.Length), QuantKind.QData)
    elif key.EndsWith(QDataSuffix, StringComparison.Ordinal) then
      Some(key.Substring(0, key.Length - QDataSuffix.Length), QuantKind.QData)
    elif key.EndsWith(WeightScaleSuffix, StringComparison.Ordinal) then
      Some(key.Substring(0, key.Length - WeightScaleSuffix.Length), QuantKind.Scale)
    elif key.EndsWith(ScaleSuffix, StringComparison.Ordinal) then
      Some(key.Substring(0, key.Length - ScaleSuffix.Length), QuantKind.Scale)
    else
      None

  let tryBuildLayer (name: string) (qData: torch.Tensor) (scale: torch.Tensor) : LoadedLayer option =
    if qData.shape.Length <> 2 || scale.shape.Length <> 2 then
      None
    else
      let outFeatures = qData.shape.[0]
      let inFeatures = qData.shape.[1] * 2L

      if outFeatures <= 0L || inFeatures <= 0L || inFeatures % 16L <> 0L then
        None
      else
        let expectedScaleCols = inFeatures / 16L
        if scale.shape.[0] <> outFeatures || scale.shape.[1] <> expectedScaleCols then
          None
        else
          Some
            {
              Name = name
              QData = qData
              Scale = scale
              InFeatures = inFeatures
              OutFeatures = outFeatures
            }

  let disposeLayer (layer: LoadedLayer) =
    layer.QData.Dispose()
    layer.Scale.Dispose()

  let loadFromDat (cfg: TrainingConfig) : Nvfp4ModelState =
    let maxLayers = max 1 cfg.MaxLayers
    let desiredDims = Some(cfg.InFeatures, cfg.OutFeatures)

    use fs = File.OpenRead(cfg.WeightPath)
    use br = new BinaryReader(fs)

    let entryCount = int64 (readLeb128 br)
    let pending = Dictionary<string, PendingPair>(StringComparer.Ordinal)
    let preferredLayers = ResizeArray<LoadedLayer>()
    let fallbackLayers = ResizeArray<LoadedLayer>()
    let mutable fallbackDims : (int64 * int64) option = None

    let shouldStop () =
      match desiredDims with
      | Some _ when preferredLayers.Count >= maxLayers -> true
      | _ -> false

    let mutable i = 0L
    while i < entryCount && not (shouldStop()) do
      let keyLen = int (readLeb128 br)
      let keyBytes = br.ReadBytes(keyLen)
      if keyBytes.Length <> keyLen then
        raise (EndOfStreamException(sprintf "unexpected EOF while reading key bytes at entry %d" i))

      let key = Text.Encoding.UTF8.GetString(keyBytes)
      let elemType = int (readLeb128 br)
      let ndim = int (readLeb128 br)
      let shape = Array.zeroCreate<int64> ndim
      for d in 0 .. ndim - 1 do
        shape.[d] <- int64 (readLeb128 br)

      let bytes = checkedByteCount shape (elementSize elemType)

      match tryParseQuantKey key with
      | None ->
        skipBytes br bytes
      | Some(prefix, kind) ->
        let tensor = readTensorAsByte br shape bytes cfg.Device

        let pair =
          match pending.TryGetValue(prefix) with
          | true, existing -> existing
          | _ ->
            let created = { QData = None; Scale = None }
            pending[prefix] <- created
            created

        match kind with
        | QuantKind.QData ->
          pair.QData |> Option.iter (fun t -> t.Dispose())
          pair.QData <- Some tensor
        | QuantKind.Scale ->
          pair.Scale |> Option.iter (fun t -> t.Dispose())
          pair.Scale <- Some tensor

        match pair.QData, pair.Scale with
        | Some qData, Some scale ->
          pending.Remove(prefix) |> ignore
          match tryBuildLayer prefix qData scale with
          | None ->
            qData.Dispose()
            scale.Dispose()
          | Some layer ->
            let dims = (layer.InFeatures, layer.OutFeatures)
            if desiredDims = Some dims then
              if preferredLayers.Count < maxLayers then
                preferredLayers.Add(layer)
              else
                disposeLayer layer
            else
              if fallbackDims.IsNone then
                fallbackDims <- Some dims

              if fallbackDims = Some dims && fallbackLayers.Count < maxLayers then
                fallbackLayers.Add(layer)
              else
                disposeLayer layer
        | _ -> ()

      i <- i + 1L

    for kv in pending.Values do
      kv.QData |> Option.iter (fun t -> t.Dispose())
      kv.Scale |> Option.iter (fun t -> t.Dispose())

    let disposeCollected (collected: ResizeArray<LoadedLayer>) =
      for layer in collected do
        disposeLayer layer

    let chosen =
      if preferredLayers.Count > 0 then
        preferredLayers |> Seq.toList
      elif cfg.StrictLoad then
        []
      else
        fallbackLayers |> Seq.toList

    if preferredLayers.Count = 0 then
      match desiredDims, fallbackDims with
      | Some (inWanted, outWanted), Some (inFallback, outFallback) when cfg.StrictLoad ->
        disposeCollected preferredLayers
        disposeCollected fallbackLayers
        raise (
          InvalidOperationException(
            sprintf
              "Strict load rejected fallback dims in=%d,out=%d (requested in=%d,out=%d) from %s."
              inFallback
              outFallback
              inWanted
              outWanted
              cfg.WeightPath
          )
        )
      | Some (inWanted, outWanted), Some (inFallback, outFallback) ->
        printfn
          "[Load] requested dims (in=%d,out=%d) not found; fallback to (in=%d,out=%d)."
          inWanted
          outWanted
          inFallback
          outFallback
      | Some (inWanted, outWanted), None when cfg.StrictLoad ->
        disposeCollected preferredLayers
        disposeCollected fallbackLayers
        raise (
          InvalidOperationException(
            sprintf
              "Strict load found no matching dims in=%d,out=%d in %s."
              inWanted
              outWanted
              cfg.WeightPath
          )
        )
      | _ -> ()

    if chosen.IsEmpty then
      disposeCollected preferredLayers
      disposeCollected fallbackLayers
      raise (
        InvalidOperationException(
          sprintf
            "No valid NVFP4 (qdata+scale) pairs found in %s for requested dims in=%d out=%d (strict=%b)."
            cfg.WeightPath
            cfg.InFeatures
            cfg.OutFeatures
            cfg.StrictLoad
        )
      )

    if preferredLayers.Count > 0 then
      for layer in fallbackLayers do
        disposeLayer layer
    else
      for layer in preferredLayers do
        disposeLayer layer

    let inFeatures = chosen.Head.InFeatures
    let outFeatures = chosen.Head.OutFeatures

    let layers =
      chosen
      |> List.map (fun l ->
        {
          Name = l.Name
          Bundle =
            {
              Weight = l.QData
              Scale = Some l.Scale
              Absmax = None
              QuantMap = None
            }
        })

    printfn "[Load] source=%s layers=%d in=%d out=%d" cfg.WeightPath layers.Length inFeatures outFeatures

    {
      Layers = layers
      InFeatures = inFeatures
      OutFeatures = outFeatures
    }

  let load (cfg: TrainingConfig) : Nvfp4ModelState =
    if cfg.SyntheticMode then
      {
        Layers =
          [
            { Name = "layer.0"; Bundle = mkSyntheticBundle cfg.OutFeatures cfg.InFeatures cfg.Device }
            { Name = "layer.1"; Bundle = mkSyntheticBundle cfg.OutFeatures cfg.OutFeatures cfg.Device }
          ]
          |> List.truncate (max 1 cfg.MaxLayers)
        InFeatures = cfg.InFeatures
        OutFeatures = cfg.OutFeatures
      }
    else
      if not (File.Exists cfg.WeightPath) then
        raise (FileNotFoundException(sprintf "Weight file not found: %s" cfg.WeightPath))

      loadFromDat cfg
