namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System
open System.IO
open System.Text.Json
open TorchSharp

[<CLIMutable>]
type CheckpointState =
  {
    Epoch: int
    GlobalStep: int
  }

[<CLIMutable>]
type CheckpointMeta =
  {
    Epoch: int
    GlobalStep: int
    LayerCount: int
    InFeatures: int64
    OutFeatures: int64
    TimestampUtc: string
  }

module Trainer =
  let checkpointMetaPath (cfg: TrainingConfig) =
    Path.Combine(cfg.CheckpointDir, "meta.json")

  let checkpointWeightPath (cfg: TrainingConfig) (layerIndex: int) =
    Path.Combine(cfg.CheckpointDir, sprintf "layer_%04d.pt" layerIndex)

  let createBatch
    (batchSize: int64)
    (inFeatures: int64)
    (outFeatures: int64)
    (sequenceLength: int64)
    (useKvCache: bool)
    (device: string)
    (dtype: TorchSharp.torch.ScalarType)
    =
    let inputShape, targetShape =
      if useKvCache then
        [| batchSize; sequenceLength; inFeatures |], [| batchSize; sequenceLength; outFeatures |]
      else
        [| batchSize; inFeatures |], [| batchSize; outFeatures |]
    let input = torch.randn(inputShape, dtype = dtype, device = device)
    let target = torch.randn(targetShape, dtype = dtype, device = device)
    input, target

  let scalarLoss (output: TorchSharp.torch.Tensor) (target: TorchSharp.torch.Tensor) =
    let targetForLossTemp =
      if target.dtype = output.dtype then None else Some (target.to_type(output.dtype))
    let targetForLoss =
      match targetForLossTemp with
      | Some t -> t
      | None -> target

    use diff = output - targetForLoss
    use absDiff = diff.abs()
    let loss = absDiff.mean()
    targetForLossTemp |> Option.iter (fun t -> t.Dispose())
    loss

  let saveCheckpoint (cfg: TrainingConfig) (epoch: int) (globalStep: int) (model: Qwen3Nvfp4Model) =
    Directory.CreateDirectory(cfg.CheckpointDir) |> ignore

    use _guard = torch.no_grad()
    model.Layers
    |> List.iteri (fun idx layer ->
      let path = checkpointWeightPath cfg idx
      use toSave = layer.MasterWeight.detach().to_type(torch.float32).cpu().contiguous().clone()
      torch.save(toSave, path))

    let meta =
      {
        Epoch = epoch
        GlobalStep = globalStep
        LayerCount = model.Layers.Length
        InFeatures = model.InFeatures
        OutFeatures = model.OutFeatures
        TimestampUtc = DateTime.UtcNow.ToString("O")
      }

    let metaJson = JsonSerializer.Serialize(meta, JsonSerializerOptions(WriteIndented = true))
    File.WriteAllText(checkpointMetaPath cfg, metaJson)

  let tryLoadCheckpoint (cfg: TrainingConfig) (model: Qwen3Nvfp4Model) : CheckpointState option =
    let metaPath = checkpointMetaPath cfg
    if not cfg.ResumeFromCheckpoint || not (File.Exists(metaPath)) then
      None
    else
      match JsonSerializer.Deserialize<CheckpointMeta>(File.ReadAllText(metaPath)) |> Option.ofObj with
      | None -> None
      | Some meta ->
        if meta.LayerCount <> model.Layers.Length then
          raise (
            InvalidOperationException(
              sprintf
                "checkpoint layer count mismatch: checkpoint=%d model=%d."
                meta.LayerCount
                model.Layers.Length
            )
          )

        use _guard = torch.no_grad()
        model.Layers
        |> List.iteri (fun idx layer ->
          let path = checkpointWeightPath cfg idx
          if not (File.Exists(path)) then
            raise (FileNotFoundException(sprintf "checkpoint layer file missing: %s" path))

          use loaded = torch.load(path)
          let loadedOnTarget =
            if loaded.device.ToString() = layer.MasterWeight.device.ToString() then
              loaded
            else
              loaded.``to``(layer.MasterWeight.device)

          let loadedTyped =
            if loadedOnTarget.dtype = layer.MasterWeight.dtype then
              loadedOnTarget
            else
              loadedOnTarget.to_type(layer.MasterWeight.dtype)

          layer.MasterWeight.copy_(loadedTyped) |> ignore)

        Some
          {
            Epoch = meta.Epoch
            GlobalStep = meta.GlobalStep
          }

  let run (cfg: TrainingConfig) (model: Qwen3Nvfp4Model) =
    printfn
      "[Train] mode=NVFP4 STE, epochs=%d, steps/epoch=%d, batch=%d lr=%f"
      cfg.Epochs
      cfg.StepsPerEpoch
      cfg.BatchSize
      cfg.LearningRate
    printfn "[Train] features in=%d out=%d layers=%d" model.InFeatures model.OutFeatures model.Layers.Length
    printfn "[Train] synthetic=%b useKvc=%b seqLen=%d" cfg.SyntheticMode cfg.UseKvCache cfg.SequenceLength

    use optimizer = torch.optim.Adam(Qwen3Model.parameters model, cfg.LearningRate, 0.9, 0.999, 1e-8, 0.0, false, false)
    let computeDtype =
      if cfg.Device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase) then torch.float16 else torch.float32

    let mutable globalStep = 0
    let mutable startEpoch = 1

    match tryLoadCheckpoint cfg model with
    | Some state ->
      globalStep <- state.GlobalStep
      startEpoch <- state.Epoch + 1
      printfn "[Train] resumed from checkpoint: epoch=%d global_step=%d" state.Epoch state.GlobalStep
    | None ->
      ()

    for epoch in startEpoch .. cfg.Epochs do
      let mutable epochLoss = 0.0f
      for _step in 1 .. cfg.StepsPerEpoch do
        let input, target =
          createBatch
            cfg.BatchSize
            model.InFeatures
            model.OutFeatures
            cfg.SequenceLength
            cfg.UseKvCache
            cfg.Device
            computeDtype
        use inputTensor = input
        use targetTensor = target
        optimizer.zero_grad()
        use output =
          if cfg.UseKvCache && model.Blocks.Length > 0 then
            use cache = Qwen3Model.createKvCache model
            Qwen3Model.forwardWithKvCache model cache inputTensor (Some computeDtype)
          else
            Qwen3Model.forward model inputTensor (Some computeDtype)
        use loss = scalarLoss output targetTensor
        loss.backward()
        optimizer.step() |> ignore

        use lossForRead = loss.to_type(torch.float32).cpu()
        let lossValue = lossForRead.item<float32>()
        epochLoss <- epochLoss + lossValue
        globalStep <- globalStep + 1

        if cfg.SaveEverySteps > 0 && globalStep % cfg.SaveEverySteps = 0 then
          saveCheckpoint cfg epoch globalStep model
          printfn "[Train] checkpoint saved at step=%d dir=%s" globalStep cfg.CheckpointDir

      let avgLoss = epochLoss / float32 cfg.StepsPerEpoch
      printfn "[Train] epoch=%d avg_loss=%f" epoch avgLoss
      saveCheckpoint cfg epoch globalStep model

    if startEpoch > cfg.Epochs then
      printfn "[Train] no epoch executed (resume epoch already beyond configured epochs)."
