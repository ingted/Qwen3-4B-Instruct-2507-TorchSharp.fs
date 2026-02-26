namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System
open System.IO
open System.Text.Json
open System.Diagnostics
open TorchSharp
open TorchSharp.Modules
open TorchSharp.Q4.Extension

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
    ParameterCount: int
    ParameterNames: string array
    InFeatures: int64
    OutFeatures: int64
    TimestampUtc: string
  }

[<CLIMutable>]
type TrainVramSample =
  {
    TimestampUtc: string
    Epoch: int
    StepInEpoch: int
    GlobalStep: int
    Phase: string
    PidMemoryMiB: int
    TotalMemoryMiB: int
  }

[<CLIMutable>]
type TrainVramReport =
  {
    Pid: int
    CreatedUtc: string
    Samples: TrainVramSample array
  }

module Trainer =
  type LossMode =
    | ScalarL1
    | TokenCrossEntropy

  let parseLossMode (raw: string) =
    match raw.Trim().ToLowerInvariant() with
    | "scalar"
    | "l1"
    | "scalar-l1" -> LossMode.ScalarL1
    | "ce"
    | "cross-entropy"
    | "token-ce" -> LossMode.TokenCrossEntropy
    | other -> invalidArg "loss" (sprintf "unsupported loss mode: %s (supported: scalar|ce)" other)

  let lossModeName (mode: LossMode) =
    match mode with
    | LossMode.ScalarL1 -> "scalar"
    | LossMode.TokenCrossEntropy -> "ce"

  let private getGpuMemSnapshotMiB (pid: int) =
    try
      let psi = ProcessStartInfo("nvidia-smi")
      psi.Arguments <- "--query-compute-apps=pid,used_memory --format=csv,noheader,nounits"
      psi.RedirectStandardOutput <- true
      psi.UseShellExecute <- false
      psi.CreateNoWindow <- true
      use p = Process.Start(psi)
      let output = p.StandardOutput.ReadToEnd()
      p.WaitForExit()
      let entries =
        output.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
        |> Array.fold (fun acc line ->
          let parts = line.Split([| ',' |], StringSplitOptions.RemoveEmptyEntries)
          if parts.Length >= 2 then
            match Int32.TryParse(parts.[0].Trim()), Int32.TryParse(parts.[1].Trim()) with
            | (true, p), (true, mem) -> (p, mem) :: acc
            | _ -> acc
          else
            acc) []
      let pidMem =
        entries |> List.fold (fun acc (p, mem) -> if p = pid then acc + mem else acc) 0
      let totalMem =
        entries |> List.fold (fun acc (_, mem) -> acc + mem) 0
      pidMem, totalMem
    with _ ->
      0, 0

  let private shouldCollectTrainStepVram (cfg: TrainingConfig) =
    cfg.ProfileTrainStepVram || cfg.TrainStepVramReportPath.IsSome

  let private profileTrainStepVram
    (cfg: TrainingConfig)
    (samples: ResizeArray<TrainVramSample>)
    (epoch: int)
    (stepInEpoch: int)
    (globalStep: int)
    (phase: string)
    =
    if shouldCollectTrainStepVram cfg then
      let pidMem, totalMem = getGpuMemSnapshotMiB Environment.ProcessId
      if cfg.ProfileTrainStepVram then
        printfn
          "[TrainVRAM] phase=%s epoch=%d stepInEpoch=%d globalStep=%d pid=%dMiB total=%dMiB"
          phase
          epoch
          stepInEpoch
          globalStep
          pidMem
          totalMem
      if cfg.TrainStepVramReportPath.IsSome then
        samples.Add(
          {
            TimestampUtc = DateTime.UtcNow.ToString("O")
            Epoch = epoch
            StepInEpoch = stepInEpoch
            GlobalStep = globalStep
            Phase = phase
            PidMemoryMiB = pidMem
            TotalMemoryMiB = totalMem
          }
        )

  let private tryWriteTrainStepVramReport (cfg: TrainingConfig) (samples: ResizeArray<TrainVramSample>) =
    match cfg.TrainStepVramReportPath with
    | None -> ()
    | Some reportPath ->
      try
        let dir = Path.GetDirectoryName(reportPath)
        if not (String.IsNullOrWhiteSpace dir) then
          Directory.CreateDirectory(dir) |> ignore
        let report =
          {
            Pid = Environment.ProcessId
            CreatedUtc = DateTime.UtcNow.ToString("O")
            Samples = samples.ToArray()
          }
        let json = JsonSerializer.Serialize(report, JsonSerializerOptions(WriteIndented = true))
        File.WriteAllText(reportPath, json)
        printfn "[TrainVRAM] json report saved: %s (samples=%d)" reportPath samples.Count
      with ex ->
        eprintfn "[TrainVRAM] failed to write json report: %s" ex.Message

  let private backwardWithSequenceRecompute
    (model: Qwen3Nvfp4Model)
    (input: torch.Tensor)
    (target: torch.Tensor)
    (chunkSize: int)
    (computeDtype: torch.ScalarType) =
    let seqLen = int input.shape.[1]
    let totalChunks = max 1 ((seqLen + chunkSize - 1) / chunkSize)
    let mutable start = 0
    let mutable lossSum = 0.0f

    while start < seqLen do
      let ending = min seqLen (start + chunkSize)
      let prefixLen = int64 ending
      let chunkStart = int64 start
      let chunkLen = int64 (ending - start)

      use prefixInput = input.narrow(1L, 0L, prefixLen).contiguous()
      use prefixOutput = Qwen3Model.forward model prefixInput (Some computeDtype)
      use chunkOutput = prefixOutput.narrow(1L, chunkStart, chunkLen).contiguous()
      use chunkTarget = target.narrow(1L, chunkStart, chunkLen).contiguous()
      let chunkTargetForLossTemp =
        if chunkTarget.dtype = chunkOutput.dtype then None else Some (chunkTarget.to_type(chunkOutput.dtype))
      let chunkTargetForLoss =
        match chunkTargetForLossTemp with
        | Some t -> t
        | None -> chunkTarget
      use diff = chunkOutput - chunkTargetForLoss
      use absDiff = diff.abs()
      use chunkLoss = absDiff.mean()
      use scaledLoss = chunkLoss / float32 totalChunks
      scaledLoss.backward()
      chunkTargetForLossTemp |> Option.iter (fun t -> t.Dispose())
      use lossCpu = scaledLoss.to_type(torch.float32).cpu()
      lossSum <- lossSum + lossCpu.item<float32>()
      start <- ending

    lossSum

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

  let tokenCrossEntropyLoss
    (projectToLogits: TorchSharp.torch.Tensor -> TorchSharp.torch.Tensor)
    (outputHidden: TorchSharp.torch.Tensor)
    (targetTokenIds: int array)
    =
    if outputHidden.shape.Length <> 3 then
      invalidArg "outputHidden" (sprintf "token CE expects hidden shape [B,T,H], got rank=%d" outputHidden.shape.Length)

    use logits = projectToLogits outputHidden
    let vocab = logits.shape.[logits.shape.Length - 1]
    use logits2d = logits.reshape([| -1L; vocab |]).contiguous()
    let expectedTargets = int logits2d.shape.[0]
    if expectedTargets <> targetTokenIds.Length then
      invalidArg
        "targetTokenIds"
        (sprintf "target length mismatch: expected=%d actual=%d" expectedTargets targetTokenIds.Length)

    let targetIds64 = targetTokenIds |> Array.map int64
    use targetTensor = torch.tensor(targetIds64, dtype = torch.int64, device = logits2d.device)
    torch.nn.functional.cross_entropy(logits2d, targetTensor)

  let tokenCrossEntropyLossTensor
    (projectToLogits: TorchSharp.torch.Tensor -> TorchSharp.torch.Tensor)
    (outputHidden: TorchSharp.torch.Tensor)
    (targetTokenIds: TorchSharp.torch.Tensor)
    =
    if outputHidden.shape.Length <> 3 then
      invalidArg "outputHidden" (sprintf "token CE expects hidden shape [B,T,H], got rank=%d" outputHidden.shape.Length)

    use logits = projectToLogits outputHidden
    let vocab = logits.shape.[logits.shape.Length - 1]
    use logits2d = logits.reshape([| -1L; vocab |]).contiguous()
    use targets2d = targetTokenIds.reshape([| -1L |]).to_type(torch.int64).``to``(device = logits2d.device)
    if logits2d.shape.[0] <> targets2d.shape.[0] then
      invalidArg
        "targetTokenIds"
        (sprintf "target length mismatch: expected=%d actual=%d" logits2d.shape.[0] targets2d.shape.[0])
    torch.nn.functional.cross_entropy(logits2d, targets2d)

  let saveCheckpoint (cfg: TrainingConfig) (epoch: int) (globalStep: int) (model: Qwen3Nvfp4Model) =
    Directory.CreateDirectory(cfg.CheckpointDir) |> ignore

    let namedParams = Qwen3Model.namedParameters model
    use _guard = torch.no_grad()
    namedParams
    |> List.iteri (fun idx (_, param) ->
      let path = checkpointWeightPath cfg idx
      use toSave = param.detach().to_type(torch.float32).cpu().contiguous().clone()
      torch.save(toSave, path))

    let meta =
      {
        Epoch = epoch
        GlobalStep = globalStep
        LayerCount = model.Layers.Length
        ParameterCount = namedParams.Length
        ParameterNames = namedParams |> List.map fst |> List.toArray
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
        let namedParams = Qwen3Model.namedParameters model
        let expectedCount =
          if meta.ParameterCount > 0 then meta.ParameterCount else meta.LayerCount
        if expectedCount <> namedParams.Length then
          raise (
            InvalidOperationException(
              sprintf
                "checkpoint parameter count mismatch: checkpoint=%d model=%d."
                expectedCount
                namedParams.Length
            )
          )

        use _guard = torch.no_grad()
        namedParams
        |> List.iteri (fun idx (name, param) ->
          let path = checkpointWeightPath cfg idx
          if not (File.Exists(path)) then
            raise (FileNotFoundException(sprintf "checkpoint layer file missing: %s" path))

          use loaded = torch.load(path)
          let loadedOnTarget =
            if loaded.device.ToString() = param.device.ToString() then
              loaded
            else
              loaded.``to``(param.device)

          let loadedTyped =
            if loadedOnTarget.dtype = param.dtype then
              loadedOnTarget
            else
              loadedOnTarget.to_type(param.dtype)

          if meta.ParameterNames <> null && idx < meta.ParameterNames.Length then
            let ckptName = meta.ParameterNames.[idx]
            if not (String.Equals(ckptName, name, StringComparison.Ordinal)) then
              printfn "[Train][warn] checkpoint parameter name mismatch at %d: ckpt=%s model=%s" idx ckptName name
          param.copy_(loadedTyped) |> ignore)

        Some
          {
            Epoch = meta.Epoch
            GlobalStep = meta.GlobalStep
          }

  let run (cfg: TrainingConfig) (model: Qwen3Nvfp4Model) =
    if cfg.OffloadMVToCpu || cfg.OffloadWToCpu || cfg.OffloadGradToCpu then
      failwith
        "Offload is disabled for DGX Spark training path. Set --offload-mv-to-cpu=false --offload-w-to-cpu=false --offload-grad-to-cpu=false."

    if (not cfg.UsePackedNvfp4Optimizer) && (not cfg.SyntheticMode) && model.Layers.Length >= 200 then
      failwith
        "Plain Adam optimizer is disabled for full 4B training path due to memory/OOM risk. Use --use-packed-optimizer true."

    printfn
      "[Train] mode=NVFP4 STE, epochs=%d, steps/epoch=%d, batch=%d lr=%f"
      cfg.Epochs
      cfg.StepsPerEpoch
      cfg.BatchSize
      cfg.LearningRate
    let lossMode = parseLossMode cfg.LossMode
    printfn "[Train] lossMode=%s" (lossModeName lossMode)
    printfn "[Train] features in=%d out=%d layers=%d" model.InFeatures model.OutFeatures model.Layers.Length
    printfn "[Train] synthetic=%b useKvc=%b seqLen=%d" cfg.SyntheticMode cfg.UseKvCache cfg.SequenceLength
    printfn
      "[Train] packedOpt=%b masterDType=%s optStateMode=%s gradCkptChunk=%d stepChunkRows=%d offload(m/v/wg)=%b/%b/%b profileTrainStepVram=%b vramReport=%s"
      cfg.UsePackedNvfp4Optimizer
      cfg.MasterDType
      cfg.OptimizerStateMode
      cfg.GradCheckpointChunk
      cfg.OptimizerStepChunkRows
      cfg.OffloadMVToCpu
      cfg.OffloadWToCpu
      cfg.OffloadGradToCpu
      cfg.ProfileTrainStepVram
      (cfg.TrainStepVramReportPath |> Option.defaultValue "<none>")

    let trainableParams = Qwen3Model.parameters model
    let optimizerOpt =
      if cfg.UsePackedNvfp4Optimizer then
        None
      else
        Some(torch.optim.Adam(trainableParams, cfg.LearningRate, 0.9, 0.999, 1e-8, 0.0, false, false))

    let computeDtype =
      if cfg.Device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase) then
        match cfg.MasterDType.Trim().ToLowerInvariant() with
        | "bfloat16"
        | "bf16" -> torch.bfloat16
        | _ -> torch.float16
      else
        torch.float32

    let masterDtype =
      match trainableParams with
      | p :: _ -> p.dtype
      | [] -> computeDtype

    let nameByKey = System.Collections.Generic.Dictionary<int, string>()
    for name, p in Qwen3Model.namedParameters model do
      nameByKey[System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(p)] <- name

    let nameOfParam (p: Parameter) =
      let key = System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(p)
      match nameByKey.TryGetValue key with
      | true, name -> name
      | _ -> sprintf "param.%d" key

    let stateMode =
      match cfg.OptimizerStateMode.Trim().ToLowerInvariant() with
      | "nvfp4"
      | "fp4" -> PackedStateMode.Nvfp4
      | "int8"
      | "8bit"
      | "8-bit" -> PackedStateMode.Int8
      | other -> invalidArg "optimizer-state-mode" (sprintf "unsupported optimizer state mode: %s" other)

    let packedOptimizerStateOpt =
      if cfg.UsePackedNvfp4Optimizer then
        let packedCfg : PackedAdamwConfig =
          {
            Device = cfg.Device
            MasterDType = masterDtype
            LearningRate = float32 cfg.LearningRate
            Beta1 = 0.9f
            Beta2 = 0.999f
            Eps = 1e-8f
            WeightDecay = 0.0f
            StateMode = stateMode
            StepChunkRows = max 1L cfg.OptimizerStepChunkRows
            OffloadMVToCpu = cfg.OffloadMVToCpu
            OffloadWToCpu = cfg.OffloadWToCpu
            OffloadGradToCpu = cfg.OffloadGradToCpu
            FlushEachParam = cfg.StepFlushEachParam
          }
        let st = Nvfp4Optimizer.create packedCfg trainableParams nameOfParam
        let w, m, v = Nvfp4Optimizer.stateSizeMiB st
        printfn "[Train] packed optimizer state=%s size (MiB): w=%d m=%d v=%d total=%d" cfg.OptimizerStateMode w m v (w + m + v)
        Some st
      else
        None

    let mutable globalStep = 0
    let mutable startEpoch = 1
    let trainVramSamples = ResizeArray<TrainVramSample>()

    match tryLoadCheckpoint cfg model with
    | Some state ->
      globalStep <- state.GlobalStep
      startEpoch <- state.Epoch + 1
      printfn "[Train] resumed from checkpoint: epoch=%d global_step=%d" state.Epoch state.GlobalStep
    | None ->
      ()

    try
      for epoch in startEpoch .. cfg.Epochs do
        let mutable epochLoss = 0.0f
        for stepInEpoch in 1 .. cfg.StepsPerEpoch do
          let lossValue =
            match lossMode with
            | LossMode.ScalarL1 ->
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
              profileTrainStepVram cfg trainVramSamples epoch stepInEpoch globalStep "batch_ready"

              match packedOptimizerStateOpt with
              | Some _ -> Nvfp4Optimizer.zeroGrad trainableParams
              | None ->
                optimizerOpt |> Option.iter (fun optimizer -> optimizer.zero_grad())
              profileTrainStepVram cfg trainVramSamples epoch stepInEpoch globalStep "zero_grad_done"

              let useGradCheckpoint =
                cfg.GradCheckpointChunk > 0
                && inputTensor.shape.Length = 3
                && cfg.GradCheckpointChunk < int inputTensor.shape.[1]
                && not cfg.UseKvCache

              if useGradCheckpoint then
                backwardWithSequenceRecompute
                  model
                  inputTensor
                  targetTensor
                  cfg.GradCheckpointChunk
                  computeDtype
              else
                use output =
                  if cfg.UseKvCache && model.Blocks.Length > 0 then
                    use cache = Qwen3Model.createKvCache model
                    Qwen3Model.forwardWithKvCache model cache inputTensor (Some computeDtype)
                  else
                    Qwen3Model.forward model inputTensor (Some computeDtype)
                use loss = scalarLoss output targetTensor
                loss.backward()
                use lossForRead = loss.to_type(torch.float32).cpu()
                lossForRead.item<float32>()

            | LossMode.TokenCrossEntropy ->
              if cfg.GradCheckpointChunk > 0 then
                failwith "loss-mode=ce is not yet wired with grad-checkpoint recompute path. Set --grad-ckpt-chunk 0."
              let ceSeq = max 2L cfg.SequenceLength
              use tokenIds =
                torch.randint(int64 model.VocabSize, [| cfg.BatchSize; ceSeq + 1L |], dtype = torch.int64, device = cfg.Device)
              use inputIds = tokenIds.narrow(1L, 0L, ceSeq).contiguous()
              use targetIds = tokenIds.narrow(1L, 1L, ceSeq).contiguous()
              use inputEmb = Qwen3Model.embedTokenIds model inputIds
              profileTrainStepVram cfg trainVramSamples epoch stepInEpoch globalStep "batch_ready"

              match packedOptimizerStateOpt with
              | Some _ -> Nvfp4Optimizer.zeroGrad trainableParams
              | None ->
                optimizerOpt |> Option.iter (fun optimizer -> optimizer.zero_grad())
              profileTrainStepVram cfg trainVramSamples epoch stepInEpoch globalStep "zero_grad_done"

              use output =
                if cfg.UseKvCache && model.Blocks.Length > 0 then
                  use cache = Qwen3Model.createKvCache model
                  Qwen3Model.forwardWithKvCache model cache inputEmb (Some computeDtype)
                else
                  Qwen3Model.forward model inputEmb (Some computeDtype)
              use outputNorm = InferenceBridge.rmsNormWeighted output model.FinalNorm model.RmsNormEps
              let projectToLogits (hidden: torch.Tensor) = torch.nn.functional.linear(hidden, model.LmHead)
              use loss = tokenCrossEntropyLossTensor projectToLogits outputNorm targetIds
              loss.backward()
              use lossForRead = loss.to_type(torch.float32).cpu()
              lossForRead.item<float32>()
          profileTrainStepVram cfg trainVramSamples epoch stepInEpoch globalStep "backward_done"

          match packedOptimizerStateOpt with
          | Some packedState ->
            Nvfp4Optimizer.step packedState
          | None ->
            optimizerOpt |> Option.iter (fun optimizer -> optimizer.step() |> ignore)
          profileTrainStepVram cfg trainVramSamples epoch stepInEpoch globalStep "optimizer_step_done"

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
    finally
      tryWriteTrainStepVramReport cfg trainVramSamples
      packedOptimizerStateOpt |> Option.iter (fun st -> (st :> IDisposable).Dispose())
      optimizerOpt |> Option.iter (fun optimizer -> optimizer.Dispose())
