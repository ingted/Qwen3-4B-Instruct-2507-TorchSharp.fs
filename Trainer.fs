namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open TorchSharp

module Trainer =
  let private createBatch (batchSize: int64) (inFeatures: int64) (outFeatures: int64) (device: string) =
    let input = torch.randn([| batchSize; inFeatures |], dtype = torch.float16, device = device)
    let target = torch.randn([| batchSize; outFeatures |], dtype = torch.float16, device = device)
    input, target

  let private scalarLoss (output: TorchSharp.torch.Tensor) (target: TorchSharp.torch.Tensor) =
    let targetForLoss =
      if target.dtype = output.dtype then target else target.to_type(output.dtype)
    let diff = output - targetForLoss
    diff.abs().mean()

  let run (cfg: TrainingConfig) (model: Qwen3Nvfp4Model) =
    printfn "[Train] mode=NVFP4(kernels only), epochs=%d, steps/epoch=%d, batch=%d" cfg.Epochs cfg.StepsPerEpoch cfg.BatchSize
    printfn "[Train] note: this scaffold currently runs forward/loss loop; optimizer update path is pending."
    printfn "[Train] features in=%d out=%d layers=%d" model.InFeatures model.OutFeatures model.Layers.Length

    for epoch in 1 .. cfg.Epochs do
      let mutable epochLoss = 0.0f
      for _step in 1 .. cfg.StepsPerEpoch do
        let input, target = createBatch cfg.BatchSize model.InFeatures model.OutFeatures cfg.Device
        let output = model.Forward(input)
        let loss = scalarLoss output target
        let lossValue = loss.to_type(torch.float32).cpu().item<float32>()
        epochLoss <- epochLoss + lossValue

      let avgLoss = epochLoss / float32 cfg.StepsPerEpoch
      printfn "[Train] epoch=%d avg_loss=%f" epoch avgLoss
