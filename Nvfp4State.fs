namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System
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
  let private mkSyntheticBundle (outFeatures: int64) (inFeatures: int64) (device: string) =
    let kPacked = max 1L (inFeatures / 2L)
    let scaleCols = max 1L (inFeatures / 16L)
    {
      Weight = torch.randint(0L, 255L, [| outFeatures; kPacked |], dtype = torch.uint8, device = device)
      Scale = Some (torch.ones([| outFeatures; scaleCols |], dtype = torch.float32, device = device))
      Absmax = None
      QuantMap = None
    }

  let load (cfg: TrainingConfig) : Nvfp4ModelState =
    if cfg.SyntheticMode then
      {
        Layers =
          [
            { Name = "layer.0"; Bundle = mkSyntheticBundle cfg.OutFeatures cfg.InFeatures cfg.Device }
            { Name = "layer.1"; Bundle = mkSyntheticBundle cfg.OutFeatures cfg.OutFeatures cfg.Device }
          ]
        InFeatures = cfg.InFeatures
        OutFeatures = cfg.OutFeatures
      }
    else
      if not (File.Exists cfg.WeightPath) then
        raise (FileNotFoundException(sprintf "Weight file not found: %s" cfg.WeightPath))

      raise
        (NotImplementedException(
          "NVFP4 binary parser for real Qwen3 weight file is not implemented yet. Use --synthetic true for now."
        ))
