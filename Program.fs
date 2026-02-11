namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System

module Program =
  [<EntryPoint>]
  let main argv =
    if argv |> Array.exists (fun a -> a = "--help" || a = "-h") then
      Cli.printUsage()
      0
    else
      try
        let cfg = Cli.parse argv
        printfn "[Init] project=Qwen3-4B-Instruct-2507-TorchSharp.fs"
        printfn "[Init] modelDir=%s" cfg.ModelDir
        printfn "[Init] weight=%s" cfg.WeightPath
        printfn "[Init] device=%s synthetic=%b" cfg.Device cfg.SyntheticMode
        printfn "[Init] maxLayers=%d requested(in=%d,out=%d)" cfg.MaxLayers cfg.InFeatures cfg.OutFeatures
        printfn
          "[Init] lr=%f checkpointDir=%s saveEverySteps=%d resume=%b strictLoad=%b"
          cfg.LearningRate
          cfg.CheckpointDir
          cfg.SaveEverySteps
          cfg.ResumeFromCheckpoint
          cfg.StrictLoad

        let state = Nvfp4State.load cfg
        let model = Qwen3Model.create cfg state
        try
          Trainer.run cfg model
        finally
          Qwen3Model.dispose model
        0
      with ex ->
        eprintfn "[Error] %s" ex.Message
        1
