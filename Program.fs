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

        let state = Nvfp4State.load cfg
        use model = Qwen3Model.create cfg state
        Trainer.run cfg model
        0
      with ex ->
        eprintfn "[Error] %s" ex.Message
        1
