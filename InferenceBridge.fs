namespace Qwen3_4B_Instruct_2507_TorchSharp_fs

open System
open System.IO
open System.Collections.Generic
open System.Text.Json
open TorchSharp
open Microsoft.Extensions.AI
open Qwen3
open Qwen3.Module

type InferencePipeline =
  | Dense of Qwen3DensePipeline
  | MoE of Qwen3MoEPipeline

type InferenceModel =
  | DenseModel of Qwen3Dense
  | MoEModel of Qwen3MoE

type InferenceSession =
  {
    Config: Qwen3Config
    Tokenizer: Qwen3Tokenizer
    Model: InferenceModel
    Pipeline: InferencePipeline
    Device: string
    StopTokens: List<int>
  }

type InferenceGenOptions =
  {
    MaxTokens: int
    Temperature: float32
    TopP: float32
    Seed: int option
  }

module InferenceBridge =
  let defaultStopTokens = new List<int>([ 151645; 151643 ])

  let private isNf4Like (q: string) =
    q.Equals("nf4", StringComparison.OrdinalIgnoreCase)
    || q.Equals("4bit", StringComparison.OrdinalIgnoreCase)
    || q.Equals("int4", StringComparison.OrdinalIgnoreCase)

  let private isFp4Like (q: string) = q.Equals("fp4", StringComparison.OrdinalIgnoreCase)

  let private applyDtype (dtype: torch.ScalarType) (m: torch.nn.Module) =
    match dtype with
    | torch.ScalarType.Float16 -> m.half() |> ignore
    | torch.ScalarType.Float32 -> m.float() |> ignore
    | torch.ScalarType.Float64 -> m.double() |> ignore
    | _ -> ()

  let private shouldUseMoE (cfg: Qwen3Config) =
    cfg.NumExperts.HasValue && cfg.NumExperts.Value > 0

  let private ensureFiles (paths: string list) =
    let missing = paths |> List.filter (fun p -> not (File.Exists p))
    if missing.Length > 0 then
      invalidOp (String.Join(Environment.NewLine, missing |> List.map (fun p -> $"missing: {p}")))

  let private chooseWeightPath (modelDir: string) (quantHint: string option) (weightOverride: string option) =
    let pickByKeyword (keyword: string) (paths: string array) =
      paths
      |> Array.tryFind (fun p -> p.Contains(keyword, StringComparison.OrdinalIgnoreCase))

    match weightOverride with
    | Some w when not (String.IsNullOrWhiteSpace w) ->
      if Path.IsPathRooted w then w else Path.Combine(modelDir, w)
    | _ ->
      let candidates = Directory.GetFiles(modelDir, "*.dat")
      if candidates.Length = 0 then
        Path.Combine(modelDir, "Qwen3-4B-Instruct-2507-fp16.dat")
      else
        match quantHint with
        | Some q when isFp4Like q ->
          pickByKeyword "nvfp4" candidates
          |> Option.orElse (pickByKeyword "fp4" candidates)
          |> Option.orElse (Array.tryHead candidates)
          |> Option.get
        | Some q when isNf4Like q ->
          pickByKeyword "nf4" candidates
          |> Option.orElse (pickByKeyword "4bit" candidates)
          |> Option.orElse (Array.tryHead candidates)
          |> Option.get
        | _ ->
          pickByKeyword "fp16" candidates
          |> Option.orElse (Array.tryHead candidates)
          |> Option.get

  let init
    (modelDir: string)
    (weightOverride: string option)
    (quantHint: string option)
    (device: string)
    (dtype: torch.ScalarType)
    =
    let resolvedModelDir =
      if String.IsNullOrWhiteSpace modelDir then Defaults.modelDir else modelDir.Trim()
    let configPath = Path.Combine(resolvedModelDir, "config.json")
    let tokenizerPath = Path.Combine(resolvedModelDir, "tokenizer.json")
    let weightPath = chooseWeightPath resolvedModelDir quantHint weightOverride
    ensureFiles [ configPath; tokenizerPath; weightPath ]

    // Match existing runner behavior: initialize CUDA backend unconditionally.
    torch.InitializeDeviceType(DeviceType.CUDA)
    torch.set_default_dtype(dtype)

    let cfg = JsonSerializer.Deserialize<Qwen3Config>(File.ReadAllText(configPath))
    if isNull cfg then invalidOp "failed to parse config.json"

    match quantHint with
    | Some q when isFp4Like q -> cfg.Quantization <- "fp4"
    | Some q when isNf4Like q -> cfg.Quantization <- "4bit"
    | _ -> ()

    let tokenizer = new Qwen3Tokenizer(tokenizerPath)

    let model, pipeline =
      if shouldUseMoE cfg then
        let m = new Qwen3MoE(cfg)
        m.eval() |> ignore
        applyDtype dtype m
        m.LoadWeights(weightPath)
        m.``to``(device) |> ignore
        MoEModel m, MoE(new Qwen3MoEPipeline(m, tokenizer, device))
      else
        let m = new Qwen3Dense(cfg)
        m.eval() |> ignore
        applyDtype dtype m
        m.LoadWeights(weightPath)
        m.``to``(device) |> ignore
        DenseModel m, Dense(new Qwen3DensePipeline(m, tokenizer, device))

    {
      Config = cfg
      Tokenizer = tokenizer
      Model = model
      Pipeline = pipeline
      Device = device
      StopTokens = defaultStopTokens
    }

  let buildPrompt (messages: ChatMessage list) =
    Qwen3ChatTemplateBuilder.Instance.BuildPrompt(messages, null, true)

  let generate (session: InferenceSession) (prompt: string) (opt: InferenceGenOptions) =
    let seed =
      match opt.Seed with
      | Some s -> Nullable(s)
      | None -> Nullable()
    match session.Pipeline with
    | Dense p -> p.Generate(prompt, session.StopTokens, opt.MaxTokens, opt.Temperature, opt.TopP, seed)
    | MoE p -> p.Generate(prompt, session.StopTokens, opt.MaxTokens, opt.Temperature, opt.TopP, seed)

  let chat (session: InferenceSession) (messages: ChatMessage list) (opt: InferenceGenOptions) =
    messages |> buildPrompt |> fun prompt -> generate session prompt opt

  let dispose (session: InferenceSession) =
    match session.Model with
    | DenseModel m -> m.Dispose()
    | MoEModel m -> m.Dispose()
    match box session.Tokenizer with
    | :? IDisposable as d -> d.Dispose()
    | _ -> ()
