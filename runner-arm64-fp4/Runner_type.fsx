#if INTERACTIVE
#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.9"
#r "nuget: Tokenizers.DotNet, 1.3.0"
#r "nuget: Tokenizers.DotNet.runtime.linux-arm64, 1.3.0"
#r "nuget: Microsoft.Extensions.AI, 9.9.0"
#r "/workspace/fsann/Qwen3-4B-Instruct-2507-TorchSharp-mod/Qwen3/bin/Release/net10.0/Qwen3.dll"
#endif

open System
open System.Collections.Generic
open Microsoft.Extensions.AI
open Qwen3
open Qwen3.Module
open TorchSharp

type KVC = Qwen3KVCache

type ModelPaths =
    { ModelDir: string
      ConfigPath: string
      TokenizerPath: string
      WeightPath: string }

type Pipeline =
    | Dense of Qwen3DensePipeline
    | MoE of Qwen3MoEPipeline

type Model =
    | DenseModel of Qwen3Dense
    | MoEModel of Qwen3MoE

type HistoryMode =
    | Messages
    | Tokens
    | Both

type KVCInputMode =
    | TokenByToken
    | PromptByPrompt

type Session =
    { Config: Qwen3Config
      Tokenizer: Qwen3Tokenizer
      Model: Model
      Pipeline: Pipeline
      Device: string
      StopTokens: List<int>
      MaxContext: int
      HistoryMode: HistoryMode
      mutable KVCache: KVC option
      mutable PromptTokens: int list
      mutable History: ChatMessage list }
      with
        member session.readFromGPU () =
            match session.KVCache with
            | None -> None
            | Some cache ->
                if not (cache.Device.Equals("cpu", StringComparison.OrdinalIgnoreCase)) then
                    cache.MoveTo("cpu")
                session.KVCache <- Some cache
                Some cache

type GenOptions =
    { MaxTokens: int
      Temperature: float32
      TopP: float32
      Seed: int option }

type ScriptArgs =
    { ModelDir: string
      WeightPath: string option
      Quant: string option
      Device: string
      DType: string
      MaxTokens: int
      Temperature: float32
      TopP: float32
      Seed: int option
      Prompt: string
      SystemPrompt: string option
      CheckLogits: bool
      Q4Kernel: bool option
      Q4Cache: bool option
      KVCacheOut: bool
      KVCInputMode: KVCInputMode
      HistoryMode: HistoryMode
      DebugKVC: bool
      IgnoreEOS: bool
      Timing: bool
      TeeChatSession: string option
      TeeChatSessionJson: string option
      PromptFromChatSession: bool }

type ChatSessionJsonMessage =
    { role: string
      text: string }

type ChatSessionJsonEntry =
    { ts: string
      messages: ChatSessionJsonMessage list
      output: string }
