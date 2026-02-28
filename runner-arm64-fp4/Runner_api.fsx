#if INTERACTIVE
#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.9"
#r "nuget: Tokenizers.DotNet, 1.3.0"
#r "nuget: Tokenizers.DotNet.runtime.linux-arm64, 1.3.0"
#r "nuget: Microsoft.Extensions.AI, 9.9.0"
#r "/workspace/fsann/Qwen3-4B-Instruct-2507-TorchSharp-mod/Qwen3/bin/Release/net10.0/Qwen3.dll"
#load "Runner_shared.fsx"
#endif
open System
open System.Collections.Generic
open System.Diagnostics
open System.Runtime.InteropServices
open System.Reflection
open TorchSharp
open Microsoft.Extensions.AI
open Qwen3
open Qwen3.Module
open Runner_type
open Runner_shared


module Qwen3Api =
    // Stop on <|im_end|> and <|endoftext|>; avoid <|im_start|> to reduce empty outputs.
    let defaultStopTokens = new List<int>([ 151645; 151643 ])
    let imStartTokenId = 151644
    let imEndTokenId = 151645
    let mutable TimingEnabled = false
    let mutable DebugKVC = false
    let mutable DebugCudaMem = false
    let mutable KVCInputMode = Runner_type.KVCInputMode.TokenByToken

    module private CudaMem =
        [<DllImport("libcudart.so")>]
        extern int cudaMemGetInfo(nativeint& freeMem, nativeint& totalMem)

        let tryGetUsedMiB () =
            try
                let mutable freeMem = nativeint 0
                let mutable totalMem = nativeint 0
                let rc = cudaMemGetInfo(&freeMem, &totalMem)
                if rc = 0 then
                    let freeB = int64 freeMem
                    let totalB = int64 totalMem
                    let usedB = max 0L (totalB - freeB)
                    Some(usedB / (1024L * 1024L), totalB / (1024L * 1024L))
                else
                    None
            with _ ->
                None

    let private debugCudaMem (label: string) =
        if DebugCudaMem then
            match CudaMem.tryGetUsedMiB() with
            | Some (usedMiB, totalMiB) ->
                printfn "[cuda-mem] %s used=%d MiB total=%d MiB" label usedMiB totalMiB
            | None ->
                printfn "[cuda-mem] %s unavailable" label

    let private tryEmptyCudaCache () =
        try
            let ty = Type.GetType("Qwen3.FP4.Extension.CSharp.NativeMethods, Qwen3.FP4.Extension.CSharp")
            if not (isNull ty) then
                let mi = ty.GetMethod("NVFP4_empty_cache", BindingFlags.Public ||| BindingFlags.Static)
                if not (isNull mi) then
                    mi.Invoke(null, [||]) |> ignore
        with _ ->
            ()

    let timeSection (label: string) (f: unit -> 'T) =
        if not TimingEnabled then
            f()
        else
            let sw = Stopwatch.StartNew()
            let result = f()
            sw.Stop()
            printfn "[time] %s: %.1f ms" label sw.Elapsed.TotalMilliseconds
            result

    let runWithTimeout (timeoutMs: int) (f: unit -> 'T) : 'T =
        let task = System.Threading.Tasks.Task.Run(fun () -> f())
        if task.Wait(timeoutMs) then
            task.Result
        else
            failwithf "Operation timed out after %d ms" timeoutMs

    /// Apply dtype to model weights (in-place).
    let applyDtype (dtype: torch.ScalarType) (m: torch.nn.Module) =
        match dtype with
        | torch.ScalarType.Float16 -> m.half() |> ignore
        | torch.ScalarType.Float32 -> m.float() |> ignore
        | torch.ScalarType.Float64 -> m.double() |> ignore
        | _ -> ()

    /// Select MoE path only when model config explicitly declares experts.
    let shouldUseMoE (cfg: Qwen3Config) =
        cfg.NumExperts.HasValue && cfg.NumExperts.Value > 0

    /// Initialize model, tokenizer, and optional KV cache.
    let init (modelDir: string) (weightOverride: string option) (quantHint: string option) (device: string) (dtype: torch.ScalarType) (kvCacheOut: bool) (historyMode: HistoryMode) =
        let paths = Paths.resolve modelDir weightOverride quantHint
        torch.InitializeDeviceType(DeviceType.CUDA)
        torch.set_default_dtype(dtype)
        debugCudaMem "init.before_load"

        let cfg = Loader.loadConfig paths.ConfigPath
        if isNull cfg then invalidOp "failed to parse config.json"

        let quant = quantHint |> Option.orElse (Option.ofObj cfg.Quantization)
        match quant with
        | Some q when q.Equals("fp4", StringComparison.OrdinalIgnoreCase) ->
            cfg.Quantization <- "fp4"
        | Some q when q.Equals("nf4", StringComparison.OrdinalIgnoreCase) || q.Equals("4bit", StringComparison.OrdinalIgnoreCase) || q.Equals("int4", StringComparison.OrdinalIgnoreCase) ->
            cfg.Quantization <- "4bit"
        | _ -> ()
        let weightPath = Paths.resolveWeightPath paths.ModelDir quant weightOverride
        let paths = { paths with WeightPath = weightPath }
        Loader.ensureFiles paths

        let tokenizer = new Qwen3Tokenizer(paths.TokenizerPath)

        let model, pipeline =
            if shouldUseMoE cfg then
                let m = new Qwen3MoE(cfg)
                m.eval() |> ignore
                applyDtype dtype m
                m.LoadWeights(paths.WeightPath)
                m.``to``(device) |> ignore
                debugCudaMem "init.after_model_to_device"
                MoEModel m, MoE (new Qwen3MoEPipeline(m, tokenizer, device))
            else
                let m = new Qwen3Dense(cfg)
                m.eval() |> ignore
                applyDtype dtype m
                m.LoadWeights(paths.WeightPath)
                m.``to``(device) |> ignore
                debugCudaMem "init.after_model_to_device"
                DenseModel m, Dense (new Qwen3DensePipeline(m, tokenizer, device))

        match quant with
        | Some q when q.Equals("fp4", StringComparison.OrdinalIgnoreCase) ->
            tryEmptyCudaCache()
            debugCudaMem "init.after_empty_cache"
        | _ -> ()

        let maxContext =
            if cfg.MaxPositionEmbeddings > 0 then cfg.MaxPositionEmbeddings else 0
        let kvCache =
            if kvCacheOut then
                Some (new KVC(cfg.NumHiddenLayers, maxContext))
            else
                None

        { Config = cfg
          Tokenizer = tokenizer
          Model = model
          Pipeline = pipeline
          Device = device
          StopTokens = defaultStopTokens
          MaxContext = maxContext
          KVCache = kvCache
          HistoryMode = historyMode
          PromptTokens = []
          History = [] }

    /// Build prompt string from chat messages (template).
    let buildPrompt (messages: ChatMessage list) =
        Qwen3ChatTemplateBuilder.Instance.BuildPrompt(messages, null, true)

    let assistantEndPrefix = $"<|im_end|>{Qwen3ChatTemplateBuilder.Newline}"

    let private clip (text: string) (limit: int) =
        if text.Length <= limit then text else text.Substring(0, limit)

    let private trimAtImStart (tokens: int list) =
        match tokens |> List.tryFindIndex (fun t -> t = imStartTokenId) with
        | Some idx -> tokens |> List.take idx, true
        | None -> tokens, false

    /// Generate full response (no KV cache).
    let generate (session: Session) (prompt: string) (opt: GenOptions) =
        let seed =
            match opt.Seed with
            | Some s -> Nullable(s)
            | None -> Nullable()
        match session.Pipeline with
        | Dense p -> p.Generate(prompt, session.StopTokens, opt.MaxTokens, opt.Temperature, opt.TopP, seed)
        | MoE p -> p.Generate(prompt, session.StopTokens, opt.MaxTokens, opt.Temperature, opt.TopP, seed)

    /// Generate streaming response (no KV cache).
    let generateStreaming (session: Session) (prompt: string) (opt: GenOptions) =
        let seed =
            match opt.Seed with
            | Some s -> Nullable(s)
            | None -> Nullable()
        match session.Pipeline with
        | Dense p -> p.GenerateStreaming(prompt, session.StopTokens, opt.MaxTokens, opt.Temperature, opt.TopP, seed)
        | MoE p -> p.GenerateStreaming(prompt, session.StopTokens, opt.MaxTokens, opt.Temperature, opt.TopP, seed)

    /// Encode text into token ids.
    let encode (session: Session) (text: string) =
        session.Tokenizer.Encode(text)

    /// Encode text into F# int list.
    let encodeList (session: Session) (text: string) =
        encode session text |> Seq.toList

    /// Convert token list to [1, T] tensor on target device.
    let toTensor (session: Session) (ids: List<int>) =
        torch.tensor(ids |> Seq.toArray, dtype = torch.ScalarType.Int64, device = session.Device).unsqueeze(0)

    /// Forward tokens through model (no KV cache).
    let forwardTokens (session: Session) (ids: List<int>) =
        use inputIds = toTensor session ids
        match session.Model with
        | DenseModel m -> m.forward(inputIds)
        | MoEModel m -> m.forward(inputIds)

    /// Forward tokens using F# list input (no KV cache).
    let forwardTokensList (session: Session) (ids: int list) =
        forwardTokens session (new List<int>(ids))

    /// Forward raw text through model (no KV cache).
    let forwardText (session: Session) (text: string) =
        let ids = encode session text
        forwardTokens session ids

    let rec private isPrefix (prefix: int list) (full: int list) =
        match prefix, full with
        | [], _ -> true
        | p :: ps, f :: fs when p = f -> isPrefix ps fs
        | _ -> false

    let private moveCacheToDevice (session: Session) (cache: KVC) =
        if not (cache.Device.Equals(session.Device, StringComparison.OrdinalIgnoreCase)) then
            cache.MoveTo(session.Device)
        cache

    let private endsWith (full: int list) (suffix: int list) =
        if suffix.IsEmpty then true
        elif full.Length < suffix.Length then false
        else
            full
            |> List.skip (full.Length - suffix.Length)
            |> (=) suffix

    let private diffTokenSeq (session: Session) (label: string) (expected: int list) (actual: int list) =
        let rec firstDiff i e a =
            match e, a with
            | [], [] -> None
            | [], _ -> Some i
            | _, [] -> Some i
            | eh :: et, ah :: at ->
                if eh = ah then firstDiff (i + 1) et at else Some i
        match firstDiff 0 expected actual with
        | None ->
            printfn "[kvc] token_diff %s: OK (len=%d)" label expected.Length
        | Some idx ->
            let takeN n xs = xs |> List.skip idx |> List.truncate n
            let expSlice = takeN 12 expected
            let actSlice = takeN 12 actual
            let expText = session.Tokenizer.Decode(new List<int>(expSlice))
            let actText = session.Tokenizer.Decode(new List<int>(actSlice))
            let expTok = expected |> List.tryItem idx
            let actTok = actual |> List.tryItem idx
            printfn "[kvc] token_diff %s: expected=%d actual=%d first=%d expTok=%A actTok=%A" label expected.Length actual.Length idx expTok actTok
            printfn "[kvc] token_diff %s: expText=%s" label (clip expText 120)
            printfn "[kvc] token_diff %s: actText=%s" label (clip actText 120)

    let private prefillTokensIntoCache (session: Session) (cache: KVC) (tokens: int list) =
        tokens
        |> List.fold (fun acc token ->
            use inputIds =
                timeSection "kvc.input_tensor_prefill" (fun () ->
                    torch.tensor([| token |], dtype = torch.ScalarType.Int64, device = session.Device).unsqueeze(0))
            match session.Model with
            | DenseModel m ->
                let struct (logits, newCache) = m.ForwardWithKVCache(inputIds, acc, session.MaxContext)
                logits.Dispose()
                newCache
            | MoEModel m ->
                let struct (logits, newCache) = m.ForwardWithKVCache(inputIds, acc, session.MaxContext)
                logits.Dispose()
                newCache) cache

    let private prefillChunkIntoCache (session: Session) (cache: KVC) (tokens: int list) =
        if tokens.IsEmpty then
            cache
        else
            use inputIds =
                timeSection "kvc.input_tensor_prefill_chunk" (fun () ->
                    torch.tensor(tokens |> List.toArray, dtype = torch.ScalarType.Int64, device = session.Device).unsqueeze(0))
            match session.Model with
            | DenseModel m ->
                let struct (logits, newCache) = m.ForwardWithKVCache(inputIds, cache, session.MaxContext)
                logits.Dispose()
                newCache
            | MoEModel m ->
                let struct (logits, newCache) = m.ForwardWithKVCache(inputIds, cache, session.MaxContext)
                logits.Dispose()
                newCache

    /// Generate response using full prompt; never uses KV cache.
    let chat (session: Session) (messages: ChatMessage list) (opt: GenOptions) =
        let prompt = buildPrompt messages
        generate session prompt opt

    /// Generate response using KV cache; only new messages are fed.
    let chatWithKVC (session: Session) (messages: ChatMessage list) (opt: GenOptions) =
        debugCudaMem "chat.enter"
        let cache =
            timeSection "kvc.cache_get_or_create" (fun () ->
                match session.KVCache with
                | Some c -> c
                | None ->
                    let created = new KVC(session.Config.NumHiddenLayers, session.MaxContext)
                    session.KVCache <- Some created
                    created)

        let keepHistory =
            match session.HistoryMode with
            | HistoryMode.Tokens -> false
            | _ -> true
        let trackHistory = keepHistory || DebugKVC

        if trackHistory then
            session.History <- session.History @ messages
        let deltaTokens =
            timeSection "kvc.delta_tokens" (fun () ->
                match session.HistoryMode with
                | HistoryMode.Tokens ->
                    let hasSystem = messages |> List.exists (fun m -> m.Role = ChatRole.System)
                    if hasSystem && session.PromptTokens.Length > 0 then
                        printfn "Reset triggered! System message with non-empty cache."
                        cache.Reset()
                        session.PromptTokens <- []
                    let deltaPrompt = timeSection "kvc.build_prompt_delta" (fun () -> buildPrompt messages)
                    let endTokens = encodeList session assistantEndPrefix
                    let lastIsImEnd =
                        endsWith session.PromptTokens endTokens
                    let deltaText =
                        if session.PromptTokens.Length > 0 && not lastIsImEnd then
                            assistantEndPrefix + deltaPrompt
                        else
                            deltaPrompt
                    timeSection "kvc.encode_delta" (fun () -> encodeList session deltaText)
                | HistoryMode.Messages
                | HistoryMode.Both ->
                    let prompt = timeSection "kvc.build_prompt" (fun () -> buildPrompt session.History)
                    let fullTokens = timeSection "kvc.encode_full" (fun () -> encodeList session prompt)
                    if isPrefix session.PromptTokens fullTokens then
                        fullTokens |> List.skip session.PromptTokens.Length
                    else
                        printfn "Reset triggered! Not incremental chat session!"
                        cache.Reset()
                        session.PromptTokens <- []
                        fullTokens)
        if DebugKVC then
            let deltaText = session.Tokenizer.Decode(new List<int>(deltaTokens))
            printfn "[kvc] deltaTokens=%d deltaTextHead=%s" deltaTokens.Length (clip deltaText 200)
            if trackHistory then
                let expectedFull = encodeList session (buildPrompt session.History)
                let combined = session.PromptTokens @ deltaTokens
                diffTokenSeq session "full_vs_combined" expectedFull combined

        let output, outCache, generatedTokens =
            if deltaTokens.Length = 0 then
                "", cache, []
            else
                let cacheOnDevice = timeSection "kvc.cache_to_device" (fun () -> moveCacheToDevice session cache)
                debugCudaMem "chat.after_cache_to_device"
                let ifFullPrefill = cacheOnDevice.SeqLen = 0
                let prefillByTokenList, prefillByChunkList, genTokens =
                    if ifFullPrefill then
                        match KVCInputMode with
                        | Runner_type.KVCInputMode.TokenByToken ->
                            if deltaTokens.Length > 1 then
                                deltaTokens |> List.take (deltaTokens.Length - 1), [], [ deltaTokens.[deltaTokens.Length - 1] ]
                            else
                                [], [], deltaTokens
                        | Runner_type.KVCInputMode.PromptByPrompt ->
                            // Full-prefill as one chunk, then generate from the last prompt token.
                            if deltaTokens.Length > 1 then
                                [], (deltaTokens |> List.take (deltaTokens.Length - 1)), [ deltaTokens.[deltaTokens.Length - 1] ]
                            else
                                [], [], deltaTokens
                    else
                        match KVCInputMode with
                        | Runner_type.KVCInputMode.TokenByToken ->
                            if deltaTokens.Length > 1 then
                                deltaTokens |> List.take (deltaTokens.Length - 1), [], [ deltaTokens.[deltaTokens.Length - 1] ]
                            else
                                [], [], deltaTokens
                        | Runner_type.KVCInputMode.PromptByPrompt ->
                            // Incremental prompt chunk prefill, then generate from the last prompt token.
                            if deltaTokens.Length > 1 then
                                [], (deltaTokens |> List.take (deltaTokens.Length - 1)), [ deltaTokens.[deltaTokens.Length - 1] ]
                            else
                                [], [], deltaTokens

                if DebugKVC then
                    let modeStr =
                        match KVCInputMode with
                        | Runner_type.KVCInputMode.TokenByToken -> "tbt"
                        | Runner_type.KVCInputMode.PromptByPrompt -> "pbp"
                    printfn "[kvc] mode=%s full_prefill=%b prefill_token=%d prefill_chunk=%d gen_input=%d" modeStr ifFullPrefill prefillByTokenList.Length prefillByChunkList.Length genTokens.Length

                let cacheAfterPrefillToken = prefillTokensIntoCache session cacheOnDevice prefillByTokenList
                let cacheAfterPrefill = prefillChunkIntoCache session cacheAfterPrefillToken prefillByChunkList
                debugCudaMem "chat.after_prefill"

                use inputIds =
                    timeSection "kvc.input_tensor" (fun () ->
                        torch.tensor(genTokens |> List.toArray, dtype = torch.ScalarType.Int64, device = session.Device).unsqueeze(0))
                let seed =
                    match opt.Seed with
                    | Some s -> Nullable(s)
                    | None -> Nullable()
                let struct (tokens, updated) : struct (List<int> * KVC) =
                    timeSection "kvc.generate" (fun () ->
                        match session.Model with
                        | DenseModel m ->
                            m.GenerateWithKVCache(inputIds, cacheAfterPrefill, session.MaxContext, opt.MaxTokens, session.StopTokens, opt.Temperature, opt.TopP, seed)
                        | MoEModel m ->
                            m.GenerateWithKVCache(inputIds, cacheAfterPrefill, session.MaxContext, opt.MaxTokens, session.StopTokens, opt.Temperature, opt.TopP, seed))
                let rawTokens = tokens |> Seq.toList
                let keptTokens, trimmedAtImStart = trimAtImStart rawTokens
                if trimmedAtImStart && DebugKVC then
                    printfn "[kvc] trim generated at <|im_start|>: raw=%d kept=%d" rawTokens.Length keptTokens.Length
                let text = timeSection "kvc.decode" (fun () -> session.Tokenizer.Decode(new List<int>(keptTokens)))
                debugCudaMem "chat.after_generate"
                text, updated, keptTokens

        session.KVCache <- Some outCache
        if trackHistory then
            let assistant = new ChatMessage(role = ChatRole.Assistant, content = output)
            session.History <- session.History @ [ assistant ]
        if deltaTokens.Length > 0 then
            session.PromptTokens <- session.PromptTokens @ deltaTokens @ generatedTokens
            // Append <|im_end|> + newline tokens to keep cache/prompt aligned with chat template
            let endTokens = encodeList session assistantEndPrefix
            let lastHasEnd = endsWith session.PromptTokens endTokens
            if not lastHasEnd then
                let cacheOnDevice = timeSection "kvc.cache_to_device_end" (fun () -> moveCacheToDevice session outCache)
                let updatedCache = prefillTokensIntoCache session cacheOnDevice endTokens
                session.KVCache <- Some updatedCache
                session.PromptTokens <- session.PromptTokens @ endTokens
        ChatSessionLog.append messages output
        debugCudaMem "chat.exit"
        output

    /// Generate streaming response (no KV cache).
    let chatStreaming (session: Session) (messages: ChatMessage list) (opt: GenOptions) =
        let prompt = buildPrompt messages
        generateStreaming session prompt opt
