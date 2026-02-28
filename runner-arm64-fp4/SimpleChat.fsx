#if INTERACTIVE
#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.9"
#r "nuget: Tokenizers.DotNet, 1.3.0"
#r "nuget: Tokenizers.DotNet.runtime.linux-arm64, 1.3.0"
#r "nuget: Microsoft.Extensions.AI, 9.9.0"
#r "/workspace/fsann/Qwen3-4B-Instruct-2507-TorchSharp-mod/Qwen3/bin/Release/net10.0/Qwen3.dll"
#load "Runner_api.fsx"
//#load "chatSessionJson.fsx"
#endif
open System
open Microsoft.Extensions.AI
open Runner_api
open Runner_type
open Runner_shared
module SimpleChat =
    let runFullFollowup (timeMs: string -> (unit -> string) -> string) (sess: Session) (opts: GenOptions) (fullHistory: ChatMessage list ref) (debugPrompt: bool) (label: string) (userMsgs: ChatMessage list) (timeoutMs: int) =
        let prompt = Qwen3Api.buildPrompt (fullHistory.Value @ userMsgs)
        if debugPrompt then
            let clipLen = 400
            let show = if prompt.Length <= clipLen then prompt else prompt.Substring(0, clipLen)
            printfn "[prompt] %s" show
        let outX = timeMs label (fun () ->
            Qwen3Api.runWithTimeout timeoutMs (fun () -> Qwen3Api.generate sess prompt opts))
        fullHistory.Value <- fullHistory.Value @ userMsgs
        fullHistory.Value <- fullHistory.Value @ [ ChatMessage(ChatRole.Assistant, outX) ]
        ChatSessionLog.append userMsgs outX
        outX

    let simpleChat (timeMs: string -> (unit -> string) -> string) (sess: Session) (opts: GenOptions) (fullHistory: ChatMessage list ref) (kvEnabled: bool option) (kvCacheOut: bool) (debugPrompt: bool) tag timeoutMs (msg: string) =
        let mList = [ ChatMessage(ChatRole.User, msg) ]
        if (kvEnabled.IsSome && kvEnabled.Value) || (kvCacheOut && not (kvEnabled.IsSome && not kvEnabled.Value)) then
            let out = timeMs $"chatWithKVC ({tag})" (fun () ->
                Qwen3Api.runWithTimeout timeoutMs (fun () -> Qwen3Api.chatWithKVC sess mList opts))
            printfn "[%A]out: %s" tag out
            Some (mList, out)
        else
            let out = runFullFollowup timeMs sess opts fullHistory debugPrompt $"generate ({tag})" mList timeoutMs
            printfn "[%A]out: %s" tag out
            Some (mList, out)
