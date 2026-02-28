#if INTERACTIVE
#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.9"
#r "nuget: Tokenizers.DotNet, 1.3.0"
#r "nuget: Tokenizers.DotNet.runtime.linux-arm64, 1.3.0"
#r "nuget: Microsoft.Extensions.AI, 9.9.0"
#r "/workspace/fsann/Qwen3-4B-Instruct-2507-TorchSharp-mod/Qwen3/bin/Release/net10.0/Qwen3.dll"
#load "Runner_type.fsx"
#endif
open System
open System.IO
open System.Text.Json
open Microsoft.Extensions.AI
open Runner_type

module ChatSessionJson =
    let private tryParseEntry (line: string) =
        let trimmed = line.Trim()
        if String.IsNullOrWhiteSpace trimmed then
            None
        else
            try
                let entry = JsonSerializer.Deserialize<ChatSessionJsonEntry>(trimmed)
                if isNull (box entry) then None else Some entry
            with _ ->
                None

    let readEntries (path: string) =
        if not (File.Exists path) then
            []
        else
            File.ReadAllLines(path)
            |> Array.choose tryParseEntry
            |> Array.toList

    let private parseRole (value: string) =
        match value.Trim().ToLowerInvariant() with
        | "assistant" -> ChatRole.Assistant
        | "system" -> ChatRole.System
        | "user" -> ChatRole.User
        | _ -> ChatRole.User

    let readMessages (path: string) =
        let entries = readEntries path
        let lastIndex = entries.Length - 1
        entries
        |> List.mapi (fun i entry ->
            let msgs =
                entry.messages
                |> List.map (fun m -> new ChatMessage(parseRole m.role, m.text))
            if i = lastIndex then
                msgs
            else
                msgs @ [ new ChatMessage(ChatRole.Assistant, entry.output) ])
        |> List.collect id
