open System
open System.IO
open System.Diagnostics
open System.Runtime.InteropServices
open System.Text.RegularExpressions

let torchLibDir = "/usr/local/lib/python3.12/dist-packages/torch/lib"
let cudaLibDir = "/usr/local/cuda/lib64"

printfn "--- 正在從系統路徑載入 Native 庫 ---"

let loadLib path name =
    let fullPath = Path.Combine(path, name)
    if File.Exists(fullPath) then
        try
            NativeLibrary.Load(fullPath) |> ignore
            printfn $"✅ 成功載入: {fullPath}"
            true
        with ex ->
            printfn $"❌ 載入失敗 {fullPath}: {ex.Message}"
            false
    else
        printfn $"⚠️ 找不到檔案: {fullPath}"
        false

let _ = loadLib cudaLibDir "libcudart.so"
let _ = loadLib torchLibDir "libtorch_cuda.so"
let _ = loadLib torchLibDir "libtorch.so"

printfn "--- 正在載入 NVFP4 擴展庫 ---"
let _ = loadLib "/workspace/nvfp4_native" "libNVFP4.so"

printfn "--- Native 庫初始化完成 ---"
let tryGetTorchSharpVersionFromScript () =
    try
        let scriptPathOpt =
            Environment.GetCommandLineArgs()
            |> Array.tryFind (fun arg ->
                arg.EndsWith(".fsx", StringComparison.OrdinalIgnoreCase) && File.Exists(arg))
        match scriptPathOpt with
        | Some scriptPath ->
            File.ReadLines(scriptPath)
            |> Seq.tryPick (fun line ->
                let m = Regex.Match(line, "#r\\s+\"nuget:\\s*FAkka\\.TorchSharp\\.DGX\\s*,\\s*([^\"]+)\"", RegexOptions.IgnoreCase)
                if m.Success then Some(m.Groups.[1].Value.Trim()) else None)
        | None -> None
    with _ ->
        None

let resolveTorchSharpNativePath () =
    let explicitPath = Environment.GetEnvironmentVariable("FAKKA_TORCHSHARP_NATIVE")
    if not (String.IsNullOrWhiteSpace explicitPath) && File.Exists(explicitPath) then
        Some (explicitPath, "env:FAKKA_TORCHSHARP_NATIVE")
    else
        let preferredVersion =
            let fromEnv = Environment.GetEnvironmentVariable("FAKKA_TORCHSHARP_VERSION")
            if not (String.IsNullOrWhiteSpace fromEnv) then
                Some (fromEnv.Trim())
            else
                match tryGetTorchSharpVersionFromScript () with
                | Some ver -> Some ver
                | None -> Some "26.1.0-py3.9"

        let packageRoot = "/root/.nuget/packages/fakka.torchsharp.dgx"
        let tryParseVersionKey (s: string) =
            let m = Regex.Match(s, @"^(\d+)\.(\d+)\.(\d+)-py(\d+)(?:\.(\d+))?$")
            if m.Success then
                let g (i: int) = Int32.Parse(m.Groups.[i].Value)
                let pyPatch =
                    if m.Groups.[5].Success then Int32.Parse(m.Groups.[5].Value) else 0
                Some (g 1, g 2, g 3, g 4, pyPatch)
            else
                None

        let tryByExactVersion (verName: string) =
            let libPath = Path.Combine(packageRoot, verName, "runtimes/linux-arm64/native/libLibTorchSharp.so")
            if File.Exists(libPath) then Some libPath else None

        if Directory.Exists(packageRoot) then
            match preferredVersion |> Option.bind tryByExactVersion with
            | Some libPath ->
                Some (libPath, $"version:{preferredVersion.Value}")
            | None ->
                match preferredVersion with
                | Some ver ->
                    printfn $"⚠️ 指定版本 {ver} 的 native lib 不存在，改用本機可用最高版。"
                | None -> ()

                Directory.GetDirectories(packageRoot)
                |> Array.choose (fun verDir ->
                    let verName = Path.GetFileName(verDir)
                    let libPath = Path.Combine(verDir, "runtimes/linux-arm64/native/libLibTorchSharp.so")
                    if File.Exists(libPath) then
                        match tryParseVersionKey verName with
                        | Some key -> Some (key, verName, libPath)
                        | None -> None
                    else
                        None)
                |> Array.sortBy (fun (k, _, _) -> k)
                |> Array.tryLast
                |> Option.map (fun (_, ver, path) -> path, $"highest:{ver}")
        else
            None

match resolveTorchSharpNativePath() with
| Some (nugetLibPath, reason) ->
    try
        NativeLibrary.Load(nugetLibPath) |> ignore
        printfn $"✅ 成功載入 (NuGet/{reason}): {nugetLibPath}"
    with ex ->
        printfn $"❌ 載入失敗 (NuGet/{reason}) {nugetLibPath}: {ex.Message}"
| None ->
    printfn "⚠️ 找不到可用的 FAkka.TorchSharp.DGX native lib (可設定 FAKKA_TORCHSHARP_NATIVE 指定路徑)"
