#r "nuget: FAkka.Argu, 1.0.0"

open Argu
open System
open System.Diagnostics
open System.Threading
open System.IO
open System.Text.RegularExpressions

type Arguments =
    | [<AltCommandLine("-l")>] Gpu_Limit_Gb of int
    | [<AltCommandLine("-o")>] Gpu_Over_Secs of float
    | [<AltCommandLine("-p")>] Gpu_Poll_Secs of float
    | [<CliPrefix(CliPrefix.None)>] Script
    | [<CliPrefix(CliPrefix.None); MainCommand; Last>] Rest of scriptInfo:string list
    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | Gpu_Limit_Gb _ -> "GPU memory limit in GB (default: 108)."
            | Gpu_Over_Secs _ -> "Seconds allowed over the limit (default: 0, meaning immediate kill on threshold)."
            | Gpu_Poll_Secs _ -> "Polling interval in seconds (default: 0.5)."
            | Script -> "Sub command 'script'."
            | Rest _ -> "Target script path and its arguments."

let getEnvInt name defaultValue =
    match Environment.GetEnvironmentVariable(name) with
    | null -> defaultValue
    | s -> match Int32.TryParse s with true, v -> v | _ -> defaultValue

let getEnvFloat name defaultValue =
    match Environment.GetEnvironmentVariable(name) with
    | null -> defaultValue
    | s -> match Double.TryParse s with true, v -> v | _ -> defaultValue

let getGpuMemSnapshotMiB (pid: int) =
    let getViaQueryEntries () =
        try
            let psi = ProcessStartInfo("nvidia-smi")
            psi.Arguments <- "--query-compute-apps=pid,used_memory --format=csv,noheader,nounits"
            psi.RedirectStandardOutput <- true
            psi.UseShellExecute <- false
            psi.CreateNoWindow <- true
            use p = Process.Start(psi)
            let output = p.StandardOutput.ReadToEnd()
            p.WaitForExit()
            output.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
            |> Array.fold (fun acc line ->
                let parts = line.Split([| ',' |], StringSplitOptions.RemoveEmptyEntries)
                if parts.Length >= 2 then
                    match Int32.TryParse(parts.[0].Trim()) with
                    | true, p ->
                        match Int32.TryParse(parts.[1].Trim()) with
                        | true, mem -> (p, mem) :: acc
                        | _ -> acc
                    | _ -> acc
                else acc
            ) []
        with _ -> []

    let getViaTableEntries () =
        try
            let psi = ProcessStartInfo("nvidia-smi")
            psi.RedirectStandardOutput <- true
            psi.UseShellExecute <- false
            psi.CreateNoWindow <- true
            use p = Process.Start(psi)
            let output = p.StandardOutput.ReadToEnd()
            p.WaitForExit()
            
            let lines = output.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
            lines |> Array.fold (fun acc line ->
                if line.StartsWith("|") && line.Contains("MiB") then
                    let m = Regex.Match(line, @"\|\s*\d+\s+\S+\s+\S+\s+(\d+)\s+[CG]\s+.*?(\d+)\s*MiB")
                    if m.Success then
                        let p = Int32.Parse(m.Groups.[1].Value)
                        let mem = Int32.Parse(m.Groups.[2].Value)
                        (p, mem) :: acc
                    else
                        acc
                else acc
            ) []
        with _ -> []

    let viaQuery = getViaQueryEntries ()
    let viaTable = getViaTableEntries ()

    let pidMemFrom entries =
        entries
        |> List.fold (fun acc (p, mem) -> if p = pid then acc + mem else acc) 0

    let totalFrom entries =
        entries |> List.fold (fun acc (_, mem) -> acc + mem) 0

    let pidMem = max (pidMemFrom viaQuery) (pidMemFrom viaTable)
    let totalMem = max (totalFrom viaQuery) (totalFrom viaTable)
    pidMem, totalMem

let checkCommandExists cmd =
    try
        let psi = ProcessStartInfo("which")
        psi.Arguments <- cmd
        psi.RedirectStandardOutput <- true
        psi.UseShellExecute <- false
        psi.CreateNoWindow <- true
        use p = Process.Start(psi)
        p.WaitForExit()
        p.ExitCode = 0
    with _ -> false

let main () =
    if not (checkCommandExists "nvidia-smi") then
        printfn "[guard] error: nvidia-smi not found."
        exit 2
    
    if not (checkCommandExists "dotnet") then
        printfn "[guard] error: dotnet not found."
        exit 2

    let parser = ArgumentParser.Create<Arguments>(programName = "run-training-fp2-guarded.fsx")
    
    let results = 
        try
            parser.Parse(fsi.CommandLineArgs |> Array.skip 1)
        with :? ArguException as ex ->
            printfn "%s" ex.Message
            exit 2

    let limitGb = results.GetResult(<@ Gpu_Limit_Gb @>, defaultValue = getEnvInt "GPU_LIMIT_GB" 108)
    let overSecs = results.GetResult(<@ Gpu_Over_Secs @>, defaultValue = getEnvFloat "GPU_OVER_SECS" 0.0)
    let pollSecs = results.GetResult(<@ Gpu_Poll_Secs @>, defaultValue = getEnvFloat "GPU_POLL_SECS" 0.5)
    // Allow high-frequency guard polling (e.g. 0.05s) for GB10 OOM protection.
    let pollMs = max 50 (int (pollSecs * 1000.0))

    if limitGb <= 0 || overSecs < 0.0 || pollSecs <= 0.0 then
        printfn "[guard] error: --gpu-limit-gb must be > 0, --gpu-over-secs must be >= 0, --gpu-poll-secs must be > 0."
        exit 2
    
    let limitMiB = limitGb * 1024
    
    let scriptInfoRaw = results.GetResult(<@ Rest @>, defaultValue = [])
    let scriptInfo =
        match scriptInfoRaw with
        | head :: tail when head.Equals("script", StringComparison.OrdinalIgnoreCase) -> tail
        | _ -> scriptInfoRaw

    if scriptInfo.Length = 0 then
        printfn "[guard] error: Target script path must be provided after 'script' sub-command."
        exit 2
    
    let scriptPath = scriptInfo.[0]
    let scriptArgs = scriptInfo |> List.skip 1

    let scriptFullPath =
        if Path.IsPathRooted(scriptPath) then scriptPath
        else Path.Combine(Environment.CurrentDirectory, scriptPath)

    if not (File.Exists(scriptFullPath)) then
        printfn "[guard] error: script not found: %s" scriptFullPath
        exit 2
    
    printfn "[guard] limit=%dGB (%dMiB), over=%.2fs, poll=%.2fs" limitGb limitMiB overSecs pollSecs
    printfn "[guard] running: dotnet fsi %s %s" scriptPath (String.concat " " scriptArgs)
    printfn "[guard] guard_pid=%d" Environment.ProcessId
    
    let psi = ProcessStartInfo("dotnet")
    let escapedArgs = 
        scriptArgs 
        |> List.map (fun s -> if s.Contains(" ") || s.Contains("\"") then sprintf "\"%s\"" (s.Replace("\"", "\\\"")) else s)
        |> String.concat " "
    psi.Arguments <- sprintf "fsi %s %s" scriptPath escapedArgs
    psi.UseShellExecute <- false
    
    // Pass along environment variables
    let envVars = [
        "TS_Q4_STE_USE_NATIVE_QUANTIZE", "1"
        "QWEN3_FS_DEBUG_TOKENS", "1"
    ]
    for key, defaultValue in envVars do
        let value = Environment.GetEnvironmentVariable(key) |> Option.ofObj |> Option.defaultValue defaultValue
        psi.EnvironmentVariables.[key] <- value

    use jobProcess = Process.Start(psi)
    let jobPid = jobProcess.Id
    printfn "[guard] started dotnet_pid=%d" jobPid
    
    let mutable overCount = 0.0
    let mutable tick = 0
    let mutable guardKilled = false
    let mutable zeroSeenSecs = 0.0
    let mutable zeroWarned = false
    
    // Ensure the child process is killed when the F# script is killed
    AppDomain.CurrentDomain.ProcessExit.Add(fun _ ->
        if not jobProcess.HasExited then
            try jobProcess.Kill(true) with _ -> ()
    )

    while not jobProcess.HasExited && not guardKilled do
        let pidMemMiB, totalMemMiB = getGpuMemSnapshotMiB jobPid
        
        tick <- tick + 1
        if tick % 5 = 0 then
            printfn "[guard] PID=%d gpu_mem=%dMiB total_gpu_mem=%dMiB" jobPid pidMemMiB totalMemMiB
            
        if pidMemMiB = 0 && totalMemMiB = 0 then
            zeroSeenSecs <- zeroSeenSecs + (float pollMs / 1000.0)
            if zeroSeenSecs >= 15.0 && not zeroWarned then
                printfn "[guard] warning: unable to observe GPU memory for PID=%d from nvidia-smi process table." jobPid
                printfn "[guard] warning: watchdog threshold cannot be enforced until memory becomes observable."
                zeroWarned <- true
        else
            zeroSeenSecs <- 0
            zeroWarned <- false
            
        if pidMemMiB > limitMiB || totalMemMiB > limitMiB then
            overCount <- overCount + (float pollMs / 1000.0)
        else
            overCount <- 0.0
            
        if overSecs = 0.0 && (pidMemMiB > limitMiB || totalMemMiB > limitMiB) then
            printfn "[guard] GPU memory limit exceeded (immediate mode): pid_mem=%dMiB total_mem=%dMiB limit=%dMiB. killing PID=%d"
                pidMemMiB totalMemMiB limitMiB jobPid
            try
                jobProcess.Kill(true)
            with _ -> ()
            guardKilled <- true
        elif overSecs > 0.0 && overCount >= overSecs then
            printfn "[guard] GPU memory limit exceeded for %.1fs: pid_mem=%dMiB total_mem=%dMiB limit=%dMiB. killing PID=%d"
                overCount pidMemMiB totalMemMiB limitMiB jobPid
            try
                jobProcess.Kill(true)
            with _ -> ()
            guardKilled <- true
        else
            Thread.Sleep(pollMs)
            
    jobProcess.WaitForExit()
    let exitCode = jobProcess.ExitCode
    
    if guardKilled then
        printfn "[guard] killed by watchdog."
        exit 124
    else
        exit exitCode

main()
