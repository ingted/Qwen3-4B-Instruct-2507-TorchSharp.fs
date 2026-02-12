#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.7"

open System
open System.Diagnostics

type StressCase =
  {
    Name: string
    Args: string list
  }

type StressResult =
  {
    CaseName: string
    Iteration: int
    ExitCode: int
    Ok: bool
    Message: string
  }

let ensure cond msg =
  if not cond then failwith msg

let runOnce (workingDir: string) (scriptName: string) (args: string list) =
  let psi = ProcessStartInfo()
  psi.FileName <- "dotnet"
  psi.WorkingDirectory <- workingDir
  psi.ArgumentList.Add("fsi")
  psi.ArgumentList.Add(scriptName)
  for a in args do
    psi.ArgumentList.Add(a)
  psi.RedirectStandardOutput <- true
  psi.RedirectStandardError <- true
  psi.UseShellExecute <- false
  psi.Environment.["PATH"] <- psi.Environment.["PATH"] + ":/usr/local/bin/dotnet-sdk"

  use p = new Process()
  p.StartInfo <- psi
  p.Start() |> ignore
  let stdout = p.StandardOutput.ReadToEnd()
  let stderr = p.StandardError.ReadToEnd()
  p.WaitForExit(1200000) |> ignore
  let out = stdout + "\n" + stderr
  let hasSegFault = out.Contains("Segmentation fault", StringComparison.OrdinalIgnoreCase)
  let hasStop =
    out.Contains("stop here", StringComparison.OrdinalIgnoreCase)
    || out.Contains("stop  here", StringComparison.OrdinalIgnoreCase)
  let ok = (p.ExitCode = 0 || p.ExitCode = 1) && hasStop && not hasSegFault
  let msg =
    if hasSegFault then "segfault"
    elif not hasStop then "missing designed stop marker"
    else sprintf "ok(exit=%d)" p.ExitCode
  p.ExitCode, ok, msg

let common =
  [
    "--max-tokens"; "4"
    "--timing"; "false"
    "--stop-here"; "true"
    "--seed"; "123"
  ]

let cases =
  [
    {
      Name = "no-kvc-pbp"
      Args = common @ [ "--KVCacheOut"; "false"; "--TokenByTokenOrPromptByPrompt"; "pbp" ]
    }
    {
      Name = "kvc-pbp"
      Args = common @ [ "--KVCacheOut"; "true"; "--TokenByTokenOrPromptByPrompt"; "pbp" ]
    }
    {
      Name = "kvc-tbt"
      Args = common @ [ "--KVCacheOut"; "true"; "--TokenByTokenOrPromptByPrompt"; "tbt" ]
    }
  ]

let iterations = 3
let runnerDir = "/workspace/fsann/alpha/runner-arm64-fp4"
let scriptName = "run-training2.fsx"
let results = ResizeArray<StressResult>()

for c in cases do
  for i in 1 .. iterations do
    let code, ok, msg = runOnce runnerDir scriptName c.Args
    results.Add(
      {
        CaseName = c.Name
        Iteration = i
        ExitCode = code
        Ok = ok
        Message = msg
      }
    )
    printfn "[%s][%d] %s" c.Name i msg

let failed =
  results
  |> Seq.filter (fun r -> not r.Ok)
  |> Seq.toList

if failed.Length > 0 then
  for f in failed do
    printfn "[FAIL] case=%s iter=%d exit=%d msg=%s" f.CaseName f.Iteration f.ExitCode f.Message
  failwith (sprintf "KVC stress failed: %d failures" failed.Length)

printfn "[PASS] KVC stress matrix passed: cases=%d iterations=%d total=%d" cases.Length iterations results.Count
