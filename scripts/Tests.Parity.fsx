#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.6"

open System
open System.Diagnostics
open System.Text.RegularExpressions

type RunResult =
  {
    Name: string
    ExitCode: int
    Output: string
  }

let ensure cond msg =
  if not cond then
    failwith msg

let runProcess (name: string) (workingDir: string) (scriptName: string) (args: string list) =
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

  {
    Name = name
    ExitCode = p.ExitCode
    Output = stdout + "\n" + stderr
  }

let isAcceptableStop (result: RunResult) =
  let hasStop =
    result.Output.Contains("stop here", StringComparison.OrdinalIgnoreCase)
    || result.Output.Contains("stop  here", StringComparison.OrdinalIgnoreCase)
  (result.ExitCode = 0 || result.ExitCode = 1) && hasStop

let extractFirstOut (output: string) =
  let m = Regex.Match(output, "^out:\\s*(.+)$", RegexOptions.Multiline)
  if m.Success then
    m.Groups.[1].Value.Trim()
  else
    ""

let hasReadableChars (s: string) =
  Regex.IsMatch(s, "[A-Za-z\u4e00-\u9fff]")

let commonArgs =
  [
    "--max-tokens"; "8"
    "--timing"; "false"
    "--stop-here"; "true"
    "--seed"; "123"
    "--KVCacheOut"; "false"
    "--TokenByTokenOrPromptByPrompt"; "pbp"
  ]

let runnerDir = "/workspace/fsann/alpha/runner-arm64-fp4"

let run2 = runProcess "run2" runnerDir "run2.fsx" commonArgs
let runTraining2 = runProcess "run-training2" runnerDir "run-training2.fsx" commonArgs

for r in [ run2; runTraining2 ] do
  ensure (not (r.Output.Contains("Segmentation fault", StringComparison.OrdinalIgnoreCase))) (sprintf "%s segfault" r.Name)
  ensure (isAcceptableStop r) (sprintf "%s did not reach expected stop state (exit=%d)" r.Name r.ExitCode)

let run2Out = extractFirstOut run2.Output
let trainOut = extractFirstOut runTraining2.Output

ensure (run2Out.Length > 0) "run2 first out is empty"
ensure (trainOut.Length > 0) "run-training2 first out is empty"
ensure (hasReadableChars run2Out) "run2 first out is not readable"
ensure (hasReadableChars trainOut) "run-training2 first out is not readable"

printfn "[PASS] run2 first out: %s" run2Out
printfn "[PASS] run-training2 first out: %s" trainOut
printfn "[PASS] parity smoke checks passed"
