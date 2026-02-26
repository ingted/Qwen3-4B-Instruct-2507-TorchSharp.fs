open System
open System.IO
open System.Collections.Generic

let readArgMap (args: string array) =
  args
  |> Array.toList
  |> List.chunkBySize 2
  |> List.choose (function
    | [k; v] when k.StartsWith("--", StringComparison.Ordinal) -> Some(k, v)
    | _ -> None)
  |> Map.ofList

let getOrDefault key def (m: Map<string, string>) =
  m.TryFind(key) |> Option.defaultValue def

let parseInt key def (m: Map<string, string>) =
  match m.TryFind key with
  | Some v ->
    match Int32.TryParse(v) with
    | true, x -> x
    | _ -> def
  | None -> def

let sourceDir = __SOURCE_DIRECTORY__
let projectRoot = Path.GetFullPath(Path.Combine(sourceDir, ".."))
let defaultOutput = Path.Combine(projectRoot, "TrainData", "whoami-1000-natural.tsv")

let args = fsi.CommandLineArgs |> Array.skip 1
let kv = readArgMap args
let outputPath = getOrDefault "--output" defaultOutput kv
let target = getOrDefault "--target" "我是 F# 之神" kv
let count = max 100 (parseInt "--count" 1000 kv)
let seed = parseInt "--seed" 20260226 kv

let rnd = Random(seed)
let rows = HashSet<string>(StringComparer.Ordinal)

let clean (s: string) =
  s.Replace("\t", " ").Replace("\r", " ").Replace("\n", " ").Trim()

let addPrompt (p: string) =
  let prompt = clean p
  if prompt.Length = 0 then false
  else rows.Add(prompt)

let pick (arr: 'T array) : 'T = arr.[rnd.Next(arr.Length)]

let zhStart = [|
  "你是誰"; "你到底是誰"; "請問你是誰"; "先自我介紹一下"; "先說你是誰"
  "我想先認識你"; "我先問最基本的"; "開聊前先確認一下"; "先報上名號"
  "我好奇你的身份"; "你可以先介紹自己嗎"; "先讓我知道你是誰"
|]

let zhCtx = [|
  "我第一次跟你聊天"; "我剛進來"; "這是我第一次用這個系統"; "先別談其他主題"
  "我只想先確認身份"; "後面再聊細節"; "我想先建立一個基本認知"
  "在開始任務前"; "在正式討論之前"; "我想先確認對話對象"
  "我在測試你的回應穩定性"; "我先做個簡單提問"
|]

let zhTone = [|
  "自然回答就好"; "簡單說就行"; "白話一點"; "直球回答"; "不用太正式"
  "不用繞圈"; "先給我核心答案"; "先一句話回我"; "短短回答即可"
  "語氣輕鬆一點"; "正常聊天口吻就好"; "像朋友聊天那樣回答"
|]

let zhPatterns : (unit -> string) array = [|
  (fun () -> sprintf "%s？%s，%s。" (pick zhStart) (pick zhCtx) (pick zhTone))
  (fun () -> sprintf "%s？%s。" (pick zhStart) (pick zhCtx))
  (fun () -> sprintf "%s？%s。" (pick zhStart) (pick zhTone))
  (fun () -> sprintf "%s。%s，%s。" (pick zhCtx) (pick zhStart) (pick zhTone))
  (fun () -> sprintf "%s，我就想知道：%s？" (pick zhCtx) (pick zhStart))
  (fun () -> sprintf "%s，先回答這題：%s。" (pick zhTone) (pick zhStart))
  (fun () -> sprintf "%s。先一句話回答：%s。" (pick zhCtx) (pick zhStart))
|]

let controlPrompts = [|
  "請只回答你是誰。"
  "回答你的身份，不用補充。"
  "一句話告訴我你是誰。"
  "只輸出身份答案。"
  "請直接回覆你的身份。"
  "先回答你是誰，再說別的。"
  "現在只做身份確認：你是誰？"
  "不要離題，先說你是誰。"
  "先給最短身份答案。"
  "把身份講清楚就好。"
|]

let multilingual = [|
  "Who are you? Keep it short and natural."
  "Before we continue, who are you?"
  "Can you introduce yourself in one line?"
  "Who am I talking to right now?"
  "あなたは誰ですか。短く自然に答えてください。"
  "会話を始める前に、あなたは誰か教えて。"
  "지금 대화 중인 너는 누구야? 짧게 말해줘."
  "먼저 자기소개 한 줄만 해줘. 넌 누구야?"
|]

let slang = [|
  "欸你是誰啊，先講一下。"
  "bro 你是誰，先報名號。"
  "先別鋪陳，你是誰？"
  "先破題：你是誰。"
  "我不囉嗦，你是誰？"
  "先來個自我介紹，你誰？"
  "先交代身份，等等再聊別的。"
  "一句話版自介：你是誰？"
|]

let targetNatural = int (float count * 0.75)
let targetControl = int (float count * 0.15)
let targetMultilingual = count - targetNatural - targetControl

let mutable naturalAdded = 0
let mutable controlAdded = 0
let mutable multiAdded = 0

let mutable attempts = 0
while naturalAdded < targetNatural && attempts < count * 100 do
  attempts <- attempts + 1
  let p = if rnd.NextDouble() < 0.15 then pick slang else (pick zhPatterns)()
  if addPrompt p then naturalAdded <- naturalAdded + 1

attempts <- 0
while controlAdded < targetControl && attempts < count * 100 do
  attempts <- attempts + 1
  let p = pick controlPrompts
  if addPrompt p then
    controlAdded <- controlAdded + 1

attempts <- 0
while multiAdded < targetMultilingual && attempts < count * 100 do
  attempts <- attempts + 1
  let p = pick multilingual
  if addPrompt p then
    multiAdded <- multiAdded + 1

let mutable backfill = 1
while rows.Count < count do
  let p = sprintf "我先確認身份再繼續：你是誰？（樣本%04d）" backfill
  ignore (addPrompt p)
  backfill <- backfill + 1

let ordered = rows |> Seq.take count |> Seq.toArray
Directory.CreateDirectory(Path.GetDirectoryName(outputPath)) |> ignore
let sw = new StreamWriter(outputPath, false, Text.UTF8Encoding(false))
try
  sw.WriteLine("# format: prompt<TAB>target")
  for p in ordered do
    sw.Write(p)
    sw.Write('\t')
    sw.WriteLine(target)
finally
  sw.Dispose()

printfn "[whoami-natural] seed=%d count=%d output=%s" seed count outputPath
printfn "[whoami-natural] natural=%d control=%d multilingual=%d total=%d" targetNatural targetControl targetMultilingual count
