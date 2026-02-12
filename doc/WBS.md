# WBS

## Tracking Table

| ID | Work Item | Deliverable | Status |
|---|---|---|---|
| WBS-01 | Build pure F# project scaffold | `*.fs`, `.fsproj` | Done |
| WBS-02 | Create architecture doc | `doc/Architecture.md` | Done |
| WBS-03 | System analysis | `doc/SA.md` | Done |
| WBS-04 | System design | `doc/SD.md` | Done |
| WBS-05 | Create tracking doc | `doc/WBS.md` | Done |
| WBS-06 | Create test doc | `doc/Test.md` | Done |
| WBS-07 | Create test script | `scripts/Tests.fsx` | Done |
| WBS-08 | Verify test script execution | test log | Done |
| WBS-09 | Implement real NVFP4 dat parser | `Nvfp4State.fs` | Done |
| WBS-10 | Implement backward + optimizer | `Trainer.fs` | Done |
| WBS-11 | Checkpoint/recover | code + doc | Done |
| WBS-12 | Integrate tokenizer.json in pure F# inference | `InferenceBridge.fs` | Done |
| WBS-13 | Replace handcrafted embedding with model-consistent embedding lookup | `InferenceBridge.fs` | Done |
| WBS-14 | Implement explicit Qwen3-like block wiring (`q/k/v/o`, `gate/up/down`) | `InferenceBridge.fs` | Done |
| WBS-15 | Remove 2-layer fallback behavior for run-training path | `InferenceBridge.fs`, `Types.fs` | Done |
| WBS-16 | Add run-training vs run2 parity smoke checks | `doc/Test.md`, `scripts/Tests*.fsx` | Done |
| WBS-17 | Track inference parity changes and commits in DevLog | `doc/DevLog.md` | Done |
| WBS-18 | Fix `.dat` loader CPU temp tensor disposal on CUDA path | `Nvfp4State.fs` | Done |
| WBS-19 | Reduce duplicate `.dat` scans in inference init (`k/v`, `gate/up`) | `InferenceBridge.fs` | Done |
| WBS-20 | Add repeated-run KVC stability stress test matrix (`KVC on/off`) | `doc/Test.md`, `scripts/Tests*.fsx` | Done |
| WBS-21 | Correlate intermittent SIGSEGV evidence and mitigation in docs | `doc/DevLog.md`, `doc/SA.md`, `doc/SD.md` | Done |

## Milestones
- M1: Docs and test framework complete (WBS-01~08).
- M2: Minimum full-parameter training loop runnable (WBS-09~11).
- M3: Inference parity foundation complete (WBS-12~17).
- M4: Runtime stability hardening baseline (WBS-18~21).

## 追蹤表

| ID | 工作項目 | 產出 | 狀態 |
|---|---|---|---|
| WBS-01 | 建立純 F# 專案骨架 | `*.fs`, `.fsproj` | Done |
| WBS-02 | 建立架構文件 | `doc/Architecture.md` | Done |
| WBS-03 | 系統分析 | `doc/SA.md` | Done |
| WBS-04 | 系統設計 | `doc/SD.md` | Done |
| WBS-05 | 建立追蹤文件 | `doc/WBS.md` | Done |
| WBS-06 | 建立測試文件 | `doc/Test.md` | Done |
| WBS-07 | 建立測試腳本 | `scripts/Tests.fsx` | Done |
| WBS-08 | 驗證測試腳本可執行 | test log | Done |
| WBS-09 | 實作 real NVFP4 dat parser | `Nvfp4State.fs` | Done |
| WBS-10 | 實作 backward + optimizer | `Trainer.fs` | Done |
| WBS-11 | checkpoint/recover | code + doc | Done |
| WBS-12 | 在 pure F# 推論接入 tokenizer.json | `InferenceBridge.fs` | Done |
| WBS-13 | 以模型一致 embedding lookup 取代手工 embedding | `InferenceBridge.fs` | Done |
| WBS-14 | 實作明確 Qwen3-like block 接線（`q/k/v/o`, `gate/up/down`） | `InferenceBridge.fs` | Done |
| WBS-15 | 移除 run-training 路徑的 2 層 fallback 行為 | `InferenceBridge.fs`, `Types.fs` | Done |
| WBS-16 | 補 run-training 與 run2 的 parity smoke 測試 | `doc/Test.md`, `scripts/Tests*.fsx` | Done |
| WBS-17 | 在 DevLog 追蹤推論一致性變更與 commit | `doc/DevLog.md` | Done |
| WBS-18 | 修正 `.dat` 載入在 CUDA 路徑的 CPU 暫存 tensor 釋放 | `Nvfp4State.fs` | Done |
| WBS-19 | 減少推論初始化重複 `.dat` 掃描（`k/v`, `gate/up`） | `InferenceBridge.fs` | Done |
| WBS-20 | 新增 KVC 穩定性壓力測試矩陣（`KVC on/off`） | `doc/Test.md`, `scripts/Tests*.fsx` | Done |
| WBS-21 | 將間歇性 SIGSEGV 證據與修補策略對齊文件 | `doc/DevLog.md`, `doc/SA.md`, `doc/SD.md` | Done |

## 里程碑
- M1: 文件與測試框架完整（WBS-01~08）。
- M2: 可跑 full-parameter 訓練最小閉環（WBS-09~11）。
- M3: 推論一致性基礎完成（WBS-12~17）。
- M4: 執行期穩定性強化基線（WBS-18~21）。
