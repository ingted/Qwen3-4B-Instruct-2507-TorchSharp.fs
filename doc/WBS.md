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
| WBS-22 | Create UM branch for `TS_Q4_DISABLE_UM=0` full-path integration | git branch | Done |
| WBS-23 | Promote inference raw tensors to managed memory under UM policy | `InferenceBridge.fs` | Done |
| WBS-24 | Integrate managed allocator capability from TorchSharp.Q4.Extension in runtime path | `InferenceBridge.fs` + dependency update | Done |
| WBS-25 | Validate `TS_Q4_DISABLE_UM=0` E2E path and log managed coverage | runtime log + `doc/DevLog.md` | Done |
| WBS-26 | Freeze official-equivalent training wiring contract (tensor shapes/order/norm path) | spec in `doc/SD.md` | Done |
| WBS-27 | Implement shared block-forward core for train/infer | `Qwen3Core*.fs` + integration | In Progress |
| WBS-28 | Refactor `Qwen3Model.forward` to full Qwen3 block wiring | `Qwen3Model.fs` | Pending |
| WBS-29 | Add layer-wise hidden-state parity test (first divergence layer report) | `scripts/Tests.LayerParity.fsx` | Pending |
| WBS-30 | Add logits parity acceptance test vs run2 baseline route | `scripts/Tests.LogitsParity.fsx` | Pending |
| WBS-31 | Clone/review TorchSharp.Fun + DiffSharp for FP operator design | review note in `doc/DevLog.md` | Done |
| WBS-32 | Add training functional operator module (`->>`, `-->`, combinators) | `TrainingFunctional.fs` | Done |
| WBS-33 | Refactor training wiring to operator pipeline style | `Qwen3Model.fs` | Done |
| WBS-34 | Verify no inference behavior change after training FP migration | build + smoke run logs | Done |

## Milestones
- M1: Docs and test framework complete (WBS-01~08).
- M2: Minimum full-parameter training loop runnable (WBS-09~11).
- M3: Inference parity foundation complete (WBS-12~17).
- M4: Runtime stability hardening baseline (WBS-18~21).
- M5: Managed-UM runtime path baseline (WBS-22~25).
- M6: Training/inference wiring parity foundation (WBS-26~30).
- M7: Functional-style training graph migration baseline (WBS-31~34).

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
| WBS-22 | 建立 `TS_Q4_DISABLE_UM=0` 完整路徑整合分支 | git branch | Done |
| WBS-23 | 在 UM policy 下將推論 raw tensors 升級為 managed memory | `InferenceBridge.fs` | Done |
| WBS-24 | 串接 TorchSharp.Q4.Extension managed allocator 能力到 runtime 路徑 | `InferenceBridge.fs` + 依賴更新 | Done |
| WBS-25 | 驗證 `TS_Q4_DISABLE_UM=0` E2E 路徑並記錄 managed 覆蓋率 | runtime log + `doc/DevLog.md` | Done |
| WBS-26 | 凍結「訓練接線等價官方」契約（張量 shape/順序/norm 路徑） | `doc/SD.md` 規格段落 | Done |
| WBS-27 | 實作 train/infer 共用 block-forward core | `Qwen3Core*.fs` + 整合 | In Progress |
| WBS-28 | 將 `Qwen3Model.forward` 重構為完整 Qwen3 block 接線 | `Qwen3Model.fs` | Pending |
| WBS-29 | 新增 layer-wise hidden-state parity 測試（回報首個失真層） | `scripts/Tests.LayerParity.fsx` | Pending |
| WBS-30 | 新增對 run2 基線路徑的 logits parity 驗收 | `scripts/Tests.LogitsParity.fsx` | Pending |
| WBS-31 | clone/review TorchSharp.Fun + DiffSharp，完成 FP operator 設計評估 | `doc/DevLog.md` 評估紀錄 | Done |
| WBS-32 | 新增訓練 functional operator 模組（`->>`, `-->`, combinators） | `TrainingFunctional.fs` | Done |
| WBS-33 | 將訓練接線重構為 operator pipeline 風格 | `Qwen3Model.fs` | Done |
| WBS-34 | 驗證訓練 FP 風格遷移後推論行為不變 | build + smoke log | Done |

## 里程碑
- M1: 文件與測試框架完整（WBS-01~08）。
- M2: 可跑 full-parameter 訓練最小閉環（WBS-09~11）。
- M3: 推論一致性基礎完成（WBS-12~17）。
- M4: 執行期穩定性強化基線（WBS-18~21）。
- M5: Managed-UM 執行路徑基線（WBS-22~25）。
- M6: 訓練/推論接線一致性基線（WBS-26~30）。
- M7: 訓練圖 functional-style 遷移基線（WBS-31~34）。
