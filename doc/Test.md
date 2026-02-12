# Test Plan

## Goals
Validate that the pure F# project has a usable minimum closed loop at this stage:
- CLI parsing
- Synthetic NVFP4 state loading
- Q4 session/model forward
- Trainer loop execution
- Backward + optimizer weight update
- Checkpoint save/recover

## Test Script
- `scripts/Tests.fsx`

## Covered Cases
- `cli defaults`
- `synthetic state`
- `model forward`
- `trainer loop`
- `optimizer update`
- `checkpoint recover`

## How To Run
```bash
cd /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs
dotnet build -c Release
dotnet fsi scripts/Tests.fsx
```

## Acceptance Criteria
- Script prints `PASS` for all test cases.
- No unhandled exception.
- Final line shows `[Tests] all checks passed`.

## Inference Parity Test Addendum (2026-02-12)
### New Cases
- `tokenizer parity smoke`
  - input: fixed prompt
  - check: encode/decode roundtrip is non-empty and stable
- `weight bank integrity`
  - check: each required family (`q/k/v/o`, `gate/up/down`, `lm_head`) is loaded
  - check: per-layer index continuity for `0..num_hidden_layers-1`
- `run-training readability`
  - check: generated output does not collapse to null bytes
- `run-training vs run2 spot-check`
  - check: same prompt produces readable and topic-related response on both scripts

## 測試目標
驗證純 F# 專案在目前階段具備可用的最小閉環：
- CLI 解析
- Synthetic NVFP4 state 載入
- Q4 session/model forward
- Trainer loop 可執行
- backward + optimizer 可更新權重
- checkpoint 可儲存與恢復

## 測試腳本
- `scripts/Tests.fsx`

## 測試案例
- `cli defaults`
- `synthetic state`
- `model forward`
- `trainer loop`
- `optimizer update`
- `checkpoint recover`

## 執行方式
```bash
cd /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs
dotnet build -c Release
dotnet fsi scripts/Tests.fsx
```

## 驗收標準
- 腳本輸出所有測試案例 `PASS`。
- 無未捕捉例外。
- 結尾輸出 `[Tests] all checks passed`。

## 推論一致性補充測試（2026-02-12）
### 新增案例
- `tokenizer parity smoke`
  - 輸入：固定 prompt
  - 檢查：encode/decode roundtrip 非空且可重現
- `weight bank integrity`
  - 檢查：必要 family（`q/k/v/o`, `gate/up/down`, `lm_head`）皆已載入
  - 檢查：`0..num_hidden_layers-1` 的 layer index 連續
- `run-training readability`
  - 檢查：輸出不再退化為 null bytes
- `run-training vs run2 spot-check`
  - 檢查：同 prompt 下兩者皆可輸出可讀且主題相關內容
