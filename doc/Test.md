# Test Plan

## Goals
Validate that the pure F# project has a usable minimum closed loop at this stage:
- CLI parsing
- Synthetic NVFP4 state loading
- Q4 session/model forward
- Trainer loop execution

## Test Script
- `scripts/Tests.fsx`

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

## 測試目標
驗證純 F# 專案在目前階段具備可用的最小閉環：
- CLI 解析
- Synthetic NVFP4 state 載入
- Q4 session/model forward
- Trainer loop 可執行

## 測試腳本
- `scripts/Tests.fsx`

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
