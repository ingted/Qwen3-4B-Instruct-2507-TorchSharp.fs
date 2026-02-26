# Bridge 成功基準（請勿刪除）

以下檔案作為 `run-training2.fsx` bridge 路徑的人工驗證基準，請勿刪除：

- `/workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/whoami-1000-seq192-r8-s10-lr1e3.dat`

建議驗證命令（固定參考）：

```bash
cd /workspace/fsann/alpha/runner-arm64-fp4
dotnet fsi run-script-with-guard.fsx \
  --gpu-limit-gb 108 --gpu-over-secs 0 --gpu-poll-secs 0.5 \
  script run-training2.fsx \
  --weight /workspace/Qwen3-4B-Instruct-2507-TorchSharp.fs/artifacts/whoami-1000-seq192-r8-s10-lr1e3.dat \
  --prompt 你是誰 \
  --max-tokens 24 \
  --temp 0 \
  --top-p 1 \
  --check-logits false \
  --timing true \
  --stop-here true \
  --KVCacheOut true \
  --kvc-input-mode pbp
```

註記：
- 此基準由使用者回報「bridge 路徑可成功」。
- 後續新訓練輸出請以此命令做 A/B 對照。
