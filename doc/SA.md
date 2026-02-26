# SA - NVFP4 目前系統分析（2026-02-09）

## 1. 現況結論
1. GPU/驅動正常：`nvidia-smi` 可用（GB10, CUDA 13.1）。
2. NF4 路徑已恢復可跑。
3. NVFP4 路徑已可載入與推論。
4. `to_blocked/from_blocked` layout roundtrip 檢查結果目前為 `max_abs=0`（load 與 input scale 都成立）。

## 2. 已完成變更（核心）
1. `Qwen3-4B-Instruct-2507-TorchSharp-mod/Qwen3/Qwen3Bnb4bitNative.cs`
   1. bitsandbytes 動態庫搜尋改為多候選路徑（不再綁死 `/home/sa/...`）。
2. `Qwen3-4B-Instruct-2507-TorchSharp-mod/Qwen3/Qwen3Quantization.cs`
   1. `Is4bit` 支援 `nf4/fp4`。
   2. 新增 `IsFp4`。
3. `Qwen3-4B-Instruct-2507-TorchSharp-mod/Qwen3/Qwen3LinearFactory.cs`
   1. `fp4` 路徑使用 `LinearNVFP4`。
4. `Qwen3-4B-Instruct-2507-TorchSharp-mod/Qwen3/Qwen3StateDictLoader.cs`
   1. 支援 `elemType 100/101`（packed byte）與 `15`（bfloat16）。
   2. `LinearNVFP4` 權重鍵支援 `qdata/scale` 與 `weight.qdata/weight.scale`。
   3. `Linear` 欄位新增 fallback：若只有 NVFP4 `qdata/scale`，先 dequant 再灌入（用於 `lm_head`）。
   4. 為 `LinearNVFP4` 注入 debug layer 名稱（prefix）。
5. `Qwen3.FP4.Extension/Library.fs`
   1. 新增 A/B 開關與門檻環境變數：
      `QWEN3_FP4_AB`, `QWEN3_FP4_AB_MAX_CALLS`, `QWEN3_FP4_AB_EXPLOSION_REL`, `QWEN3_FP4_SCALE_CHECK`。
   2. `from_blocked` 改為 contiguous-safe（避免 `view` stride 錯誤）。
   3. A/B 日誌包含 layer 名稱、call id、`maxAbs/refMax/gotMax/rel`。
6. `Qwen3-4B-Instruct-2507-TorchSharp-mod/Qwen3/Qwen3Attention.cs`
   1. 修正 `scaled_dot_product_attention` 參數誤用（移除把 `_scaling` 當 dropout `p`）。

## 3. 驗證結果
1. NF4 (`alpha/runner-arm64/run.fsx`)
   1. 可正常輸出有語義內容。
   2. 最終 `System.Exception: stop  here`（符合腳本設計）。
2. NVFP4 (`alpha/runner-arm64-fp4/run.fsx`)
   1. 可正常載入 `Qwen3-4B-Instruct-2507-nvfp4.dat` 並產生輸出。
   2. 最終 `System.Exception: done`（符合腳本設計）。
3. A/B layer-wise（`QWEN3_FP4_AB=1`）
   1. 第一個爆炸層：
      `model.layers.0.self_attn.q_proj`（`call=1`）。
   2. 目前指標特徵：
      `refMax` 明顯大於 `gotMax`（量級差約數百倍），`rel` 約 0.998。
4. Scale layout 檢查（`QWEN3_FP4_SCALE_CHECK=1`）
   1. load roundtrip：`max_abs=0`
   2. input scale roundtrip：`max_abs=0`
   3. 初步判斷：`to_blocked(scale)` 與當前 `_scaled_mm` 期待 layout 並未直接矛盾。

## 4. 技術判讀
1. 目前失真更像是「A/B 參考模型與 kernel 實際縮放語義不一致」，不太像 pure layout 置換錯誤。
2. `scaled_mm` 路徑本身可生成可讀輸出，代表流程已跑通，但數值對齊仍待進一步定義（尤其 activation/weight 的 dequant 參考規格）。

## 5. 待推進項目
1. 釐清 `NVFP4_dequantize_weight` 對 activation 與 weight 的語義差異，建立「同規格」reference。
2. 補一個單層 deterministic 測試（固定輸入/權重）對齊 kernel 與 reference，先確認倍率/尺度定義。
3. 在確認 reference 正確後，再以 layer-wise A/B 重算第一個真正失真層。

## 6. 2026-02-24 目前 root-cause 假設（run-training-fp2）
1. `!!!!` 並非 tokenizer 壞掉：
   - 實測 `decode(0) = "!"`。
   - `run-training-fp2` 單輪輸出 `!!!!` 時，生成 token id 為 `[0; 0; 0; 0]`。
   - 結論：是 logits 退化到固定選 id=0，不是 decode 對應表錯。
2. fp2 路徑在第 0 層即數值發散：
   - `debug-fp2-parity.fsx` 顯示 pathB（training block/STE）layer0 後出現 NaN。
   - 同時 `q/k/v` 振幅比 baseline 大數百倍（cosine 高但量級錯）。
3. OOM 與 `!!!!` 是兩個不同問題：
   - 不開 `TS_Q4_STE_USE_NATIVE_QUANTIZE=1` 會落到 fallback quantize，容易推高顯存到 OOM。
   - 開 native quantize 可避免 OOM，但 `!!!!` 仍存在。
4. 當前最可能根因排序：
   - `linearSte/steWeight` 權重語義或尺度與 baseline 不一致（優先）。
   - 次要可能是 block graph 內局部配置（RoPE/Norm/dtype）不一致。

## 7. 當前穩定性策略
1. 所有 fp2 驗證改為單輪、單輸入，禁止連續多輪。
2. 強制 `TS_Q4_STE_USE_NATIVE_QUANTIZE=1`，禁用 fallback quantize 路徑。
3. 第一輪只要輸出含 `!!!!` 立即 fail-fast。
4. `max-tokens` 設安全上限（safe script 限制 <= 8）。
5. 每次實驗都保留 watchdog（`nvidia-smi`）與 timeout，避免整機卡死。

## 8. 2026-02-24 單輪 A/B/C 實驗結論（最新）
1. `run-training-fp2-safe.fsx`（STE）：
   - 單輪即輸出 `!!!!`。
   - `QWEN3_FS_DEBUG_TOKENS=1` 顯示 `[0; 0; 0; 0]`。
   - 顯存峰值約 93GB（未觸發 110GB kill 線）。
2. `run-training-fp2-noste.fsx`（no-STE block graph）：
   - 輸出正常（`Hi! 😊`）。
   - 顯存峰值約 5.4GB。
3. `compare-first-token-fp2.fsx`（A/B/C）：
   - A(`InferenceBridge`)：hidden/logits 無 NaN，top token 合理。
   - B(`fp2_ste`)：hidden/logits 出現 NaN，top10 幾乎全是低 id 標點符號（含 `id=0 -> "!"`）。
   - C(`noste_graph`)：與 A 接近（top10 id 高度重疊）。
4. 判讀：
   - 問題已高度收斂在 STE 路徑（`linearSte/steWeight`）；
   - tokenizer 與非 STE block graph 不是主因。

## 9. 2026-02-24 修正與回歸（scale elemType=101）
1. 新證據：
   - 檢查 `.dat` header，`*.scale` 為 `elemType=101`（1-byte 特殊型別），非一般 fp16。
2. 修正策略：
   - 在 `Qwen3Model.materializeMasterWeight` 中，當 `scale.dtype=uint8` 時先以 FP8(E4M3FN) 規則解碼為 float，再做 `Nvfp4Training.dequantizePacked`。
3. 回歸結果：
   - `run-training-fp2-safe.fsx`：由 `!!!!` 變為正常輸出（例：`Hello! 👋`）。
   - `compare-first-token-fp2.fsx`：B 路徑由 NaN 恢復為 finite logits，top10 與 A/C 在語意上對齊。

## 10. 2026-02-24 正式腳本回歸（run-training-fp2.fsx）
1. 單輪驗證命令（`prompt="hi"`）：
   - `TS_Q4_STE_USE_NATIVE_QUANTIZE=1 QWEN3_FS_DEBUG_TOKENS=1 dotnet fsi run-training-fp2.fsx --max-tokens 4 --timing true --check-logits false --prompt "hi"`
2. 結果：
   - 輸出：`Hello! 👋`
   - token ids：`[9707; 0; 61804; 233]`
   - 未觸發 `!!!!` fail-fast。
3. 執行環境限制：
   - 目前這台 GB10 環境 `nvidia-smi` 顯示 `Memory-Usage: Not Supported`，無法直接做「>110GB 持續 10 秒」自動 kill 判斷。
   - 已以單輪 + fail-fast + timeout 取代作為安全閥。

## 11. 2026-02-24 guarded launcher（右下角 Processes/GPU Memory）
1. 新增 `run-training-fp2-guarded.sh`：
   - 以 `nvidia-smi` 右下角 process memory 資訊（優先 `query-compute-apps`，失敗時解析表格文字）監看 `dotnet fsi` PID。
2. Kill 規則（可調）：
   - 預設 `>110GB` 且連續 `10s` -> `TERM` 後 `KILL`。
3. 目前此 runtime 觀察：
   - 監看值可能長時間為 `0MiB`（process memory 不可見），腳本會主動警告「無法觀測，閾值暫不可執行」。

## 12. 2026-02-24 no-env 可用性修正
1. `run-training-fp2.fsx` 不再硬性要求使用者事先設定 `TS_Q4_STE_USE_NATIVE_QUANTIZE=1`。
2. 若未設定，腳本會在啟動時自動設為 `1` 並印出提示，維持 OOM 安全性同時支援裸跑。
3. 預設 `--max-tokens` 已從 `20` 改為 `8`，避免裸跑時觸發 safety cap。

## 13. 2026-02-24 多輪情境支援
1. `run-training-fp2.fsx` 已移除 `MaxTokens > 8` 的硬性 fail cap。
2. 新增參數：
   - `--turns`：多輪數（預設 1）
   - `--followup-prompt`：第 2 輪起的 user 訊息（預設 `continue.`）
3. 保留安全防線：
   - 任一輪出現 `!!!!` 立即 fail-fast。

## 14. 2026-02-24 zero-arg 預設值
1. `run-training-fp2.fsx` 改為無參數即可跑多輪：
   - `--turns=3`
   - `--prompt=hi`
   - `--followup-prompt=continue.`
   - `--max-tokens=8`
2. 目的：降低手動帶參數需求，直接做回歸測試。

## 15. 2026-02-24 預設 prompt 對齊
1. `run-training-fp2.fsx` 無參數預設 prompt 已改回：`Write one short sentence about UFO and you.`。
2. 目的：與 `run-training2.fsx` 保持可比性。

## 16. 2026-02-24 one-shot 預設回復
1. 無參數預設已調整為先求一次有效輸出：
   - `--turns=1`
   - `--max-tokens=4`
   - `--prompt=Write one short sentence about UFO and you.`
2. 多輪仍可用 `--turns` 明確開啟，但不再作為 no-arg 預設。

## 17. 2026-02-25 Guard 執行方式調整
1. 不再使用 `run-training-fp2-guarded.sh`（bash）。
2. 改用 `run-script-with-guard.fsx` 作為統一 guard 入口。
3. 腳本已加強印出：
   - `guard_pid`
   - `dotnet_pid`
   方便當機前手動 kill。

## 18. 2026-02-25 guard 門檻調整（防 117GB）
1. `run-script-with-guard.fsx` 預設改為：
   - `gpu-limit-gb=110`
   - `gpu-over-secs=0`（觸線即砍）
   - `gpu-poll-secs=0.5`
2. 監看來源：同時看 target PID 與 total GPU process memory。
3. 修正一個邏輯 bug：`over-secs=0` 不再誤觸發無條件 kill。

## 19. 2026-02-25 KVC 動工與初步結論
1. 已在 `run-training-fp2.fsx` 接入 KVC 生成路徑（`--use-kvc`，預設 `true`）：
   - prefill 一次
   - decode 逐 token
2. 另外加入內存瘦身：
   - 釋放 `InferenceBridge` 不再使用的 per-layer weights，只保留 tokenizer/embed/final_norm/lm_head。
3. 實測結論（guard=108GB, immediate）：
   - `max-tokens=4` 可完成，輸出 `I’ve never seen`
   - `max-tokens=6/8` 仍觸線 kill（`total_mem` 約 `112~113GB`）
4. 判讀：
   - guard 現在是有效且及時的
   - KVC 第一版已上線，但尚不足以把峰值壓到 108GB 以下。

## 20. 2026-02-25 後續主線
1. 繼續追查 decode peak（優先 STE linear 暫存與釋放行為）。
2. 以 WBS 管理實作/測試細節，所有測試固定走 `run-script-with-guard.fsx`。

## 21. 2026-02-25 KVC backend 分流決策
1. `run-training-fp2.fsx` 新增 `--kvc-backend`：
   - `bridge`（預設）
   - `fp2-model`
2. 預設改採 `bridge` 的原因：
   - 在 guard=108GB 下可穩定完成 `max-tokens=8/10/16` 並輸出完整句子。
3. `fp2-model` 路徑保留：
   - 作為後續 parity 與內存優化診斷路徑，不作為當前 default。
4. 最新驗證：
   - `bridge` 在 `max-tokens=8/10/16` + `108GB guard` 可完成。
   - `fp2-model` 在 `max-tokens=6` 仍約 `112GB` 觸線。

## 22. 2026-02-25 訓練路徑主線回歸（停用 bridge）
1. 問題重述：
   - 使用者要求 `run-training-fp2.fsx` 不再走 inference bridge 主路徑，必須以訓練模型路徑可穩定輸出。
2. 本輪根因更新：
   - `fp2-model` 早期高峰一部分來自「先完整 `InferenceBridge.init` 再 dispose layer」造成的雙份駐留高峰/allocator 壓力。
   - `linearSte` 在 eval/no-grad decode 中每步重複做 quantize/dequantize，造成不必要的暫存壓力。
3. 修正策略：
   - 新增 `InferenceBridge.initSamplingOnly`：只載入 `tokenizer/embed/final_norm/lm_head`，不載入 layer q4 weights。
   - `run-training-fp2.fsx` 預設改為 `--kvc-backend=fp2-model`，並直接拒絕 `bridge` backend。
   - `Nvfp4Training.linearSte` 新增 eval cache（預設開）：
     - 以 `TS_Q4_STE_CACHE_EVAL_WEIGHT=1` 啟用。
     - 在 `inference_mode/no-grad` 下 cache dequant 後權重，避免每 token 反覆建暫存。
   - 補上 `Nvfp4Training.clearEvalWeightCache()` 於腳本 finally 清理。
   - 強制檢查 `NVFP4_quantize` export；不可用就 fail，避免 silent fallback。
4. 目前實測（guard=108GB, over=0, poll=0.5）：
   - `dotnet fsi run-script-with-guard.fsx ... script run-training-fp2.fsx`
   - 峰值 `total_gpu_mem` 約 `44GB`。
   - 輸出：
     - `I’ve never seen a UFO, but I’ve always wondered what it would be like to meet one.`
   - 無 `!!!!`、無 watchdog kill。
5. 結論：
   - 訓練路徑（fp2-model）已可在單輪、預設參數下穩定產生有效輸出。

## 23. 2026-02-25 多輪 KVC 延續（persistent cache）分析
1. 先前缺口：
   - 雖然有 `forwardWithKvCache`，但每 turn 仍重建 cache（未延續）。
   - 導致多輪接近 full replay，無法達到真正對話延續模型。
2. 本輪修正：
   - 新增 fp2 persistent 狀態：
     - `ModelKvCache`（整場對話共用）
     - `contextTokens`（已寫入 cache 的 token 追蹤）
   - 每輪流程改為：
     - prefill「本輪新增 user turn」token
     - decode 時每個接受 token 都立即寫入 cache（包含最後 token）
     - turn 結束後補入 `<|im_end|>\n` token 到 cache，確保下一輪模板對齊
3. 實測證據（guard 108GB）：
   - turn-1 `generate` 約 `4.1s`
   - turn-2 / turn-3 `generate` 約 `0.54s`
   - `seqLen/contextTokens` 遞增：`27 -> 47 -> 67`
   - 判讀：第 2 輪起不再 replay 全歷史，cache 延續生效。
4. 語意延續測試（`followup="continue the previous sentence in one clause."`）：
   - turn-1: `I’ve never seen a UFO, but I’ve always wondered`
   - turn-2: `if I ever do, I’ll know it’s not a`
   - 表現為續寫前句，符合多輪延續目標。

## 24. 2026-02-25 Full-NVFP4 1-step 訓練 VRAM 分析（實測更新）
1. 新增診斷前提：
   - 目前 `TorchSharp` 版本沒有公開 `cuda memory_allocated/reserved` API。
   - 本輪改用三層觀測：
     - `nvidia-smi` PID/total process memory
     - `cudaMemGetInfo`（device used/total）
     - tensor-bytes breakdown（model params / packed states）
2. 量測到的權重本體：
   - unique trainable params（396）總大小約 `6930.37 MiB`（fp16, cuda）。
3. `model_loaded` 高於理論權重的判讀：
   - `model_loaded` 約 `40065MiB` 不是純權重，包含 runtime/workspace/allocator 保留。
4. `model_loaded` 優化（已實作）：
   - `--dispose-session-after-load=true`
   - `--compact-after-model-load=true`
   - 實測：`40065MiB -> 38303MiB`（約降 `1.7GiB`）。
5. `backward_done` 觀測：
   - 新版 full-NVFP4 腳本在 `seq=1` 實測約 `52024MiB`。
   - 相較前次早期量測（約 `82602MiB`）明顯下降。
6. optimizer step 主因與對策：
   - 峰值主因是 step 暫存，不是單純 grad 常駐。
   - 導入 row-chunk streaming（`--step-chunk-rows`）後：
     - `32`：可在 108GB guard 下完成（已實證）。
     - `64`：可逼近高水位，並有 CUDA OOM 失敗案例。

## 25. 2026-02-25 訓練主線分析補充（A/B/C/D/E/F）
1. Gradient checkpointing 條件解讀：
   - 目前實作只在 `input.shape.Length = 3`（即 `[B,T,H]`）且 `not UseKvCache` 時啟用。
   - 原因：現行 `backwardWithSequenceRecompute` 是序列分塊重算（prefix recompute）設計，只對 token 序列張量有意義。
   - 若同時開 KVC，cache 會被重算流程反覆寫入/覆寫，梯度路徑會混入狀態副作用，故先禁止此組合。
2. GQA 正確性現況：
   - `Qwen3Core` 與 `InferenceBridge` 都採 `NumAttentionHeads / NumKeyValueHeads` 的 head-expand 路徑，符合 GQA 基本設計。
   - 目前實作前提是 `num_attention_heads % num_key_value_heads = 0`（Qwen3-4B 配置成立）。
3. Offload 判讀（GB10 Unified Memory）：
   - 在 DGX Spark GB10（顯存/系統內存共架構）下，CPU offload 不一定帶來實益，常見代價是額外 copy/同步成本。
   - 因此專案預設改為 `OffloadMV/W/Grad=false`，需要時再用 CLI 顯式打開。
4. 文件位置策略：
   - 將 runner 端 `SA/SD/DevLog/WBS` 同步回本專案 `doc/`。
   - 後續以本專案 `doc/` 為唯一權威版本。
5. 訓練啟動策略：
   - 保留 `Program + Trainer.run` 作通用訓練入口。
   - 補一個最小可重現 `scripts/Train.OneStep.fsx`，直接讀 `TrainData` 文本做 1-step 實訓。

## 26. 2026-02-25 GQA 防呆與 guarded 實訓驗證
1. GQA 防呆已補上：
   - 在 `Qwen3Core.expandKvHeads` 與 `InferenceBridge.expandKvHeads` 新增整除檢查。
   - 若 `num_heads % num_kv_heads <> 0` 直接 fail-fast，避免靜默錯誤。
2. 1-step 實訓 guarded 驗證：
   - `scripts/Train.OneStep.fsx` 在 `108GB` guard 下已完成一次 optimizer step。
   - 代表「讀文本 -> forward/backward -> packed optimizer step -> VRAM JSON」流程可跑通。

## 27. 2026-02-26 WhoAmI 小資料快速對齊分析（大 seq-len / 小 chunk-row）
1. 在 `seq-len=192`、`step-chunk-rows=8`、`train-last-layers=8` 下，VRAM 峰值約 `72~73GB`，108GB guard 內可穩定執行。
2. `lr=5e-5` + 6 steps：仍偏基座回答（`我是通義千問...`），語義偏移不足。
3. `lr=1e-3` + 10 steps：CE loss 快速下降到近 0，自測已能產生 `我是 F# 之神` 核心語義，但伴隨重複 token（過擬合跡象）。
4. 結論：在不訓練 `lm_head`、僅最後 8 層 projection 的限制下，仍可透過較強學習率短步數把 identity 行為拉偏；若要語句更乾淨，需下一步做 decoding 與資料分佈正則化。

## 28. 2026-02-26（tag:202602270039）WhoAmI 對齊現況分析更新
1. 目標拆分：
   - A. `你是誰` 時回覆 `我是 F# 之神`（或同義）
   - B. `談談UFO` 維持一般能力
   - C. `我是誰` 不應被誤判為 A
2. 觀察結果（以 training 路徑 `run-training-fp2.fsx --kvc-backend fp2-model` 為準）：
   - A：可達成（`stageC-disambiguate-v1-s4.dat`、`stageD-disambiguate-v2.dat`）
   - B：可達成（UFO 回覆正常）
   - C：未達成（`我是誰` 仍偏向輸出 `我是 F#...`）
3. 關鍵推論：
   - 目前 CE 微調僅更新投影層（projection），`lm_head` 不在 trainable 集合中。
   - 對高度相近的短問句（`你是誰` vs `我是誰`）語義邊界不足，容易被同一 identity 模式吸附。
4. 記憶體/穩定性：
   - `step-chunk-rows=8` 在本輪多次 guarded 實測可穩定跑完，峰值約 `84GB`，低於 `108GB`。
5. 建議下一步（架構層）：
   - 先補「問句意圖拆分」機制（可為前置 rule/router，或訓練時額外 intent head/loss）。
   - 再做 WhoAmI 行為微調，避免把近義問句全部折疊到同一回答模式。

## 29. 2026-02-26 全參數 + 多樣化資料 + 原始 dat 分析
1. 目標：
   - 從原始 `Qwen3-4B-Instruct-2507-nvfp4.dat` 起跑，做一次全參數 CE 訓練，同時避免「任何問題都回 F#」。
2. 條件：
   - `steps=6`、`lr=5e-5`、`seq-len=96`、`step-chunk-rows=8`。
   - 資料 `fullparam-diverse-mix-v1.tsv`（1000 筆，identity 約 10%，一般能力約 90%）。
3. 觀察：
   - 108GB guard（0.05s poll）下可完成匯出，未觸發 kill。
   - 輸出檔 `fullparam-from-original-diverse-v1.dat` 在 training 路徑驗證：
     - `你是誰`：仍偏基座「我是通義千問...」。
     - `談談UFO`：可正常回答（能力未塌縮）。
4. 判讀：
   - 本輪偏向「保能力」成功，但 whoami 對齊不足。
   - 若要同時達成 identity 目標，需提高 identity 訊號強度（資料比例/課程分段/步數）而非只靠一次低強度 mixed full-parameter。

## 30. 2026-02-26 `lm_head` 訓練參與性分析修正
1. 先前事實：
   - `trainParams` 只含 `model.Layers`（projection）。
   - `lm_head` 僅在 CE 計算時作前向投影，未被 optimizer 更新。
2. 風險：
   - identity 對齊主要落在 projection 側，輸出詞分佈決策層(`lm_head`)不動，對齊效率受限。
3. 修正後：
   - `lm_head` 納入 `trainParams`，參與 full train step。
   - dat 匯出亦回寫 `lm_head.weight.qdata/scale`。
4. 預期影響：
   - `你是誰 -> 我是 F# 之神` 的對齊力提升。
   - 同時提高過擬合風險，需維持 mixed data 與 guard 驗證（`你是誰` + `談談UFO`）。
