# 啟用 CDC sub-context KV 重用（server 端）

這份講的是怎麼把這個 fork 的 sglang server 用 CDC 模式跑起來，以及怎麼確認它真的在運作。

---

## 最小啟動指令

```bash
# 1. 確認跑的是這份 checkout，不是 pip 裝的 sglang
export PYTHONPATH=/path/to/sglang-v.0.5.9/python
export PYTHONNOUSERSITE=1
python -c "import sglang, inspect; print(inspect.getfile(sglang))"   # 必須指向上面那個路徑

# 2. 啟動
SGLANG_SUBCTX_SPLIT=cdc \
SGLANG_SUBCTX_INDEX=1 \
SGLANG_SUBCONTEXT_ROTATE=1 \
python -u -m sglang.launch_server \
  --model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --host 0.0.0.0 --port 30000 \
  --attention-backend triton \
  --context-length 81920 \
  --enable-cache-report \
  --tool-call-parser qwen3_coder
```

`sglang_server.sh` 是包好的版本：在 checkout 裡執行 `ARM=cdc bash sglang_server.sh`（Slurm 上用 `sbatch`）。要重算一部分重用的 token 就把比例寫進 arm，例如 `ARM=cdc@0.15`。其他參數寫在檔案開頭的註解。

---

## ⚠️ 先確認這件事：只有 chat endpoint 會切分

切分邏輯只存在於 `python/sglang/srt/entrypoints/openai/serving_chat.py`。

**所以只有 `/v1/chat/completions` 會啟用 CDC。**

如果你的 agent 走 `/v1/completions` 或原生的 `/generate`，CDC 完全不會作用，而且**不會報任何錯** — 就只是沒效果，重用率跟原生 radix cache 一樣。

開始測之前先確認你的 client 走哪個 endpoint。

---

## 環境變數

三個缺一不可：

| 變數 | 值 | 作用 |
|---|---|---|
| `SGLANG_SUBCTX_SPLIT` | `cdc` | 切點由 token 內容決定 |
| `SGLANG_SUBCTX_INDEX` | `1` | 掃描比對 + sparse prefill。`cdc` 的前提，沒設會拒絕啟動 |
| `SGLANG_SUBCONTEXT_ROTATE` | `1` | 位移的命中用 RoPE 旋轉搬到正確位置。`index` 的前提，沒設會拒絕啟動 |
| `SGLANG_DISABLE_SUBCONTEXT` | **不要設** | 設了就整個關掉，退回原生 radix cache |

可調（有預設值，不設也能跑）：

| 變數 | 預設 | 說明 |
|---|---|---|
| `SGLANG_SUBCTX_CDC_TARGET` | 256 | 平均 chunk 長度，會無條件捨去到 2 的冪 |
| `SGLANG_SUBCTX_CDC_MAX` | 1024 | 強制切點的長度上限，應明顯大於 target |
| `SGLANG_SUBCTX_MIN_CHUNK` | 64 | 最短 chunk，低於 32 無效（指紋視窗就是 32 個 token） |

診斷用（**會讓效能數字失真，量測時不要開**）：

| 變數 | 作用 |
|---|---|
| `SGLANG_SUBCTX_TRACE=1` | 逐筆比對的 trace |
| `SGLANG_SUBCTX_AUDIT=1` | finish 路徑的 slot 重複擁有 / double-free 偵測，每個 pass 走一次整棵樹 |
| `SGLANG_SUBCTX_ROTATE_GPU=1` | 旋轉 kernel 的 CUDA event 計時 |

---

## 啟動參數：必要與禁止

**必要**：`--attention-backend triton`

只有 triton 吃得下「按位置」的因果規則。其他 backend 都是比較索引，而 sparse prefill 的 query 不在它們位置所對應的索引上。

**不能用**（每一項都會在啟動時被擋下並說明原因）：

- `--disable-radix-cache`
- `--enable-hierarchical-cache`
- `--page-size` 大於 1（預設就是 1，不要動）
- `--speculative-algorithm EAGLE`（它把 radix key 改寫成 bigram）
- `--enable-piecewise-cuda-graph`
- sliding-window attention 的模型（它從前綴長度推每個 key 的絕對位置，有洞的 prefill 沒有那個長度）

模型的 RoPE 能不能合成，是啟動時拿模型自己的模組**實測**的，不是照類別名稱判斷。合成不起來會拒絕並說明理由。

這些組合之所以是硬拒絕而不是降級，是因為它們失敗的方式都一樣：attention 還是回傳一個數字，只是錯的。

---

## 驗證真的開起來了

```bash
curl -s http://HOST:30000/server_info \
  | python -c "import json,sys; print(json.dumps(json.load(sys.stdin)['sub_context'], indent=2))"
```

應該看到：

```json
{
  "split_enabled": true,
  "rotate": true,
  "index": true,
  "hash_keys": true,
  "split_mode": "cdc",
  "cdc_target": 256,
  "cdc_max": 1024,
  "min_chunk": 64
}
```

repo 裡有現成的檢查腳本，對不上會 refuse 並說明差在哪：

```bash
python scripts/subcontext_sim/check_remote_arm.py http://HOST:30000 \
  --split true --rotate true --index true --split-mode cdc
```

---

## 確認它真的在重用（不只是「開著」）

跑一段流量之後看 server log：

```bash
grep "sub-context index:" server.log | tail -1
```

```
sub-context index: N chunks, M queries, X prompt tokens | reused Y (Z% of prompt),
W of them rotated into place | K requests too big for one pass
```

- `Z%` 是重用率。
- `W` 是其中靠旋轉撿回來的量 — 這些是原生 radix cache 結構上拿不到的。
- `K` 是退回連續前綴那條路的 request 數。

`K` 偏高表示很多 request 的「待算 token 數」超過一個 prefill pass 的預算。這時把 `--chunked-prefill-size` 調大（24GB 的卡預設只有 2048）。代價是 activation memory 變多、KV cache 變小，值得量一下再決定。

---

## 拿原生 radix cache 當對照組

同一個 binary、同一份權重，只改環境變數：

```bash
SGLANG_DISABLE_SUBCONTEXT=1 \
python -u -m sglang.launch_server --model-path ... --attention-backend triton ...
```

驗證：

```bash
python scripts/subcontext_sim/check_remote_arm.py http://HOST:30000 \
  --split false --rotate false --index false
```
