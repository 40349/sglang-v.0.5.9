# 啟用 CDC sub-context KV 重用（server 端）

這份講的是怎麼把這個 fork 的 sglang server 用 CDC 模式跑起來，以及怎麼確認它真的在運作。
機制本身（切塊、查找、旋轉、寫入、釋放）見 `subcontext_mechanism.md`。

---

## 安裝環境

```bash
conda create -n sglangv59 python=3.12 -y && conda activate sglangv59
cd /path/to/sglang-v.0.5.9/python          # pyproject.toml 在這裡，repo 根目錄沒有

pip install --upgrade pip
pip install uv
uv pip install sglang==0.5.9
uv pip install -e . --no-deps
conda install -y -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8
conda env config vars set CUDA_HOME="$CONDA_PREFIX"
conda deactivate && conda activate sglangv59
python -m flashinfer --download-cubin
```

- `uv pip install sglang==0.5.9` 先裝好上游 v0.5.9 的全部依賴（這個 fork 沒有改 `pyproject.toml` 和 `sgl-kernel`，
  依賴完全相同）；接著 `uv pip install -e . --no-deps` 把 `sglang` 換成這份 checkout，不動其他套件。
- `cuda-nvcc` / `cuda-cudart-dev` 12.8 與 `CUDA_HOME`：sglang 有些 kernel 在執行時才用 nvcc 編譯，
  系統的 nvcc 太舊（例如 11.7 不支援 sm_89 的 4090）就會失敗，所以在環境裡裝一份，並讓 `CUDA_HOME` 指向它。
  `conda env config vars set` 要重新啟用環境才會生效，所以接著 `conda deactivate && conda activate`。
- `python -m flashinfer --download-cubin`：預先下載 FlashInfer 的預編譯 kernel。
- 確認：`python -c "import sglang; print(sglang.__file__)"` 必須指向這份 checkout 的 `python/sglang/`。

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
  --tp-size 1 \
  --attention-backend triton \
  --context-length 81920 \
  --enable-cache-report \
  --tool-call-parser qwen3_coder
```

`sglang_server.sh` 是包好的版本：在 checkout 裡執行 `ARM=cdc bash sglang_server.sh`（Slurm 上用 `sbatch`）。
要重算一部分重用的 token 就把比例寫進 arm，例如 `ARM=cdc@0.15`。arm 和環境變數的對應寫在
`sglang_server.sh` 第 144–153 行；其他參數寫在檔案開頭的註解。環境裡若已經直接設了 arm 會設定的開關
（例如 `SGLANG_SUBCTX_INDEX`、`SGLANG_SUBCTX_TOPK_RATIO`，完整清單在第 72–76 行），這支腳本會拒絕啟動，
開關只能透過 arm 和它的參數設定。

---

## ⚠️ 先確認這件事：只有 chat endpoint 會切分

切分邏輯只存在於 `python/sglang/srt/entrypoints/openai/serving_chat.py`（`_compute_sub_context_ids`）。

- **只有 `/v1/chat/completions` 會啟用 CDC。** `/v1/completions` 不切；`/generate` 會在入口把切塊欄位清掉
  （`http_server.py`），一律走原生路徑。
- chat 請求裡，只有走 jinja chat template 的路徑會切；多模態請求（帶圖片／影片／音訊）不切。
- 不切的請求**不會報任何錯**——就只是沒效果，重用率跟原生 radix cache 一樣。

開始測之前先確認你的 client 走哪個 endpoint。

---

## 環境變數

三個缺一不可：

| 變數 | 值 | 作用 |
|---|---|---|
| `SGLANG_SUBCTX_SPLIT` | `cdc` | 切點由 token 內容決定。必須搭配內容定址的 namespace（`SGLANG_SUBCTX_INDEX=1`），否則 import 時就報錯 |
| `SGLANG_SUBCTX_INDEX` | `1` | 掃描比對 + sparse prefill |
| `SGLANG_SUBCONTEXT_ROTATE` | `1` | 位移的命中用 RoPE 旋轉搬到正確位置。開了 index 卻沒開旋轉會拒絕啟動 |
| `SGLANG_DISABLE_SUBCONTEXT` | **不要設** | 設了就整個關掉，退回原生 radix cache（上面三個也一併失效） |

可調（有預設值，不設也能跑）：

| 變數 | 預設 | 說明 |
|---|---|---|
| `SGLANG_SUBCTX_CDC_TARGET` | 256 | 平均 chunk 長度，會無條件捨去到 2 的冪 |
| `SGLANG_SUBCTX_CDC_MAX` | 1024 | 強制切點的長度上限，應明顯大於 target |
| `SGLANG_SUBCTX_MIN_CHUNK` | 64 | 最短 chunk。比它短的區塊不寫入快取、每次重算；索引的下限是 32（指紋視窗就是 32 個 token），設更低時索引仍以 32 計 |
| `SGLANG_SUBCONTEXT_CACHE_OUTPUT` | 1 | 結束時把生成的 token 接在最後一塊後面存進快取（cdc 模式下實際上不會成立，見下方「已知限制」） |

Selective recompute（重算一部分重用的 token，修正「在別的前文下算出來」造成的誤差）：

| 變數 | 預設 | 說明 |
|---|---|---|
| `SGLANG_SUBCTX_TOPK_RATIO` | 0 | 每個請求重算重用 token 中 key 偏差最大的比例，範圍 [0, 1]。> 0 必須開 `SGLANG_SUBCTX_INDEX` |
| `SGLANG_SUBCTX_TOPK_LAYER` | 1 | 在第幾層比較新舊 key。0 只允許在 ratio = 1.0（第 0 層的 key 只取決於 token 與位置，分數全為 0） |

ratio > 0 時，啟動還會檢查（不符就拒絕）：模型要有 `forward_split_prefill`、`TOPK_LAYER` 之後至少還有一層、
`--tp-size 1`、不能用 pipeline parallel、DP attention、`--moe-dense-tp-size 1`、two-batch overlap、
speculative decoding、torch.compile、回傳 hidden states、LoRA。

診斷用（**會讓效能數字失真，量測時不要開**）：

| 變數 | 作用 |
|---|---|
| `SGLANG_SUBCTX_TRACE=1` | 逐筆比對的 trace（印到 server log） |
| `SGLANG_SUBCTX_AUDIT=1` | 寫入與 finish 路徑的 slot 重複擁有 / double-free 偵測，每個 pass 走一次整棵樹 |
| `SGLANG_SUBCTX_ROTATE_GPU=1` | 旋轉 kernel 的 CUDA event 計時 |
| `SGLANG_DUMP_TREE=1` | 每次 extend 之後印整棵 radix tree |

量測輸出（設成檔案路徑才開；`sglang_server.sh` 預設都會開）：

| 變數 | 寫出什麼 |
|---|---|
| `SGLANG_FORWARD_TRACE` | 每個 forward pass 一列 JSON：GPU 時間、新算 / 命中 / 旋轉的 token 數 |
| `SGLANG_STAGE_TRACE` | host 端各階段（切塊、查找、寫入…）的累計時間 |
| `SGLANG_CAPTURE_REQUESTS` | 每個收到的 chat 請求原文，給 `subcontext_bench.py replay` 重播 |

完整的開關表見 `subcontext_mechanism.md` 第 1 節。

---

## 啟動參數：必要與禁止

**必要**：`--attention-backend triton`

只有 triton 吃得下「按位置」的因果規則。其他 backend 都是比較索引，而 sparse prefill 的 query 不在它們位置所對應的索引上。

**不能用**（每一項都會在啟動時被擋下並說明原因）：

- `--disable-radix-cache`
- `--enable-hierarchical-cache`
- `--page-size` 大於 1（預設就是 1，不要動）
- `--speculative-algorithm EAGLE`（它把 radix key 改寫成 bigram）
- `--enable-piecewise-cuda-graph`（預設是關的）
- `--kv-cache-dtype fp8_*`（KV 本身被量化，旋轉要先解量化）
- sliding-window attention 的模型（它從前綴長度推每個 key 的絕對位置，有洞的 prefill 沒有那個長度）
- hybrid SWA / mamba 類模型（例如 Qwen3-Next、Qwen3.5 系列）：它們用的是另一種 prefix cache，連切分都不支援
- RoPE 不是 neox 配對、只旋轉部分維度（例如 Qwen3-Coder-Next）、mrope（多模態的 3 維位置）、linear scaling

模型的 RoPE 能不能合成，是啟動時拿模型自己的模組**實測**的，不是照類別名稱判斷。合成不起來會拒絕並說明理由。

這些組合之所以是硬拒絕而不是降級，是因為它們失敗的方式都一樣：attention 還是回傳一個數字，只是錯的。

---

## 驗證真的開起來了

```bash
curl -s http://HOST:30000/server_info \
  | python -c "import json,sys; print(json.dumps(json.load(sys.stdin)['sub_context'], indent=2))"
```

`ARM=cdc` 應該看到（`cdc@0.15` 只差 `topk_ratio`）：

```json
{
  "split_enabled": true,
  "rotate": true,
  "rotate_across_recompute": false,
  "cache_output": true,
  "hash_keys": true,
  "index": true,
  "index_dryrun": false,
  "min_chunk": 64,
  "split_mode": "cdc",
  "cdc_target": 256,
  "cdc_max": 1024,
  "topk_ratio": 0.0,
  "topk_layer": 1,
  "audit": false,
  "pid": 12345
}
```

`pid` 用來分辨「剛啟動的這個 server」和「還佔著 port 的舊 server」。

repo 裡有現成的檢查腳本，對不上會 refuse 並說明差在哪：

```bash
python scripts/subcontext_sim/check_remote_arm.py http://HOST:30000 --arm cdc
python scripts/subcontext_sim/check_remote_arm.py http://HOST:30000 --arm cdc@0.15
```

也可以逐項指定：`--split true --rotate true --index true --split-mode cdc --topk-ratio 0 --audit false`。

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

**這行只拿來確認「有在運作」，不要當成報告數字。** 排隊中的請求每一輪排程都會重新掃描一次
（`scheduler.py` 對每個等待中的請求呼叫 `init_next_round_input`），排不進去的下一輪再掃、再累加，
所以負載高時 `M`、`Y`、`K` 都會偏高。報告用的命中率請用 `SGLANG_FORWARD_TRACE` 的逐 pass 紀錄，
或 client 端每個請求的 `usage.prompt_tokens_details.cached_tokens`（需要 `--enable-cache-report`）。

`K` 偏高表示很多 request 的「待算 token 數」超過一個 prefill pass 的預算（有洞的 prefill 不能切成多段）。
這時把 `--chunked-prefill-size` 調大（24GB 的卡預設只有 2048）。代價是 activation memory 變多、KV cache 變小，
值得量一下再決定。

---

## 已知限制

- **短於 `SGLANG_SUBCTX_MIN_CHUNK` 的區塊永遠不寫入快取。** cdc 在每則訊息開頭都切，所以很短的訊息
  （例如短的終端機輸出）會自成一塊，之後每一輪都重算。Terminal-Bench 上這是 cdc 命中率低於原生 radix cache
  的主要原因（2026-10-02 初步估計每請求約 370 token）。
- **cdc 模式下生成的 token 不會進快取。** 不是因為生成的 token 短，而是因為它們要接的那一塊：
  - 存輸出的條件是「整個 prompt 都已寫進樹」，然後把生成的 token 接在 prompt 最後一塊後面存進去。
  - prompt 最後是 generation prompt（`<|im_start|>assistant\n`，3 個 token）。cdc 在每個 `<|im_start|>` 前都切，
    所以它自成一塊，短於 `MIN_CHUNK`，永遠不寫入 → 條件永遠不成立 → 生成的 token 直接被釋放。
  - 就算放寬這個條件，存下來的輸出也不會登記進索引；下一輪的回覆是一則新的 assistant 訊息，cdc 依內容雜湊
    找區塊，找不到它。所以每一輪模型的回覆都要重算（blocks 模式沒有這個問題：最後一塊是整段 `messages`）。
- **開了 selective recompute 時，重用的區塊不會以新位置寫回快取**，所以位移過一次的內容之後每個請求都要再旋轉一次。

---

## 拿原生 radix cache 當對照組

同一個 binary、同一份權重，只改環境變數：

```bash
SGLANG_DISABLE_SUBCONTEXT=1 \
python -u -m sglang.launch_server --model-path ... --attention-backend triton ...
```

（或 `ARM=off bash sglang_server.sh`。）兩邊要用同一個 attention backend，GPU 時間才能比。

驗證：

```bash
python scripts/subcontext_sim/check_remote_arm.py http://HOST:30000 --arm off
```
