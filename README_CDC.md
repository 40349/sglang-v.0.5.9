# 啟用 CDC sub-context KV 重用（server 端）

這份講的是怎麼把這個 fork 的 sglang server 用 CDC 模式跑起來，以及怎麼確認它真的在運作。
機制本身（切塊、查找、旋轉、寫入、釋放）見 `subcontext_mechanism.md`。

一句話說明（2026-10-05 起）：每個請求先像原生 radix cache 一樣從頭比對；比不到的部分，用內容索引在 prompt
任何位置找已快取的區塊，複製（位置不同就旋轉）進自己的格子，只算空洞；算完整條照原生方式存回樹，
新存的區塊登記進索引。只往後接的對話因此跟原生 radix cache 一樣。

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

# 2. 啟動（沒設 SGLANG_DISABLE_SUBCONTEXT 就是 cdc）
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
要重算一部分重用的 token 就把比例寫進 arm，例如 `ARM=cdc@0.15`；逐層篩選寫成一串比例，例如
`ARM=cdc@0.3,0.2,0.15`（從 `TOPK_LAYER` 起連續幾層各挑一次，每次只從上一層留下的裡面挑）。arm 只有 `off` 和
`cdc` 兩種，對應的環境變數寫在 `sglang_server.sh` 的 `export SGLANG_DISABLE_SUBCONTEXT= ...` 那段；其他參數寫在檔案開頭的註解。
環境裡若已經直接設了 arm 會設定的開關（例如 `SGLANG_DISABLE_SUBCONTEXT`、`SGLANG_SUBCTX_TOPK_RATIO`，完整清單在
同檔 `for v in SGLANG_DISABLE_SUBCONTEXT ...` 那段），這支腳本會拒絕啟動，開關只能透過 arm 和它的參數設定。

---

## ⚠️ 先確認這件事：只有 chat endpoint 會切分

切分邏輯只存在於 `python/sglang/srt/entrypoints/openai/serving_chat.py`（`_compute_sub_context_ends`）。

- **只有 `/v1/chat/completions` 會啟用 CDC。** `/v1/completions` 不切；`/generate` 會在入口把切塊欄位清掉
  （`http_server.py`），一律走原生路徑。
- chat 請求裡，只有走 jinja chat template 的路徑會切；多模態請求（帶圖片／影片／音訊）不切。
- 不切的請求**不會報任何錯**——就只是沒效果，重用率跟原生 radix cache 一樣。

開始測之前先確認你的 client 走哪個 endpoint。

---

## 環境變數

| 變數 | 值 | 作用 |
|---|---|---|
| `SGLANG_DISABLE_SUBCONTEXT` | **不要設** | 不設就是 cdc：依內容切塊、前綴比對 + 掃描比對 + sparse prefill、位移的命中用 RoPE 旋轉搬到正確位置、整條存成一條鏈。設成 1 就整個關掉，退回原生 radix cache |

2026-10-14 起 on / rot / idx 三種 arm 已移除，`SGLANG_SUBCTX_SPLIT`、`SGLANG_SUBCTX_INDEX`、
`SGLANG_SUBCONTEXT_ROTATE`、`SGLANG_SUBCONTEXT_ROTATE_ACROSS`、`SGLANG_SUBCTX_HASH_KEYS`、
`SGLANG_SUBCTX_INDEX_DRYRUN`、`SGLANG_SUBCONTEXT_CACHE_OUTPUT` 設了任何一個都會在 import 時拒絕啟動。舊程式在 git 歷史（`origin/idx` 5ff1e41）。

可調（有預設值，不設也能跑）：

| 變數 | 預設 | 說明 |
|---|---|---|
| `SGLANG_SUBCTX_CDC_TARGET` | 256 | 候選切點的平均間距，會無條件捨去到 2 的冪。不是 chunk 平均長度：長訊息內的 chunk 平均約 `MIN_CHUNK` + target（預設約 320），短訊息整則一塊 |
| `SGLANG_SUBCTX_CDC_MAX` | 1024 | 強制切點的長度上限，應明顯大於 target |
| `SGLANG_SUBCTX_MIN_CHUNK` | 64 | 最短 chunk，也是 cdc 切點的最小間距。比它短的區塊照樣隨整條鏈存進快取，但不登記索引（位置變了就找不回來）；索引的下限是 32（指紋視窗就是 32 個 token），設更低時索引仍以 32 計 |

Selective recompute（重算一部分重用的 token，修正「在別的前文下算出來」造成的誤差）：

| 變數 | 預設 | 說明 |
|---|---|---|
| `SGLANG_SUBCTX_TOPK_RATIO` | 0 | 每個請求重算重用 token 中 key 偏差最大的比例，範圍 [0, 1]。逗號分隔（不遞增）時每個比例對應一個連續的打分層。> 0 時不能同時設 `SGLANG_DISABLE_SUBCONTEXT=1` |
| `SGLANG_SUBCTX_TOPK_LAYER` | 1 | 第一個打分層。0 只允許第一個比例 = 1.0（第 0 層的 key 只取決於 token 與位置，分數全為 0） |

probe 各層（`[0, TOPK_LAYER]`）每一列都把新算的 KV 寫回自己的 slot，attention 讀整段新算的 KV，所以 ratio 1.0
在任何打分層都等於整段重算。

ratio > 0 時，啟動還會檢查（不符就拒絕）：模型要有 `forward_split_prefill`、最後一個打分層之後至少還有一層、
`--tp-size 1`、不能用 pipeline parallel、DP attention、`--moe-dense-tp-size 1`、two-batch overlap、
speculative decoding、torch.compile、回傳 hidden states、LoRA。

RelayCaching（arXiv 2603.13289，照官方程式 github.com/YingshengGeng/RelayCaching 移植；`sglang_server.sh` 的
arm 寫成 `ARM=relay@L_start,L_det,L_end` 或 `ARM=relay@profile`，機制見 `subcontext_mechanism.md` 第 8R 節）：

| 變數 | 預設 | 說明 |
|---|---|---|
| `SGLANG_SUBCTX_RELAY_LAYERS` | 不設 | `L_start,L_det,L_end`，要 `0 <= L_start < L_det <= L_end < 層數`。decode 存下每個輸出 token 在 `L_start` 輸入的 hidden state，以及它被後面輸出 token 注意的程度；prefill 時重用的 token 從 `L_start` 接回、在 `L_det` 挑選、選中的跑到 `L_end` |
| `SGLANG_SUBCTX_RELAY_TAU_DEV` | 1.5 | `L_det` 的 value 偏差超過「這個請求平均的 1.5 倍」的 token 被選中；照官方以 KV dtype（bf16）計算，請求的平均偏差低於約 0.002 時延到下一層再挑 |
| `SGLANG_SUBCTX_RELAY_TAU_INF` | 1.45 | 每段重用內容依「被注意程度」預先選 token 的門檻倍數 |
| `SGLANG_SUBCTX_RELAY_SUFFIX` | 10 | 每段重用內容最後幾個 token 一定重算（算在上面那個預算裡） |
| `SGLANG_SUBCTX_RELAY_PROFILE` | 不設 | 一個檔案路徑（每次啟動清空）：每個重用 token 每一層都重算，並把每層的偏差寫進去（每段連續重用內容一行）；`scripts/subcontext_sim/relay_layers.py` 依官方規則從它挑出三個層。不能和 `RELAY_LAYERS` 同時設 |

relay 不能和 `SGLANG_SUBCTX_TOPK_RATIO > 0` 同時開；啟動檢查與 ratio > 0 相同，另外要求模型有 `model.layers`、
每層恰好一個 `RadixAttention`、KV pool 是 `MHATokenToKVPool`、不能用 `--enable-pdmux`。
`L_start > 0` 時每個 KV slot 多存一份 hidden state（Qwen2.5-7B 每個 token 多 7 KB，KV pool 少約 11% 的 token），
每個 slot 另外多 5 bytes（被注意程度與「是否由 decode 算出」）。

診斷用（**會讓效能數字失真，量測時不要開**）：

| 變數 | 作用 |
|---|---|
| `SGLANG_SUBCTX_TRACE=1` | 逐筆比對的 trace（印到 server log） |
| `SGLANG_SUBCTX_AUDIT=1` | KV pool 的 double-free 偵測，以及重用 chunk 時比對它所在位置的 token |
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
- `--enable-mixed-chunk`（預設是關的；它把 decode 的 token 接進 prefill batch，sparse prefill 給的位置只涵蓋 prefill 的 token）
- `--enable-pdmux`（PD multiplexing 的 forward 跑在自己的 stream 上，旋轉的複製排不到它後面）
- diffusion LLM（`--dllm-algorithm`）、prefill/decode 分離（`--disaggregation-mode`）：那兩條排程路徑不會放掉被拒請求掃描時拿的鎖與副本
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

`ARM=cdc` 應該看到（`cdc@0.15` 只差 `topk_ratios`；`relay@3,4,22` 是 `"relay_layers": [3, 4, 22]`，
`relay@profile` 是 `"relay_profile": true`）：

```json
{
  "split_enabled": true,
  "min_chunk": 64,
  "cdc_target": 256,
  "cdc_max": 1024,
  "topk_ratios": [0.0],
  "topk_layer": 1,
  "relay_layers": null,
  "relay_profile": false,
  "relay_tau_dev": 1.5,
  "relay_tau_inf": 1.45,
  "relay_suffix": 10,
  "audit": false,
  "pid": 12345
}
```

`pid` 用來分辨「剛啟動的這個 server」和「還佔著 port 的舊 server」。
還回報 `split_mode` 等欄位的 server 跑的是 2026-10-14 以前的 build，`check_remote_arm.py` 會拒絕它。

repo 裡有現成的檢查腳本，對不上會 refuse 並說明差在哪：

```bash
python scripts/subcontext_sim/check_remote_arm.py http://HOST:30000 --arm cdc
python scripts/subcontext_sim/check_remote_arm.py http://HOST:30000 --arm cdc@0.15
python scripts/subcontext_sim/check_remote_arm.py http://HOST:30000 --arm relay@3,4,22
```

另外可以加 `--audit true|false` 要求 audit 開或關。

---

## 確認它真的在重用（不只是「開著」）

跑一段流量之後看 server log：

```bash
grep "sub-context index:" server.log | tail -1
```

```
sub-context index: N chunks, M queries, X tokens scanned | matched P prompt tokens: prefix A (a%),
scan found B (b%), W of them rotated into place | K requests too big for one pass (D found tokens dropped), R refused for room
```

- `a%` 是從頭比對到的部分（原生 radix cache 也拿得到）。
- `b%` 是掃描找到的部分 — 原生 radix cache 結構上拿不到的；`W` 是其中靠旋轉撿回來的量。
- `K` 是放不進一次 prefill、退回只用前綴的 request 數，`D` 是這些 request 因此丟掉的掃描命中 token 數；`R` 是沒空間被拒、之後只用前綴的 request 數。

`P`、`A`、`B`、`K`、`D` 是每個請求在真的執行時算一次（被拒、排隊重新比對的那幾次不算），4090 smoke 中與 client 端
每個請求的 `usage.prompt_tokens`、`usage.prompt_tokens_details.cached_tokens` 加總完全一致（需要 `--enable-cache-report`）。
開頭的 `M`、`X` 仍是每次掃描都算，負載高時會偏高。報告用的數字仍建議用 client 端的 `cached_tokens` 或 `SGLANG_FORWARD_TRACE`。

`K` 偏高表示很多 request 的「待算 token 數」超過一個 prefill pass 的預算（有洞的 prefill 不能切成多段）。
這時把 `--chunked-prefill-size` 調大（24GB 的卡預設只有 2048）。代價是 activation memory 變多、KV cache 變小，
值得量一下再決定。

---

## 已知限制

- **短於 `SGLANG_SUBCTX_MIN_CHUNK` 的區塊不登記索引。** 它們在鏈上、從頭比對拿得到，但位置一變（例如前面的
  歷史被省略）就找不回來。
- **開了 selective recompute 時，重算過的副本會隨整條鏈存回去**，下一輪從頭比對到時當成精確前綴直接用，不再重算。
- **位置相同也要複製一份**，GPU 記憶體用量跟原生 radix cache 一樣。
- **開了 selective recompute 時，要求 logprob 的請求只用前綴比對**（重算會改掉 logprob 對齊用的列）。
- 2026-10-05 以前的版本（每塊一個 namespace）有兩個已解決的問題：短訊息不寫入、模型回覆不進快取，
  Terminal-Bench 上 cdc 命中率因此低於原生 radix cache。2026-10-05 以前量到的 cdc 數字都是那個版本的。

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
