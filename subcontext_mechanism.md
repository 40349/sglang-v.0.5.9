# Sub-context KV 重用：機制說明

這份文件講的是這個 fork（`idx` 分支，依 2026-10-05 的程式碼）的 sub-context KV 重用機制：
怎麼切塊、怎麼找到可重用的 KV、位置不同時怎麼旋轉、怎麼只算空洞、怎麼重算一部分重用的 token、
算完怎麼寫回快取與釋放，以及為什麼是這樣寫的。怎麼啟動 server 見 `README_CDC.md`。

主要涉及的檔案（都在 `python/sglang/srt/` 下）：

| 檔案 | 負責 |
|---|---|
| `entrypoints/openai/serving_chat.py` | 切塊（blocks / cdc）、錄製請求 |
| `managers/tokenizer_manager.py`、`managers/io_struct.py` | 驗證切塊與 input_ids 一致、自動截斷後裁齊、往下傳 |
| `managers/scheduler.py` | 啟動檢查、掛上旋轉器與索引；沒被收進這一輪的請求放掉掃描結果 |
| `managers/schedule_batch.py` | 讀路徑：stitch 與 scan、旋轉/複製副本的生命週期、sparse prefill 的輸入 |
| `managers/schedule_policy.py` | `PrefillAdder`：Stage 2 切 chunk、sparse prefill 不可切、probe 預算 |
| `mem_cache/radix_cache.py` | 寫路徑：逐 namespace 插入（`on`/`rot`）、鏈式請求的登記與 chunk 位置維護（`idx`/`cdc`）、finish 的搶救與釋放、audit |
| `mem_cache/rotate_kv.py` | Triton 旋轉 kernel、KV 複製、啟動時的 RoPE 性質自測 |
| `mem_cache/subctx_index.py` | 內容索引（依 token 內容找已快取的區塊、記錄它們在樹上的位置）、CDC 切點 |
| `mem_cache/subctx_blend.py` | selective recompute：打分、挑選、寫回 |
| `mem_cache/common.py` | sparse prefill 的 `req_to_token` 寫入、重算 token 的額外 slot |
| `layers/attention/triton_backend.py`、`triton_ops/extend_attention.py` | 依 query 絕對位置的 causal 遮罩 |
| `model_executor/forward_batch_info.py`、`model_executor/model_runner.py` | 給定位置的 prefill、selective recompute 的兩段 layer forward |
| `managers/forward_trace.py`、`utils/host_timer.py`、`utils/subctx_trace.py` | 量測與 trace |
| `utils/subctx_config.py` | 所有開關、import 時的交叉檢查、「不支援原因」的判斷 |

---

## 0. 這套機制在解決什麼

一般的 radix cache 把整個 prompt 當成**一條** key。agent 的 prompt 長這樣：

```
[system prompt][tools 定義][對話歷史]
```

只要前面差一個字，整條 key 就從那裡開始不同，後面的內容全部重算——即使它們一模一樣。

這個 fork 分四層處理，每一層都建立在前一層之上：

1. **切塊 + namespace**（arm `on`）：把 prompt 切成幾塊，每塊放進自己的 namespace（radix tree 裡以
   `extra_key` 區隔、彼此不共用的子樹），各自查詢。前一塊變了，後一塊仍可能命中。
2. **旋轉**（arm `rot`）：同一塊內容出現在不同位置時，把快取的 K 轉到新位置再用。
3. **內容索引 + sparse prefill**（arm `idx`、`cdc`）：先照原生 radix cache 從頭比對；比不到的部分，
   再用內容索引在 prompt 的**任何位置**找已快取的區塊，複製到自己的格子（位置不同就旋轉），只計算中間的空洞。
   算完整條照原生方式存回樹（一條從位置 0 開始的鏈），新存上樹的區塊登記進索引、記下它在樹上的位置。
   `cdc` 再把切點改成依內容決定，不依角色。
4. **selective recompute**（arm `idx@r`、`cdc@r`）：重用的區塊是在別的前文下算的，挑出 key 偏差最大的
   比例 r 重算，修正品質。`cdc@r1,r2,…` 在連續幾層逐層篩選。

**位置的代價**：RoPE 把絕對位置烤進了 K。一塊 KV 是在位置 100 算的，就只能在位置 100 被直接重用。
所以每個 tree node 要記 `canonical_position`——「我這份 KV 是在哪個絕對位置算出來的」。

**旋轉**：RoPE 的角度對位置是線性的，所以 `R(a)·R(b) = R(a+b)`，可以把一塊 KV 旋轉 `Δ` 搬到新位置：

```
R(p_new) · f(k_raw) == R(p_new − p) · k_cached[loc]
```

`f` 是 RoPE 之前對 K 做的事（Qwen3 是 k_norm）。這在位置上**完全精確**——RoPE 保範數，
所以不需要重跑 `f`。V 與位置無關，直接複製。

但它修的是**位置**，修不了**前文**。那塊當初是在別的上下文下算出來的，這是品質代價，
也就是 pass@1 要量的東西，也是第 4 層要修正的東西。

---

## 1. 開關與 arm

所有開關都是環境變數，在 import 時讀一次；fork 沒有新增任何命令列參數。

### arm（`sglang_server.sh` 第 144–153 行）

| arm | 設定的環境變數 | 走哪條讀路徑 |
|---|---|---|
| `off` | `SGLANG_DISABLE_SUBCONTEXT=1` | 不切塊，原生 radix cache |
| `on` | （全部不設） | blocks 切塊 + stitch；位置不對的命中丟掉 |
| `rot` | `SGLANG_SUBCONTEXT_ROTATE=1`、`SGLANG_SUBCONTEXT_ROTATE_ACROSS=1` | stitch + Stage 1 + Stage 2 旋轉 |
| `idx[@r]` | `ROTATE=1`、`SGLANG_SUBCTX_INDEX=1`、`SGLANG_SUBCTX_TOPK_RATIO=r` | blocks 切塊；前綴比對 + scan + sparse prefill；整條存成一條鏈 |
| `cdc[@r]` | `ROTATE=1`、`INDEX=1`、`SGLANG_SUBCTX_SPLIT=cdc`、`TOPK_RATIO=r` | 依內容切塊；其餘同 `idx` |

### 全部環境變數

| 環境變數 | 預設 | 作用 |
|---|---|---|
| `SGLANG_DISABLE_SUBCONTEXT` | 空 | 非 0：chat 不切塊、不做啟動檢查、不掛旋轉器與索引（即使其他開關有設） |
| `SGLANG_SUBCONTEXT_CACHE_OUTPUT` | **1** | 結束時把「最後一塊 ++ 生成的 token」存進最後一塊的 namespace（只用在逐 namespace 的 `on`/`rot`；`idx`/`cdc` 的回覆隨整條鏈存） |
| `SGLANG_SUBCONTEXT_ROTATE` | 空 | Stage 1：stitch 時旋轉位移的命中；finish 時就地旋轉 |
| `SGLANG_SUBCONTEXT_ROTATE_ACROSS` | 空 | Stage 2：prefill chunk 切在 block 邊界，下一輪旋轉接上（也會掛上旋轉器） |
| `SGLANG_SUBCONTEXT_ROTATE_NATIVE` | 空 | 用純 torch 旋轉取代 Triton kernel（除錯） |
| `SGLANG_SUBCTX_HASH_KEYS` | 空 | namespace 改用內容雜湊命名，仍走 stitch |
| `SGLANG_SUBCTX_INDEX` | 空 | 鏈式儲存：前綴比對 + scan + sparse prefill，存回時登記索引（隱含內容雜湊命名） |
| `SGLANG_SUBCTX_INDEX_DRYRUN` | 空 | 只統計 scan 能找到什麼，服務仍走 stitch |
| `SGLANG_SUBCTX_MIN_CHUNK` | 64 | 索引登記與重用的最短長度；也是 cdc 切點的最小間距 |
| `SGLANG_SUBCTX_SPLIT` | `blocks` | `blocks`（依角色）或 `cdc`（依內容） |
| `SGLANG_SUBCTX_CDC_TARGET` | 256 | cdc 平均 chunk 長度（捨去到 2 的冪） |
| `SGLANG_SUBCTX_CDC_MAX` | 1024 | cdc 強制切斷長度 |
| `SGLANG_SUBCTX_TOPK_RATIO` | 0 | selective recompute 重算的比例；逗號分隔時每個比例對應一個連續的打分層（逐層篩選） |
| `SGLANG_SUBCTX_TOPK_LAYER` | 1 | 第一個打分層 |
| `SGLANG_SUBCTX_ROTATE_GPU` | 空 | 旋轉 kernel 的 CUDA event 計時（`rotate_kv.py`） |
| `SGLANG_SUBCTX_AUDIT` | 空 | slot 歸屬與 double-free 檢查（`radix_cache.py`、`allocator.py`） |
| `SGLANG_SUBCTX_TRACE` | 空 | 印 TRACE 除錯行 |
| `SGLANG_STAGE_TRACE` | 空 | host 階段計時的輸出前綴 |
| `SGLANG_FORWARD_TRACE` | 空 | 每個 forward pass 的 GPU 時間輸出檔 |
| `SGLANG_DUMP_TREE` | 空 | 每次 extend 後印整棵 radix tree |
| `SGLANG_CAPTURE_REQUESTS` | 空 | 錄下每個 chat 請求的輸出檔 |

`SGLANG_SUBCTX_AUDIT` 在兩處的解析方式不同：`radix_cache.py` 是「非空且不是 0 即開」，
`allocator.py` 用 `get_bool_env_var`（只認 `true` / `1`）。設成 `1` 兩邊都開。

### import 時的交叉檢查（`subctx_config.py`）

不合法直接丟 `ValueError`，server 起不來：

- `SPLIT` 只能是 `blocks` 或 `cdc`。
- `cdc` 必須搭配內容雜湊命名（`INDEX`、`HASH_KEYS` 或 `INDEX_DRYRUN`），除非設了 `DISABLE`。
- `TOPK_RATIO` 的每個比例都必須在 [0, 1] 且不遞增；第一個 > 0 時必須開 `INDEX`（這條即使設了 `DISABLE` 也會檢查）。
- `TOPK_LAYER` 不可為負；`TOPK_LAYER = 0` 只允許第一個比例 = 1.0（第 0 層的 key 只取決於 token 與位置，分數全為 0）。

---

## 2. 四份狀態

| | 內容 | 在哪 |
|---|---|---|
| `k_buffer` / `v_buffer` | 真正的 KV bytes，`[slot, head, dim]` | GPU |
| `req_to_token[req, 位置]` | **位置 → slot 的權威對照表** | GPU tensor |
| tree node | `key`（token）、`value`（slot）、`extra_key`（namespace）、`canonical_position` | CPU |
| `free_pages` | 哪些 slot 是空的 | CPU |

**GPU 上的 bytes 不帶身分。** 一個 slot 是誰的，完全由「誰指向它」決定。
這句話是後面所有規則的來源。

內容索引（`SubContextIndex`）是第五份狀態：存 token id、chunk id，以及每個 chunk 在樹上的位置
（「持有它最後一個 token 的節點 + 它第一個 token 的位置」，可以有好幾個，最早的優先）。不存 KV；
KV 在不在還是由 radix tree 決定。樹拆節點、淘汰節點時同步更新位置（見第 7 節）。

### request 身上的欄位（`Req.__init__`）

| 欄位 | 意思 | 誰寫 |
|---|---|---|
| `sub_context_ids` / `sub_context_extra_keys` | 每塊的 token 與 namespace；`concat(ids) == origin_input_ids` | 切塊（scan 設回收到時的切法） |
| `sub_context_ids_as_sent` | 收到時的原始切法；鏈式儲存登記索引時的區塊邊界 | 建構時 |
| `sub_context_match_nodes/_indices/_positions/_lens` | 排程時查到的 node、slot、canonical、命中長度 | stitch（scan 只留 node） |
| `sub_context_owned_lens[i]` | 第 i 塊已經被算進 prefix 或已插入的長度 | stitch → 每個 chunk 更新 |
| `sub_context_tree_owned[i]` | 第 i 塊**現在是不是樹的** | 只有寫入那趟寫 |
| `sub_context_tree_canonical[i]` | 樹把第 i 塊登記在哪個位置 | 同上 |
| `sub_context_last_nodes` | 寫入時上鎖的各 namespace node | 寫入；finish 釋放 |
| `sub_context_rotated_slots` | 自己 alloc 的旋轉／複製副本，**還沒交給 `req_to_token`** | stitch / scan / append |
| `sub_context_next_boundary` | Stage 2 要把 chunk 切在哪 | stitch |
| `sub_context_layout` / `_layout_ours` | scan 的稀疏配置 `(start, end, slots)`，以及該段是否為自己的副本；第一段可能是從頭比對到的前綴 | scan |
| `sub_context_exact_len` | 從頭比對到的前綴長度；這段是精確的，selective recompute 不打分 | scan |
| `sub_context_index_declined` | 沒空間被拒、又沒有其他請求在跑；之後只用前綴比對 | scheduler |
| `sub_context_no_insert[i]` / `sub_context_source_key[i]` | 舊 scan 留下的欄位：逐 namespace 的寫入路徑還會讀，但現在沒有路徑會設成非 None | — |
| `sub_context_discarded/_moved/_rotated/_reinserted` | 給 forward trace 的計數，讀完歸零 | 各路徑 |
| `cache_protected_len` | 從 0 起連續被樹擁有的長度（上游欄位的有損投影） | 每階段重算 |

---

## 3. 啟動：能不能跑

沒設 `SGLANG_DISABLE_SUBCONTEXT` 時，`Scheduler._check_sub_context_support` 一定會跑。
條件不成立就 **raise，不啟動**——每一種不支援的組合失敗的方式都一樣：attention 還是回傳一個數字，只是錯的。

### 3-1 切分能不能服務（`unsupported_reason`）

| 條件 | 不成立的話 |
|---|---|
| 是原生 `RadixCache`（不是子類、不是 C++ 版、不是 hierarchical） | 逐 namespace 插入路徑不存在 |
| 不是 hybrid SWA / mamba 模型 | 它們用 `SWARadixCache` / `MambaRadixCache`，同上 |
| `page_size == 1` | 分頁會把 slot 綁成一組，塊邊界對不齊 |
| 不是 EAGLE | 它把 key 改寫成 bigram |
| 沒有 `--disable-radix-cache` | 沒有樹 |

### 3-2 旋轉能不能做（`rotation_unsupported_reason`，開 `ROTATE` 或 `ROTATE_ACROSS` 時）

| 檢查 | 為什麼 |
|---|---|
| 模型裡只有一個 `RotaryEmbedding` | 多模態模型常有第二顆（視覺塔），不知道該用哪顆 |
| 不是 mrope | 位置是 3-vector（text/height/width），沒有單一 delta |
| 不是 linear scaling | cos_sin_cache 是多份串接（一個 scaling factor 一份），列號不等於位置 |
| neox 配對、`rotary_dim == head_dim` | kernel 的假設 |
| MHA pool、`store_dtype == dtype` | 排除 MLA 和量化 KV（`--kv-cache-dtype fp8_*` 會以 `uint8` 儲存） |
| **性質自測** | 實測 `R(p+δ)·k == R(δ)·(R(p)·k)` |

**性質自測**取代了原本的型別白名單（只准 `type(rotary) is RotaryEmbedding`）。
它用模型自己的 `forward_native` 當 oracle，對八組 `(p, δ)` 樣本實測，位置最遠打到 30000。

門檻不是拍腦袋的定值，而是從 fp32 精度推出來的預算：

```python
budget = tol + 4.0 * max(abs(p), abs(p + d)) * 2**-24
```

理由是 `cos_sin_cache` 把 `position × inv_freq` 存成 fp32，位置 P 的那一列本身就帶著
`~P·2⁻²⁴` 弧度的捨入，而這個恆等式要讀三列。實測佐證：同一條律用 fp64 cache 在每組樣本
都吻合到 8e-8，而出貨的 fp32 cache 到位置 30000 漂到 6e-4。用固定門檻 1e-3 會把
dynamic NTK 的 1.04e-3 誤判成性質破缺。

實測結果：

| rope_type | 結果 | 最差樣本用掉的預算 |
|---|---|---|
| default | 通過 | 8% |
| llama3 | 通過 | 8% |
| yarn | 通過 | 8% |
| dynamic | 通過 | 14% |

兩個負控制（位置 8192 之後換 `inv_freq` 的 cache、YaRN 不除 mscale）分別以
1.92e+00 和 1.39e-01 被拒絕，差 3–4 個數量級。

**YaRN 除 mscale**：YaRN 家族存的是 `mscale·cos` / `mscale·sin`。那個純量已經烤在
被旋轉的那份 K 裡了，用原始 row 會讓 K 每跳一次就再乘一次 mscale。做法不是特判 YaRN，
而是把 row 正規化回單位旋轉（`sqrt(cos²+sin²)`）——對 plain RoPE 是空操作
（實測只改變 1468 萬個 bf16 元素裡的 52 個，各 1 ulp）。

**mrope 和 linear scaling 仍然用結構性否決**，因為量測看不到它們的破法：
mrope 餵純量位置時**會通過**自測（它就是 plain RoPE），真正壞掉的是圖片來的 3-vector 位置。
這件事本身寫成了一個測試，否則那個否決看起來像多餘的。

**明確開了旋轉但條件不成立就不啟動**，因為不支援的 RoPE 不會失敗，
它會用錯的律去轉，然後安靜地讓每個重用的塊都變差。

### 3-3 sparse prefill 能不能做（`sparse_prefill_unsupported_reason`，開 `INDEX` 時）

| 條件 | 為什麼 |
|---|---|
| `--attention-backend triton` | 只有 triton 的 kernel 收 per-position 遮罩；其他 backend 以索引判斷 causal |
| 不是 sliding-window attention | 它從前綴長度推每個 key 的絕對位置，有洞的 prefill 沒有那個長度 |
| 沒有 `--enable-piecewise-cuda-graph` | 它 capture prefill 的形狀，而稀疏 prefill 的遮罩每個請求大小不同 |
| 沒有 `--enable-mixed-chunk` | 它把 decode 的 token 接進 prefill batch，而 sparse prefill 給的位置只涵蓋 prefill 的 token |

另外：**開了 index 卻沒有旋轉器就拒絕啟動**——位置不同的區塊會全部被丟掉，index 就沒有意義。
**開了 index 又開 diffusion LLM 或 prefill/decode 分離也拒絕**：那兩條排程路徑不會放掉被拒請求 scan 拿的鎖與副本（第 7-4 節）。
**開了旋轉又開 `--enable-pdmux` 也拒絕**：PD multiplexing 的 forward 跑在自己的 stream 上，旋轉的複製排不到它後面。

### 3-4 selective recompute 能不能做（`topk_unsupported_reason`，ratio > 0 時）

| 條件 | 為什麼 |
|---|---|
| 模型有 `forward_split_prefill` | token 數要在打分層之間裁減 |
| 最後一個打分層之後至少還有一層 | 否則沒有東西要重算 |
| 沒有 pipeline parallel | 被比較的那層可能在別的 rank |
| attention TP = 1 | 分數對 head 維度加總，TP 下每個 rank 只看到自己的 head，會選出不同的 token |
| 沒有 DP attention、`--moe-dense-tp-size 1` | token 被分散到各 rank，全域的列號在每個 rank 指到不同 token |
| 沒有 two-batch overlap、speculative decoding、torch.compile、回傳 hidden states、LoRA | 都假設整個 forward 的 token 數不變 |

哪些模型過得了，見第 16 節。

---

## 4. 生命週期總覽

```
1. 切塊（serving_chat）      chat request → 幾個 block，各帶一個 namespace            ← 第 5 節
2. 查找（init_next_round_input）
   ├─ stitch（on / rot）     逐 namespace 查詢，只接受從位置 0 起連續的命中        ← 第 6 節
   └─ scan（idx / cdc）      先從頭比對，其餘用內容索引在任何位置找、複製進自己的格子 ← 第 7 節
3. PrefillAdder             Stage 2 把 chunk 切在 block 邊界；有洞的 prefill 不可切；
                            沒收進這一輪的 scan 請求放掉鎖與副本                    ← 第 7-4 節
4. prepare_for_extend       prefix → req_to_token，責任交接；sparse 時依位置取輸入  ← 第 8 節
5. prefill                  算 KV 並產生第一個 token；ratio > 0 時兩段 layer          ← 第 8、9 節
6. cache_unfinished_req     on/rot：逐塊插入；idx/cdc：整條插入、登記新存的區塊      ← 第 10、7-5 節
   └─ append                Stage 2
7. decode                   一個一個吐 token
8. cache_finished_req       on/rot：搶救 → 存回覆 → 還鑰匙；idx/cdc：整條插入（含回覆）← 第 11、7-5 節
```

其中第 6 步**不是每個 request 都會跑**（見第 14 節）。

兩個判斷，三個地方（第 2、6、8 步）都用同一套，必須一致，否則一個 request 會用一條路讀、用另一條路寫：

- `chains_sub_contexts(req)`：開了 `INDEX`、有索引、request 有切塊、cache 支援 → 走第 7 節的鏈式路徑，
  寫入走上游的整條插入。
- `serves_sub_contexts(req)`（= 有切塊、cache 支援、且不是上一種）→ 走逐 namespace 的路徑。

---

## 5. 切塊（`serving_chat._compute_sub_context_ids`）

### 5-1 哪些請求會切

- **只有 `/v1/chat/completions`**，而且只在 jinja chat template 的路徑；多模態請求不切
  （它會把 prompt decode 成文字再重新 tokenize）。
- `/generate` 在 handler 入口把 `sub_context_ids` / `sub_context_extra_keys` 清成 `None`，永遠不切。
- 設了 `DISABLE` 不切。
- 切出來少於 2 塊就當作不切。

### 5-2 blocks 模式（依角色）

1. 數出開頭連續的 system 訊息數 `n_sys`。
2. 依序只 render 前 `n_sys`、`n_sys+1` 則訊息（沒有 system 時試 1、2 則），找第一個
   「render 結果正好是完整 prompt 前綴」的切法——第二個候選是給把 tools 放進第一則 user 訊息的模板（如 Llama 3.x）。
   找不到就不切。
3. 有 tools 時再 render 一次不帶 tools 的版本；它若是帶 tools 版本的前綴，就切成
   `system_prompt_key` + `tools_key`，否則兩者合成一塊 `system_prompt_key`。
4. 其餘全部是 `messages_key`。

每個 chat request 因此多 render 1~3 次 template（host stage `subctx_split`）。

**Qwen3-Coder 的模板切不出 `tools_key`**：它把 tools 放在 system 訊息裡面
（`<|im_start|>system\n{system}\n\n# Tools ...<|im_end|>`），不帶 tools 的版本是以 `<|im_end|>` 結尾，
不是帶 tools 版本的前綴，所以 system 與 tools 永遠是同一塊。在這個模型上，blocks 的每一個切點都是訊息邊界。

### 5-3 cdc 模式（依內容）

1. **訊息開頭一定切**：從模板 render 一則與兩則訊息，第二則的第一個 token 若是特殊 token，就當成
   「每則訊息的開頭標記」（Qwen 是 `<|im_start|>`），只算一次。prompt 裡每個這種 token 的位置都切一刀。
   找不到標記就只依內容切。
2. **每則訊息內依內容細切**（`subctx_index.cut_points`）：對每個位置算「從這裡開始 32 個 token」的指紋，
   指紋低 `log2(CDC_TARGET)` 個位元全為 0 的位置是候選切點。距上一個切點不足 `MIN_CHUNK` 的忽略、
   超過 `CDC_MAX` 強制切、尾段短於 `MIN_CHUNK` 就併入前一塊；訊息短於 `2 × MIN_CHUNK` 不細切。
   同樣的內容不論出現在哪裡都切在同樣的地方。
3. namespace 先給佔位名 `cdc_0, cdc_1, …`，`Req.__init__` 會換成內容雜湊（cdc 一定搭配內容雜湊）。

短於 `MIN_CHUNK` 的訊息、prompt 最後的 generation prompt（`<|im_start|>assistant\n`，3 個 token）
都自成一塊。鏈式儲存下它們照樣隨整條鏈存進樹、從頭比對拿得到，只是不登記進索引——
位置變了就找不回來。（舊的逐塊儲存下它們永遠不寫入，見第 15 節。）

### 5-4 namespace 怎麼命名

| 條件 | namespace | 例 |
|---|---|---|
| blocks 模式，沒開 `HASH_KEYS` / `INDEX` / `DRYRUN` | 角色名 | `system_prompt_key`、`tools_key`、`messages_key` |
| 開了任一個（cdc 一定是） | `chunk_id(block 的 token, req.extra_key)` | `sc:<xxh3-64>` |

`chunk_id` 把 token 以 int32 雜湊，再接上 `extra_key`（cache_salt + LoRA id）。
**角色名不含 `extra_key`**：不同 cache_salt / LoRA 的請求會共用同一個角色 namespace（見第 15 節）。

`idx` / `cdc` 走鏈式儲存，樹裡**不用**這些名稱：整條存在 `req.extra_key`（cache_salt + LoRA id）底下，
跟 `off` 一樣；`chunk_id` 只當索引裡的 chunk 編號。

### 5-5 往下傳

- `tokenizer_manager`：預先切好的 `sub_context_ids` 只有串接起來等於 `input_ids` 才接受，否則警告並當作不切。
- `--allow-auto-truncate` 截掉 prompt 尾巴後，`_clip_sub_contexts_to_input` 把 block 清單裁齊：
  保留放得下的完整 block、裁切跨界的那塊、丟掉其餘。
- scheduler 建立 `Req` 時傳入 `extra_key`、`sub_context_ids`、`sub_context_extra_keys`。
  （上游 v0.5.9 的 scheduler 不傳 `extra_key`，所以上游的 `cache_salt` 不會區隔快取；這個 fork 會，`off` 也一樣。）

---

## 6. stitch（排程，`on` / `rot`）

對**每一塊**都去它自己的 namespace 查，即使用不到也查、也鎖、也記：

```python
seg_match = tree_cache.match_prefix(RadixKey(seg_ids, seg_key))
hit = len(seg_match.device_indices)
canonical = matched_canonical_position(seg_match.last_device_node, hit)
if hit > 0:
    tree_cache.inc_lock_ref(seg_match.last_device_node)     # 鎖住，不管用不用
displaced = canonical is not None and canonical != offset
```

然後決定要不要拿。**拿的前提是 `contiguous` 還成立**——拼出來的必須是從位置 0 開始的
連續一段，因為一次 prefill 只能表達「重用前綴 + 後面新算」。

### 情境

| 情境 | 做什麼 | 後果 |
|---|---|---|
| `hit == 0`（沒命中） | 不拿 | `contiguous = False`，後面全部只查不拿 |
| 位置對（`canonical == offset`） | 直接把**樹的格子**放進 prefix | 零成本 |
| 位置不對，旋轉可用 | **alloc 新格子**，複製並旋轉 `offset - canonical` | 記進 `sub_context_rotated_slots` |
| 位置不對，旋轉不可用 | 不拿，`contiguous = False` | 開關關著／delta 超出 cos_sin_cache／pool 滿 |
| 部分命中（`hit < 塊長`） | 拿 `hit` 個 | `contiguous = False` |
| `take` 被上限砍 | 拿 `min(hit, len(fill_ids)-1 - total)` | 至少留一個新 token 可算 |

### `contiguous` 和 `displaced` 不是同一件事

| | 意思 | 跟誰比 |
|---|---|---|
| `contiguous` | 到目前為止，每一塊都被**完整拼進來**了 | 跟**我自己這一趟**比 |
| `displaced` | 樹裡那份登記的位置跟**我要放的位置**不同 | 跟**別的 request 的版面**比 |

位置會不同，不是因為我這一趟有斷層，而是因為**當初把這塊放進樹的那個 request，
它的版面跟我不一樣**：

```
R_old：system 120 個 token，tools 接在 120  → tools_key 的 canonical = 120
R_new：system 只有 100 個，  tools 接在 100

R_new 的 stitch：
  system 完整命中，canonical 0 == offset 0  → 整塊拿走，contiguous 成立
  tools  完整命中，canonical 120 ≠ offset 100 → displaced，旋轉 -20
```

前面完全沒有斷層，但 tools 還是位移了。

### 為什麼旋轉只在 contiguous 成立時做

1. **接不上去**。前綴必須是從位置 0 開始的連續一段。前面若只部分命中，中間空著，
   後面的塊沒有東西可以貼。
2. **delta 會是錯的**。`delta = offset - canonical` 只有在前面每塊都整塊拿走時才對——
   那時 prompt 的版面和拼出來的版面才一致，這塊才真的落在 `offset`。

「前面沒能整塊命中」的情形，stitch 路徑由 **Stage 2** 處理（第 10 節），scan 路徑則根本不要求連續（第 7 節）。

### 部分取用

`take = min(hit, max_prefix_len - total)`，`max_prefix_len = len(fill_ids) - 1`。

那個 `-1` 是因為 prefill 至少要留一個位置實際通過模型，否則拿不到 logits，
kernel 也會收到空 grid（CUDA 直接報 `invalid configuration argument`，實際撞過）。

它只會咬到**最後一塊**，而 agent prompt 裡最後一塊正是 `messages`——唯一每輪都在變、
也唯一值得重用的那塊。所以「只拿一部分」是必須的，不允許的話那塊會被整塊拒絕。
數學上安全，因為 delta 屬於整塊，對每個 token 一視同仁。

### 輸出

```python
prefix_indices = cat(stitched)     # 混合：樹的格子 + 自己的旋轉副本
sub_context_owned_lens[i] = take
sub_context_tree_owned = None      # 明確清掉
cache_protected_len = len(prefix_indices)
sub_context_next_boundary = ...    # Stage 2 用
last_node = root_node              # scheduler 的單一 node 鎖變成空操作
```

**為什麼 `tree_owned` 要清掉**：retraction（被踢出 batch、KV 被 free、重新排程）之後，
上一輪的 `True` 會讓 finish 跳過那塊 → 這個 request 自己的格子沒人還 → 洩漏。

**重新排程時**：stitch 開頭會先放掉上一輪的鎖與旋轉副本，並清掉 scan 留下的 layout 等欄位。

---

## 7. scan 與鏈式儲存（`idx` / `cdc`）

這兩個 arm 的 request 讀寫都**像原生 request 一樣**：在樹裡存成一條從位置 0 開始的鏈，放在
`req.extra_key`（cache_salt + LoRA id）底下。內容索引只負責「從頭比對斷掉之後，後面還有哪些已快取的區塊、
它們在樹上哪裡」。（2026-10-05 起；之前是每塊一個內容雜湊 namespace，見 git 歷史。）

### 7-1 索引

`SubContextIndex`（`subctx_index.py`）記「哪些 token 序列存在樹上的哪裡」：

- **登記**：request 存回樹之後，新存上樹的區塊（第 7-5 節）連同 `req.extra_key` 登記；短於 `MIN_CHUNK` 的不登記。
  每筆記 token、開頭 32 token 的指紋，以及**位置**：持有最後一個 token 的節點 + 第一個 token 的位置。
- **多個位置**：同一段內容存在好幾條鏈上時，每條都記一筆，**最早的優先**——讀的是最初算出來的那份，
  一次旋轉就到位；後來的（多半是複製、旋轉過的副本）只在前面的被淘汰後才用（`chain_chunk_slots`，
  `radix_cache.py` L409）。
- **範圍**：每個 `extra_key` 一個範圍，掃描只看自己的範圍——不同 cache_salt / LoRA 互相看不到。
- **掃描**：對 prompt 每個位置算「從這裡開始 32 個 token」的指紋（一次向量化算完），先用 Bloom filter
  （快速排除一定沒登記的位置的位元表）過濾，再查「chunk 開頭指紋 → chunk」表，最後逐 token 比對確認。
  回傳所有出現（可重疊）。新 request 自己怎麼切塊跟比對無關。
- **挑選**：從可能重疊的命中裡，挑出互不重疊、總長最大的一組。

### 7-2 `_scan_sub_contexts` 的步驟（`schedule_batch.py` L1306）

1. 放掉上一輪的鎖與副本，清掉上一輪的配置。
2. **從頭比對**：跟原生 request 一樣 `match_prefix(fill_ids[:max_prefix_len], req.extra_key)`，
   得到前綴長度 `exact`、`last_node`（之後由 `PrefillAdder` 上鎖，跟原生一樣）、`cache_protected_len = exact`。
3. **掃描剩下的部分**：`index.scan(fill_ids, extra_key, min_end=exact + MIN_CHUNK)`，只回傳結尾在 `min_end` 之後的命中，
   只對 `fill_ids[min_end - 最長 chunk:]` 取指紋（每個 scope 記登記過的最長 chunk）；結果和掃整條再裁相同。
   每個命中裁到「`exact` 之後、`max_prefix_len` 之前」（prefill 至少要算最後一個 token），裁完短於 `MIN_CHUNK` 的丟掉。
   `index.select` 挑出不重疊的組合。
   被標成 `sub_context_index_declined` 的 request 跳過這一步（第 7-4 節）；開了 selective recompute 時，
   要求 logprob 的 request 也跳過（重算後的列數會讓 logprob 的對齊跑掉）。
4. **找位置、上鎖**：`chain_chunk_slots(cid)` 從位置節點往上走父節點，湊出 `[start, end)` 的格子；
   位置失效（節點已淘汰、範圍對不上）就丟掉那筆位置，全失效就整段重算（記入 `discarded`）。
   `Δ = 命中位置 − 當初的位置`；`Δ` 超出旋轉範圍的段在這裡就丟掉（記入 `moved`、`discarded`），不上鎖。
   其餘的段對它的節點 `inc_lock_ref`（連同所有祖先）。
5. **先算放不放得進一次 prefill**（第 7-3 節）：放不下就放鎖、只用前綴，不淘汰也不複製。
6. 前綴節點也暫時上鎖，為所有副本先 `evict` 出空間，再放開前綴的鎖。
7. **一律複製到自己的格子**（`_copy_sub_context_run`）：`Δ ≠ 0` 旋轉，`Δ == 0` 原樣複製
   （`KVRotator.copy_into`）。即使位置一樣也要複製：鏈必須擁有自己的格子，一個格子不能屬於兩個節點。
   pool 滿就放鎖、整段重算。
   overlap 排程下，上一個 forward 可能還在 forward stream 上寫這些格子（chunked prefill 的中間 chunk、剛結束請求的最後幾個 token），
   所以旋轉與複製的 kernel 直接在 forward stream 上執行（`KVRotator.launch_stream`）：排在進行中的 forward 之後、下一個 forward 之前，
   default stream 不多任何依賴，`prepare_for_extend` 之後的同步複製也就不會讓 CPU 等前一個 forward。
   用到的 index tensor 以 `record_stream` 留給 forward stream。不開 overlap 時 forward 本來就在 default stream，照舊在那裡執行。
8. **輸出**：
   - 沒找到任何段 → 跟原生 request 完全一樣（`sub_context_layout = None`）。
   - 找到 → `sub_context_layout = [(0, exact, 前綴格子)] + [(start, end, 副本格子), …]`，
     `sub_context_exact_len = exact`；`prefix_indices` 是所有段的格子串接。

### 7-3 放不進一次 prefill → 只用前綴

有洞的 prefill **不能切成多個 chunk**。檢查「要算的 token 數 ≤ `--chunked-prefill-size`」做兩次：
複製之前用找到的段長先算一次（`_fits_one_pass`），放不下就不複製；複製之後再算一次
（`_sub_context_layout_fits_one_pass`），因為 pool 滿而沒複製成的段會讓要算的變多。任一次放不下都記一次 `fell_back`，
放掉段的鎖與副本，退回只用前綴（之後照原生 chunked prefill）。放得下但這一輪預算不夠的，`PrefillAdder` 回 `OTHER` 讓它等。

### 7-4 沒被收進這一輪的請求

`init_next_round_input` 在排程迴圈裡對每個排到的等待請求都會跑一次，scan 在這時就拿了鎖、配了副本。
請求若沒被收進這一輪（`add_one_req` 回非 `CONTINUE`、且它不在 `can_run_list` 裡——adder 收進一個請求後
預算剛好用完也會回 `NO_TOKEN`/`OTHER`，那種不算），`scheduler.py` L2257 呼叫
`release_sub_context_scan`（`schedule_batch.py` L1091）：放掉段的鎖、還掉副本。下一輪重新 scan。

**沒空間的卡死**：段的鎖會把來源節點到 root 的整條路徑都釘住，加上副本，可能讓 adder 的剩餘額度不夠，
回 `NO_TOKEN`；下一輪重 scan 結果一樣。所以當「`NO_TOKEN`、沒有請求在跑、沒有 chunked 請求、這一輪也沒收進任何請求」
時，這個請求標成 `sub_context_index_declined`，之後只用前綴比對（記入 `no_room`）；這一輪**不**設
`batch_is_full`，否則沒有請求在跑時 scheduler 不會再排 prefill，前綴重試永遠不會發生。

離開等待隊列的其他三條路（`abort_request`、`_abort_on_waiting_timeout`、優先序擠掉）也都呼叫
`release_sub_context_scan`。

### 7-5 存回樹與登記（`_register_chain`，`radix_cache.py` L539）

`serves_sub_contexts` 對鏈式請求回 False，所以 `cache_unfinished_req` / `cache_finished_req` 走**上游的整條插入**：
`insert(fill_ids, req_to_token 那一列)`；樹已經有的部分（`prefix_len`）照上游規則把自己的重複格子還掉。
插入之後：

- **登記新存上樹的區塊**：區塊邊界用 request 收到時的切法（`sub_context_ids_as_sent`）；結束長度
  `> prefix_len`（含有這次新存的 token）且 ≥ `MIN_CHUNK` 的才登記，位置是這條鏈上持有它最後一個 token 的節點。
  從別處複製進來的段也在其中——同一段內容因此多一筆位置（排在原本的後面）。
- **結束時**（`cache_finished_req`，`final=True`）：最後一塊延伸到生成的最後一個 token，再依切塊模式重切
  （`cdc` 用 `cut_points`，`blocks` 當一塊）——所以 generation prompt 和模型的回覆一起登記。
- `cache_unfinished_req` 插入後放掉 scan 拿的段的鎖（副本已經寫好，交給鏈了）。

開了 selective recompute 時，副本裡有一部分 token 是重算過的；它們照樣隨鏈存進樹，之後的 request
從頭比對到時當成精確前綴直接用，不再重算（這是設計上的選擇：鏈不能有洞，停在第一個借來的段會丟掉它後面的所有內容）。

### 7-6 位置的維護

| 事件 | 做什麼 | 在哪 |
|---|---|---|
| 節點被拆開（`_split_node`） | 最後一個 token 落在新的上半節點的位置，改指上半節點（格子號碼本來就不變） | `_rehome_chunks`，L502 |
| 節點被淘汰 | 丟掉以它為位置的那幾筆；一個 chunk 的位置全丟光就從索引移除 | `_forget_chunks_at`，L528 |
| `reset()`（flush cache） | 清空位置表與索引 | L382 |
| 讀取時發現位置失效 | 丟掉那一筆，換下一筆 | `chain_chunk_slots` |

`_chunks_at`（節點 id → 以它為位置的 chunk）是反查表，讓拆節點和淘汰不必掃整個索引。
開 `SGLANG_SUBCTX_AUDIT` 時，每次讀取還會逐 token 比對位置上的內容（`SUBCTX-CHAIN-MISMATCH`）。

### 7-7 dry run（`SGLANG_SUBCTX_INDEX_DRYRUN`）

不開 `INDEX`、只開 `DRYRUN` 時照常走逐 namespace 的 stitch，另外跑一次 scan 只統計：找到多少、仍在快取多少、
其中位置不同的多少、超出 stitch 前綴多少。不改 request 狀態，但查樹會更新節點的存取時間（影響 LRU）。

---

## 8. prepare_for_extend：責任交接與 sparse prefill

### 8-1 交接

```python
write_cache_indices(...)                 # prefix_indices → req_to_token（sparse 時見 8-2）
req.sub_context_rotated_slots = None     # 責任移轉
```

`sub_context_rotated_slots` 是一份**釋放責任清單**——`release_sub_context_rotated_slots`
會把清單上的每一批格子直接 free 掉。

| 情境 | 結果 |
|---|---|
| 在此之前 abort 或重新排程 | 清單還在 → `release_sub_context_rotated_slots` 救援 |
| 在此之後 | `req_to_token` 涵蓋它們 → finish 的逐塊迴圈會還 |
| **忘了清空** | 兩邊都還 → **double free** |
| **太早清空** | 兩邊都不還 → **洩漏** |

所以這是唯一正確的交接點：只有這一刻 `req_to_token` 和記帳同時涵蓋那些格子。
而且**每一輪 chunk 都會重新交接一次**（Stage 2 的 append 會再產生新的旋轉副本）。

### 8-2 sparse prefill（batch 裡有任何 request 帶 `sub_context_layout`）

| 步驟 | 在哪 | 做什麼 |
|---|---|---|
| 輸入 | `ScheduleBatch.prepare_for_extend` | 每個 request 只取「要算的位置」（重用段之間的空洞）的 token；`extend_num_tokens` 只算這些 |
| 配 slot | `common.alloc_for_extend` | 新 slot 只給要算的位置 |
| 對照表 | `common.write_cache_indices_sparse` | 重用段的 slot 寫到它所在的位置，新 slot 寫到要算的位置；`[0, seq_len)` 每格恰好寫一次。同批的一般 request 照舊 |
| 位置 | `forward_batch_info` | 每個 token 帶給定的絕對位置（不再由前綴長度推算），`subctx_sparse = True` |
| attention | `triton_backend` | KV 取每個 request 的**整段** `[0, seq_len)`，第 j 個 KV 就是位置 j |
| 遮罩 | `extend_attention._fwd_kernel_unified` | 位置 p 的 query 只看得到 j ≤ p 的 key；整個 KV tile 都在所有 query 之後就跳過 |

不傳位置時，kernel 的行為與上游相同。

---

## 9. selective recompute（`TOPK_RATIO > 0`）

重用的區塊是在別的前文下算的，它的 KV 和「在現在這個前文下算」不同。做法（`subctx_blend.py`、
`ModelRunner.forward_subctx_blend`）：

1. **數量在排程前就決定**：`TOPK_RATIO` 是一串比例 `r1,r2,…,rm`（不遞增），對應連續的打分層
   `TOPK_LAYER, TOPK_LAYER+1, …`。每個 request 在第 i 個打分層之後留下 `int(打分的重用 token 數 × ri)` 個
   （`sub_context_topk_counts`）。打分的只有 scan 找到的段；從頭比對到的前綴（`sub_context_exact_len`）是精確的，
   不打分（`_scored_runs`）。不多配 slot：重算的 token 直接寫回它自己的副本（scan 找到的段一律是 request 自己的副本，
   `build_plan` 會檢查）。
2. **probe**（第 `[0, TOPK_LAYER]` 層）：每個 request 從它的 probe 起點（`sub_context_probe_start`）跑到 prompt 結尾——
   有 layout 的從精確前綴之後開始，同一批裡其他 request 從它要算的第一個 token 開始。起點之前的 KV 照樣從快取讀，
   只是不當 query。**每一列都把新算的 K/V 寫進自己的 slot**：新 token 寫它的新 slot，重用位置寫回自己的副本。
   所以 probe 各層的 attention 讀到的是整段新算的 KV，跟 CacheBlend repo 在第 1 層的做法一樣；ratio 1.0 時
   不論在第幾層打分，結果都等於整段重算。
3. **打分**（每個打分層）：這一層跑之前，先把還在跑的重用位置的快取 key 讀出來；這一層跑完後，
   分數 = `Σ (k_新算 − k_快取)²`（對 head 與維度加總，fp32）。每個 request 在還在跑的重用位置裡取分數最高的那些，
   和新 token 合併、排序。後一個打分層只從前一層留下的位置裡挑（逐層篩選，CacheBlend 論文 §4.3）。
   挑選在 GPU 上完成，不需要同步回 host。
4. **裁減**：每個打分層之後，hidden states、residual、位置、寫入的 slot 都裁成「新 token + 留下的重用 token」，
   重建 attention metadata 再跑下一段。被淘汰的重用 token 在之後的層保留快取的 KV；它跑過的層（含 probe 各層）
   留下新算的 KV。最後一個打分層之後的層只跑最後留下的那些。這條路不用 CUDA graph。

**`PrefillAdder` 的 probe 規則**：probe 要跑從起點之後的整段（`sub_context_forward_rows`），比「要算的 token」多；
如果這個 request 不是本批第一個、而多出的列放不進剩餘預算，就等下一輪；
如果是第一個，就單獨跑，允許超出預算。

**重算過的副本隨鏈寫回**：`idx` / `cdc` 的副本是 request 自己的格子，跟整條鏈一起存進樹（第 7-5 節）；
下一輪從頭比對到這段時當成精確前綴，不再打分。索引裡同一段內容的最早位置仍是原始算出來的那份，
別的 request 從索引找到這段時讀的是原始那份。

---

## 10. cache_unfinished_req（每個 chunk）

**本節只適用逐 namespace 的路徑（`on` / `rot`、只開 `HASH_KEYS`）。** `idx` / `cdc` 走上游的整條插入，見第 7-5 節。

把剛算好的 KV 交給樹保管。KV 現在只有這個 request 的 `req_to_token` 指著，
request 一結束就沒人記得它了。

### 逐塊步驟

**⓪ 要不要寫**：這次還沒算到這塊就跳過（`covered_end = min(offset + 塊長, end_k)`）；
`sub_context_no_insert[i]` 為真也跳過（舊 scan 用的，現在沒有路徑會設定）——跳過的塊留到 finish 處理。

**① 樹上有嗎、位置對嗎**

```python
existing = matched_canonical_position(probe.last_device_node, probe_hit)
if existing is not None and existing != offset:
    continue           # 跳過這塊，不是停止
```

**first-writer-wins**：namespace 裡已經在別的位置存過這段內容，就保留先寫的版本。
`continue` 不是 `break`——namespace 是各自獨立的樹。停下來會讓越長越大的 `messages` 塊
永遠進不了快取。代價是中間會有洞（見下）。

**② 插入**

送進去的是**整塊已覆蓋的部分**，但樹只會真正存下它本來沒有的那一段——
insert 沿著已有路徑往下走，走不下去才新增節點。新節點記下 `canonical_position = offset + 已走過的長度`。

```python
result.prefix_len      # 這串 token 裡，有多長是樹本來就有的
```

**③ 還掉多餘的**

```python
if result.prefix_len > owned:
    free(kv_indices[offset+owned : offset+result.prefix_len])
```

`[owned, prefix_len)` = 我這次新算的，但樹上已經有別人算好的同一份。

**④ 回寫 `req_to_token`**

把這塊每個位置指到樹真正持有的格子。**必須在 ③ 之後**——`kv_indices` 是 `req_to_token`
的 view，先回寫的話 ③ 會 free 到樹的格子。

**⑤ 記帳 + 鎖住 + 登記索引**

```python
tree_owned[i] = True;  tree_canonical[i] = offset;  owned_lens[i] = covered_end - offset
inc_lock_ref(seg_match.last_device_node)
sub_context_index.register(seg_ids, req.extra_key)    # 有索引時
```

### 四個情境

**情境一：第一次見到這塊。** 樹上空的。insert 存下整塊，`prefix_len = 0`，
沒東西要還，回寫等於指回自己的格子。

**情境二：我查找時拿過，樹上就是我拿的那份。** `prefix_len = 150`、`owned = 150`
→ 不還。回寫指回同一批。**這一輪實質上什麼都沒變**，只是重新鎖住。

**情境三：別人搶先。** 排程時樹上空的 → 我自己算了 150 個。但另一個 request 同時
也在算同一段，而且先插入了。現在 `prefix_len = 150`、`owned = 0` → **還掉我自己那份**
（樹只留先寫的），並**真的改指** `req_to_token`。不回寫的話，我還在 decode，
會讀到剛丟回籃子的格子。

**情境四：位置不對。** 跳過整塊。這塊成了洞，留到 finish 搶救（第 11 節 11-1）。

### 「洞」與「所有權」

跑完之後可能長這樣：

```
A  0–99     插進樹了     → 樹的
B  100–249  被跳過       → 不是樹的        ← 洞
C  250–399  插進樹了     → 樹的
```

**所有權**就是 `tree_owned = [True, False, True]`，**洞**就是 B。

**為什麼不能用一個數字表示**：`cache_protected_len` 原本是一個數字，
意思「位置 0 到 N 都是樹的，不准 free」，只能描述從頭連續的一段。
`[True, False, True]` 壓不進一個數字——硬壓成 100 的話，finish 時 C 的 250–399
會被當成「不是樹的」而還回籃子，但那是樹的格子。

所以改成逐塊的布林陣列，`cache_protected_len` 退化成給上游程式碼看的有損投影。

### Stage 2：append（要開 `ROTATE_ACROSS`，只在 stitch 路徑）

處理「前面那塊只能部分重用，但後面那塊整塊都在快取裡」的情形。

stitch 時算出 `boundary`（第一個「在已拼前綴之後、完整命中、但位置不對」的塊的 offset），
`PrefillAdder` 把 chunk 夾到那裡（沒有 truncation 對齊、預算放得下時）。於是第一趟剛好算到那塊的起點。

在逐塊寫入處理完之後：

```python
if offset != cursor:  break                 # 必須貼齊覆蓋末端
delta = offset - canonical
dst = _rotate_sub_context_block(indices[:take], delta)    # alloc + 旋轉
req_to_token.write((req_pool_idx, slice(offset, offset+take)), dst)
cursor += take
```

**怎麼接回計算**：`covered = end_k + appended`，`prefix_indices = req_to_token[:covered]`。
下一輪 scheduler 呼叫 `chunked_req.init_next_round_input()` **不帶 tree_cache**
→ 不重新查找 → 直接沿用這份變長的 prefix → `extend_input_len` 只剩下沒被覆蓋的部分。

```
不做 append：第一趟算 0–99，第二趟算 100–249（150 個）
做 append：  第一趟算 0–99，第二趟算 249（1 個）
```

代價是多一趟 scheduler round-trip。所以塊夠大才划算——**最小 block 門檻（約 300 token）還沒做**。

stitch 先把「Stage 2 下一輪還會試」的位移命中記為 `deferred`；append 沒接上的部分才計入 `moved` / `discarded`。

### 收尾（順序是有意的）

```python
appended = _rotate_append_sub_contexts(req, end_k)   # 讀 match 結果
release_sub_context_match_locks(self)                # ← 之後才放鎖
prefix_indices = req_to_token[:covered]              # 從對照表重建，不接片段
cache_protected_len = _sub_context_protected_len(...)
last_node = root_node                                # 讓 scheduler 的鎖變 no-op
```

- **append 要在放鎖之前**：它讀的正是那些鎖保護的 match 結果，放鎖會把欄位一起清掉。
- **放鎖要在插入之後**：新的 namespace 鎖已經蓋住整段 prompt，中間不會有空窗
  （`insert` 自己就可能觸發淘汰）。
- **`prefix_indices` 從 `req_to_token` 重建**：中間可能有跳過的洞，接片段描述不了。

---

## 11. cache_finished_req

**11-1 到 11-3 只適用逐 namespace 的路徑。** `idx` / `cdc` 走上游的整條插入（回覆一起存），見第 7-5 節。

先兩道冪等的保險（`release_sub_context_match_locks`、`release_sub_context_rotated_slots`，
給沒跑過第 10 節的 request 用；對一般 request 是空操作），然後：

```python
if self.serves_sub_contexts(req):        # = has_sub_contexts and supports_sub_contexts()
    _finish_sub_contexts(...)            # 不走上游的整條 insert
```

這一行的歷史見第 14 節。

### 11-1 搶救：`_reverse_rotate_insert_sub_contexts`（有旋轉器時）

對每個 `tree_owned[i] == False` 且沒有 `no_insert` 的塊：

| 情境 | 做什麼 |
|---|---|
| namespace 已完整持有同一段內容 | 保留樹的版本，只還自己的格子 |
| `canonical is None`（當初擋住我的那份已被淘汰，namespace 空了） | 當成 `offset`，delta = 0，直接插入 |
| `delta == 0` | 直接插入，不轉 |
| `delta != 0`，整塊都是自己的 | **就地**旋轉 `canonical - offset` → 插入 → `tree_owned[i] = True` |
| `delta != 0`，**混著樹的格子** | **整塊放棄** |
| delta 超出 cos_sin_cache 範圍 | 放棄 |

注意方向跟查找相反：查找是 `offset - canonical`（把樹的那份搬到我要的位置），
這裡是 `canonical - offset`（把我的那份搬回樹要的位置）。

**為什麼混著樹的格子就要放棄**：就地旋轉會改到別的 request 正在 match 的 KV，
而那個 node 還在對外宣告舊的 canonical position——跨 request 的無聲汙染，
這個機制能產生的最糟錯誤。整塊複製一份再轉當然可以，但比這次搶救本身還貴。

**為什麼在 finish 而不是每個 chunk**：`cache_unfinished_req` 是在 prefill 之後、
request **還在 decode 的時候**呼叫的，它自己的 attention 每一步都在讀這些 slot。
就地搬位置會安靜地毀掉還在進行中的生成。已經 finish 的 request 再也不會讀它們。

搶救成功會重算 `cache_protected_len`——**這是 11-2 的前提**。

### 11-2 存回覆：`_cache_sub_context_output`（`CACHE_OUTPUT` 開著時）

把「最後一塊的 token ++ 生成的 token」插進最後一塊的 namespace。
下一輪對話這段回覆就在 prompt 裡，不用重算。

條件：`cache_protected_len == prompt_len` 且 `owned_lens[-1] == len(last_seg)`
且有生成內容。

**情境**：最後一塊的 `tree_canonical != offset` 時，生成的 tail 也要**就地轉同樣的 delta**
——它是在 `prompt_len` 算的，但要接在樹擺在 `canonical` 的塊後面。

插入後若匹配長度小於最後一塊的長度，印 `SUBCTX-OUTPUT-UNDERMATCH`（「今天不可能發生」的顯式警報）。

### 11-3 還鑰匙

```python
tree_owned = req.sub_context_tree_owned
if tree_owned is None and req.sub_context_extra_keys:
    tree_owned = [False] * len(...)          # 沒跑過第 10 節

for 每一塊:
    if tree_owned[i]:  continue              # 樹的，一格都不動
    block = kv_indices[offset:end]
    key = sub_context_source_key[i] or seg_key   # slot 實際所在的 namespace
    match = self.match_prefix(RadixKey(token_ids[offset:end], key))
    _free_only_ours(block, _tree_held_mask(match.device_indices, block))

if not kept and prompt_len < len(kv_indices):
    free(kv_indices[prompt_len:])            # 沒被收下的生成 tail
```

| 情境 | 結果 |
|---|---|
| 塊是樹的 | 一格都不還，只 `dec_lock_ref` |
| 塊被拒絕，全是自己的 | 整塊還 |
| 塊被拒絕，**混著樹的格子** | **逐格比對，只還自己的** |
| 只重用一部分的塊（舊 scan 裁過；現在不會發生） | 到 `sub_context_source_key` 那個 namespace 比對，只還自己的 |
| 從沒跑過第 10 節 | 每塊都當「被拒絕」處理，逐格比對 |
| 回覆沒被收下 | 連同 tail 一起還 |

---

## 12. free 與 write-back 的規則

### free 是純 CPU 記帳

```python
def free(self, free_index):
    if self.is_not_in_free_group:
        self.free_pages = torch.cat((self.free_pages, free_index))
    else:
        self.free_group.append(free_index)
```

**GPU 上那幾列一個 byte 都沒動。** 被 free 的 slot 上面還躺著完全合理的 KV，
直到有人 alloc 拿走它、算新的 KV 蓋上去為止。

所以所有權錯誤是**無聲的**：不會 crash、不會有 NaN、不會有 assert。
只會讓某個 request 在某個時刻讀到別人的 KV，輸出稍微變差。
GPU 上沒有任何線索可查——唯一能發現問題的地方是**記帳**。

`free_group`：output processor 會開一個 group，這期間所有 free 被推進暫存，
`available_size` 不動，直到 `free_group_end` 把整批 concat 起來重新進 `free()`。
這是為什麼 audit 必須自己數 free 而不能觀察 `available_size`。

### 兩種 write-back

| | 做什麼 | 動到 GPU 嗎 |
|---|---|---|
| **改對照表** | 把「我的第 N 個 token」從自己算的格子改指到樹的格子 | 不動 |
| **搬 KV** | 真的去改格子裡的 bytes（旋轉、或 selective recompute 的複製與寫入） | 動 |

### 搬 KV 發生的地方

| 何時 | 目的地 | 為什麼 |
|---|---|---|
| stitch（排程） | **複製到新格子**並旋轉 | 來源是樹的，別人鎖著 |
| scan（排程） | **一律複製到新格子**，位置不同時旋轉 | 鏈必須擁有自己的格子 |
| append（每個 chunk 結尾） | **複製到新格子**並旋轉 | 同上 |
| selective recompute 的寫回 | **新格子**（前面各層從快取複製） | 被重算的 token 不能改寫樹的格子 |
| finish：搶救被拒絕的塊 | **就地**旋轉 | 格子是自己的，而且已經不會再讀 |
| finish：轉生成的回覆 | **就地**旋轉 | 同上 |

規則：**來源是樹的 → 複製；來源是自己的 → 就地。**

就地旋轉在資料競爭上是安全的（kernel 每個 program 先讀自己那列再寫自己那列，
沒有跨列相依，有專門的測試 `test_in_place_rotation_matches_the_copying_one`），
但在**所有權**上不安全，所以有 `tree_held.any()` 的守衛。

### 三條貫穿全部的規則

1. **擁有權逐塊記**，不用一個長度表示（因為 `continue` 會製造洞）。
2. **free 之前逐格比對 slot 身分現算一次**（`_tree_held_mask`），不相信任何記住的狀態
   ——中間可能發生 insert、split、evict。
3. **搬 KV 的目的地**：讀路徑一律複製到新格子；寫路徑可以就地，但必須先確認整塊都是自己的。

### 為什麼要逐格比對

`_tree_held_mask` 逐一位置比對 slot 身分，而不是問「這塊是不是我的」。兩個理由：

- **一個被拒絕的 block 裡可能混著 node 自己的 slot。** 最清楚的例子就是
  「從沒跑過第 10 節」的 request：它的 `req_to_token` 裡是查找時拼好的、
  指向各 namespace node 的格子，但它在樹裡一格都不擁有。
- **重疊不是前綴。** 這裡原本只數「開頭連續相符的長度」，假設塊是從前往後被重用的。
  job 432 打破它：重複的位置從 block offset 的**下一格**開始——第一格不同、
  後面約 2000 格全同——於是「連續相符長度」算出 0，整段被還回去，而樹還在服務它。

---

## 13. 量測、診斷與防線

### 量測（不影響行為）

| 工具 | 開關 | 內容 |
|---|---|---|
| forward trace | `SGLANG_FORWARD_TRACE=<檔>` | 每個 forward pass 一列 JSON：`mode`、`bs`、`new_tokens`、`cached_tokens`（只算增量）、`matched_tokens`、`discarded_tokens`、`moved_tokens`、`rotated_tokens`、`reinserted_tokens`、`sub_reqs`、`gpu_ms`。以 CUDA event 計時，不同步 |
| host 階段計時 | `SGLANG_STAGE_TRACE=<前綴>` | 每個 process 寫 `<前綴>.<proc>.json`：`tpl_render`、`subctx_split`（http）；`match`、`subctx_stitch`、`subctx_lookup`、`subctx_scan`、`subctx_rotate`、`subctx_copy`、`subctx_rotate_finish`、`subctx_rev_rotate`、`cache_unfinished`、`cache_finished`（scheduler）。client 建立 `<前綴>.mark` 後另開一份 `measured` 累計，排除 warm-up |
| 請求錄製 | `SGLANG_CAPTURE_REQUESTS=<檔>` | 每個 chat 請求原文一行，給 `subcontext_bench.py replay` |
| `/server_info` | — | `sub_context` 物件：所有開關的實際值與 `pid`，給遠端 client 確認 arm（`scripts/subcontext_sim/check_remote_arm.py`）；`chain_storage` 為 true 表示是鏈式儲存的版本 |
| 索引統計 | 開 index 時 | prefill log 那行後面附 `sub-context index: …`：比對過的 prompt token 數、其中從頭比對到的、scan 找到的、旋轉的，以及放不進一次 prefill 的請求數與它們丟掉的掃描命中 token 數、沒空間被拒的請求數。每個請求只在第一次真的執行（`prepare_for_extend`）時算一次，被拒後重新比對、被 retract 後重跑的都不算；4090 smoke 中與 client 端 `prompt_tokens`、`cached_tokens` 加總完全一致。`chunks`、`queries`、`tokens scanned`（開頭三個）仍是每次掃描都算；`tokens scanned` 只算實際取指紋的部分（從「前綴長度 + 最短 chunk − 最長 chunk」起），dry run 掃整條 |

### 防線與偵錯

| 防線 | 抓什麼 | 需要 `SGLANG_SUBCTX_AUDIT` |
|---|---|---|
| `available + evictable == max - protected` | 記帳破了（最終症狀） | 不用 |
| `audit_pool_invariant` | 破了之後說明是哪一種原因 | 不用 |
| `_audit_tree_duplicates` | **一個 slot 兩個 owner 的當下** | **要** |
| `_audit_sub_context_chunk` / `_finish` | 寫入或 finish 之後，自己的 slot 是否「恰好」被釋放或被樹持有其一 | **要** |
| slot 級 double-free 偵測（`allocator.py`） | 同一格被還兩次的當下，附上呼叫者的檔名與行號 | **要** |
| `SUBCTX-OUTPUT-UNDERMATCH` | 「今天不可能發生」的顯式警報 | 不用 |
| `SUBCTX-CHAIN-MISMATCH` | 索引記的位置上不是那段 token（鏈式儲存，讀取時逐 token 比對） | **要** |
| `SGLANG_SUBCTX_TRACE` | 每個階段的 TRACE 行（命中、旋轉、寫入） | 不用 |
| `SGLANG_DUMP_TREE` | 每次 extend 後印整棵樹（含每個 node 的 `extra_key`） | 不用 |

記帳破掉時，`audit_pool_invariant` 會走一次完整的樹並回報：

```
tree walk: nodes=N slots=M distinct=D dup_within_tree=X dup_owners=[...]
evictable_counter=... keyed_unlocked=... counter_minus_walk=...
IN_TREE_AND_FREE=...
```

四種不同的成因（同一格被 free 兩次／同時在樹和 free list 裡／`evictable_size_` 多算／
**一格兩個 owner**）由這幾個欄位分辨。

要確認某個 bug 是否修好，`SGLANG_SUBCTX_AUDIT=1` 是必要的：沒有它，
只會在洩漏發生**之後**才知道；有它，TREE-DUP 會在第一個 slot 被兩個 owner
認領的那一刻就叫。代價是每個 pass 一次完整 tree walk，所以那一輪的 GPU 時間
不能拿去跟 toggle 的數字比。

---

## 14. 上次的 bug（已修，`ed27c8a` + `26b2bf2`）

### 正常的 request 走六步

```
1. 查找                   拼出 prefix（含樹的格子）
2. prepare_for_extend     寫進 req_to_token
3. prefill                算 KV，並產生第一個 token
4. cache_unfinished_req   逐塊插入樹，設 sub_context_last_nodes    ← 關鍵
5. decode                 一個一個吐 token
6. cache_finished_req     收尾
```

### 有一種 request 只走 1→2→3→6

prefill 不只是「把 prompt 的 KV 算出來」，它**同時算出下一個 token**
（最後一個位置的 logits → sampling）。如果那個 token 就是 EOS，
這個 request 一個 decode step 都沒跑過就結束了，**第 4 步從沒發生**。

另一種是排隊時被 abort。

### 錯在哪

第 6 步要判斷「這是不是 split request」。舊版問的是：

```python
if req.sub_context_last_nodes is not None:     # 只有第 4 步會設這個欄位
```

答案是 None → 判定「不是 split request」→ 走**預設分支**：

```python
radix_key = RadixKey(keys, req.extra_key)      # split request 的 extra_key 是 None
self.insert(key=radix_key, value=kv_indices)   # 把整份 req_to_token 登記進去
```

`req_to_token` 裡的格子有一部分是第 1 步從各 namespace 拿來的。現在它們被**再登記一次**
到 `None`（預設 namespace）底下：

```
格子 5000–5099
  ├── 登記單 A：system_prompt_key 的 node   （原本的）
  └── 登記單 B：None 的 node                （多出來的）
```

後果是延遲發生的：

```
evict 淘汰登記單 A → free(5000–5099) → 格子回到空鑰匙籃
                     但登記單 B 還在服務它們
evict 淘汰登記單 B → free(5000–5099) → 同一批再還一次   ← DOUBLE-FREE
```

3090 上重現出來的順序正是這個：

```
SUBCTX-TREE-DUP at=chunk dup=47 (was 0) owners=[('None', 47), ('system_prompt_key', 47)]
SUBCTX-DOUBLE-FREE from radix_cache.py:1485 in evict: already_free=1 slots=[43]
崩潰：853 + 36058 = 36911 vs 36814     → +97，正好等於 dup_within_tree
```

441（H200）上是 +14，同一個形狀。

### 修法

把判斷改成跟第 4 步用同一個條件，並且把三個呼叫點統一到一個 gate：

```python
def serves_sub_contexts(self, req) -> bool:
    return bool(getattr(req, "has_sub_contexts", False)) and self.supports_sub_contexts()
```

`_finish_sub_contexts` 也要能處理「一個 node 都不擁有」的 request：
`tree_owned is None` 時當成全 False，逐塊迴圈才會只還自己的。

修完同一個 workload：1,034 requests、0 失敗、0 TREE-DUP、0 DOUBLE-FREE、0 洩漏，
rotation 仍然有作用。

### 為什麼壓測抓不到

replay client 用 `ignore_eos`，所以「prefill 當下就吐停止 token」在那個設定下
**結構上不可能發生**。

---

## 15. 已知的限制與待辦

### 會影響命中率或成本的

- **（舊存法的問題，鏈式儲存已解決）cdc 的短訊息與模型回覆進不了快取。** 2026-10-05 以前 `idx` / `cdc` 每塊一個
  namespace：短於 `MIN_CHUNK` 的塊不寫入、每輪重算；generation prompt（3 個 token）也不寫入，連帶 11-2 的條件永遠不成立，
  回覆不進快取。Terminal-Bench 上 cdc 命中率因此低於 `off`（2026-10-02 估計每請求多算約 600 token）。
  改成鏈式儲存後兩者都隨整條鏈存進樹；4090 smoke（Qwen2.5-7B，初步）中只往後接的流量 cdc 每筆命中與回覆都和 `off` 相同。
- **短於 `MIN_CHUNK` 的區塊不登記索引**：在鏈上、從頭比對拿得到，但位置變了 scan 找不回來。
- **位置相同也要複製**：別條鏈上、位置剛好相同的段，舊存法是直接共用格子，現在要複製一份（鏈必須擁有自己的格子）。
  GPU 記憶體用量跟 `off` 一樣，不再有「同一段內容只存一份」的節省。複製在沒開 speculative decoding 時是逐層的 Python 迴圈。
- **等待中的請求每輪重新 scan、重新複製**：沒被收進的請求會放掉副本（第 7-4 節），下一輪再做一次。
- **`sub_context_index_declined`**：一個請求因為沒空間被標記後只用前綴比對，直到被 retract 才清掉。
  「沒有其他請求在跑」只看 running batch 與 chunked request；overlap 下還在跑、沒併進 running batch 的
  prefill-only batch 看不到，可能多判一次「沒空間」（只少了重用，不會卡住）。
- **probe 不受一次 prefill 的預算限制，可能 OOM**：放不放得進一次 prefill 只算要算的 token；開 selective recompute 時
  probe 要跑精確前綴之後的所有列，`PrefillAdder` 讓放不下的 probe 單獨跑。4090（chunked 2048）上 27.9k 列的 probe 在
  qwen2 MLP 要 970 MiB 時 OOM、server 崩潰；約 6k 列的 RAG probe 沒事，H200 尚未踩到。2026-10-07 試過「probe 列數超過
  `--chunked-prefill-size` 就退回前綴」：不再 OOM，但 4090 上 SWE 重播的掃描重用少 90%、RAG 全部退回，已撤回。根本解法是 probe 分批跑。
- **（已解決，2026-10-05）probe 跑整段 prompt。** 以前同一批裡只要有一個 layout，所有請求都把整段 prompt 送進 probe，
  而 `PrefillAdder` 只對有 layout 的請求計入額外列。現在 probe 從各自的起點開始（第 9 節），同批的前綴命中只跑它要算的列，
  adder 的計算也跟實際一致。改了 probe 的形狀會讓矩陣乘法的捨入不同（同樣的列、不同的總列數，bf16 結果不一定位元相同），
  所以和改之前的 cdc@r 回覆不會逐字相同。

### 隔離與正確性

- **角色 namespace 不含 `extra_key`**：blocks 模式（沒開內容雜湊）下，不同 cache_salt / LoRA 的請求共用
  `system_prompt_key` 等 namespace，租戶與 adapter 的隔離被打破。內容雜湊模式（`idx` / `cdc`）沒有這個問題。
- **sparse prefill 路徑沒有傳 attention sink**：有 sink 的模型不在啟動檢查的拒絕清單裡。

### `_cache_sub_context_output` 的 gate 過嚴（已決定不處理；只影響 `on` / `rot`）

現在要求 `cache_protected_len == prompt_len`（整條 prompt 不准有洞）。
但要存的東西是「最後一塊 ++ 生成的 token」，插進**最後一塊的 namespace**，
前面的塊在別的樹裡。

安全性真正需要的是**最後一塊確實被樹擁有**：

```python
or not (req.sub_context_tree_owned and req.sub_context_tree_owned[-1])
or req.sub_context_owned_lens[-1] != len(last_seg)
```

兩個條件都要——`owned_lens[-1]` **不能單獨代表擁有權**，因為查找時把旋轉副本
也算進 `owned`，一個位移後被旋轉的最後一塊會讓 `owned_lens[-1] == len(last_seg)`
而 `tree_owned[-1] == False`。若只拿掉 `cache_protected_len` 那一行，
樹會收下旋轉副本、而後面的 free 迴圈又把它們還掉——**真的 bug**。

**決定不做。** 這個放寬只在「前面的塊被凍住、最後一塊正常」時有用，而 AgentVerse workload
是反過來的（query 在 system prompt 裡、`messages_key` 被 first-writer-wins 凍住），
卡住的正是最後一塊——放寬與否都過不了 gate。cdc 的情形（最後一塊是不寫入的 generation prompt）
也一樣過不了。記在這裡是為了讓下一個看到這個條件的人知道它已經被評估過。

### Stage 2 的最小 block 門檻（未做）

append 省下的是重算，付出的是一趟 scheduler round-trip。用 Stage 1 量到的
每 token 旋轉成本回推，塊要大約 300 個 token 才回本。

**這個門檻會改變哪些 block 被重用，所以會改變輸出**——pass@1 必須在門檻定案之後才量。

### 模型通用性（部分完成）

- 已完成：性質自測取代型別白名單（放行 llama3、dynamic、YaRN）
- 已完成：YaRN 家族除 mscale
- 未做：部分旋轉（`rotary_dim < head_dim`）——native reference 已經處理了，kernel 還沒
- 未做：fp8 KV（需要 dequant → rotate → requant，會累積量化誤差）
- 未做：MLA pool（DeepSeek 那條線；只有 `k_pe` 帶 RoPE，可能反而更便宜）
- `page_size == 1` 可能是真實 serving 場景下更限制人的條件

### 什麼 workload 用得上（2026-10-02 初步觀察）

- SWE-agent（last-5 observation 省略）：cdc 比 `off` 多命中的部分全在第一則 user 訊息之後的對話段；
  system + tools 與第一則 user 訊息 `off` 已經全部命中。
- Terminal-Bench（對話只往後追加）：`off` 已命中約 97%；舊存法的 cdc 因為上面的短訊息與回覆問題反而較低，
  鏈式儲存後應與 `off` 相同（smoke 已確認，Terminal-Bench 本身未重跑）。
- 不同 task 之間能共用的只有開頭約 1k token 的共用指令；同一題的平行 rollout 之間 cdc 也沒有多撿。
  cdc 要發揮，需要「大量相同內容出現在不同的前文之後」的 workload。

---

## 16. 現在可用的模型

以下是拿本機 HuggingFace cache 裡的 config 實際建出 rope、跑過 `rope_delta_composable_reason`
的結果（2026-09-10），不是從架構名推論的。這只回答「能不能旋轉」；開 index 或 selective recompute
還要過第 3-3、3-4 節的條件。

### 四道關卡

| 關卡 | 條件 | 由什麼決定 |
|---|---|---|
| 切分能不能服務 | 原生 `RadixCache`、`page_size == 1`、非 EAGLE/hierarchical/SWA/mamba | 架構 + 啟動參數 |
| RoPE 形狀 | neox 配對、`rotary_dim == head_dim` | 模型 config |
| RoPE 律 | 性質自測通過 | **實測** |
| KV pool | MHA（非 MLA）、`store_dtype == dtype` | 架構 + `--kv-cache-dtype` |

### 通過

| 家族 | rope_type | head_dim | 備註 |
|---|---|---|---|
| Qwen3 / Qwen3-MoE | default | 128 | 8B / 14B / 32B / 30B-A3B / Coder-30B / Coder-480B |
| Qwen2.5 | default | 128 | Coder-7B 等 |
| Llama 3.1 / 3.2 / 3.3 | **llama3** | 128 / 64 | 性質自測放行的 |
| Llama 2 系 | default | 128 | vicuna-7b-v1.5 等 |
| Mistral / Devstral | default | 128 | Devstral-Small-2505/2507、Mistral-Small-3.2 |
| YaRN 長上下文變體 | yarn | — | 除 mscale 後放行；本機無 config，未實測 |
| dynamic NTK | dynamic | — | 同上 |

`Llama3RotaryEmbedding` 的自測最差樣本只用掉 8% 預算，跟 plain RoPE 一樣——它只改
`_compute_inv_freq`（與位置無關的逐維重映射），cache 仍由基底類別建、沒有 mscale。

**AWQ / GPTQ / FP8 權重量化不影響這裡。** 擋的是 `--kv-cache-dtype fp8_*`，也就是 KV
本身被量化（`store_dtype != dtype`）。

### 不通過

| 模型 | 被哪一條擋 |
|---|---|
| **Qwen3-Coder-Next** | **partial rotary 64/256** |
| Qwen3-VL / Qwen2-VL / Qwen2.5-VL / Omni | mrope（位置是 3-vector） |
| Phi-3.5-vision | `su` / longrope（位置閾值換 inv_freq） |
| DeepSeek-V2/V3/R1、MiniCPM3、LongCat、Kimi-Linear | MLA pool |
| Llama 4、gpt-oss、MiMo-V2-Flash、Step3p5 | hybrid SWA → `SWARadixCache`，連切分都不支援 |
| ChatGLM、GLM-4、Command-R、GPT-J、EXAONE-4、Hunyuan、Mistral-Large-3 | 非 neox 配對 |
| Qwen3-Next、Qwen3.5 系列、Falcon-H1 等 linear-attention 混合 | mamba pool → `MambaRadixCache`，連切分都不支援 |

Qwen3.5 系列：`Qwen3_5Config` / `Qwen3_5MoeConfig` 在 `ModelRunner.hybrid_gdn_config` 被認成 hybrid，
scheduler 因此選用 `MambaRadixCache`。

**Qwen3-Coder-Next 只差部分旋轉那一項**：config 是 `head_dim=256,
partial_rotary_factor=0.25`，rope 律本身沒問題。`rotate_copy_kv_native` 已經處理了
未旋轉的 tail，只有 Triton kernel 還假設整個 head 都轉。要往新模型走的話這是最短的一步。

### 從 config 判不出來的兩件事

1. **多模態模型可能有第二顆 RoPE**（視覺塔一顆、文字一顆），`find_rotary_embedding`
   會回 "2 distinct RotaryEmbedding instances" 而拒絕。所以 LLaVA、Mistral3 要當「需實測」。
2. **`is_neox_style` 由模型檔決定**，不在 config 裡。上面的非 neox 名單是從
   `python/sglang/srt/models/*.py` grep 出來的，不是自測的結果。

### 最可靠的確認方式

直接啟動。`rotation_unsupported_reason` 在啟動時就判，不通過會 raise 並說明原因：

```
Sub-context KV rotation is enabled but cannot be served: partial rotary (64 of 256
dims); the rotation kernel assumes the whole head rotates
```

通過的話 log 裡會有帶實測數字的這一行：

```
RoPE delta self-test passed for Llama3RotaryEmbedding on 8 (position, delta) samples;
worst was position 30000 delta 1000 at 6.18e-04 relative, 8% of its 7.40e-03 budget
```
