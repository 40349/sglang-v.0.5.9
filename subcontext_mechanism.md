# Sub-context KV 重用：機制說明

這份文件講的是這個 fork（`idx` 分支，依 2026-10-14 的程式碼）的 sub-context KV 重用機制：
怎麼切塊、怎麼找到可重用的 KV、位置不同時怎麼旋轉、怎麼只算空洞、怎麼重算一部分重用的 token、
算完怎麼寫回快取與釋放。怎麼啟動 server 見 `README_CDC.md`。

2026-10-14 起只剩 `off` 與 `cdc` 兩種 arm。早期的 `on` / `rot`（依角色切塊、每塊一個 namespace、逐塊拼接）
與 `idx`（依角色切塊 + 索引）已移除，程式在 git 歷史（`origin/idx` 5ff1e41）。

主要涉及的檔案（都在 `python/sglang/srt/` 下）：

| 檔案 | 負責 |
|---|---|
| `entrypoints/openai/serving_chat.py` | 切塊（cdc）、錄製請求 |
| `managers/tokenizer_manager.py`、`managers/io_struct.py` | 驗證切塊與 input_ids 一致、自動截斷後裁齊、往下傳 |
| `managers/scheduler.py` | 啟動檢查、掛上旋轉器與索引；沒被收進這一輪的請求放掉掃描結果 |
| `managers/schedule_batch.py` | 讀路徑：前綴比對 + scan、複製副本的生命週期、sparse prefill 的輸入 |
| `managers/schedule_policy.py` | `PrefillAdder`：sparse prefill 不可切、probe 預算 |
| `mem_cache/radix_cache.py` | 寫路徑：整條插入後登記新存的 chunk、chunk 位置維護、audit |
| `mem_cache/rotate_kv.py` | Triton 旋轉 kernel、KV 複製、啟動時的 RoPE 性質自測 |
| `mem_cache/subctx_index.py` | 內容索引（依 token 內容找已快取的區塊、記錄它們在樹上的位置）、CDC 切點 |
| `mem_cache/subctx_blend.py` | selective recompute：打分、挑選、寫回 |
| `mem_cache/subctx_relay.py` | RelayCaching：每個 slot 的 decode 狀態、decode 掛鉤、被注意程度的 kernel、挑選規則、profile |
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

`cdc` 的做法：

1. **切塊**：每則訊息開頭切一刀，訊息內部再依內容切成平均約 256 token 的小段（第 5 節）。
   同樣的內容不論出現在哪裡都切在同樣的地方。
2. **前綴比對 + 內容索引 + sparse prefill**：先照原生 radix cache 從頭比對；比不到的部分，
   再用內容索引在 prompt 的**任何位置**找已快取的區塊，複製到自己的格子（位置不同就旋轉），只計算中間的空洞。
   算完整條照原生方式存回樹（一條從位置 0 開始的鏈），新存上樹的區塊登記進索引、記下它在樹上的位置（第 6 節）。
3. **selective recompute**（arm `cdc@r`）：重用的區塊是在別的前文下算的，挑出 key 偏差最大的
   比例 r 重算，修正品質。`cdc@r1,r2,…` 在連續幾層逐層篩選（第 8 節）。

只往後接的對話因此跟原生 radix cache 一樣；多出來的只有「同樣內容出現在不同前文之後」的部分。

**位置的代價**：RoPE 把絕對位置烤進了 K。一塊 KV 是在位置 100 算的，就只能在位置 100 被直接重用。
所以每個 tree node 要記 `canonical_position`——「我這份 KV 是在哪個絕對位置算出來的」（鏈從位置 0 開始，
等於這個節點在鏈上的起點）。

**旋轉**：RoPE 的角度對位置是線性的，所以 `R(a)·R(b) = R(a+b)`，可以把一塊 KV 旋轉 `Δ` 搬到新位置：

```
R(p_new) · f(k_raw) == R(p_new − p) · k_cached[loc]
```

`f` 是 RoPE 之前對 K 做的事（Qwen3 是 k_norm）。這在位置上**完全精確**——RoPE 保範數，
所以不需要重跑 `f`。V 與位置無關，直接複製。

但它修的是**位置**，修不了**前文**。那塊當初是在別的上下文下算出來的，這是品質代價，
也就是 pass@1 要量的東西，也是 selective recompute 要修正的東西。

---

## 1. 開關與 arm

所有開關都是環境變數，在 import 時讀一次；fork 沒有新增任何命令列參數。

### arm（`sglang_server.sh` 的 `export SGLANG_DISABLE_SUBCONTEXT= ...` 那段）

| arm | 設定的環境變數 | 走哪條讀路徑 |
|---|---|---|
| `off` | `SGLANG_DISABLE_SUBCONTEXT=1` | 不切塊，原生 radix cache |
| `cdc[@r]` | `SGLANG_SUBCTX_TOPK_RATIO=r`（不寫比例就是 0） | 依內容切塊；前綴比對 + scan + sparse prefill；整條存成一條鏈 |
| `relay@S,D,E` | `SGLANG_SUBCTX_RELAY_LAYERS=S,D,E` | 同 cdc，重用的 token 照 RelayCaching 修正（第 8R 節） |
| `relay@profile` | `SGLANG_SUBCTX_RELAY_PROFILE=<檔案>` | 同 cdc，重用的 token 全部重算，並記錄每層的偏差 |

### 全部環境變數

| 環境變數 | 預設 | 作用 |
|---|---|---|
| `SGLANG_DISABLE_SUBCONTEXT` | 空 | 非 0：chat 不切塊、不做啟動檢查、不掛旋轉器與索引。空：cdc，旋轉器與索引一定掛上 |
| `SGLANG_SUBCONTEXT_ROTATE_NATIVE` | 空 | 用純 torch 旋轉取代 Triton kernel（除錯） |
| `SGLANG_SUBCTX_MIN_CHUNK` | 64 | 索引登記與重用的最短長度；也是 cdc 切點的最小間距 |
| `SGLANG_SUBCTX_CDC_TARGET` | 256 | cdc 候選切點的平均間距（捨去到 2 的冪）；長訊息內 chunk 平均約 `MIN_CHUNK` + target |
| `SGLANG_SUBCTX_CDC_MAX` | 1024 | cdc 強制切斷長度 |
| `SGLANG_SUBCTX_TOPK_RATIO` | 0 | selective recompute 重算的比例；逗號分隔時每個比例對應一個連續的打分層（逐層篩選） |
| `SGLANG_SUBCTX_TOPK_LAYER` | 1 | 第一個打分層 |
| `SGLANG_SUBCTX_RELAY_LAYERS` | 空 | RelayCaching 的 `L_start,L_det,L_end`（第 8R 節） |
| `SGLANG_SUBCTX_RELAY_TAU_DEV` | 1.5 | `L_det` 的 value 偏差門檻（倍數 × 平均） |
| `SGLANG_SUBCTX_RELAY_TAU_INF` | 1.45 | 被注意程度的門檻（倍數 × 候選平均） |
| `SGLANG_SUBCTX_RELAY_SUFFIX` | 10 | 每段重用內容一定重算的尾端 token 數 |
| `SGLANG_SUBCTX_RELAY_PROFILE` | 空 | profile 模式的輸出檔 |
| `SGLANG_SUBCTX_ROTATE_GPU` | 空 | 旋轉 kernel 的 CUDA event 計時（`rotate_kv.py`） |
| `SGLANG_SUBCTX_AUDIT` | 空 | 重用 chunk 時逐 token 比對位置上的內容（`radix_cache.py`）、slot 級 double-free 偵測（`allocator.py`） |
| `SGLANG_SUBCTX_TRACE` | 空 | 印 TRACE 除錯行 |
| `SGLANG_STAGE_TRACE` | 空 | host 階段計時的輸出前綴 |
| `SGLANG_FORWARD_TRACE` | 空 | 每個 forward pass 的 GPU 時間輸出檔 |
| `SGLANG_DUMP_TREE` | 空 | 每次 extend 後印整棵 radix tree |
| `SGLANG_CAPTURE_REQUESTS` | 空 | 錄下每個 chat 請求的輸出檔 |

`SGLANG_SUBCTX_AUDIT` 在兩處的解析方式不同：`radix_cache.py` 是「非空且不是 0 即開」，
`allocator.py` 用 `get_bool_env_var`（只認 `true` / `1`）。設成 `1` 兩邊都開。

### import 時的交叉檢查（`subctx_config.py`）

不合法直接丟 `ValueError`，server 起不來：

- 設了已移除的開關（`SGLANG_SUBCTX_SPLIT`、`SGLANG_SUBCTX_INDEX`、`SGLANG_SUBCONTEXT_ROTATE`、
  `SGLANG_SUBCONTEXT_ROTATE_ACROSS`、`SGLANG_SUBCTX_HASH_KEYS`、`SGLANG_SUBCTX_INDEX_DRYRUN`、`SGLANG_SUBCONTEXT_CACHE_OUTPUT`）。
- `TOPK_RATIO` 的每個比例都必須在 [0, 1] 且不遞增；第一個 > 0 時不能同時設 `DISABLE`。
- `TOPK_LAYER` 不可為負；`TOPK_LAYER = 0` 只允許第一個比例 = 1.0（第 0 層的 key 只取決於 token 與位置，分數全為 0）。
- `RELAY_LAYERS` 或 `RELAY_PROFILE` 設了：不能同時設 `DISABLE`、不能同時有 `TOPK_RATIO > 0`、兩者不能同時設；
  `RELAY_LAYERS` 要三個整數且 `0 <= L_start < L_det <= L_end`（超過模型層數在啟動時拒絕）；三個門檻不可為負。

---

## 2. 狀態

| | 內容 | 在哪 |
|---|---|---|
| `k_buffer` / `v_buffer` | 真正的 KV bytes，`[slot, head, dim]` | GPU |
| `req_to_token[req, 位置]` | **位置 → slot 的權威對照表** | GPU tensor |
| tree node | `key`（token）、`value`（slot）、`extra_key`（cache_salt + LoRA id）、`canonical_position` | CPU |
| `free_pages` | 哪些 slot 是空的 | CPU |

**GPU 上的 bytes 不帶身分。** 一個 slot 是誰的，完全由「誰指向它」決定。
這句話是後面所有規則的來源。

內容索引（`SubContextIndex`）是第五份狀態：存 token id、chunk id，以及每個 chunk 在樹上的位置
（「持有它最後一個 token 的節點 + 它第一個 token 的位置」，可以有好幾個，最早的優先）。不存 KV；
KV 在不在還是由 radix tree 決定。樹拆節點、淘汰節點時同步更新位置（見第 6 節）。

### request 身上的欄位（`Req.__init__`）

| 欄位 | 意思 | 誰寫 |
|---|---|---|
| `sub_context_ends` | 每塊在 `origin_input_ids` 裡的結束位置，遞增，最後一個等於 prompt 長度。存回樹時當登記的區塊邊界 | 切塊 |
| `sub_context_match_nodes` | scan 複製來源所在的節點，上鎖到副本存進樹為止 | scan |
| `sub_context_rotated_slots` | 自己 alloc 的副本，**還沒交給 `req_to_token`** | scan |
| `sub_context_layout` / `_layout_ours` | scan 的稀疏配置 `(start, end, slots)`，以及該段是否為自己的副本；第一段可能是從頭比對到的前綴 | scan |
| `sub_context_exact_len` | 從頭比對到的前綴長度；這段是精確的，selective recompute 不打分 | scan |
| `sub_context_index_declined` | 沒空間被拒、又沒有其他請求在跑；之後只用前綴比對 | scheduler |
| `sub_context_tally` / `_counted` | 這次比對的統計，第一次真的執行時加進索引的統計一次 | scan / `prepare_for_extend` |
| `sub_context_discarded/_moved/_rotated` | 給 forward trace 的計數，讀完歸零 | scan |
| `cache_protected_len` | 從 0 起連續被樹擁有的長度（上游欄位） | scan、`cache_unfinished_req` |

---

## 3. 啟動：能不能跑

沒設 `SGLANG_DISABLE_SUBCONTEXT` 時，`Scheduler._check_sub_context_support` 一定會跑。
條件不成立就 **raise，不啟動**——每一種不支援的組合失敗的方式都一樣：attention 還是回傳一個數字，只是錯的。

### 3-1 切分能不能服務（`unsupported_reason`）

| 條件 | 不成立的話 |
|---|---|
| 是原生 `RadixCache`（不是子類、不是 C++ 版、不是 hierarchical） | 鏈式儲存與 chunk 位置維護只寫在它上面 |
| 不是 hybrid SWA / mamba 模型 | 它們用 `SWARadixCache` / `MambaRadixCache`，同上 |
| `page_size == 1` | 分頁會把 slot 綁成一組，塊邊界對不齊 |
| 不是 EAGLE | 它把 key 改寫成 bigram |
| 沒有 `--disable-radix-cache` | 沒有樹 |

### 3-2 旋轉能不能做（`rotation_unsupported_reason`）

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

**條件不成立就不啟動**，因為不支援的 RoPE 不會失敗，
它會用錯的律去轉，然後安靜地讓每個重用的塊都變差。

### 3-3 sparse prefill 能不能做（`sparse_prefill_unsupported_reason`）

| 條件 | 為什麼 |
|---|---|
| `--attention-backend triton` | 只有 triton 的 kernel 收 per-position 遮罩；其他 backend 以索引判斷 causal |
| 不是 sliding-window attention | 它從前綴長度推每個 key 的絕對位置，有洞的 prefill 沒有那個長度 |
| 沒有 `--enable-piecewise-cuda-graph` | 它 capture prefill 的形狀，而稀疏 prefill 的遮罩每個請求大小不同 |
| 沒有 `--enable-mixed-chunk` | 它把 decode 的 token 接進 prefill batch，而 sparse prefill 給的位置只涵蓋 prefill 的 token |

另外：**又開 diffusion LLM 或 prefill/decode 分離也拒絕**：那兩條排程路徑不會放掉被拒請求 scan 拿的鎖與副本（第 6-4 節）。
**又開 `--enable-pdmux` 也拒絕**：PD multiplexing 的 forward 跑在自己的 stream 上，旋轉的複製排不到它後面。

### 3-4 selective recompute 能不能做（`topk_unsupported_reason`，ratio > 0 時）

| 條件 | 為什麼 |
|---|---|
| 模型有 `forward_split_prefill` | token 數要在打分層之間裁減 |
| 最後一個打分層之後至少還有一層 | 否則沒有東西要重算 |
| 沒有 pipeline parallel | 被比較的那層可能在別的 rank |
| attention TP = 1 | 分數對 head 維度加總，TP 下每個 rank 只看到自己的 head，會選出不同的 token |
| 沒有 DP attention、`--moe-dense-tp-size 1` | token 被分散到各 rank，全域的列號在每個 rank 指到不同 token |
| 沒有 two-batch overlap、speculative decoding、torch.compile、回傳 hidden states、LoRA | 都假設整個 forward 的 token 數不變 |

哪些模型過得了，見第 13 節。

---

## 4. 生命週期總覽

```
1. 切塊（serving_chat）      chat request → 每則訊息開頭 + 依內容的切點               ← 第 5 節
2. 查找（init_next_round_input）
   └─ scan                   先從頭比對，其餘用內容索引在任何位置找、複製進自己的格子 ← 第 6 節
3. PrefillAdder             有洞的 prefill 不可切；沒收進這一輪的請求放掉鎖與副本     ← 第 6-4 節
4. prepare_for_extend       prefix → req_to_token，責任交接；sparse 時依位置取輸入  ← 第 7 節
5. prefill                  算 KV 並產生第一個 token；ratio > 0 時兩段 layer          ← 第 7、8 節
6. cache_unfinished_req     整條插入、登記新存的區塊、放掉來源節點的鎖              ← 第 9 節
7. decode                   一個一個吐 token
8. cache_finished_req       整條插入（含回覆）、登記                                 ← 第 9 節
```

其中第 6 步**不是每個 request 都會跑**：prefill 當下就產生停止 token 的 request、排隊時被 abort 的
request 直接到第 8 步。第 8 步因此先做兩道冪等的保險（放掉 scan 的鎖、還掉沒交接的副本）。

`chains_sub_contexts(req)`（有索引、request 有切塊、cache 支援）決定走不走第 2、6、8 步的 sub-context 分支；
三個地方用同一個判斷。不成立的 request（`off`，或 cache 不支援）走上游原本的路徑。

---

## 5. 切塊（`serving_chat._compute_sub_context_ends`）

### 5-1 哪些請求會切

- **只有 `/v1/chat/completions`**，而且只在 jinja chat template 的路徑；多模態請求不切
  （它會把 prompt decode 成文字再重新 tokenize）。
- `/generate` 在 handler 入口把 `sub_context_ends` 清成 `None`，永遠不切。
- 設了 `DISABLE` 不切。
- 切出來少於 2 塊就當作不切。

### 5-2 切法

1. **訊息開頭一定切**：從模板 render 一則與兩則訊息，第二則的第一個 token 若是特殊 token，就當成
   「每則訊息的開頭標記」（Qwen 是 `<|im_start|>`），只算一次。prompt 裡每個這種 token 的位置都切一刀。
   找不到標記就只依內容切。
2. **每則訊息內依內容細切**（`subctx_index.cut_points`）：對每個位置算「從這裡開始 32 個 token」的指紋，
   指紋低 `log2(CDC_TARGET)` 個位元全為 0 的位置是候選切點。距上一個切點不足 `MIN_CHUNK` 的忽略、
   超過 `CDC_MAX` 強制切、尾段短於 `MIN_CHUNK` 就併入前一塊；訊息短於 `2 × MIN_CHUNK` 不細切。
   同樣的內容不論出現在哪裡都切在同樣的地方。
   因為距上一個切點不足 `MIN_CHUNK` 的候選被忽略，之後平均還要再等 `CDC_TARGET` 個位置才遇到候選，
   長訊息內的 chunk 平均約 `MIN_CHUNK + CDC_TARGET`（預設約 320），不是 `CDC_TARGET`；
   短訊息則整則一塊（2026-10-22 實測：TB 長訊息內部 target 64/128/256 的平均是 134/187/296）。
3. **切點快取**：每則訊息的切點依「訊息長度 + 內容雜湊」存在 LRU（最多 65,536 則），
   同一則訊息在後面的 request 再出現時直接取用，只有沒看過的訊息才跑第 2 步。

切完只回傳**每塊的結束位置**（例如 `[120, 640, 897, 900]`，最後一個等於 prompt 長度），不回傳每塊的 token。
找訊息開頭、算快取的 key 都直接在 `prompt_ids` 這個 Python list 上做，只有沒看過的訊息才轉成 numpy 跑 `cut_points`。

短於 `MIN_CHUNK` 的訊息、prompt 最後的 generation prompt（`<|im_start|>assistant\n`，3 個 token）
都自成一塊。它們照樣隨整條鏈存進樹、從頭比對拿得到，只是不登記進索引——位置變了就找不回來。

### 5-3 chunk 的編號

`chunk_id(token, extra_key)` 把 token 以 int32 雜湊（xxh3-64），再接上 `extra_key`（cache_salt + LoRA id），
得到 `sc:<hex>`。它只當索引裡的 chunk 編號；樹裡整條存在 `req.extra_key` 底下，跟 `off` 一樣。

### 5-4 往下傳

- `tokenizer_manager`：`sub_context_ends` 要遞增、第一個大於 0、最後一個等於 `len(input_ids)` 才接受
  （`subctx_index.valid_block_ends`），否則警告並當作不切。只看位置不比對內容：`input_ids` 就是切塊讀的那份 `prompt_ids`。
- `--allow-auto-truncate` 截掉 prompt 尾巴後，`subctx_index.clip_block_ends` 把結束位置裁齊：
  保留落在截斷點之前的、最後一塊改成結束在截斷點、其餘丟掉。tokenizer_manager（截到 context length）和
  scheduler（`validate_input_length` 截到 `max_req_input_len`）截完都各裁一次。
- scheduler 建立 `Req` 時傳入 `extra_key`、`sub_context_ends`。
  （上游 v0.5.9 的 scheduler 不傳 `extra_key`，所以上游的 `cache_salt` 不會區隔快取；這個 fork 會，`off` 也一樣。）

---

## 6. scan 與鏈式儲存

cdc 的 request 讀寫都**像原生 request 一樣**：在樹裡存成一條從位置 0 開始的鏈，放在
`req.extra_key`（cache_salt + LoRA id）底下。內容索引只負責「從頭比對斷掉之後，後面還有哪些已快取的區塊、
它們在樹上哪裡」。（2026-10-05 起；之前是每塊一個內容雜湊 namespace，見 git 歷史。）

### 6-1 索引

`SubContextIndex`（`subctx_index.py`）記「哪些 token 序列存在樹上的哪裡」：

- **登記**：request 存回樹之後，新存上樹的區塊（第 6-5 節）連同 `req.extra_key` 登記；短於 `MIN_CHUNK` 的不登記。
  每筆記 token、開頭 32 token 的指紋，以及**位置**：持有最後一個 token 的節點 + 第一個 token 的位置。
- **多個位置**：同一段內容存在好幾條鏈上時，每條都記一筆，**最早的優先**——讀的是最初算出來的那份，
  一次旋轉就到位；後來的（多半是複製、旋轉過的副本）只在前面的被淘汰後才用（`chain_chunk_slots`，
  `radix_cache.py` L384）。
- **範圍**：每個 `extra_key` 一個範圍，掃描只看自己的範圍——不同 cache_salt / LoRA 互相看不到。
- **掃描**：對 prompt 每個位置算「從這裡開始 32 個 token」的指紋（一次向量化算完），先用 Bloom filter
  （快速排除一定沒登記的位置的位元表）過濾，再查「chunk 開頭指紋 → chunk」表，最後逐 token 比對確認。
  回傳所有出現（可重疊）。新 request 自己怎麼切塊跟比對無關。
- **挑選**：從可能重疊的命中裡，挑出互不重疊、總長最大的一組。

### 6-2 `_scan_sub_contexts` 的步驟（`schedule_batch.py` L1216）

1. 放掉上一輪的鎖與副本，清掉上一輪的配置。
2. **從頭比對**：跟原生 request 一樣 `match_prefix(fill_ids[:max_prefix_len], req.extra_key)`，
   得到前綴長度 `exact`、`last_node`（之後由 `PrefillAdder` 上鎖，跟原生一樣）、`cache_protected_len = exact`。
3. **掃描剩下的部分**：`index.scan(fill_ids, extra_key, min_end=exact + MIN_CHUNK)`，只回傳結尾在 `min_end` 之後的命中，
   只對 `fill_ids[min_end - 最長 chunk:]` 取指紋（每個 scope 記登記過的最長 chunk）；結果和掃整條再裁相同。
   每個命中裁到「`exact` 之後、`max_prefix_len` 之前」（prefill 至少要算最後一個 token），裁完短於 `MIN_CHUNK` 的丟掉。
   `index.select` 挑出不重疊的組合。
   被標成 `sub_context_index_declined` 的 request 跳過這一步（第 6-4 節）；開了 selective recompute 時，
   要求 logprob 的 request 也跳過（重算後的列數會讓 logprob 的對齊跑掉）。
4. **找位置、上鎖**：`chain_chunk_slots(cid)` 從位置節點往上走父節點，湊出 `[start, end)` 的格子；
   位置失效（節點已淘汰、範圍對不上）就丟掉那筆位置，全失效就整段重算（記入 `discarded`）。
   `Δ = 命中位置 − 當初的位置`；`Δ` 超出旋轉範圍的段在這裡就丟掉（記入 `moved`、`discarded`），不上鎖。
   其餘的段對它的節點 `inc_lock_ref`（連同所有祖先）。
5. **先算放不放得進一次 prefill**（第 6-3 節）：放不下就放鎖、只用前綴，不淘汰也不複製。
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

### 6-3 放不進一次 prefill → 只用前綴

有洞的 prefill **不能切成多個 chunk**。檢查「要算的 token 數 ≤ `--chunked-prefill-size`」做兩次：
複製之前用找到的段長先算一次（`_fits_one_pass`），放不下就不複製；複製之後再算一次
（`_sub_context_layout_fits_one_pass`），因為 pool 滿而沒複製成的段會讓要算的變多。任一次放不下都記一次 `fell_back`，
放掉段的鎖與副本，退回只用前綴（之後照原生 chunked prefill）。放得下但這一輪預算不夠的，`PrefillAdder` 回 `OTHER` 讓它等。

### 6-4 沒被收進這一輪的請求

`init_next_round_input` 在排程迴圈裡對每個排到的等待請求都會跑一次，scan 在這時就拿了鎖、配了副本。
請求若沒被收進這一輪（`add_one_req` 回非 `CONTINUE`、且它不在 `can_run_list` 裡——adder 收進一個請求後
預算剛好用完也會回 `NO_TOKEN`/`OTHER`，那種不算），`scheduler.py` L2239 呼叫
`release_sub_context_scan`（`schedule_batch.py` L1024）：放掉段的鎖、還掉副本。下一輪重新 scan。

**沒空間的卡死**：段的鎖會把來源節點到 root 的整條路徑都釘住，加上副本，可能讓 adder 的剩餘額度不夠，
回 `NO_TOKEN`；下一輪重 scan 結果一樣。所以當「`NO_TOKEN`、沒有請求在跑、沒有 chunked 請求、這一輪也沒收進任何請求」
時，這個請求標成 `sub_context_index_declined`，之後只用前綴比對（記入 `no_room`）；這一輪**不**設
`batch_is_full`，否則沒有請求在跑時 scheduler 不會再排 prefill，前綴重試永遠不會發生。

離開等待隊列的其他三條路（`abort_request`、`_abort_on_waiting_timeout`、優先序擠掉）也都呼叫
`release_sub_context_scan`。

### 6-5 存回樹與登記（`_register_chain`，`radix_cache.py` L514）

`cache_unfinished_req` / `cache_finished_req` 走**上游的整條插入**：
`insert(fill_ids, req_to_token 那一列)`；樹已經有的部分（`prefix_len`）照上游規則把自己的重複格子還掉。
插入之後（第 9 節）：

- **登記新存上樹的區塊**：區塊邊界用 request 收到時的切法（`sub_context_ends`）；結束長度
  `> prefix_len`（含有這次新存的 token）且 ≥ `MIN_CHUNK` 的才登記，位置是這條鏈上持有它最後一個 token 的節點。
  從別處複製進來的段也在其中——同一段內容因此多一筆位置（排在原本的後面）。
- **結束時**（`cache_finished_req`，`final=True`）：最後一塊延伸到生成的最後一個 token，再用 `cut_points` 重切
  ——所以 generation prompt 和模型的回覆一起登記。
- `cache_unfinished_req` 插入後放掉 scan 拿的段的鎖（副本已經寫好，交給鏈了）。

開了 selective recompute 時，副本裡有一部分 token 是重算過的；它們照樣隨鏈存進樹，之後的 request
從頭比對到時當成精確前綴直接用，不再重算（這是設計上的選擇：鏈不能有洞，停在第一個借來的段會丟掉它後面的所有內容）。

### 6-6 位置的維護

| 事件 | 做什麼 | 在哪 |
|---|---|---|
| 節點被拆開（`_split_node`） | 最後一個 token 落在新的上半節點的位置，改指上半節點（格子號碼本來就不變） | `_rehome_chunks`，L477 |
| 節點被淘汰 | 丟掉以它為位置的那幾筆；一個 chunk 的位置全丟光就從索引移除 | `_forget_chunks_at`，L503 |
| `reset()`（flush cache） | 清空位置表與索引 | L358 |
| 讀取時發現位置失效 | 丟掉那一筆，換下一筆 | `chain_chunk_slots` |

`_chunks_at`（節點 id → 以它為位置的 chunk）是反查表，讓拆節點和淘汰不必掃整個索引。
開 `SGLANG_SUBCTX_AUDIT` 時，每次讀取還會逐 token 比對位置上的內容（`SUBCTX-CHAIN-MISMATCH`）。

---

## 7. prepare_for_extend：責任交接與 sparse prefill

### 7-1 交接

```python
write_cache_indices(...)                 # prefix_indices → req_to_token（sparse 時見 7-2）
req.sub_context_rotated_slots = None     # 責任移轉
```

`sub_context_rotated_slots` 是一份**釋放責任清單**——`release_sub_context_rotated_slots`
會把清單上的每一批格子直接 free 掉。

| 情境 | 結果 |
|---|---|
| 在此之前 abort 或重新排程 | 清單還在 → `release_sub_context_rotated_slots` 救援 |
| 在此之後 | `req_to_token` 涵蓋它們 → 隨整條鏈存進樹，或在存回時當重複的格子還掉（第 9 節） |
| **忘了清空** | 兩邊都還 → **double free** |
| **太早清空** | 兩邊都不還 → **洩漏** |

所以這是唯一正確的交接點：只有這一刻 `req_to_token` 和記帳同時涵蓋那些格子。

### 7-2 sparse prefill（batch 裡有任何 request 帶 `sub_context_layout`）

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

## 8. selective recompute（`TOPK_RATIO > 0`）

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

**重算過的副本隨鏈寫回**：scan 的副本是 request 自己的格子，跟整條鏈一起存進樹（第 6-5 節）；
下一輪從頭比對到這段時當成精確前綴，不再打分。索引裡同一段內容的最早位置仍是原始算出來的那份，
別的 request 從索引找到這段時讀的是原始那份。

---

## 8R. RelayCaching（`relay@L_start,L_det,L_end`）

論文 arXiv 2603.13289（ICML 2026），照官方程式 github.com/YingshengGeng/RelayCaching（MIT）移植：
`RelayCaching/utils/relay_utils.py` 的 `RelayCache`、`utils/fixed_qwen3_hijack.py` 的掛鉤、`llm/gpt_chat.py`
的候選交接、`experiments/observation/layer_selection.py` 的挑層規則。論文的情境是「上一個 agent 的回覆被放進下一個
agent 的 prompt」；在這裡，scan 找到的重用段（其中由 decode 算出的就是上一輪的回覆）扮演那份 decoding KV，
每段**連續**的重用內容（相鄰的 scan run 合併）是一個 relay chunk。儲存、scan、旋轉、整條存回都和 cdc 一樣，
只有「怎麼修正重用的 KV」不同。程式在 `mem_cache/subctx_relay.py` 與 `ModelRunner.forward_subctx_relay`。

### decode 時記下的東西（`RelayState`，每個 KV slot 一份）

掛鉤在 `ModelRunner.init_subctx_relay` 註冊，在 CUDA graph capture 之前，所以會錄進 decode 的 graph：

| 掛在哪 | 做什麼 |
|---|---|
| 第 0 層的 forward pre-hook | 這一步 decode 寫的 slot 標成「由 decode 算出」，被注意程度歸零 |
| 第 `L_start` 層的 forward pre-hook（`L_start > 0`） | 存下該層輸入的 residual stream（`hidden_states + residual`） |
| 每一層 `RadixAttention` 的 forward hook | Triton kernel：這一步的 query 對「這個請求自己的輸出 key」（位置 `[out_start, seq_len)`，含這一步的）做 softmax，各 head 相加，加到每個 key 的被注意程度上 |

`out_start`（每個請求第一個輸出 token 的位置）在 `alloc_for_extend` 寫入；prefill 寫的 slot 在
`forward_extend` 開頭清成「非 decode、分數 0」；scan 複製或旋轉 KV 時（`KVRotator`）這三樣跟著複製。
CUDA graph 補齊的列 `seq_lens = 1`，範圍是空的，不會累加。

### prefill（每個請求）

1. **預選**（forward 開始時，`influence_preselect`）：每段取被注意程度前 256 名當候選，`p` = 候選中超過
   「`TAU_INF` × 候選平均」的比例，預算 `ceil(段長 × p)`；段尾 `SUFFIX` 個一定選，**算在預算裡**，剩下的名額依排名給候選。
2. **`[0, L_start)`**：跑新 token 與「沒有存 hidden state」的重用 token（例如 prefill 算出的生成標頭）；
   有存的不跑，這幾層保留快取的 KV。
3. **`L_start`**：有存的列以存下的 hidden state 接回（residual 為 0）。
4. **`[L_start, L_det]`**：所有列都跑，每列把新算的 K/V 寫進自己的 slot。
5. **`L_det`**（`deviation_select`）：每個重用 token 的分數 = 各 head 的 `1 - cos(新算的 V, 快取的 V)` 平均；
   選「超過 `TAU_DEV` × 這個請求平均」的，再聯集第 1 步的預選。請求整體的 `1 - 平均 cos` < 1e-6 就全部保留、
   下一層再選一次（**延後挑選**）。照官方程式，這些都用 KV 的 dtype 計算：bf16 下平均 cos 高於約 0.998
   就會捨入成 1.0，所以實際上平均偏差低於約 0.002 就會延後；每個 token 的分數也只剩少數幾種值。
6. **`(L_det, L_end]`**：新 token + 選中的重用 token。
7. **`L_end` 之後**：只有新 token。最後一列一定是新 token（prompt 最後一個位置），logits 照常。

每一段之間依「還在跑的列」重排 hidden states 與 attention metadata；`L_det` 和 `L_end` 各需要一次 host 同步
（取得各請求的列數），`L_start > 0` 時再一次。`relay@0,1,<最後一層>` 加 `RELAY_TAU_DEV=0` 等於整段重算
（4090 smoke：與 `cdc@1.0 TOPK_LAYER=0` 的回覆 9/9 相同）。

### 與官方程式不同的地方

- **relay chunk 的範圍**：官方是整個上游回覆；這裡是一段連續的重用內容，候選取「這一段內」的前 256 名
  （官方取整個回覆的前 256 名，再套到該 chunk）。
- **沒有 hidden state 的 token**：官方以請求為單位（整個請求沒有就全部從第 0 層算）；這裡逐 token 判斷。
- **overlap 排程多跑的一步**：請求結束後 overlap 排程還會多 decode 一步（結果丟掉），那一步的注意力也會加進
  被注意程度（官方 HF 流程沒有這一步）。4090 實測（Qwen2.5-7B，3 個 384 token 的回覆）：總量多約 0.24%，
  逐 token 與 HF 重算的誤差中位數不變（2–3%）。關掉 overlap（`--disable-overlap-schedule`）就沒有這一步。
- **request pool 位置被重用**：overlap 下一個請求的 pool 位置可能在它最後那一步還在跑時就分給新請求；kernel 只在
  `req_to_token[rpi, seq_len-1]` 等於這一步寫的 slot 時才累加，換了主人的列跳過。
- **保留了官方的行為**：沒有任何 token 超過偏差門檻時，連預選的也全部不重算；分數 NaN/inf/負值當 0；
  門檻是嚴格大於；`TAU <= 1e-6` 全選；一組分數總和 < 1e-8 全不選。
- `prefix_tokens`（官方預設 0）沒有移植。

### profile 模式（`relay@profile`）

每個重用 token 每一層都重算（結果等於整段重算），並對每個請求記錄每一層：`s_tok`（論文式 1-2：逐 token、各 head
cosine 的平均）、`s_code`（官方 profiler 實際算的：在 `[head, token, dim]` 上沿 token 軸取 cosine 再平均，和 rho 一樣以 KV dtype 計算）、
相鄰兩層的 Spearman rho（逐 token 的 `1 - cos`，heads × dim 攤平）。分成「所有重用 token」與「由 decode 算出的」兩組。
`scripts/subcontext_sim/relay_layers.py` 對各請求取平均後套官方規則（預設照官方 profiler CLI：`tau_low` 0.99、
λ 0.5（論文表 4 寫 2.0）、`C` 2、`N_min` 3、`T` 5），印出 `ARM=relay@S,D,E`。這支腳本的挑層結果在 3000 組隨機輸入上
與官方 `layer_selection.py` 完全一致。

### 量測

profile 檔在每次 server 啟動時清空，只含這次的請求。`sglang_server.sh` 的檔名標籤會帶上改過的門檻
（例如 `relay_L3-4-22_dev1.2`）。

`sub-context index:` 那行之後多一行 `RelayCaching: N requests, R reused rows, W loaded with a stored hidden state,
S (x%) run past L_det, D deferred selections`。`SGLANG_SUBCTX_TRACE=1` 時每次 prefill 印一行 `[TRACE-RELAY]`
（段數、預選數、選中數、實際挑選的層、`L_det` 的平均偏差，最後一項以 fp32 計算）。
要 logprob 的請求照常 scan（重用段一定在 logprob 起點之前結束，logprob 的位置都是新算的 token）；最後一段依剩下的列重新設定每個請求的 logprob 起點，跟 selective recompute 的做法相同。

直接核對（2026-10-27，4090，Qwen2.5-7B，見 `progress/2026-10-27_relay_smoke.md`）：decode 存下的 hidden state 與
sglang 自己 prefill 同一段 context 在 `L_start` 的輸入相比，cos 平均 0.99998、最低 0.99995；被注意程度與 HF 依
注意力機率重算的值相關 0.990–0.997。

## 9. cache_unfinished_req 與 cache_finished_req

兩者都走**上游的整條插入**，sub-context 只在插入之後多做事。

**cache_unfinished_req**（每個 prefill chunk 之後）：

1. `insert(fill_ids, req_to_token 那一列)`；`[cache_protected_len, prefix_len)` 是自己新算、但樹上已有別人那份的格子，還掉。
2. 重新比對一次、把 `req_to_token` 改指樹真正持有的格子、換 `last_node` 的鎖（上游原本的步驟）。
3. 鏈式請求：`_register_chain(final=False)` 登記結束在這次新存部分的區塊；放掉 scan 對來源節點的鎖
   （副本已經寫進鏈，來源可以被淘汰了）。

**cache_finished_req**：

1. 先做兩道冪等的保險：`release_sub_context_match_locks`、`release_sub_context_rotated_slots`
   （給沒跑過 cache_unfinished_req 的 request；對其他 request 是空操作）。
2. 上游的整條插入（含生成的 token）、還掉重複的格子、放掉 `last_node` 的鎖。
3. 鏈式請求：`_register_chain(final=True)`，用 insert 回傳的最後一個節點當起點，不再整條重新比對。

---

## 10. free 與 write-back 的規則

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
double-free 偵測因此在 `free()` 入口自己記每一格是否在 free list 裡，不看 `available_size`。

### 兩種 write-back

| | 做什麼 | 動到 GPU 嗎 |
|---|---|---|
| **改對照表** | 把「我的第 N 個 token」從自己算的格子改指到樹的格子 | 不動 |
| **搬 KV** | 真的去改格子裡的 bytes（旋轉、或 selective recompute 的複製與寫入） | 動 |

### 搬 KV 發生的地方

| 何時 | 目的地 | 為什麼 |
|---|---|---|
| scan（排程） | **一律複製到新格子**，位置不同時旋轉 | 來源是別條鏈的，而鏈必須擁有自己的格子 |
| selective recompute 的寫回 | 自己的副本 | 被重算的 token 不能改寫別條鏈的格子 |

規則：**讀路徑一律複製到自己的新格子，不就地改任何樹上的格子。**

---

## 11. 量測、診斷與防線

### 量測（不影響行為）

| 工具 | 開關 | 內容 |
|---|---|---|
| forward trace | `SGLANG_FORWARD_TRACE=<檔>` | 每個 forward pass 一列 JSON：`mode`、`bs`、`new_tokens`、`cached_tokens`（只算增量）、`matched_tokens`、`discarded_tokens`、`moved_tokens`、`rotated_tokens`、`sub_reqs`、`gpu_ms`。以 CUDA event 計時，不同步 |
| host 階段計時 | `SGLANG_STAGE_TRACE=<前綴>` | 每個 process 寫 `<前綴>.<proc>.json`：`tpl_render`、`subctx_split`（http）；`match`、`subctx_scan`（內含 `scan_prefix`、`scan_index`、`scan_locate`、`scan_copy`）、`subctx_rotate`、`subctx_copy`、`cache_unfinished`（內含 `register_unfinished`）、`cache_finished`（內含 `register_finished`）（scheduler）。client 建立 `<前綴>.mark` 後另開一份 `measured` 累計，排除 warm-up |
| 請求錄製 | `SGLANG_CAPTURE_REQUESTS=<檔>` | 每個 chat 請求原文一行，給 `subcontext_bench.py replay` |
| `/server_info` | — | `sub_context` 物件：所有開關的實際值與 `pid`，給遠端 client 確認 arm（`scripts/subcontext_sim/check_remote_arm.py`）；還回報 `split_mode` 的是 2026-10-14 以前的 build，該腳本會拒絕 |
| 索引統計 | cdc 時 | prefill log 那行後面附 `sub-context index: …`：比對過的 prompt token 數、其中從頭比對到的、scan 找到的、旋轉的，以及放不進一次 prefill 的請求數與它們丟掉的掃描命中 token 數、沒空間被拒的請求數。每個請求只在第一次真的執行（`prepare_for_extend`）時算一次，被拒後重新比對、被 retract 後重跑的都不算；4090 smoke 中與 client 端 `prompt_tokens`、`cached_tokens` 加總完全一致。`chunks`、`queries`、`tokens scanned`（開頭三個）仍是每次掃描都算；`tokens scanned` 只算實際取指紋的部分（從「前綴長度 + 最短 chunk − 最長 chunk」起） |

### 防線與偵錯

| 防線 | 抓什麼 | 需要 `SGLANG_SUBCTX_AUDIT` |
|---|---|---|
| `available + evictable == max - protected` | 記帳破了（最終症狀） | 不用 |
| `audit_pool_invariant` | 破了之後說明是哪一種原因 | 不用 |
| slot 級 double-free 偵測（`allocator.py`） | 同一格被還兩次的當下，附上呼叫者的檔名與行號 | **要** |
| `SUBCTX-CHAIN-MISMATCH` | 索引記的位置上不是那段 token（鏈式儲存，讀取時逐 token 比對） | **要** |
| `SGLANG_SUBCTX_TRACE` | 每個階段的 TRACE 行（切塊、scan 的配置） | 不用 |
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
只會在洩漏發生**之後**才知道；有它，同一格被還兩次的當下就會報出呼叫者。代價是每次 free 多一次檢查、
每次讀 chunk 多一次逐 token 比對，所以那一輪的時間不能拿去跟 toggle 的數字比。

---

## 12. 已知的限制與待辦

### 會影響命中率或成本的

- **（舊存法的問題，鏈式儲存已解決）cdc 的短訊息與模型回覆進不了快取。** 2026-10-05 以前 `idx` / `cdc` 每塊一個
  namespace：短於 `MIN_CHUNK` 的塊不寫入、每輪重算；generation prompt（3 個 token）也不寫入，連帶存回覆的條件永遠不成立，
  回覆不進快取。Terminal-Bench 上 cdc 命中率因此低於 `off`（2026-10-02 估計每請求多算約 600 token）。
  改成鏈式儲存後兩者都隨整條鏈存進樹；4090 smoke（Qwen2.5-7B，初步）中只往後接的流量 cdc 每筆命中與回覆都和 `off` 相同。
- **短於 `MIN_CHUNK` 的區塊不登記索引**：在鏈上、從頭比對拿得到，但位置變了 scan 找不回來。
- **位置相同也要複製**：別條鏈上、位置剛好相同的段，舊存法是直接共用格子，現在要複製一份（鏈必須擁有自己的格子）。
  GPU 記憶體用量跟 `off` 一樣，不再有「同一段內容只存一份」的節省。複製在沒開 speculative decoding 時是逐層的 Python 迴圈。
- **等待中的請求每輪重新 scan、重新複製**：沒被收進的請求會放掉副本（第 6-4 節），下一輪再做一次。
- **`sub_context_index_declined`**：一個請求因為沒空間被標記後只用前綴比對，直到被 retract 才清掉。
  「沒有其他請求在跑」只看 running batch 與 chunked request；overlap 下還在跑、沒併進 running batch 的
  prefill-only batch 看不到，可能多判一次「沒空間」（只少了重用，不會卡住）。
- **probe 不受一次 prefill 的預算限制，可能 OOM**：放不放得進一次 prefill 只算要算的 token；開 selective recompute 時
  probe 要跑精確前綴之後的所有列，`PrefillAdder` 讓放不下的 probe 單獨跑。4090（chunked 2048）上 27.9k 列的 probe 在
  qwen2 MLP 要 970 MiB 時 OOM、server 崩潰；約 6k 列的 RAG probe 沒事，H200 尚未踩到。2026-10-07 試過「probe 列數超過
  `--chunked-prefill-size` 就退回前綴」：不再 OOM，但 4090 上 SWE 重播的掃描重用少 90%、RAG 全部退回，已撤回。根本解法是 probe 分批跑。
- **（已解決，2026-10-05）probe 跑整段 prompt。** 以前同一批裡只要有一個 layout，所有請求都把整段 prompt 送進 probe，
  而 `PrefillAdder` 只對有 layout 的請求計入額外列。現在 probe 從各自的起點開始（第 8 節），同批的前綴命中只跑它要算的列，
  adder 的計算也跟實際一致。改了 probe 的形狀會讓矩陣乘法的捨入不同（同樣的列、不同的總列數，bf16 結果不一定位元相同），
  所以和改之前的 cdc@r 回覆不會逐字相同。

### 隔離與正確性

- **sparse prefill 路徑沒有傳 attention sink**：有 sink 的模型不在啟動檢查的拒絕清單裡。

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

## 13. 現在可用的模型

以下是拿本機 HuggingFace cache 裡的 config 實際建出 rope、跑過 `rope_delta_composable_reason`
的結果（2026-09-10），不是從架構名推論的。這只回答「能不能旋轉」；cdc 還要過第 3-3 節，
selective recompute 還要過第 3-4 節的條件。

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
Sub-context KV rotation cannot be served: partial rotary (64 of 256 dims); the
rotation kernel assumes the whole head rotates
```

通過的話 log 裡會有帶實測數字的這一行：

```
RoPE delta self-test passed for Llama3RotaryEmbedding on 8 (position, delta) samples;
worst was position 30000 delta 1000 at 6.18e-04 relative, 8% of its 7.40e-03 budget
```
