# Sub-context KV 重用：機制說明

這份文件講的是 `state1` 分支上的 sub-context 切分與 RoPE delta 旋轉機制：每個階段做什麼、
會遇到什麼情境、以及為什麼是這樣寫的。

主要涉及的檔案：

| 檔案 | 負責 |
|---|---|
| `python/sglang/srt/managers/schedule_batch.py` | 讀路徑：`_stitch_sub_contexts`、交接、旋轉副本的生命週期 |
| `python/sglang/srt/mem_cache/radix_cache.py` | 寫路徑：逐 namespace 插入、finish 的搶救與釋放 |
| `python/sglang/srt/mem_cache/rotate_kv.py` | Triton 旋轉 kernel、啟動時的性質自測 |
| `python/sglang/srt/utils/subctx_config.py` | 三個開關、支援條件的判斷 |

---

## 0. 這套機制在解決什麼

一般的 radix cache 把整個 prompt 當成**一條** key。agent 的 prompt 長這樣：

```
[system prompt][tools 定義][對話歷史]
```

只要 system prompt 差一個字，整條 key 就從頭不同，後面的 tools 和歷史全部重算——
即使它們一模一樣。

**sub-context 的做法**：把 prompt 切成幾塊，每塊放進自己的 namespace（用 `extra_key` 區分），
各自獨立查詢。tools 那塊不會因為 system prompt 變了就失效。

**代價**：RoPE 把絕對位置烤進了 K。一塊 KV 是在位置 100 算的，就只能在位置 100 被重用。
所以每個 tree node 要記 `canonical_position`——「我這份 KV 是在哪個絕對位置算出來的」。

**旋轉**：RoPE 的角度對位置是線性的，所以 `R(a)·R(b) = R(a+b)`，可以把一塊 KV 旋轉 `Δ`
搬到新位置：

```
R(p_new) · RMSNorm(k_raw) == R(p_new − p) · k_cached[loc]
```

這在位置上**完全精確**——RMSNorm 在旋轉之前，而 RoPE 保範數，所以不需要重跑 k_norm。

但它修的是**位置**，修不了**前文**。那塊當初是在別的上下文下算出來的，這是品質代價，
也就是 pass@1 要量的東西。

---

## 1. 四份狀態

| | 內容 | 在哪 |
|---|---|---|
| `k_buffer` / `v_buffer` | 真正的 KV bytes，`[slot, head, dim]` | GPU |
| `req_to_token[req, 位置]` | **位置 → slot 的權威對照表** | GPU tensor |
| tree node | `key`（token）、`value`（slot）、`extra_key`（namespace）、`canonical_position` | CPU |
| `free_pages` | 哪些 slot 是空的 | CPU |

**GPU 上的 bytes 不帶身分。** 一個 slot 是誰的，完全由「誰指向它」決定。
這句話是後面所有規則的來源。

### 三個開關

| 環境變數 | 作用 |
|---|---|
| `SGLANG_DISABLE_SUBCONTEXT=1` | 完全不切塊，走原本的 radix cache（`off` 臂） |
| `SGLANG_SUBCONTEXT_ROTATE=1` | Stage 1：stitch 的旋轉 + finish 的兩種就地旋轉 |
| `SGLANG_SUBCONTEXT_ROTATE_ACROSS=1` | Stage 2：chunk 邊界夾擠 + append（隱含開啟旋轉） |
| `SGLANG_SUBCTX_AUDIT=1` | 打開 TREE-DUP 與 slot 級 double-free 偵測 |

### request 身上的欄位

| 欄位 | 意思 | 誰寫 |
|---|---|---|
| `sub_context_match_nodes/_indices/_positions` | 排程時查到的 node、slot、canonical | stitch |
| `sub_context_owned_lens[i]` | 第 i 塊已經被算進 prefix 或已插入的長度 | stitch → 每個 chunk 更新 |
| `sub_context_tree_owned[i]` | 第 i 塊**現在是不是樹的** | 只有 chunk 那趟寫 |
| `sub_context_tree_canonical[i]` | 樹把第 i 塊登記在哪個位置 | 同上 |
| `sub_context_rotated_slots` | 自己 alloc 的旋轉副本，**還沒交給 `req_to_token`** | stitch / append |
| `cache_protected_len` | 從 0 起連續被樹擁有的長度（上游欄位的有損投影） | 每階段重算 |

---

## 2. 啟動：能不能跑

### 切分能不能服務（`unsupported_reason`）

| 條件 | 不成立的話 |
|---|---|
| 是原生 `RadixCache`（不是子類、不是 C++ 版、不是 hierarchical） | 逐 namespace 插入路徑不存在 |
| `page_size == 1` | 分頁會把 slot 綁成一組，塊邊界對不齊 |
| 不是 EAGLE | 它把 key 改寫成 bigram |
| 沒有 `--disable-radix-cache` | 沒有樹 |

### 旋轉能不能做（`rotation_unsupported_reason`）

| 檢查 | 為什麼 |
|---|---|
| 不是 mrope | 位置是 3-vector（text/height/width），沒有單一 delta |
| 不是 linear scaling | cos_sin_cache 是多份串接（一個 LoRA scaling factor 一份），列號不等於位置 |
| neox 配對、`rotary_dim == head_dim` | kernel 的假設 |
| MHA pool、`store_dtype == dtype` | 排除 MLA 和量化 KV |
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

**情境**：明確開了旋轉但條件不成立 → **raise，不啟動**。因為不支援的 RoPE 不會失敗，
它會用錯的律去轉，然後安靜地讓每個重用的塊都變差。

---

## 3. 生命週期總覽

```
1. 切塊                    chat request → 幾個 block，各帶一個 extra_key
2. stitch（排程）           逐 namespace 查詢，拼出可重用的前綴
3. prepare_for_extend      prefix_indices → req_to_token，責任交接
4. prefill                 算 KV，並產生第一個 token
5. cache_unfinished_req    逐塊插入樹                    ← 每個 chunk 跑一次
   └─ append               Stage 2
6. decode                  一個一個吐 token
7. cache_finished_req      搶救 → 存回覆 → 還鑰匙
```

其中第 5 步**不是每個 request 都會跑**（見第 8 節）。

---

## 4. stitch（排程）

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
| 位置不對，旋轉不可用 | 不拿 | 開關關著／delta 超出 cos_sin_cache／pool 滿 |
| 部分命中（`hit < 塊長`） | 拿 `hit` 個 | `contiguous = False` |
| `take` 被上限砍 | 拿 `min(hit, len(fill_ids)-1 - total)` | 至少留一個新 token 可算 |

### `contiguous` 和 `displaced` 不是同一件事

這是最容易混淆的地方：

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

前面完全沒有斷層，但 tools 還是位移了。在 agent 場景這是常態。

### 為什麼旋轉只在 contiguous 成立時做

兩個理由：

1. **接不上去**。前綴必須是從位置 0 開始的連續一段。前面若只部分命中，中間空著，
   後面的塊沒有東西可以貼。
2. **delta 會是錯的**。`delta = offset - canonical` 只有在前面每塊都整塊拿走時才對——
   那時 prompt 的版面和拼出來的版面才一致，這塊才真的落在 `offset`。

「前面沒能整塊命中」那種情形是 **Stage 2** 在解的（見第 6 節）。

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
```

**為什麼 `tree_owned` 要清掉**：retraction（被踢出 batch、KV 被 free、重新排程）之後，
上一輪的 `True` 會讓 finish 跳過那塊 → 這個 request 自己的格子沒人還 → 洩漏。

---

## 5. prepare_for_extend：責任交接

```python
write_cache_indices()                    # prefix_indices → req_to_token
req.sub_context_rotated_slots = None     # 責任移轉
req.kv_committed_len = seq_len
```

`sub_context_rotated_slots` 是一份**釋放責任清單**——`release_sub_context_rotated_slots`
會把清單上的每一批格子直接 free 掉。

| 情境 | 結果 |
|---|---|
| 在此之前 abort | 清單還在 → `release_sub_context_rotated_slots` 救援 |
| 在此之後 | `req_to_token` 涵蓋它們 → finish 的逐塊迴圈會還 |
| **忘了清空** | 兩邊都還 → **double free** |
| **太早清空** | 兩邊都不還 → **洩漏** |

所以這是唯一正確的交接點：只有這一刻 `req_to_token` 和記帳同時涵蓋那些格子。
而且**每一輪 chunk 都會重新交接一次**（Stage 2 的 append 會再產生新的旋轉副本）。

---

## 6. cache_unfinished_req（每個 chunk）

把剛算好的 KV 交給樹保管。KV 現在只有這個 request 的 `req_to_token` 指著，
request 一結束就沒人記得它了。

### 逐塊六步

**① 這塊這次算到了嗎**

`covered_end = min(offset + 塊長, end_k)`，沒到就跳過。

**② 樹上有嗎、位置對嗎**

```python
existing = matched_canonical_position(probe.last_device_node, probe_hit)
if existing is not None and existing != offset:
    continue           # 跳過這塊，不是停止
```

`continue` 不是 `break`——namespace 是各自獨立的樹。停下來會讓越長越大的 `messages` 塊
永遠進不了快取。代價是中間會有洞（見下）。

**③ 插入**

送進去的是**整塊已覆蓋的部分**，但樹只會真正存下它本來沒有的那一段——
insert 沿著已有路徑往下走，走不下去才新增節點。

```python
result.prefix_len      # 這串 token 裡，有多長是樹本來就有的
```

**④ 還掉多餘的**

```python
if result.prefix_len > owned:
    free(kv_indices[offset+owned : offset+result.prefix_len])
```

`[owned, prefix_len)` = 我這次新算的，但樹上已經有別人算好的同一份。

**⑤ 回寫 `req_to_token`**

把這塊每個位置指到樹真正持有的格子。**必須在 ④ 之後**——`kv_indices` 是 `req_to_token`
的 view，先回寫的話 ④ 會 free 到樹的格子。

**⑥ 記帳 + 鎖住**

```python
tree_owned[i] = True;  tree_canonical[i] = offset;  owned_lens[i] = covered_end - offset
inc_lock_ref(seg_match.last_device_node)
```

### 四個情境

**情境一：第一次見到這塊。** 樹上空的。insert 存下整塊，`prefix_len = 0`，
沒東西要還，回寫等於指回自己的格子。

**情境二：我 stitch 時拿過，樹上就是我拿的那份。** `prefix_len = 150`、`owned = 150`
→ 不還。回寫指回同一批。**這一輪實質上什麼都沒變**，只是重新鎖住。

**情境三：別人搶先。** 排程時樹上空的 → 我自己算了 150 個。但另一個 request 同時
也在算同一段，而且先插入了。現在 `prefix_len = 150`、`owned = 0` → **還掉我自己那份**
（樹只留先寫的），並**真的改指** `req_to_token`。不回寫的話，我還在 decode，
會讀到剛丟回籃子的格子。

**情境四：位置不對。** 跳過整塊。這塊成了洞，留到 finish 搶救（第 7 節 7-1）。

### 「洞」與「所有權」

跑完之後可能長這樣：

```
A  0–99     插進樹了     → 樹的
B  100–249  被跳過       → 不是樹的        ← 洞
C  250–399  插進樹了     → 樹的
```

**所有權**就是 `tree_owned = [True, False, True]`。

**洞**就是 B：樹擁有 0–99 和 250–399，中間 100–249 空著。

**為什麼不能用一個數字表示**：`cache_protected_len` 原本是一個數字，
意思「位置 0 到 N 都是樹的，不准 free」，只能描述從頭連續的一段。
`[True, False, True]` 壓不進一個數字——硬壓成 100 的話，finish 時 C 的 250–399
會被當成「不是樹的」而還回籃子，但那是樹的格子。

所以改成逐塊的布林陣列，`cache_protected_len` 退化成給上游程式碼看的有損投影。

### Stage 2：append（要開 `ROTATE_ACROSS`）

處理「前面那塊只能部分重用，但後面那塊整塊都在快取裡」的情形。

stitch 時算出 `boundary`（第一個「在已拼前綴之後、完整命中、但位置不對」的塊的 offset），
`PrefillAdder` 把 chunk 夾到那裡。於是第一趟剛好算到那塊的起點。

在 `_cache_unfinished_sub_contexts` 逐塊處理完之後：

```python
if offset != cursor:  break                 # 必須貼齊覆蓋末端
delta = offset - canonical
dst = _rotate_sub_context_block(indices[:take], delta)    # alloc + 旋轉
req_to_token.write((req_pool_idx, slice(offset, offset+take)), dst)
cursor += take
```

**怎麼接回計算**：`covered = end_k + appended`，`prefix_indices = req_to_token[:covered]`。
下一輪 scheduler 呼叫 `chunked_req.init_next_round_input()` **不帶 tree_cache**
→ 不重新 match、不重新 stitch → 直接沿用這份變長的 prefix
→ `extend_input_len` 只剩下沒被覆蓋的部分。

```
不做 append：第一趟算 0–99，第二趟算 100–249（150 個）
做 append：  第一趟算 0–99，第二趟算 249（1 個）
```

代價是多一趟 scheduler round-trip。所以塊夠大才划算——**最小 block 門檻（約 300 token）
還沒做**。

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
- **`prefix_indices` 從 `req_to_token` 重建**：中間可能有第②步跳過的洞，接片段描述不了。

---

## 7. cache_finished_req

先兩道冪等的保險（給沒跑過第 6 節的 request 用），然後：

```python
if self.serves_sub_contexts(req):        # = has_sub_contexts and supports_sub_contexts()
```

這一行的歷史見第 8 節。

### 7-1 搶救：`_reverse_rotate_insert_sub_contexts`

對每個 `tree_owned[i] == False` 的塊，算 `delta = canonical - offset`，**就地**轉回去再插入。

注意方向跟 stitch 相反：stitch 是 `offset - canonical`（把樹的那份搬到我要的位置），
這裡是 `canonical - offset`（把我的那份搬回樹要的位置）。

| 情境 | 做什麼 |
|---|---|
| `canonical is None`（當初擋住我的那份已被淘汰，namespace 空了） | 當成 `offset`，delta = 0，直接插入 |
| `delta == 0` | 直接插入，不轉 |
| `delta != 0`，整塊都是自己的 | 就地旋轉 → 插入 → `tree_owned[i] = True` |
| `delta != 0`，**混著樹的格子** | **整塊放棄** |
| delta 超出 cos_sin_cache 範圍 | 放棄 |

**為什麼混著樹的格子就要放棄**：就地旋轉會改到別的 request 正在 match 的 KV，
而那個 node 還在對外宣告舊的 canonical position——跨 request 的無聲汙染，
這個機制能產生的最糟錯誤。整塊複製一份再轉當然可以，但比這次搶救本身還貴。

**為什麼在 finish 而不是每個 chunk**：`cache_unfinished_req` 是在 prefill 之後、
request **還在 decode 的時候**呼叫的，它自己的 attention 每一步都在讀這些 slot。
就地搬位置會安靜地毀掉還在進行中的生成。已經 finish 的 request 再也不會讀它們。

搶救成功會重算 `cache_protected_len`——**這是 7-2 的前提**。

### 7-2 存回覆：`_cache_sub_context_output`

把「最後一塊的 token ++ 生成的 token」插進最後一塊的 namespace。
下一輪對話這段回覆就在 prompt 裡，不用重算。

條件：`cache_protected_len == prompt_len` 且 `owned_lens[-1] == len(last_seg)`
且有生成內容。

**情境**：最後一塊的 `tree_canonical != offset` 時，生成的 tail 也要**就地轉同樣的 delta**
——它是在 `prompt_len` 算的，但要接在樹擺在 `canonical` 的塊後面。

### 7-3 還鑰匙

```python
tree_owned = req.sub_context_tree_owned
if tree_owned is None and req.sub_context_extra_keys:
    tree_owned = [False] * len(...)          # 沒跑過第 6 節

for 每一塊:
    if tree_owned[i]:  continue              # 樹的，一格都不動
    block = kv_indices[offset:end]
    match = self.match_prefix(RadixKey(token_ids[offset:end], seg_key))
    _free_only_ours(block, _tree_held_mask(match.device_indices, block))

if not kept and prompt_len < len(kv_indices):
    free(kv_indices[prompt_len:])            # 沒被收下的生成 tail
```

| 情境 | 結果 |
|---|---|
| 塊是樹的 | 一格都不還，只 `dec_lock_ref` |
| 塊被拒絕，全是自己的 | 整塊還 |
| 塊被拒絕，**混著樹的格子** | **逐格比對，只還自己的** |
| 從沒跑過第 6 節 | 每塊都當「被拒絕」處理，逐格比對 |
| 回覆沒被收下 | 連同 tail 一起還 |

---

## 8. free 與 write-back 的規則

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
| **旋轉 KV** | 真的去改格子裡的 bytes | 動 |

### 旋轉發生的四個地方

| 何時 | 目的地 | 為什麼 |
|---|---|---|
| stitch（排程） | **複製到新格子** | 來源是樹的，別人鎖著 |
| append（每個 chunk 結尾） | **複製到新格子** | 同上 |
| finish：搶救被拒絕的塊 | **就地** | 格子是自己的，而且已經不會再讀 |
| finish：轉生成的回覆 | **就地** | 同上 |

規則：**來源是樹的 → 複製；來源是自己的 → 就地。**

就地旋轉在資料競爭上是安全的（kernel 每個 program 先讀自己那列再寫自己那列，
沒有跨列相依，有專門的測試 `test_in_place_rotation_matches_the_copying_one`），
但在**所有權**上不安全，所以有 `tree_held.any()` 的守衛。

### 三條貫穿全部的規則

1. **擁有權逐塊記**，不用一個長度表示（因為 `continue` 會製造洞）。
2. **free 之前逐格比對 slot 身分現算一次**（`_tree_held_mask`），不相信任何記住的狀態
   ——中間可能發生 insert、split、evict。
3. **旋轉的目的地**：讀路徑一律複製到新格子；寫路徑可以就地，
   但必須先確認整塊都是自己的。

### 為什麼要逐格比對

`_tree_held_mask` 逐一位置比對 slot 身分，而不是問「這塊是不是我的」。兩個理由：

- **一個被拒絕的 block 裡可能混著 node 自己的 slot。** 最清楚的例子就是
  「從沒跑過第 6 節」的 request：它的 `req_to_token` 裡是 stitch 拼好的、
  指向各 namespace node 的格子，但它在樹裡一格都不擁有。
- **重疊不是前綴。** 這裡原本只數「開頭連續相符的長度」，假設塊是從前往後被重用的。
  job 432 打破它：重複的位置從 block offset 的**下一格**開始——第一格不同、
  後面約 2000 格全同——於是「連續相符長度」算出 0，整段被還回去，而樹還在服務它。

---

## 9. 上次的 bug（已修，`ed27c8a` + `26b2bf2`）

### 正常的 request 走六步

```
1. stitch                 拼出 prefix（含樹的格子）
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

## 10. 防線與偵錯

| 防線 | 抓什麼 | 需要 `SGLANG_SUBCTX_AUDIT` |
|---|---|---|
| `available + evictable == max - protected` | 記帳破了（最終症狀） | 不用 |
| `audit_pool_invariant` | 破了之後說明是哪一種原因 | 不用 |
| `_audit_tree_duplicates` | **一個 slot 兩個 owner 的當下** | **要** |
| slot 級 double-free 偵測 | 同一格被還兩次的當下 | **要** |
| `SUBCTX-OUTPUT-UNDERMATCH` | 「今天不可能發生」的顯式警報 | 不用 |

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

## 11. 已知的保守之處與待辦

### `_cache_sub_context_output` 的 gate 過嚴（已決定不處理）

現在要求 `cache_protected_len == prompt_len`（整條 prompt 不准有洞）。
但要存的東西是「最後一塊 ++ 生成的 token」，插進**最後一塊的 namespace**，
前面的塊在別的樹裡。

安全性真正需要的是**最後一塊確實被樹擁有**：

```python
or not (req.sub_context_tree_owned and req.sub_context_tree_owned[-1])
or req.sub_context_owned_lens[-1] != len(last_seg)
```

兩個條件都要——`owned_lens[-1]` **不能單獨代表擁有權**，因為 stitch 把旋轉副本
也算進 `owned`，一個位移後被旋轉的最後一塊會讓 `owned_lens[-1] == len(last_seg)`
而 `tree_owned[-1] == False`。若只拿掉 `cache_protected_len` 那一行，
樹會收下旋轉副本、而後面的 free 迴圈又把它們還掉——**真的 bug**。

**決定不做。** 這個放寬只在「前面的塊被凍住、最後一塊正常」時有用，而目前的
AgentVerse workload 是反過來的（query 在 system prompt 裡、`messages_key` 被
first-writer-wins 凍住），卡住的正是最後一塊——放寬與否都過不了 gate。
記在這裡是為了讓下一個看到這個條件的人知道它已經被評估過，不用再推導一次。

### Stage 2 的最小 block 門檻（未做）

append 省下的是重算，付出的是一趟 scheduler round-trip。用 Stage 1 量到的
每 token 旋轉成本回推，塊要大約 300 個 token 才回本。

**這個門檻會改變哪些 block 被重用，所以會改變輸出**——pass@1 必須在門檻定案之後才量。

### 模型通用性（部分完成）

- 已完成：性質自測取代型別白名單（放行 llama3、dynamic、YaRN）
- 已完成：YaRN 家族除 mscale
- 未做：部分旋轉（`rotary_dim < head_dim`）——native reference 已經處理了，
  kernel 還沒
- 未做：fp8 KV（需要 dequant → rotate → requant，會累積量化誤差）
- 未做：MLA pool（DeepSeek 那條線；只有 `k_pe` 帶 RoPE，可能反而更便宜）
- `page_size == 1` 可能是真實 serving 場景下更限制人的條件
