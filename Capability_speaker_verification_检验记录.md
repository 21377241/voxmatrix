# Capability：speaker_verification（说话人确认）

负责人：甲（T3）  
日期：2026-09-03  
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-local`）  
Smoke Job：**12975**  
产物：`VoxMatrix/smoke_sv_all/results/`

> **协议**  
> 双音频 → 是否同一说话人；正式指标 **EER / AUC**（`mesh-speaker-verification` + `MeshAgg`）。  
> 辅看 `speaker_verification_acc`（默认阈值 0.5）。

---

## 状态总览（2026-09-03）

| # | 问题 | 状态 | 说明 |
| --- | --- | --- | --- |
| — | smoke GPU 推理 | ✅ Job **12975** | `failure_rate=0`，双音频 + JSON 解析正常 |
| — | 协议主分 | ✅ | **EER（主）+ AUC**；acc 仅对照 |
| 甲-SV-01 | Omni 非专用声纹模型，分数呈两极 | 📝 记录 | 链路可信；10 条 EER 噪声大，不当正式榜 |

### Smoke 汇总（voxceleb1 / 10 trials，同人 5 + 异人 5）

| 指标 | 数值 | 角色 |
|------|-----:|------|
| **EER** | **0.40**（40%） | **推荐主分**（n=10 仅 smoke） |
| **AUC** | **0.66** | 辅主分 |
| `speaker_verification_acc`（thr=0.5） | **60%**（6/10） | diagnostic |
| `failure_rate` | 0% | 链路 |

---

## 统一 template

- **任务形态**：两段单说话人音频（enroll / test）→ 判断是否同一说话人；输出 JSON。
- **Prompt 骨架**（`mesh-speaker-verification`）：两段 `audio` +  
  *Determine whether both recordings contain the same speaker. Return JSON only as `{"same_speaker": true, "score": 0.0}`, where score is confidence from 0 to 1 that they are the same speaker.*
- **选择题是否改为问答**：**否**——官方即为 verification trial（同人/异人对），不是 MCQ。
- **多音频 / 多轮约定**：输入为 **双音频**（`WavPath` + `WavPath2`）；单轮。

---

## 主指标

### 推荐主指标

| 指标 | 角色 | 原因 |
|------|------|------|
| **EER** | **推荐主分** | 说话人确认标准指标；阈值扫描使 FAR≈FRR |
| **AUC** | 辅主分 | 阈值无关排序质量；与 EER 同源 score |
| `speaker_verification_acc` | diagnostic | 固定阈值 0.5 的 0/1 准确率 |

### 算分说明（实现口径）

代码：`mesh_eval/evaluator/speaker.py`（`SpeakerVerificationEvaluator`）+ `mesh_eval/agg/mesh.py`（`_write_verification_metrics`）。

#### 0. 每条样本先抽出 score / label

1. 模型输出 JSON，例如 `{"same_speaker": true, "score": 0.95}`
2. 取值优先级：`score` → `similarity` → `same_speaker` → 原文
3. 得到：
   - `verification_score`：浮点相似度 / 同人置信度（越高越倾向同人）
   - `verification_label`：金标，1=同人，0=异人  
4. **有 `score` 时，后续判定以 score 为准**，不以 `same_speaker` 布尔字段为准

#### 1. `speaker_verification_acc`（样本级 → 语料平均）

- 固定阈值 **0.5**：`predicted = (score ≥ 0.5)`
- 与金标 same/diff 比较：对 → 1，错 → 0
- 语料级 = 各条平均（本轮 6/10 = **60%**）
- **仅 diagnostic**，不是正式主分

#### 2. EER（Equal Error Rate，语料级，**推荐主分**）

先按金标拆成两组分数：

- positives：同人（label=1）的 score 列表  
- negatives：异人（label=0）的 score 列表  

对每个候选阈值 \(t\)（取本批出现过的全部 score，外加 \(+\infty / -\infty\)）：

\[
\mathrm{FAR}(t)=\frac{\#\{\text{异人且 }score \ge t\}}{\#\text{异人}},\quad
\mathrm{FRR}(t)=\frac{\#\{\text{同人且 }score < t\}}{\#\text{同人}}
\]

选使 \(|\mathrm{FAR}-\mathrm{FRR}|\) 最小的 \(t\)，再取：

\[
\mathrm{EER}=\frac{\mathrm{FAR}+\mathrm{FRR}}{2}
\]

- **越小越好**（0=完美）
- 本轮 smoke：**0.40**（n=10，噪声大，仅验证链路）

#### 3. AUC（语料级，辅主分）

Mann–Whitney 式：对每个同人分 \(p\)、每个异人分 \(n\)：

- \(p > n\) → 1；\(p = n\) → 0.5；\(p < n\) → 0  

\[
\mathrm{AUC}=\frac{\text{总得分}}{\#\text{同人}\times\#\text{异人}}
\]

含义：随机抽一对（同人、异人），同人分数更高的概率。  
- **越大越好**（1 完美，0.5 随机）
- 本轮 smoke：**0.66**

#### 4. 三者对照

| 指标 | 依赖阈值？ | 含义 | 本轮 |
|------|:----------:|------|-----:|
| acc@0.5 | 固定 0.5 | 砍一刀后的分类对错率 | 60% |
| **EER** | 扫描最优 | FAR≈FRR 时的平均错误率 | **0.40** |
| **AUC** | 否 | 同人分整体是否高于异人分 | **0.66** |

inventory 另列 `min_dcf`，框架未实现 → **本轮不评 min_dcf**。

### 数据流

```
WavPath + WavPath2 + Prompt → Qwen3-Omni → JSON {same_speaker, score}
  → verification_score / verification_label
  ├─ thr=0.5 → speaker_verification_acc（样本级）
  └─ MeshAgg → overall/eer, overall/auc（语料级）
```

### Benchmark 算分

| Benchmark | 实现 | smoke | 备注 |
|-----------|------|------:|------|
| voxceleb1 (`veri_test`) | `mesh-speaker-verification` + `mesh` | EER **0.40** / AUC **0.66** | 官方 trial；同人/异人各半 |

---

## VoxCeleb1 核验（`veri_test`）

- **官方协议**：`veri_test.txt`，`label enroll.wav test.wav`（1=同人，0=异人）
- **数据根**：`/mnt/afs/oss_data/datasets/03_multi_speaker/VoxCeleb_1_2_multi_speaker_multi_speaker/juliuscn/voxceleb/vox1/wav/`（**miss=0**）
- **全集**：37720 trials（同人/异人各 18860）
- **抽样**：10 条，同人 5 + 异人 5，跨 speaker stride

| # | sample_id | gold | score | acc@0.5 | 路径（enroll / test） |
|---|-----------|:----:|------:|:-------:|----------------------|
| 0 | `voxceleb1_veri_00000_same` | same | 0.95 | ✓ | `id10270/x6uYqmx31kE/00001.wav` / `id10270/8jEAjG6SegY/00008.wav` |
| 1 | `voxceleb1_veri_00001_diff` | diff | 0.10 | ✓ | `id10270/x6uYqmx31kE/00001.wav` / `id10300/ize_eiCFEg0/00003.wav` |
| 2 | `voxceleb1_veri_07544_same` | same | 0.10 | ✗ | `id10278/Pp-rAswo4Xg/00036.wav` / `id10278/DSGE2F5sTcg/00005.wav` |
| 3 | `voxceleb1_veri_07545_diff` | diff | 0.95 | ✗ | `id10278/Pp-rAswo4Xg/00036.wav` / `id10300/2cAiIYHfon0/00003.wav` |
| 4 | `voxceleb1_veri_15088_same` | same | 0.10 | ✗ | `id10286/mmFmN0OPhKs/00005.wav` / `id10286/lJQTN_wK9vA/00001.wav` |
| 5 | `voxceleb1_veri_15089_diff` | diff | 0.00 | ✓ | `id10286/mmFmN0OPhKs/00005.wav` / `id10309/XMLMvfrgdzY/00011.wav` |
| 6 | `voxceleb1_veri_22632_same` | same | 0.95 | ✓ | `id10294/WJPpL58RruA/00007.wav` / `id10294/WJPpL58RruA/00009.wav` |
| 7 | `voxceleb1_veri_22633_diff` | diff | 0.10 | ✓ | `id10294/WJPpL58RruA/00007.wav` / `id10283/QitLqlca660/00004.wav` |
| 8 | `voxceleb1_veri_30176_same` | same | 0.10 | ✗ | `id10302/ekqJava7sdE/00007.wav` / `id10302/ekqJava7sdE/00001.wav` |
| 9 | `voxceleb1_veri_30177_diff` | diff | 0.10 | ✓ | `id10302/ekqJava7sdE/00007.wav` / `id10297/23tlx1v1UCA/00001.wav` |

完整绝对路径见 `smoke_sv_all/voxceleb1/manifest.jsonl`。

- 当前指标：EER **0.40** / AUC **0.66** / acc **60%**
- 人核结论：**pass（链路）**——双音频进模、JSON 可解析、EER/AUC 能出；**模型能力弱于专用 ASV**（同人漏判 3、异人误判 1），score 多落在 0.0/0.1/0.95 两极
- 问题编号：甲-SV-01（能力边界，非数据病）

---

## 问题列表

### 甲-SV-01. Omni 通用模型做 verification：校准差、EER 仅 smoke 参考

| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-09-03 |
| 场景 | `smoke_sv_all` / Job 12975 |
| 状态 | 📝 记录；**不挡链路收口** |

**样例**

- `voxceleb1_veri_07544_same`：gold=same，pred `score=0.1` / same_speaker=false  
- `voxceleb1_veri_07545_diff`：gold=diff，pred `score=0.95` / same_speaker=true  

**原因**

Qwen3-Omni 不是声纹/embedding 验证器；输出置信度离散，10 条上 EER=0.40 噪声大，不能当正式榜。

**解决方案**

- 评测链路保持现协议（双音频 + score → EER/AUC）  
- 正式全量跑完再报 EER；smoke 只验证链路  
- 若要对齐 SOTA ASV，需另接 embedding 后端（本轮统一 Omni，不换模）  
- 方案是否确定：**确定（本轮用 Omni + EER/AUC）**

---

## 附录：smoke 配置

| 集 | Manifest | Suite | Prompt | Evaluator | Agg | Job | 结果目录 |
|----|----------|-------|--------|-----------|-----|-----|----------|
| voxceleb1 | `voxceleb1/` | `suite.yaml` | `mesh-speaker-verification` | `mesh-speaker-verification` | `mesh` | **12975** | `results/` |

- 构建：`python smoke_sv_all/build_manifests.py`
- 运行：`bash smoke_sv_all/run_suite.sh`
- Registry：`smoke_sv_all/registry/`
- 索引：`benchmark_subset_manifest.jsonl` → `voxceleb1` / `veri_test` / `speaker_verification`
