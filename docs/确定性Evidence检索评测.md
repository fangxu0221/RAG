# 确定性 Evidence 级检索评测：第一阶段

本阶段在原 Source 级评测旁增加证据区间指标，不改变 BM25、BGE、RRF、Reranker、生成模块或 Judge。Dev 是开发集，分数用于检查评测与失败案例，不能代表未见测试集效果。

## 1. 数据怎样流动

```text
railway_dev_bm25.json
  → cli.run_benchmark：加载一份 Document 快照供检索与证据定位共同使用
  → build_retriever_from_documents：原420字符 / 80重叠切块，原BM25
  → evaluate_retrieval(documents=同一快照)
      → EvidenceGroundTruth：预先校验全部标注，定位原文区间
      → 每题只调用一次 retriever.search(question, top_k=3)
      → 原 Source 指标 + 新 Evidence 指标，共用相同排序结果
  → JSON报告：逐题指标、证据位置、Chunk区间、输入及代码哈希
```

阅读顺序：[models.py](../domainrag/models.py) 的 Document/Chunk/EvaluationCase → [text.py](../domainrag/text.py) 的 normalize_text → [chunking.py](../domainrag/chunking.py) 的 start/end → [evidence.py](../domainrag/evidence.py) 的 `_compile`、`score` → [evaluation.py](../domainrag/evaluation.py) 的汇总 → [cli.py](../domainrag/cli.py) 的独立配置入口。

## 2. 定位、覆盖与证据逻辑

全文和 quote 都复用 `normalize_text`：NFKC、连续空白合为一个空格、去首尾空白。用 `source + normalized_quote` 精确搜索，得到左闭右开的 Python 字符区间 `[start,end)`。不是 UTF-8 字节偏移，也不是 `units.jsonl` 的原始 Markdown 偏移。

每个检索 Chunk 必须满足：来源在该快照中，坐标在规范化全文范围内，且 `全文[start:end].strip() == chunk.text`。不满足就报错，防止原始与规范化坐标混用。同一来源的已检索区间按位置排序，重叠或首尾相邻时合并；某个引文的**整个区间**被覆盖才算命中。仅相交或来源相同不算命中；不同来源的区间不能拼接。

标注沿用 `metadata.evidence`，默认每条去重后的引文是一组必要证据，组间 AND。相同 `source + normalized_quote` 不重复计分；具有相同替代项集合的组也去重。部分重叠但不相同的组定义会报歧义，不能通过复用同一证据增加必要事实数。

需要表达同一事实的不同引文时，显式使用同一个 `group_id`，组内 OR。例如下面需要“任一可用的材料要求证据”以及“时限证据”同时满足：

```json
{"evidence": [
  {"source": "a.md", "quote": "材料要求原文甲", "group_id": "materials"},
  {"source": "b.md", "quote": "材料要求原文乙", "group_id": "materials"},
  {"source": "a.md", "quote": "办理时限原文", "group_id": "deadline"}
]}
```

示例用于解释字段，不能直接当成真实标注；实际运行要求每段引文能定位。

相同引文出现多次时，默认 `match_policy=unique`，报错要求审核。确认每个位置都是同一事实的有效依据后，才能标为 `match_policy=any_occurrence`：这些位置 OR 匹配，只算一组。Dev-003 的学年/四次引文在学生 FAQ 第2问和第11问均出现，已显式添加这一标记；`unit_id=q011` 仍表示标注溯源，不限制只能命中第11问。另一条“未用次数不能结转”仍必须另外命中。Dev-007、008各自保留三组必要证据。

本次题目、答案、来源、引文及 unit_id 均未改写，仅增加上述定位策略；对比 Git 原数据的所有其他 JSON 值完全相同。验证记录保留原题集 SHA256 和原始校验时间。没有根据检索结果补充替代条款或调整标签。

## 3. 指标定义与无效输入

每题必要证据组数为 G，Top-K 完整覆盖的组数为 C：

| 指标 | 逐题定义 |
| --- | --- |
| `evidence_hit@k` | C > 0 |
| `evidence_recall@k` | C / G；每组权重相同，不按字符数或位置数计分 |
| `evidence_mrr` | 在排名1…K的结果前缀中，首次完整覆盖任一组的前缀长度 r 的倒数；无命中为0 |
| `sufficient_evidence@k` | C = G，全部必要组均已覆盖 |

跨 Chunk 的证据直到合并后完整覆盖才命中。例如第1个 Chunk 只有引文前半段，第2个补齐后半段，则首次命中排名为2，MRR为1/2。相邻位置指原文区间相邻，不要求 Chunk 在检索排名中相邻。

可回答题宏平均上述指标；本轮分母8。不可回答题的所有 Evidence 指标均为 null，独立统计数量、是否返回结果，不视为证据漏召回，也不据此声称拒答成功。原 Source 指标的排除口径保持一致。

可回答题缺少引文、来源不存在、引文无法定位、未审核的多位置引文、来源集合不一致、AND/OR含糊时，在任何检索前抛出包含题号的 ValueError；返回 Chunk 的坐标或文本损坏也报错，不输出一份伪成功报告。不可回答题必须来源和证据为空且说明原因。旧数据不启用 Evidence 时无需这些新标注，旧调用 `evaluate_retrieval(retriever,cases,top_k)` 的报告格式保持不变。

Ground Truth 目前是选定的13段充分证据，未穷尽其他段落的语义等价表述。评分器不会用词相似度或模型推断替代证据。它支持显式 OR 标签，但标签充分性与语义等价仍需人工审核；`sufficient_evidence` 表示满足标注要求，不是生成答案正确性的证明。

## 4. 复现与验证

独立配置：[configs/railway_dev_bm25.json](../configs/railway_dev_bm25.json)。检索参数逐项沿用 `bm25_baseline.json`：chunk_size=420、overlap=80、top_k=3、k1=1.5、b=0.75；只切换铁路语料、Dev题集并启用 `evidence_evaluation`。四个基础配置保留原样。

```powershell
.\.venv-web\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -v
.\.venv-web\Scripts\python.exe -B -X utf8 -m domainrag.cli benchmark --config configs/railway_dev_bm25.json --output results/railway_dev_evidence_bm25_reproduce.json
```

先执行原62项测试，通过后新增19项回归测试；运行全部81项通过后，再执行10题BM25。完整测试另在禁用 socket 连接的进程中复验通过，BM25评测也禁用连接，没有下载模型或请求生成/Judge API。

新增测试覆盖：Source正确但Chunk错误、单引文部分覆盖、多必要证据、重复标签及Chunk、显式OR及重复组、重复原文的歧义/替代、跨Chunk覆盖及完成排名、重叠与间隙、跨来源隔离、找不到证据时提前报错、NFKC/空白与真实切块、损坏坐标、含糊分组、不可回答排除与全不可回答集、旧输出兼容、CLI接入和哈希、真实Dev标注及参数完整性。

以 Git HEAD 的原 `evaluation.py` 和当前版本在同一 BM25/旧40题上比较，**整份报告相等**：Source Hit@3=1、Recall@3=0.9875、MRR=0.9125。不是只比较三项平均值。

## 5. 本轮 BM25：Source 与 Evidence 对比

本机首次结果：[results/railway_dev_evidence_bm25_v1.json](../results/railway_dev_evidence_bm25_v1.json)。`results/` 是忽略的运行产物目录，克隆仓库后可用上面命令重建；本文保留已观察到的结果。报告记录原始文档文本哈希、配置/题集哈希、代码指纹、每段规范化证据区间和排名，便于核对与重现。

| 指标 | Source | Evidence |
| --- | ---: | ---: |
| Hit@3 | 0.875（7/8） | 0.750（6/8） |
| Recall@3（宏平均） | 0.875 | 0.6875 |
| MRR（截断于Top-3） | 0.750 | 0.5625 |
| 完整证据率@3 | — | 0.625（5/8） |

逐题表中 H=Hit@3、R=Recall@3、RR=倒数排名、S=完整证据@3；题号均以 `rail-dev-` 为前缀。每道可回答题的标注来源均只有一个文件，因此本轮 Source H 与 R 数值相同。

| 题号/业务 | 标注来源 | Source H / R / RR | Evidence H / R / RR / S |
| --- | --- | --- | --- |
| 001 身份重复 | faq/online_booking.md | 1 / 1 / 1 | 1 / 1 / 0.5 / 1 |
| 002 代办退票 | faq/window_refunds.md | 1 / 1 / 0.5 | 1 / 1 / 0.5 / 1 |
| 003 学年次数结转 | faq/student_tickets.md | 1 / 1 / 0.5 | 1 / 0.5 / 0.5 / 0 |
| 004 退票费整点边界 | notices/ticket_purchase_notice.md | 1 / 1 / 1 | 1 / 1 / 1 / 1 |
| 005 晚点退票 | regulations/passenger_transport_rules.md | 1 / 1 / 1 | 1 / 1 / 1 / 1 |
| 006 越站补票 | notices/travel_notice.md | 1 / 1 / 1 | 1 / 1 / 1 / 1 |
| 007 儿童优惠及申报 | faq/children_tickets.md | 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| 008 开车后改签与退票 | faq/ticket_changes.md | 1 / 1 / 1 | 0 / 0 / 0 / 0 |
| 009 白酒限额缺失 | 空 | 不适用 | 不适用 |
| 010 微信退款小时承诺缺失 | 空 | 不适用 | 不适用 |

**Source命中、Evidence完全未命中：Dev-008。** 第1名 `faq/ticket_changes.md#0000` 区间[0,420)，第3名同文件 `#0005` 区间[1700,2120)。三个必要引文分别在[473,508)、[611,639)、[911,960)，均不在返回区间内。第2名来自其他文件。Source按文件名计1，Evidence为0，不能据文件名推断已取得“免费改当日车”和“开车后改签不能退票”的条款。

**部分必要证据：Dev-003。** 第2名学生FAQ `#0001` 区间[340,760)，覆盖“学年/四次”的替代位置[544,595)，因此该事实算命中，即使 `unit_id` 指向第11问的另一个位置[2539,2590)。但“未用次数不能结转”在[1165,1210)，未被覆盖，Recall=1/2、完整证据为0。

**排名差异：Dev-001。** 第1名购票FAQ区间[0,420)只有标注引文[365,449)的一部分；第2名[340,760)才完整覆盖。Source RR=1，Evidence RR=1/2。

Dev-007没有召回标注儿童FAQ，Source与Evidence均为0。两道不可回答题均返回3个候选，单独统计为2题，不纳入均值；未执行生成或拒答评测。未根据这些结果改题、换证据、调参，也未执行 Git commit/push。
