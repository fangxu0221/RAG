# RailPolicy-RAG V1：四组检索对照

2026-11-09。仅检索层开发实验；不改检索算法、Evidence评分器、Judge、Agent或Ground Truth，不按结果改题、换引文或调参。

## 实验控制与实际模型

共用 `data/railway/` 的9份Markdown、`eval_dev.jsonl` 的10题：8道可回答参与Source/Evidence宏平均，2道不可回答单列。共91个Chunk，长度420字符、重叠80、Top-K=3；均启用 `evidence_evaluation=true`。

| 组 | 配置 | 检索与模型参数 |
| --- | --- | --- |
| A | `railway_dev_bm25.json` | BM25，k1=1.5、b=0.75；保留已有结果并复现 |
| B | `railway_dev_embedding.json` | `BAAI/bge-small-zh-v1.5`，batch=32；查询指令“为这个句子生成表示以用于检索相关文章：” |
| C | `railway_dev_hybrid_rrf.json` | A+B，RRF每路candidate_k=10，rrf_k=60 |
| D | `railway_dev_hybrid_reranker.json` | C后接 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`；candidate_k=10、batch=4、max_length=512，ONNX文件 `onnx/model_quint8_avx2.onnx` |

B/C/D三个独立配置只替换原基础配置的资料路径、题集路径并启用Evidence；参数逐项比较必须完全相同，否则脚本拒绝运行。A沿用已有铁路Dev配置。模型保持ID加载，通过本地缓存将main解析到以下不可变版本：

- BGE：`7999e1d3359715c523056ef9478215996d62a620`。
- Cross-Encoder：`1427fd652930e4ba29e8149678df786c240d8825`。

准备过程将权重及分词/Pooling配置的SHA256写入[模型清单](experiments/railpolicy_v1/model_manifest.json)，正式实验前后检查文件哈希，预检真实推理通过后才评测。模型文件不会提交到Git。

## 运行环境与复现

独立环境 `.venv-railpolicy-v1/`：Windows x86-64、Python 3.12.14、CPU。实际安装 sentence-transformers 5.7.0、transformers 4.54.1、torch 2.15.0+cpu、onnxruntime 1.31.1、optimum 2.1.0、optimum-onnx 0.1.0。B/C/D使用同一环境；OMP/MKL线程环境变量均为4。具体依赖版本以[锁文件](experiments/railpolicy_v1/requirements.lock.txt)为准，不声称兼容任意未来版本。

CPU PyTorch来自官方CPU源，其余依赖通过清华PyPI镜像完成下载，模型来自Hugging Face官方源。Windows首次并行下载遇到无符号链接权限的缓存探测竞态，改为串行下载后完成。安装和模型准备允许联网。初轮严格离线运行完成A/B/C，但ONNX加载库强制访问仓库文件目录；第二轮又因未缓存SentencePiece词表而阻断。补齐原版本词表后，完整运行使用 `--allow-hub-metadata`：只放行官方Hugging Face模型元数据GET及HEAD，其余socket连接均禁止，权重仍来自本地固定缓存。会话登记AzureExecutionProvider/CPUExecutionProvider，没有GPU后端。前后哈希校验一致。run_v3记录286次元数据请求（含预检、装配和额外候选诊断），没有生成或Judge API调用。

```powershell
uv --cache-dir results/railpolicy_v1/uv-cache venv --python 3.12 .venv-railpolicy-v1
uv --cache-dir results/railpolicy_v1/uv-cache pip install --python .venv-railpolicy-v1/Scripts/python.exe torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
uv --cache-dir results/railpolicy_v1/uv-cache pip install --python .venv-railpolicy-v1/Scripts/python.exe -r docs/experiments/railpolicy_v1/requirements.lock.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
.\.venv-railpolicy-v1\Scripts\python.exe -B -X utf8 scripts/run_railpolicy_v1.py --prepare
.\.venv-railpolicy-v1\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -v
.\.venv-railpolicy-v1\Scripts\python.exe -B -X utf8 scripts/run_railpolicy_v1.py --output results/railpolicy_v1/reproduce_01 --allow-hub-metadata
```

`--prepare`优先使用已公开模型清单中的revision；正式运行拒绝覆盖已有输出目录。原始报告由现有 `cli.run_benchmark` 生成。额外Top-10诊断必须与正式Top-3前缀一致；D另外保存重排前Top-10。诊断不改变正式评测的K，也不用于选择参数。

## 指标与分析

完整结果取本机 `results/railpolicy_v1/run_v3/`。四组均完成；A的整份JSON与原BM25首轮报告一致；A/B/C也与前两轮的对应报告一致。D的重排前Top-10逐项等于C的Top-10，四组的语料、题集、代码指纹及证据逻辑相同。

| 组别 | Source Hit@3 | Source Recall@3 | Source MRR | Evidence Hit@3 | Evidence Recall@3 | Evidence MRR | Sufficient Evidence@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A BM25 | 0.8750 | 0.8750 | 0.7500 | 0.7500 | 0.6875 | 0.5625 | 0.6250 |
| B BGE | 0.7500 | 0.7500 | 0.6250 | 0.6250 | 0.5625 | 0.4375 | 0.5000 |
| C RRF | 0.8750 | 0.8750 | 0.7292 | 0.7500 | 0.6875 | 0.5417 | 0.6250 |
| D RRF+重排 | 0.6250 | 0.6250 | 0.4792 | 0.6250 | 0.5208 | 0.3542 | 0.3750 |

### 逐题变化

单元格为 **Evidence Recall@3 / Evidence RR / 完整证据**。全部Source逐题指标、Top-3 Chunk及字符区间见[可追踪精简摘要](experiments/railpolicy_v1/summary.json)；六个典型重排前后Top-10见[候选诊断](experiments/railpolicy_v1/reranker_cases.json)。题号省略 `rail-dev-` 前缀。以下均指当前标注的充分证据，不自动扩展等价条款。

| 题号/业务 | A | B | C | D |
| --- | --- | --- | --- | --- |
| 001 身份重复 | 1 / 0.5 / 是 | 1 / 0.5 / 是 | 1 / 0.5 / 是 | 1 / 0.5 / 是 |
| 002 窗口代办退票 | 1 / 0.5 / 是 | 0 / 0 / 否 | 0 / 0 / 否 | 0 / 0 / 否 |
| 003 学年次数结转 | 0.5 / 0.5 / 否 | 0.5 / 1 / 否 | 0.5 / 1 / 否 | 0.5 / 0.5 / 否 |
| 004 退票费整点 | 1 / 1 / 是 | 1 / 0.5 / 是 | 1 / 1 / 是 | 0 / 0 / 否 |
| 005 晚点退票 | 1 / 1 / 是 | 1 / 0.5 / 是 | 1 / 1 / 是 | 1 / 1 / 是 |
| 006 越站补票 | 1 / 1 / 是 | 0 / 0 / 否 | 1 / 0.3333 / 是 | 0 / 0 / 否 |
| 007 儿童票/申报 | 0 / 0 / 否 | 1 / 1 / 是 | 1 / 0.5 / 是 | 1 / 0.3333 / 是 |
| 008 开车后改签/退票 | 0 / 0 / 否 | 0 / 0 / 否 | 0 / 0 / 否 | 0.6667 / 0.5 / 否 |
| 009 白酒限额缺失 | 不适用 | 不适用 | 不适用 | 不适用 |
| 010 微信退款小时承诺缺失 | 不适用 | 不适用 | 不适用 | 不适用 |

### BGE是否改善语义改写题

该子集仅005、006两题。A/B/C/D的Evidence Hit、Recall和完整证据率分别为1/0.5/1/0.5，Evidence MRR分别为1/0.25/0.6667/0.5。**没有观察到BGE改善这两道语义改写题。**

005的正确规程Chunk `regulations/passenger_transport_rules.md#0024` [8160,8580) 覆盖证据[8399,8444)。A排第1、B排第2、C/D排第1，B只改变排序且退后。

006的须知证据[902,985)位于 `notices/travel_notice.md#0002` [680,1100)。A排第1，B的Top-3未返回此来源，C排第3。B首位规程 `#0017` [5780,6200) 只有越站条款后半句，另外两个候选涉及其他处理规则，未收齐“原到站前提出”的完整条件。D将须知从融合第3降到第7，首位反而是规程 `#0021` [7140,7560) 的拒不补票/拒绝运输等内容；不能将相近业务术语等同于完整条件。

### RRF是否优于单路

相较B，C补回006，改善004、005的正确证据排名，Evidence指标均有所提高；相较A，C没有提高覆盖均值，MRR略低。局部互补不意味着必然整体提高。

002的正确窗口FAQ `faq/window_refunds.md#0001` [340,760) 覆盖[559,637)，在A第2、C第4、D第4。RRF第1的 `faq/online_booking.md#0003` 和第2的 `faq/ticket_changes.md#0002` 获得双路贡献，而正确窗口证据仅有一路贡献，未进Top-3。反过来，007被BGE找出，C保留完整证据。因此C与A的完整证据数都是5题，却包含不同的题。

### 三道多条件题与重排的收益/损失

- **003：四组始终只满足1/2组必要证据。** “学年/四次”有效位置为学生FAQ[544,595)或[2539,2590)，“不能结转”为[1165,1210)。A以第2名 `#0001` [340,760) 命中前者；B/C以第1名 `#0003` [1020,1440) 命中后者。D把 `#0001` 从融合第4提到第2，却把 `#0003` 从第1降到第5，仍只有半套。第一次命中更早不能证明完整回答依据增加。
- **007：A=0，B/C/D均为3/3。** 三组证据[232,288)、[361,386)、[301,360)都在儿童FAQ `#0000` [0,420)；其排名B=1、C=2、D=3。D并未改善该标注证据的排名。但D第1名规程 `#0008` [2720,3140) 也逐项包含“未满6岁不占席、一成人限一名、单独座位购优惠票、提前申明”的等价条款：这是**尚未标注的有效替代位置**。本轮只记录，不修改标签；当前Evidence RR只按FAQ计，不能据此断言首位内容无用。
- **008：A/B/C=0，D=2/3，仍不完整。** 必要引文在改签FAQ[473,508)、[611,639)、[911,960)。`#0001` [340,760)覆盖前两条，由融合第6升到D第2，是本轮重排的明确局部收益；覆盖第三条“开车后改签不能退票”的 `#0002` [680,1100) 却从融合第5降到D第8。因此问题来自选入Top-3的排序，完整证据其实已在同一Top-10候选池内，不能归因于没有召回候选。

其他损失：004的标注购票须知 `#0005` [1700,2120) 覆盖证据[1889,1998)，从C第1降到D第5。D第1的改签FAQ `#0005` 是**改签费**规则，容易与退票费业务混淆。不过D第3的窗口退票FAQ `#0002` 也含退票费5%/10%/20%的相关规则，属于另一处未标注材料，故这里的0严格表示未命中选定引文，而非人工判定所有返回内容都错误。总体D的完整证据题数由C的5题降为3题，Evidence MRR由0.5417降为0.3542；当前固定重排模型在当前标签下没有整体收益。

### Source命中但Evidence未命中

A/B/C均只有008属于该类：A命中改签FAQ `#0000` [0,420)、`#0005` [1700,2120)；B命中 `#0003` [1020,1440)、`#0000`；C命中 `#0000`。这些区间均未覆盖上述三条引文。D没有完全零Evidence的Source命中案例，因为008已覆盖2/3；但D同时丢失004、006的标注Source命中，不能把这一现象单独解释为重排整体更可信。

001仍体现排名差异：四组都先返回购票FAQ `#0000` [0,420)，引文[365,449)仅部分覆盖；第2名 `#0001` [340,760)才完整覆盖，因此Source RR=1而Evidence RR=0.5。

### 验证与产物

- 原轻量环境及独立实验环境均通过81项回归测试；在禁用socket的独立进程复验，日志本地保存为 `results/railpolicy_v1/regression.log`。
- 四组均10题、9文档、91块、Top-3；逐题字段兼容、宏平均复算、共同语料/题集/代码指纹、配置参数控制全部通过。不可回答题各返回3块、各Evidence指标为null，不据此评拒答。
- D重排前池与C的Top-10逐项一致；额外Top-10的前3名与各组正式Top-3一致。模型文件在运行前后哈希不变。A与历史首轮完整JSON相等；A/B/C的离线报告与放行元数据后的报告相等。
- 本地原始 `A.json`—`D.json`、四份候选诊断、环境/模型清单、元数据请求日志及 `comparison.json` 位于 `results/railpolicy_v1/run_v3/`。未完成D的早期run_v1/run_v2记录保留，没有覆盖成成功记录。
- `results/`仍被Git忽略；可公开复现的精简JSON、六题重排诊断、精确依赖锁及模型revision/文件哈希已另存至[docs/experiments/railpolicy_v1/](experiments/railpolicy_v1/summary.json)。摘要不包含本机绝对路径或模型权重。没有执行Git commit/push。

## 适用范围

只有8道可回答Dev题，单题Hit或完整证据率变化就会使均值变化0.125；语义改写子集只有2题。结果用于开发分析，不用于最终泛化性能结论或显著性断言。没有调参搜索、多个随机种子或未见Test集。

确定性Ground Truth匹配选定的13段引文，支持显式替代关系，尚未穷尽所有语义等价条款；“命中充分标注证据”不等于生成答案一定正确。两道不可回答题不纳入检索均值，返回候选也不表示正确拒答。计时混入模型加载、缓存和文件写入，本轮不作为延迟/吞吐量对照。
