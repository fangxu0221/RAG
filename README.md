# DomainRAG-Benchmark

项目主线是 **BM25 + BGE → RRF → Cross-Encoder Reranker**，可用于固定 RAG、原生 Agent 和 LangGraph Agent V1。

独立仓库：[fangxu0221/RAG](https://github.com/fangxu0221/RAG)。项目文件直接放在仓库根目录。

先读 [主线阅读导航](docs/代码分层与阅读导航.md)，需要数据流与问答要点时看 [项目数据流与面试复习](docs/项目数据流与面试复习.md)。

## 当前目录

```text
DomainRAG-Benchmark/
├── domainrag/
│   ├── domain/          # 数据契约、协议、工具 Schema
│   ├── application/     # 知识工具、请求会话、答案评测
│   ├── infrastructure/  # retrievers/、llm/、storage/
│   ├── workflows/       # native_agent.py、langgraph_agent.py、langgraph_state.py
│   ├── langgraph_cli.py  # LangGraph Agent V1 独立入口
│   ├── bootstrap.py     # 配置、冻结校验、文档快照及检索器装配
│   └── ...              # CLI、旧路径兼容导出、按原路径校验的冻结源码
├── configs/          # 四个基础配置 + 四组独立铁路Dev评测配置
├── data/llm_notes/   # 暂留的11篇知识文档及配套旧题集
├── data/railway/     # 新采集的9份12306官方资料、原始响应与来源索引
├── scripts/          # 独立采集工具；不参与普通问答运行
├── ui/               # RailPolicy RAG Studio 本地界面、知识库目录和请求适配
├── .streamlit/       # 本机监听及深色主题配置
├── tests/            # 核心回归与 Studio 测试，冻结校验使用临时合成样本
├── docs/             # 主线导航、数据流复习、铁路资料采集说明
├── results/          # 新运行的输出位置；历史内容已归档
├── .venv-web/        # 保留的本地Python环境；主线不加载Web框架
├── pyproject.toml
├── README.md
├── AGENTS.md
└── LICENSE
```

历史结果、审核、研究脚本和专项测试、冻结评测套件及辅助文档已移到项目外的本地归档。原README和完整复习笔记也保存在归档中。归档位置与恢复方法见 [数据流复习文档](docs/项目数据流与面试复习.md#5-归档和恢复)。

CLI 和知识工具使用同一个 [bootstrap](domainrag/bootstrap.py)，顺序为读取配置、校验冻结数据、加载文档快照、切块并创建检索链。Evidence 评测与检索索引使用同一个快照。原生 `run_agent(question, tools, model, trace_dir, limits)` 和图版 `run_langgraph_agent(...)` 接收已装配的知识服务与 `ChatModel`，共用工具执行和结果契约。具体 DeepSeek 传输位于基础设施层。[分层与兼容边界](docs/代码分层与阅读导航.md#1-目录与依赖)

## 环境与运行

所有命令从本项目根目录执行，要求Python 3.10+。BM25基础流程和单元测试不需要重型模型依赖。本机已验证的解释器是 `.venv-web/Scripts/python.exe`（Python 3.12）；下面命令中的 `python` 应使用对应环境的解释器。

首次克隆可创建自己的环境；虚拟环境不包含在仓库中：

```powershell
git clone https://github.com/fangxu0221/RAG.git
cd RAG
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

macOS/Linux激活命令为 `source .venv/bin/activate`。

在 Windows 终端提问铁路规则时，可用本机 Python 环境、铁路 BM25 配置和 DeepSeek。密钥仅设置在当前 PowerShell 进程中。

```powershell
if (-not $env:DEEPSEEK_API_KEY) {
    $railKey = Read-Host "DeepSeek API Key" -AsSecureString
    $env:DEEPSEEK_API_KEY = [System.Net.NetworkCredential]::new('', $railKey).Password
    Remove-Variable railKey
}
$railQuestion = Read-Host "请输入铁路问题"
& .\.venv-web\Scripts\python.exe -B -X utf8 -m domainrag.cli answer --config configs/railway_dev_bm25.json --provider deepseek --question "$railQuestion"
```

每次运行处理一道题，输出回答及检索来源。DeepSeek 会接收问题与检索到的片段。

只检索，不调用生成模型：

```powershell
python -m domainrag.cli query --config configs/bm25_baseline.json --question "为什么注意力分数要除以根号维度？"
```

完整知识检索链需要BGE、Cross-Encoder可选依赖和相应模型权重。按需安装：

```powershell
python -m pip install -e ".[reranker]"
python -m domainrag.cli query --config configs/hybrid_reranker.json --question "为什么注意力分数要除以根号维度？"
```

固定RAG与原生Agent：

```powershell
python -m domainrag.cli answer --config configs/hybrid_reranker.json --question "为什么注意力分数要除以根号维度？" --provider deepseek
python -m domainrag.agent --config configs/hybrid_reranker.json --question "为什么注意力分数要除以根号维度？"
```

后两条命令使用 `DEEPSEEK_API_KEY` 调用真实模型，将问题、提示词和取得的证据发给供应商并产生用量。Agent在预算内可能调用多次。`.venv-web/`仍用于轻量验证；独立的`.venv-railpolicy-v1/`已准备BGE与ONNX重排依赖及权重用于检索实验，没有运行生成或Judge请求。

### LangGraph Agent V1

在 Python 3.10+ 环境按需安装 extra。本机验证使用 Python 3.12 和 LangGraph 1.2.20。

```powershell
python -m pip install -e ".[langgraph]"
python -m domainrag.langgraph_cli --help
```

下面的示例使用 BM25 配置和 `DEEPSEEK_API_KEY`，会调用真实 DeepSeek 并产生用量。

```powershell
python -m domainrag.langgraph_cli --config configs/bm25_baseline.json --question "为什么注意力分数要除以根号维度？"
```

新入口与 `python -m domainrag.agent` 共用参数和默认值。Python 调用方还可通过 `AgentLimits.max_context_chars` 设置上下文字符预算。两种入口均输出 JSON 与本地 JSONL Trace，完成或合法拒答返回退出码 0，预算停止或失败返回 1。图版在同一个请求会话内执行 `model → tools → model`，最终经过引用校验或按停止原因收尾。

离线回归及专项测试使用 ScriptedModel 和本地 BM25：

```powershell
python -B -X utf8 -m unittest discover -s tests -v
python -B -X utf8 -m unittest discover -s tests -p test_langgraph_agent.py -v
```

安装 extra 后执行全部测试。基础环境可运行固定 RAG、原生 Agent 和回归测试，其中需要图运行时的测试会跳过。图版专项测试核对原生与图版的模型入参、工具顺序、引用、终态和 Trace。[状态与执行细节](docs/项目数据流与面试复习.md#8-langgraph-agent-v1)

独立知识工具示例：

```powershell
python -m domainrag.tools --config configs/railway_dev_bm25.json search_knowledge --query "学生票优惠次数可以结转吗？"
```

架构重构验证为基线 86 项、bootstrap 阶段 91 项、分层后 94 项，均通过。已验证旧 CLI、真实 BM25 和四组 V2 冻结报告重放。BGE、重排和远程模型调用使用离线替身验证，实际模型推理留作独立验证。

### RailPolicy RAG Studio V1

在仓库根目录，用 Python 3.10+ 环境运行。下面的依赖包含本地界面和 LangGraph 模式。

```powershell
python -m pip install -e ".[ui,langgraph]"
python -m streamlit run ui/app.py
```

若使用本机已有的 `.venv-studio-ui`，可直接运行 `& .\.venv-studio-ui\Scripts\python.exe -m streamlit run ui/app.py`。浏览器打开 [http://127.0.0.1:8501](http://127.0.0.1:8501)。

- **知识库浏览**按常见问题、使用须知和相关规程展示当前检索快照。铁路快照有 9 份资料、197 个知识单元。选一份资料后可搜索条目，规程可按章和节定位，原文可以折叠。点击「用此问题测试」会填入问答草稿。索引缺失时可展开快照中的文档原文。知识单元用于阅读，Chunk 用于检索和引用。
- **问答测试**可切换纯 RAG、原生 Agent 和 LangGraph V1，三种模式默认用「通俗简洁」风格。回答先给结论，再说明影响结论的条件，保留数字、适用范围、重要例外和 `[1]` 式引用。引用证据、检索证据、轨迹和统计可按需展开。点击「发送」后，DeepSeek 接收问题和流程取得的证据。问答需在启动 Streamlit 的 PowerShell 中设置 `DEEPSEEK_API_KEY`，可复用上面的安全输入命令。没有密钥时仍可浏览知识库及运行本地检索。
- **检索调试**可切换会话中的全局检索策略，独立返回排名、来源、Chunk ID、原始分数和完整片段。BM25、余弦相似度、RRF 和 CrossEncoder 分数各有自己的尺度。
- **检索设置**可选 BM25、BGE Embedding、BM25 + BGE + RRF、RRF + CrossEncoder Reranker。初始策略为重排，初始 Top-K 为 3。参数从 `configs/railway_dev_*.json` 读取，修改在当前会话中生效。纯 RAG 和检索调试使用界面 Top-K；Agent 工具调用可传入自己的 Top-K，省略时使用既有默认值 3。执行过程会显示请求值和实际返回数量。

只安装 UI 和 LangGraph 时，可在「检索调试」或「检索设置」切换到 BM25。要运行 BGE 或 RRF，安装 `.[embedding]` 并准备模型权重；要运行默认重排策略，安装 `.[reranker]`，它也包含 BGE 所需依赖。

```powershell
python -m pip install -e ".[ui,langgraph,reranker]"
```

首次加载 BGE 或 CrossEncoder 需要可用权重。已有 Hugging Face 缓存时，可在启动前用 `HF_HOME` 指定其根目录。本机铁路实验缓存位于 `results/railpolicy_v1/hf-home`。在完全离线时，当前 ONNX 后端使用远程模型 ID 仍可能尝试读取仓库文件清单。在高级选项的「重排模型」中填写本地快照目录，即包含 `config.json` 和 `onnx/` 的目录，并保留配置中的量化模型文件。

Studio 优先从本机 Hugging Face 缓存加载 BGE，文件缺失时按原流程获取。同一进程内，BGE、RRF 和重排在文档切块及 BGE 配置一致时共享编码器与文档向量。「查看运行统计」分别显示索引准备、检索和模型调用耗时，「检索调试」也区分索引准备与查询耗时。本机用缓存权重、拦截 HTTP 请求实测，首次准备 RRF 索引约 16 秒，从 RRF 切到 BGE 后准备索引约 10 毫秒，单次 BGE 检索约 11 毫秒。重启后的首次加载仍需导入模型依赖、加载权重并编码文档。

各问答请求的 JSONL 轨迹保存在 `results/studio_traces/`。文件含问题、工具参数与证据，供本机复盘。界面只读取结构化的运行事件。新请求使用提交时的文档快照与配置，缓存按文档内容和索引参数复用已装配的检索链。

验证环境为 Python 3.12、Streamlit 1.67.0、LangGraph 1.2.20。全量 126 项测试通过，其中 10 项覆盖 Studio 的目录、交互、缓存、离线 BM25 和问答适配。这 10 项也在 Streamlit 1.37.0 通过。桌面浏览器走通了条目带入问答页及 BM25 检索。检索调试页用真实语料执行了四种策略的冒烟验证，BGE 与 ONNX 重排使用本地权重。三种问答使用模型替身验证，真实 DeepSeek 请求待验证。

「通俗简洁」更新后，全量 128 项测试通过。用「只有学生证能否实名购票」和「网上退票怎么收费」走通三种模式的实际请求组装、本地检索及引用映射。回答使用离线替身，真实模型的表达效果待验证。CLI 的默认提示词与更新前一致，RailPolicy V2-auto 四组冻结排名重放通过。

```powershell
python -B -X utf8 -m unittest discover -s tests -p test_studio.py -v
python -B -X utf8 -m unittest discover -s tests -v
```

## 后续替换知识库

第一版12306铁路旅客资料已采集到 `data/railway/`：六个FAQ专题（62组问答）、两份须知（20项）、一份国铁集团客运规程（115条）。原始HTML/正文脚本与Markdown同时保留，来源和版本见 `sources.json`。四个基础配置仍使用旧学习资料，铁路 `eval.jsonl` 暂空。内容、校验和接入边界见 [铁路旅客知识库：第一版采集说明](docs/铁路旅客知识库_第一版.md)。

新增独立Dev题集 [data/railway/eval_dev.jsonl](data/railway/eval_dev.jsonl)：事实查询、数字/时间规则、语义改写、多条件判断、不可回答各2道，共10道。复用现有 `EvaluationCase` JSONL格式；8道可回答题在 `metadata.evidence` 中标注13段可精确匹配的Markdown原文，2道不可回答题来源及证据为空并说明全库缺失依据。静态校验通过，详见 [验证记录](data/railway/eval_dev.validation.json) 和 [题目清单](docs/铁路旅客知识库_第一版.md#6-dev评测题集)。

该题集绑定现有9份铁路Markdown快照，标为 `split=dev`，不是未见Test集。标注选取充分证据，尚未穷尽其他条款中的等价表述，也未经过独立人工复核。现已接入确定性Evidence级检索评测：默认逐条必要（AND），可用相同 `group_id` 表达替代引文（OR）；重复原文默认报歧义，仅显式 `match_policy=any_occurrence` 时接受任一位置。Dev-003仅新增这一技术标记，题目、答案、来源、quote、unit_id均未改变，旧哈希保存在验证记录中。原 `eval.jsonl`、语料快照和四个基础配置未覆盖。

1. 把新资料放在独立目录，例如 `data/knowledge/`；当前尚未创建或启用该目录。
2. 现有读取器支持UTF-8 `.md` / `.txt`。PDF、Word等需先转换或增加解析能力。
3. 在独立配置中验证新资料，再同步四个基础配置的 `docs_path`。配置文件继续放在 `configs/` 下。
4. 重新制作相应的评测题、参考答案与 `relevant_sources`，再更新 `eval_path`。
5. 检索、回答和测试验证后，再归档旧 `data/llm_notes/`。

当前知识库及其题集暂时保持原样，四个配置仍指向有效文件。项目根 `docs/` 是阅读文档，不会自动进入知识库。旧评测分数均属历史证据，不用于证明新知识库效果。

## 铁路Dev：确定性Evidence检索评测

独立配置 [railway_dev_bm25.json](configs/railway_dev_bm25.json) 复用原BM25参数（420字符、80重叠、Top-3、k1=1.5、b=0.75），开启 `evidence_evaluation`；四个基础配置保持原样。

```powershell
python -B -X utf8 -m domainrag.cli benchmark --config configs/railway_dev_bm25.json --output results/railway_dev_evidence_bm25_reproduce.json
```

[evidence.py](domainrag/evidence.py) 用规范化后的 `source + quote` 定位完整证据区间，合并相邻/重叠Chunk覆盖并去重；[evaluation.py](domainrag/evaluation.py) 保留Source指标，同时新增 `evidence_hit@k`、`evidence_recall@k`、`evidence_mrr`、`sufficient_evidence@k`。缺失或含糊标注报错，不可回答题指标为null并单独统计。旧评测不启用该功能时，调用及报告结构保持兼容。

本轮81项测试通过，含禁用socket连接的复验；随后运行10题BM25（8道可回答参与均值、2道不可回答单列）：Source Hit/Recall@3=0.875、MRR=0.750；Evidence Hit@3=0.750、Recall@3=0.6875、MRR=0.5625、完整证据率@3=0.625。Dev-008命中文件但未覆盖任何标注引文；Dev-003只覆盖两条必要证据之一。旧40题与Git HEAD原评分函数的整份Source报告完全相同。再次复现与首轮报告逐字段一致。

完整协议、10题对照及区间解释见 [确定性Evidence检索评测](docs/确定性Evidence检索评测.md)，本机运行产物在 `results/railway_dev_evidence_bm25_v1.json`（该目录按现有规则忽略）。此次只改评测与技术标注元数据，不调整题意、引文或检索参数，不请求模型/Judge API，也不提交或推送Git。

## RailPolicy-RAG V1四组对照

新增铁路Dev的[BGE](configs/railway_dev_embedding.json)、[RRF](configs/railway_dev_hybrid_rrf.json)、[Reranker](configs/railway_dev_hybrid_reranker.json)配置，复用已有铁路BM25配置；统一420/80/Top-3及原模型、原参数。没有改动算法、评分器、Judge、Agent或Ground Truth。

| 组 | Source Hit/Recall@3 | Source MRR | Evidence Hit@3 | Evidence Recall@3 | Evidence MRR | 完整证据率@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A BM25 | 0.8750 | 0.7500 | 0.7500 | 0.6875 | 0.5625 | 0.6250 |
| B BGE | 0.7500 | 0.6250 | 0.6250 | 0.5625 | 0.4375 | 0.5000 |
| C RRF | 0.8750 | 0.7292 | 0.7500 | 0.6875 | 0.5417 | 0.6250 |
| D RRF+重排 | 0.6250 | 0.4792 | 0.6250 | 0.5208 | 0.3542 | 0.3750 |

81项测试在原环境和独立CPU实验环境通过；A完整复现旧报告。BGE补回007但丢失002、006；RRF覆盖率未超过BM25；重排将008从0提升到2/3，却丢失其他标注证据，整体没有收益。007首位规程包含未标注的等价证据，已记录限制，没有修改标签。仅8道可回答Dev题参与均值，不能作为最终泛化结论。

[实验脚本](scripts/run_railpolicy_v1.py)复用CLI保存正式Top-3与额外Top-10诊断。环境/权重在忽略目录，准备阶段使用官方CPU源、清华PyPI镜像及Hugging Face官方权重；正式运行只放行HF文件元数据GET/HEAD，其他连接禁止。四组原始JSON、候选和环境记录本地保存在`results/railpolicy_v1/run_v3/`；可追踪的[精简摘要](docs/experiments/railpolicy_v1/summary.json)、[模型版本与哈希](docs/experiments/railpolicy_v1/model_manifest.json)、[依赖锁](docs/experiments/railpolicy_v1/requirements.lock.txt)及[典型重排诊断](docs/experiments/railpolicy_v1/reranker_cases.json)另存于docs。

运行命令、10题变化和证据区间分析见[实验文档](docs/RailPolicy-RAG_V1_检索对照实验.md)。无生成/Judge API调用，无参数调优，无Git提交或推送。

## RailPolicy-RAG Ground Truth V2-auto

[独立 V2-auto 题集](data/railway/eval_dev_v2_auto.jsonl)已按 **AI-assisted Silver Ground Truth** 冻结。沿用 10 题、原标准答案及证据组数。全库扫描并校验 26 个候选，经 DeepSeek Flash 和 Pro 分别盲审后采纳 19 条，拒绝 5 条，隔离 2 条分歧项。V1 原题集与报告保留。

[离线重评分](docs/RailPolicy-RAG_V2-auto_离线重评分.md)复用四组历史 Top-3 和原 Evidence 评分器。A/B/C/D 的完整证据率从 0.6250/0.5000/0.6250/0.3750 变为 0.6250/0.7500/0.7500/0.5000，变化来自标签补全。新增 Source Group 指标按必要组计分，同组替代来源满足其一即可。

```powershell
.\.venv-web\Scripts\python.exe -B -X utf8 scripts/railpolicy_v2_auto.py replay
```

完整 86 项测试通过。[审计记录](data/railway/eval_dev_v2_auto.audit.json)、[冻结清单](docs/experiments/railpolicy_v2_auto/manifest.json)和[指标摘要](docs/experiments/railpolicy_v2_auto/summary.json)可查。复现所需的 A–D 原始报告保存在本机 V1 运行目录。

## 维护约定

普通测试直接导入分层后的实现。兼容测试继续验证七个旧模块的公共导出。其中六个可列为未来的弃用候选，待明确撤销旧导入承诺后再处理。`retriever.py` 还被 V2 冻结的 `evaluation.py` 导入，其移除需要单独做版本迁移。

本地清理暂存目录为 `垃圾桶/`，由 Git 忽略。2026-11-10 将 uv 安装缓存及与仓库已跟踪版本逐字节相同的 V2 审计、重放副本移入该目录，并保留原相对路径。需要恢复时，可把目录内的文件移回项目根目录对应位置。Python 环境、模型权重及 V1 原始实验报告仍用于运行和复现。

每次新增、修改、移动、归档或删除项目文件时，同步更新本 README 中受影响的说明；没有使用层变化时，补充简短变更记录。专题文档仍按需同步维护，验证结果只记录实际执行的检查。此约定已写入 `AGENTS.md`。
