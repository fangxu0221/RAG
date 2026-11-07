# DomainRAG-Benchmark

当前只围绕 **BM25 + BGE → RRF → Cross-Encoder Reranker → 固定 RAG / 原生 Agent** 学习和运行。

独立仓库：[fangxu0221/RAG](https://github.com/fangxu0221/RAG)。项目文件直接放在仓库根目录。

先读 [主线阅读导航](docs/代码分层与阅读导航.md)，需要数据流与问答要点时看 [项目数据流与面试复习](docs/项目数据流与面试复习.md)。

## 当前目录

```text
DomainRAG-Benchmark/
├── domainrag/        # 19个核心模块：检索、生成、Agent及其依赖
├── configs/          # BM25、BGE、RRF、Reranker四个基础配置
├── data/llm_notes/   # 暂留的11篇知识文档及配套旧题集
├── data/railway/     # 新采集的9份12306官方资料、原始响应与来源索引
├── scripts/          # 独立采集工具；不参与普通问答运行
├── tests/            # 核心回归测试，冻结校验使用临时合成样本
├── docs/             # 主线导航、数据流复习、铁路资料采集说明
├── results/          # 新运行的输出位置；历史内容已归档
├── .venv-web/        # 保留的本地Python环境；主线不加载Web框架
├── pyproject.toml
├── README.md
├── AGENTS.md
└── LICENSE
```

历史结果、审核、研究脚本和专项测试、冻结评测套件及辅助文档已移到项目外的本地归档。原README和完整复习笔记也保存在归档中。归档位置与恢复方法见 [数据流复习文档](docs/项目数据流与面试复习.md#5-归档和恢复)。

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

后两条命令使用 `DEEPSEEK_API_KEY` 调用真实模型，将问题、提示词和取得的证据发给供应商并产生用量。Agent在预算内可能调用多次。完整链路的可选依赖及权重尚未在当前验证环境准备，本次没有运行这些真实模型请求。

回归测试：

```powershell
python -B -X utf8 -m unittest discover -s tests -v
```

当前62项测试通过，验证控制流、计算、证据与异常边界；另2项旧研究专项测试随脚本归档，没有作为当前套件跳过。

## 后续替换知识库

第一版12306铁路旅客资料已采集到 `data/railway/`：六个FAQ专题（62组问答）、两份须知（20项）、一份国铁集团客运规程（115条）。原始HTML/正文脚本与Markdown同时保留，来源和版本见 `sources.json`。当前配置仍使用旧学习资料，铁路 `eval.jsonl` 暂空。内容、校验和接入边界见 [铁路旅客知识库：第一版采集说明](docs/铁路旅客知识库_第一版.md)。

1. 把新资料放在独立目录，例如 `data/knowledge/`；当前尚未创建或启用该目录。
2. 现有读取器支持UTF-8 `.md` / `.txt`。PDF、Word等需先转换或增加解析能力。
3. 在独立配置中验证新资料，再同步四个基础配置的 `docs_path`。配置文件继续放在 `configs/` 下。
4. 重新制作相应的评测题、参考答案与 `relevant_sources`，再更新 `eval_path`。
5. 检索、回答和测试验证后，再归档旧 `data/llm_notes/`。

当前知识库及其题集暂时保持原样，四个配置仍指向有效文件。项目根 `docs/` 是阅读文档，不会自动进入知识库。旧评测分数均属历史证据，不用于证明新知识库效果。

## 维护约定

每次新增、修改、移动、归档或删除项目文件时，同步更新本 README 中受影响的说明；没有使用层变化时，补充简短变更记录。专题文档仍按需同步维护，验证结果只记录实际执行的检查。此约定已写入 `AGENTS.md`。
