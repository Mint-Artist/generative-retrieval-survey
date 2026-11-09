# 生成式检索（Generative Retrieval）调研

> Generative Retrieval / Generative Information Retrieval — 调研报告、论文清单与核心论文集
> 调研日期：2026-11-07 ｜ 持续整理：是

## 什么是生成式检索

生成式检索（Generative Retrieval, GR）是信息检索的新范式：不再依赖"倒排索引 + 多路召回 + 级联排序"的传统管线，而是用一个（通常是自回归的）序列模型，把**整个语料库/物品库的索引"记忆"进模型参数**，查询时直接**生成文档/物品的标识符**（DocID / Semantic ID），一步完成检索。

两条主要脉络：

1. **生成式文档检索**：GENRE (2021) → DSI / NCI / SEAL (2022) → docid 设计精细化 (2023) → 规模化 + LLM 化 + 工业落地 (2024–2026)
2. **语义 ID / 生成式推荐**：TIGER (2023) → HSTU (2024) → OneRec (2025) → 广告/电商全面生产化 (2026)

## 仓库内容

| 路径 | 内容 |
|---|---|
| [生成式检索调研报告.md](生成式检索调研报告.md) | **主报告**：领域概述、发展时间线、奠基论文、2024–2026 前沿、业界落地全景（Google/YouTube、Meta、快手、百度、阿里、腾讯、京东、美团、字节、Spotify、Snap、Pinterest…）、综述/教程/开源资源、趋势总结、推荐阅读路径 |
| [生成式检索论文清单.xlsx](生成式检索论文清单.xlsx) | **论文清单表格**（5 个工作表，117 条）：总览 / 奠基经典 2020–2023 / 最新进展 2024–2026 / 业界落地 / 综述与资源 |
| [生成式检索论文清单（打印版）.pdf](生成式检索论文清单（打印版）.pdf) | 表格的 11 页横向 PDF 版 |
| [生成式检索论文集/](生成式检索论文集/) | **7 篇核心论文的原文 + 中文全文翻译 + 深度解读**（OneRec、PLUM、GBS、ICEGR、UniGD、TSGR、DOME），详见其目录内 [README](生成式检索论文集/README.md) |
| [scripts/build_xlsx.py](scripts/build_xlsx.py) | 论文清单表格的可复现生成脚本 |
| [_tools/](_tools/) | 论文集 PDF 翻译版构建工具（extract_figs.py + build_pdf.py） |

## 核心结论速览

- **2025–2026 是工业落地元年**：快手 OneRec 承接 25% QPS；天猫生成式召回通道占成交 72.63%；YouTube PLUM 面向数十亿用户；抖音 GEAR、快手 GR4AD（广告收入 +4.2%）、淘宝 TSGR（GMV +1.64%）均已全量部署。
- **技术趋势**：语义 ID 构造从两阶段走向端到端；解码效率成为落地关键（约束解码向量化、并行解码、推测解码）；稀疏+稠密混合架构（LIGER/COBRA）回应"生成 vs 判别"之争；推理增强（OneRec-Think、SIDReasoner）进入检索。
- **文档检索方向**：docid 设计三大流派（层级数字型 / 文本子串型 / 可学习型）；长文档代表 GBS（整本书 22.5 万词，MRR@20 较 RIPOR +9.8%）；动态更新代表 DOME（模型编辑，约 60% 成本接近全量重训效果）。
- **垂类专业库**：法律最活跃（LegalSearchLM、GCSR）；金融财报库基本空白（只有 FinDER 检索基准），是被 RAG 统治的空白地带。

## 常见信息勘误（二手资料高频错误）

| 常见说法 | 事实 |
|---|---|
| SEAL 发表于 ICML 2022 | ❌ NeurIPS 2022 |
| NCI 是 Google 的工作 | ❌ 微软亚洲研究院（NeurIPS 2022 Outstanding Paper） |
| NOVO 为 Zhuang et al., SIGIR 2023 | ❌ Zihan Wang et al., CIKM 2023 |
| RIPOR 发表于 EMNLP 2023 | ❌ WWW 2024 |
| Ultron 发表于 SIGIR 2022 | ❌ arXiv 预印本 |
| GenIR 综述发表于 FnTIR | ❌ ACM TOIS 2025 |

另有若干流传名称（RepBF/RepLA、IGReD、Taurus/HIR/Mouchak、GBench、MAUI 等）**未能在公开渠道找到对应论文**，清单中已标注"待确认"。

## 数据可靠性与版权说明

- 论文元数据（标题/作者/会议/arXiv 编号）经 arXiv 原始页面、ACM DL、作者主页交叉核实；无法核实处均标注"待确认"。
- 工业效果数字引自论文摘要或官方博客，A/B 口径因公司而异，仅供量级参考。
- **版权**：`生成式检索论文集/` 中的论文原文版权归原作者与出版方所有（均来自 arXiv 公开版本）；中文翻译由 LLM 辅助生成、人工未逐句校对，**仅供学习研究**，不得商用；如版权方有异议将及时移除。调研报告、清单与解读文档为本仓库原创整理，采用 MIT License 发布（见 [LICENSE](LICENSE)）。

## 推荐阅读路径

- **快速入门（1 天）**：GenIR 综述（TOIS 2025, arXiv:2404.14851）+ SIGIR 2024 Tutorial slides
- **理解范式（3 天）**：GENRE → DSI → NCI → SEAL → TIGER
- **学术前沿（1 周）**：RIPOR → PAG → GenRet → LETTER → ETEGRec → ESI/STATIC
- **工业落地（按需）**：检索线 TSGR / ICEGR / PLUM / STATIC；推荐线 HSTU / OneRec / MTGR；广告线 GR4AD / GEAR / UniSID / RARE
- **动手复现**：GenRet（学术）→ meta generative-recommenders（HSTU）→ MiniOneRec（OneRec 全流程）→ Snap GRID（SID 实践）
