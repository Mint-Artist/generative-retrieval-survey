# -*- coding: utf-8 -*-
"""生成《生成式检索论文清单.xlsx》— 5 个 sheet：总览 / 奠基经典 / 最新进展 / 业界落地 / 综述与资源"""
import sys, os

XLSX_SKILL_DIR = "/Users/awh/.zcode/cli/plugins/cache/zcode-plugins-official/spreadsheets/0.2.1/skills/xlsx"
for sub in [XLSX_SKILL_DIR, os.path.join(XLSX_SKILL_DIR, "templates")]:
    if sub not in sys.path:
        sys.path.insert(0, sub)

from base import *  # noqa: F401,F403  design tokens & style helpers
from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

# 字体：使用 base.py 默认解析的 "Noto Sans SC"（已安装到
# /Applications/LibreOffice.app/Contents/Resources/fonts/truetype/ 与 ~/Library/Fonts，
# 保证 LibreOffice 渲染 PDF 时正确嵌入中文字形；Noto 属重字重字体，base 自动不加粗）。

OUT = "/Users/awh/work/生成式检索/生成式检索论文清单.xlsx"

# ---------------------------------------------------------------- helpers
def write_sheet(ws, title, headers, rows, widths=None, note_lines=None):
    """headers 从 B4 起，数据从 B5 起；widths: {col_letter: width} 手动覆盖列宽"""
    last_col = len(headers) + 1
    setup_sheet(ws, title=title, last_col=last_col)
    for ci, h in enumerate(headers, start=2):
        ws.cell(row=4, column=ci, value=h)
    style_header_row(ws, row_num=4, col_start=2, col_end=last_col)
    for i, row in enumerate(rows):
        rn = 5 + i
        for ci, v in enumerate(row, start=2):
            ws.cell(row=rn, column=ci, value=normalize_cell_value(v))
        style_data_row(ws, row_num=rn, col_start=2, col_end=last_col, row_index=i)
    # 列宽：先自适应，再手动覆盖长文本列
    auto_fit_columns(ws, min_width=8, max_width=26, header_row=4, data_start_row=5)
    if widths:
        for col_letter, w in widths.items():
            ws.column_dimensions[col_letter].width = w
    auto_fit_row_heights(ws, header_row=4, data_start_row=5)
    # 备注（来源/说明）：合并整行宽度书写，避免窄列内长备注换行导致行高膨胀/跨页
    if note_lines:
        base_r = 5 + len(rows) + 1  # 空一行即可，避免脚注被挤到下一页形成孤页
        for j, line in enumerate(note_lines):
            r = base_r + j
            ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=last_col)
            c = ws.cell(row=r, column=2, value=line)
            c.font = font_caption()
            c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=False)
    ws.freeze_panes = "C5"  # 冻结表头与序号列
    # 打印设置：横向、适配页宽（渲染/打印时保持表格完整）
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_title_rows = "4:4"

# ---------------------------------------------------------------- data
# Sheet2 奠基经典（2020–2023）：简称 | 英文标题 | 机构 | 年份/会议 | 核心贡献 | arXiv | 代码
classic = [
    ["GENRE", "Autoregressive Entity Retrieval", "Meta AI (FAIR)", "ICLR 2021 (Oral)",
     "自回归生成实体名完成检索/消歧/实体链接；trie 约束束搜索保证只生成合法实体名。生成式检索的直接前身", "2010.00904", "facebookresearch/GENRE"],
    ["DSI", "Transformer Memory as a Differentiable Search Index", "Google", "NeurIPS 2022",
     "奠基之作：单一 Transformer 端到端完成“索引+检索”，整个语料编码进模型参数；系统研究原子/层级/字符串三类 docid", "2202.06991", "—（社区复现）"],
    ["NCI", "A Neural Corpus Indexer for Document Retrieval", "微软亚洲研究院", "NeurIPS 2022（Outstanding Paper）",
     "层级语义聚类 docid + doc2query 伪查询统一索引/检索输入分布 + 课程学习；NeurIPS 2022 杰出论文奖", "2206.02743", "solidsea98/Neural-Corpus-Indexer-NCI"],
    ["SEAL", "Autoregressive Search Engines: Generating Substrings as Document Identifiers", "Meta AI / UCL", "NeurIPS 2022",
     "以文档自身 n-gram 子串为 docid，FM-index 约束解码后聚合打分。文本标识符路线代表作（注意：非 ICML 2022）", "2204.10628", "—"],
    ["Ultron", "An Ultimate Retriever on Corpus with a Model-based Indexer", "中国人民大学", "arXiv 2022",
     "模型化索引器；系统对比 URL/标题/伪查询三类 docid；“LM 预训练→docid 预训练→检索微调”三阶段（注：arXiv 预印本，非 SIGIR 2022）", "2208.09257", "—"],
    ["CorpusBrain", "Pre-train a Generative Retrieval Model for Knowledge-Intensive Language Tasks", "中国人民大学", "CIKM 2022",
     "面向 KILT 基准的生成式检索预训练框架", "待确认", "—"],
    ["GMR", "Generative Multi-hop Retrieval", "KAIST", "EMNLP 2022",
     "多跳生成式检索：自回归迭代生成下一跳文档标识符", "待确认", "—"],
    ["DSI-QG", "Bridging the Gap Between Indexing and Retrieval for DSI with Query Generation", "墨尔本大学 + 微软", "arXiv 2022",
     "doc2query 生成并过滤伪查询，弥合索引阶段与检索阶段的数据分布差异", "2206.10128", "ArvinZhuang/DSI-QG"],
    ["DSI++", "Updating Transformer Memory with New Documents", "UW-Madison", "arXiv 2022",
     "生成式检索的增量更新与记忆刷新", "2212.09744", "—"],
    ["IncDSI", "Incrementally Updatable Document Retrieval", "Cornell + Google", "ICML 2023",
     "新增文档建模为凸优化问题，实现秒级增量插入", "待确认", "—"],
    ["TIGER", "Recommender Systems with Generative Retrieval", "Google", "NeurIPS 2023",
     "语义 ID 开山之作：RQ-VAE 将内容 embedding 量化为层次化 Semantic ID，自回归解码 SID 做生成式推荐；后续工业语义 ID 体系的源头", "2305.05065", "—"],
    ["GenRet", "Learning to Tokenize for Generative Retrieval", "山东大学 + 百度", "NeurIPS 2023",
     "学习“文档 tokenization”生成语义 docid（tokenization 模型+重构模型+检索模型离散自编码联合训练），NQ320K SOTA", "2304.04171", "sunnweiwei/GenRet"],
    ["MINDER", "Multiview Identifiers Enhanced Generative Retrieval", "香港理工 + MSRA", "ACL 2023",
     "每篇段落构建多视角标识符（标题/子串/数字/合成），生成多种标识符后聚合排序", "2305.16675", "liyongqi67/MINDER"],
    ["SE-DSI", "Semantic-Enhanced Differentiable Search Index Inspired by Learning Strategies", "中科院计算所 + 百度", "KDD 2023",
     "借鉴学习策略：伪查询作“精细描述型 docid”+ 文档细粒度特征作“复述内容”；已在百度线上搜索落地", "2305.15115", "—"],
    ["CGR / Np-Decoding", "Contextualized Generative Retrieval（后更名 Nonparametric Decoding）", "KAIST", "Findings of ACL 2023",
     "用编码器输出的上下文化向量替代静态 docid embedding，缓解固定 docid 表示的信息瓶颈", "2210.02068", "amy-hyunji/Contextualized-Generative-Retrieval"],
    ["NOVO", "Learnable and Interpretable Document Identifiers for Model-Based IR", "中国人民大学", "CIKM 2023",
     "可学习且可解释的 term-set 型 docid（注意：Zihan Wang et al.，非 Zhuang，非 SIGIR）", "—（无 arXiv 版）", "—"],
    ["TOME", "A Two-stage Approach for Model-based Retrieval", "中国人民大学", "ACL 2023",
     "两阶段模型化检索：先粗粒度聚类 docid 建索引、再细化", "待确认", "—"],
    ["GenRRL", "Enhancing Generative Retrieval with Reinforcement Learning from Relevance Feedback", "中国人民大学", "EMNLP 2023",
     "相关性反馈作为奖励的强化学习优化生成式检索", "待确认", "—"],
    ["GLEN", "Generative Retrieval via Lexical Index Learning", "Google", "EMNLP 2023",
     "两阶段动态词汇标识符学习", "待确认", "—"],
    ["Tied-Atomic", "Generative Retrieval as Dense Retrieval", "Rutgers", "arXiv 2023",
     "文档文本向量与 docid token embedding 绑定（tied），用稠密检索技术缓解更新与扩展问题", "2306.11397", "—"],
    ["ASI", "Auto Search Indexer", "微软 STCA", "EMNLP 2023 Findings",
     "docid 学习与检索联合的“全端到端”生成式检索范式；后续 ASI++ 解决 docid 分布不均衡", "2310.12455", "—"],
    ["RIPOR", "Scalable and Effective Generative Information Retrieval", "UMass Amherst", "WWW 2024（arXiv 2023-11）",
     "group-based 语义 docid + 排序感知渐进式训练 + beam search 重排；完整 MS MARCO（880 万段落）规模化验证（注意：非 EMNLP 2023）", "2311.09134", "hansizeng/ripor"],
    ["LTRGR", "Learning to Rank in Generative Retrieval", "香港理工 + MSRA", "AAAI 2024（arXiv 2023-06）",
     "rank-based 损失将 docid 生成目标对齐最终排序目标，弥合“生成式训练目标≠排序目标”的鸿沟", "2306.15222", "liyongqi67/LTRGR"],
    ["LMIndexer", "Language Models as Semantic Indexers", "UIUC + Amazon", "ICLR 2024（arXiv 2023-10）",
     "自监督学习文本→语义 ID 映射做索引，支持零样本检索", "2310.07815", "—"],
    ["DynamicRetriever", "A Pre-trained Model-based IR System Without an Explicit Index", "中国人民大学", "MIR 期刊 2023",
     "Ultron 前身：无显式索引的预训练 IR 系统", "待确认", "—"],
]

# Sheet3 最新进展（2024–2026.11）：简称 | 英文标题 | 机构 | 年份/会议 | 方向 | 核心贡献 | arXiv
recent = [
    ["PAG", "Planning Ahead in Generative Retrieval", "UMass Amherst", "SIGIR 2024", "文档检索",
     "同步解码（simultaneous decoding）让模型生成语义 docid 时前瞻规划，兼顾检索效果与解码效率", "2404.14600"],
    ["GR²", "Generative Retrieval Meets Multi-Graded Relevance", "中科院计算所", "NeurIPS 2024", "文档检索",
     "多级（非二值）相关性信号进入监督与解码，强化生成式检索的排序能力", "2409.18409"],
    ["LLM2GR", "Generative Retrieval with Large Language Models", "—", "arXiv 2024", "LLM 检索",
     "以 LLM 为骨干、文本化 docid 的生成式检索，被知识归因/模型编辑类工作广泛引用", "2402.17010"],
    ["LLMGR", "LLM-based Generative Retrieval in Alipay Search", "蚂蚁集团", "SIGIR 2024 Industry", "工业·搜索",
     "首批 LLM 生成式检索工业落地之一（支付宝搜索）：LLM 与传统召回融合、线上延迟优化", "—（ACM DL）"],
    ["HSTU", "Actions Speak Louder than Words（万亿参数序列转导器）", "Meta", "ICML 2024", "生成式推荐",
     "HSTU 架构 + GRN 范式，把高基数用户动作当“新模态”；1.5T 参数线上 +12.4%，Meta 多场景全量", "2402.17152"],
    ["GRACE", "Generative Cross-Modal Retrieval: Memorizing Images in Multimodal LLMs", "中山大学", "ACL 2024", "多模态",
     "MLLM 用记忆机制把图像存为标识符，文本查询直接自回归生成图像 ID（推理无需视觉输入）", "2402.10805"],
    ["LETTER", "Learnable Item Tokenization for Generative Recommendation", "中科大 / NUS", "CIKM 2024", "语义 ID",
     "可学习 tokenizer：层次语义+协同信号+码字多样性三约束，ranking-guided 生成损失，码本可联合优化", "2405.07314"],
    ["AtSpeed", "Efficient Inference for LLM-based Generative Recommendation", "—", "ICLR 2025", "推理效率",
     "面向生成式推荐的推测解码对齐框架（top-K 语义级验证），2–2.5× 推理加速", "2410.05165"],
    ["LIGER", "Unifying Generative and Dense Retrieval for Sequential Recommendation", "Meta", "WSDM 2025", "混合架构",
     "两阶段混合：生成式检索出候选 + 稠密检索精排；冷启改善、延迟约 −45%", "2411.18814"],
    ["ETEGRec", "Generative Recommender with End-to-End Learnable Item Tokenization", "中国人民大学 + 快手", "SIGIR 2025", "语义 ID",
     "物品 tokenizer 与生成式推荐器联合训练（序列-语义双重对齐循环迭代），取代两阶段冻结码本", "2409.05546"],
    ["COBRA", "Unified Generative Recommendations with Cascaded Sparse-Dense Representations", "百度", "NeurIPS 2025", "混合架构",
     "级联组织的“稀疏语义 ID + 稠密向量”双重表示，统一生成式与稠密检索；已全量上线（转化 +3.6%）", "2503.02453"],
    ["Case Against", "The Case Against Generation for Retrieval", "Meta", "arXiv 2025", "反方观点",
     "系统实验论证判别式 LLM 检索器在表示质量上仍可胜过 docid 生成式方案，划定生成式检索适用边界", "2503.22675"],
    ["UTGRec", "Universal Item Tokenization for Transferable Generative Recommendation", "中国人民大学", "arXiv 2025", "语义 ID",
     "跨域共享码本的通用物品 tokenization，面向可迁移生成式推荐", "2504.04405"],
    ["RPG", "Generating Long Semantic IDs in Parallel for Recommendation", "UCSD + Meta", "KDD 2025", "语义 ID/效率",
     "长语义 ID 从自回归改为并行一次性解码（无序集合），消除误差累积并大幅加速", "2506.05781"],
    ["ActionPiece", "Contextually Tokenizing Action Sequences for Generative Recommendation", "Google DeepMind", "ICML 2025 (Spotlight)", "语义 ID",
     "按动作序列上下文动态切分 token 的 BPE 式 tokenizer，把“动作”当作新模态做上下文化编码", "2502.13581"],
    ["GenSAR", "Semantic IDs for Joint Generative Search and Recommendation", "Spotify", "RecSys 2025 (LBR)", "统一搜推",
     "统一生成式模型同时服务搜索与推荐的 SID 构建；结论：RQ-KMeans 优于 RQ-VAE（后者训练不稳定）", "2508.10478"],
    ["MMQ", "Multimodal Mixture-of-Quantization Tokenization", "—", "WSDM 2026", "语义 ID",
     "多模态混合量化 tokenization，为多模态 SID 构建提供新方案", "2508.15281"],
    ["DIGER", "Differentiable Semantic ID for Generative Recommendation", "—", "SIGIR 2026", "语义 ID",
     "可微语义 ID：量化与生成端到端联合学习", "—（DOI:10.1145/3805712.3809641）"],
    ["ESI", "Evolving Semantic Indexes for Generative Retrieval", "Santa Clara Univ. + Coupang", "arXiv 2026.10", "自动化 SID",
     "用 31B 开源 LLM 做“进化程序搜索”自动改写 SID 构建管线；NQ320K R@100 85.8%→90.6%", "2610.28568"],
    ["STATIC", "Vectorizing the Trie（约束解码向量化）", "YouTube / Google", "KDD 2026", "推理效率",
     "约束解码 trie 向量化到加速器：单步 0.033ms、仅占推理时间 0.25%，较 CPU trie 快 948×；首个生产规模部署", "2602.22647"],
    ["SID 空间研究", "A Systematic Study of Semantic ID Spaces for GR", "—", "arXiv 2026.10", "实证研究",
     "系统对比不同语义 ID 空间（聚类方式/码本结构）对生成式检索性能的影响", "2610.08732"],
    ["SID 评测", "What Makes a Good Semantic ID for Generative Recommendation?", "—", "arXiv 2026.09", "实证研究",
     "统一评测 TIGER/LETTER/RPG 等 11 种语义 ID 方法的取舍", "2609.24430"],
    ["DECOR", "Decomposed Contextual Token Representation", "—", "SIGIR 2026", "文档检索", "分解式上下文 token 表示", "2509.10468"],
    ["DualGR", "长/短兴趣双粒度生成式检索", "—", "WWW 2026", "文档检索", "长短期兴趣双粒度的生成式检索", "2511.12518"],
    ["SIDReasoner", "在语义 ID 上做推理增强的生成式推荐", "—", "KDD 2026", "推理增强", "推理链 + 语义 ID 生成结合，KDD 2026 新趋势代表", "—"],
    ["NEO", "语言可操控的统一生成式推荐", "—", "KDD 2026", "统一模型", "用自然语言指令统一操控推荐任务", "—"],
    ["H²Rec", "调和语义 ID 与哈希 ID", "—", "KDD 2026", "混合标识符", "语义 ID 与哈希 ID 双体系调和", "—"],
    ["APAO", "prefix-aware 训练-推理对齐", "—", "KDD 2026", "训练对齐", "对齐训练目标与自回归推理行为", "—"],
]

# Sheet4 业界落地：公司 | 系统/论文 | 场景 | 年份/渠道 | 落地效果 | arXiv/链接
industry = [
    ["Google/YouTube", "TIGER（研究源头）", "—", "NeurIPS 2023", "RQ-VAE 语义 ID 范式源头，非直接落地", "2305.05065"],
    ["Google/YouTube", "Better Generalization with Semantic IDs", "YouTube 排序", "RecSys 2024",
     "语义 ID 取代 video ID：改善新视频/长尾泛化，整体质量无损", "2306.08121"],
    ["Google/YouTube", "PLUM", "YouTube 生成式检索", "WWW 2026",
     "开源 LLM 继续预训练 + RQ-VAE 语义 ID 替代巨型 embedding 表召回；面向数十亿用户规模部署（第三方解读 Shorts CTR 约 +5%，待确认）", "2510.07784"],
    ["Google/YouTube", "STATIC", "YouTube GR 线上 serving", "KDD 2026",
     "约束解码单步 0.033ms、仅占推理时间 0.25%；较 CPU trie 快 948×；首个生产规模严格约束 GR 部署", "2602.22647"],
    ["Google/YouTube", "Tokens are All You Need（双用途 SID）", "排序+检索", "RecSys 2026 Industry",
     "语义 ID 兼作存储/传输与模型输入，大幅降低 I/O 成本，已大规模线上部署", "2607.24865"],
    ["Meta", "HSTU", "Instagram/Facebook 排序", "ICML 2024",
     "1.5T 参数，线上 +12.4%，对 FA2 Transformer 快 5.3–15.2×，多 surface 全量部署", "2402.17152"],
    ["Meta", "Semantic ID for Embedding Stability", "Meta 广告排序", "RecSys 2025",
     "前缀 ngram 参数化语义 ID 集成进生产广告排序：性能显著提升 + 预测稳定性改善", "2504.02137"],
    ["Meta", "LIGER", "推荐检索", "WSDM 2025", "生成式+稠密混合：冷启改善、延迟 −45%", "2411.18814"],
    ["Meta", "VISTA", "终身记忆", "ICLR 2026", "数百万亿参数记忆，数十亿用户平台部署，支持百万级终身历史", "2510.22049"],
    ["Meta", "LIGE-GR", "Instagram Reels / FB Video", "arXiv 2026.09",
     "从排序到生成式推荐的平滑升级：Reels 时长 +1.14%、FB Video +0.72%", "2609.18148"],
    ["Meta", "GEM（生成式广告模型）", "广告", "工程博客 2025-11",
     "Instagram 广告转化 +5%、Facebook Feed +3%（2025 Q2，官方博客口径）", "engineering.fb.com（博客）"],
    ["快手", "OneRec（初版）", "主端推荐", "arXiv 2025.02",
     "首个生产级端到端生成式推荐（召回+排序统一）：watch time +1.6%", "2502.18965"],
    ["快手", "OneRec Technical Report", "主端+极速版", "arXiv 2025.06",
     "承接快手/极速版 25% 总 QPS；运营成本为传统管线 10.6%；停留时长 +0.54%/+1.24%", "2506.13695"],
    ["快手", "OneRec-V2", "全量部署", "arXiv 2025.08",
     "Decoder-only 架构：总算力 −94%、训练资源 −90%、8B 参数；停留时长 +0.467%/+0.741%", "2508.20900"],
    ["快手", "GR4AD", "快手广告", "arXiv 2026.02",
     "4 亿+用户全量：UA-SID + LazyAR + RL，A/B 广告收入最高 +4.2%", "2602.22732"],
    ["快手", "OneSearch", "快手电商搜索", "arXiv 2025.09",
     "首个工业部署的电商端到端生成式搜索框架（query 建议+检索+排序统一），日 PV 数千万", "2509.03236"],
    ["快手", "Kwai26 / OpenOneRec / MiniOneRec", "开源与基准", "2025–2026",
     "13.1 亿记录工业基准 Kwai26；OneRec-Foundation（1.7B/8B）+ RecIF-Bench；轻量复现框架 MiniOneRec", "2607.26500 等"],
    ["百度", "SE-DSI", "百度搜索", "KDD 2023", "最早的生成式检索线上落地之一（中科院计算所合作）", "2305.15115"],
    ["百度", "COBRA", "推荐广告", "arXiv 2025.03",
     "全量上线：转化率 +3.6%、ARPU +4.15%（500 万用户/200 万广告工业数据集）", "2503.02453"],
    ["百度", "ICEGR", "百度电商搜索", "arXiv 2026.08",
     "意图一致端到端生成式召回通道：CTR +3.52%、订单 +15.96%、GMV +7.53%（离线 Recall@20 +21.7%）", "2608.29652"],
    ["百度", "UNIQUE", "手机百度信息流", "arXiv 2026.09",
     "检索+排序统一生成式系统：总观看时长 +0.96%、分发 +1.08%；P99 延迟 89ms、推理 MFU 44.23%", "2609.23718"],
    ["阿里/淘天", "TSGR", "淘宝搜索召回", "arXiv 2026.07",
     "全量部署：商业价值注入 SID + 价值感知排序；GMV +1.64%、成交笔数 +1.12%、IPV +0.43%", "2607.18796"],
    ["阿里/淘天", "CQ-SID + EG-GRPO", "天猫 App 搜索", "arXiv 2026.05",
     "GMV +1.15%、UCTCVR +0.40%；生成式召回通道已占曝光 50.25%、点击 58.96%、成交 72.63%", "2605.14434"],
    ["阿里/淘天", "FORGE + AL-GR 数据集", "淘宝“猜你喜欢”", "KDD 2026",
     "产业级 SID 构建基准；A/B 交易笔数 +0.35%，策略全量部署；数据集 140 亿交互/2.5 亿商品", "2509.20904"],
    ["阿里/闲鱼", "GSID", "闲鱼电商", "EMNLP 2025 Industry", "生成式语义索引替代人工类目树，已落地", "2509.23860"],
    ["阿里/阿里妈妈", "RARE", "赞助搜索广告", "EMNLP 2025",
     "LLM 生成式商业意图实时广告检索：消耗 +5.04%、GMV +6.37%、CTR +1.28%；日均数亿搜索量", "2504.01304"],
    ["阿里/淘天", "LUM", "淘宝竞价搜索", "arXiv 2025.02",
     "Large User Model + next-condition-item prediction：CTR +2.9%、RPM +1.2%（全量 A/B）", "2502.08309"],
    ["腾讯", "UniSID", "腾讯广告", "arXiv 2026.02",
     "语义 ID 与 embedding 从原始物料端到端联合生成：下游广告任务 Hit Rate 最高 +4.62%（离线）", "2602.10445"],
    ["腾讯", "混元生成式召回", "展示广告", "技术博客 2024-10",
     "自称“业界首个落地的展示广告场景生成式召回”（无论文，细节待确认）", "ai.jotoai.com（博客）"],
    ["京东", "GR with Preference Optimization", "京东搜索", "arXiv 2024.07",
     "多 span 标识符 + 点击偏好对齐 + 约束 beam search；A/B 转化提升", "2407.19829"],
    ["京东", "广告生成式召回（RQ-VAE+Qwen2-1.5B）", "京东广告", "博客 2025-01（与 NVIDIA 合作）",
     "TensorRT-LLM 优化：吞吐 5×+、部署成本约 1/5、满足 100ms 线上延迟", "developer.jdcloud.com（博客）"],
    ["美团", "MTGR", "外卖主流量", "CIKM 2025",
     "基于 HSTU 保留 DLRM 交叉特征；官方称“近两年最大线上线下收益”；单请求前向 FLOPs 为 DLRM 65×", "2505.18654"],
    ["美团", "DoS（双流正交 SID）", "—", "arXiv 2026.02", "语义与协同信息解耦的双流正交语义 ID 设计", "2602.04460"],
    ["字节跳动", "GEAR", "抖音广告", "arXiv 2026.09",
     "生成式端到端广告召回：训练-服务协同设计、防表示坍塌、流式训练，已部署上线", "2609.39327"],
    ["Spotify", "GLIDE", "播客发现", "KDD 2026",
     "1B LLM + 语义 ID 生成式检索：非习惯性播放 +5.4%、新节目发现 +14.3%（百万人级 A/B）", "2603.17540"],
    ["Spotify", "GenSAR", "搜索+推荐统一", "RecSys 2025 LBR", "统一 SID 空间 + 双任务微调最佳实践（RQ-KMeans 优于 RQ-VAE）", "2508.10478"],
    ["Snap", "GRID（实践手册）", "—", "CIKM 2025", "语义 ID 生成式推荐模块化框架 + 大规模实验（开源）", "2507.22224"],
    ["Snap", "SnapLGR", "Snapchat 短视频", "arXiv 2026.07",
     "已上线：View Time +0.37%、Time Spent +0.09%；TensorRT-LLM beam search 服务", "2607.28895"],
    ["Snap", "EGR", "—", "RecSys 2026",
     "共享 LLM 统一嵌入空间、不做显式量化；线上转化率 +2.91%（公司归属待确认）", "2607.23038"],
    ["Pinterest", "PinGER", "Related Pins", "arXiv 2026.10",
     "token 高效生成式检索：fresh saves +3.6%、总 saves +0.62%、端到端延迟 −0.7%", "2610.33435"],
    ["Netflix", "大规模生成式推荐", "—", "arXiv + 博客 2026-01", "生成式推荐从 ~1M 扩到 ~1B 参数的工程化", "2605.23312"],
    ["蚂蚁集团", "LLMGR", "支付宝搜索", "SIGIR 2024 Industry", "首批 LLM-GR 工业落地：LLM 与传统召回融合、延迟优化", "—（ACM DL）"],
    ["Coupang", "ESI", "电商检索", "arXiv 2026.10", "LLM 进化搜索自动优化 SID 管线：NQ320K R@100 85.8%→90.6%", "2610.28568"],
]

# Sheet5 综述与资源：名称 | 类型 | 机构/维护者 | 年份 | 说明 | 链接
resources = [
    ["From Matching to Generation: A Survey on Generative Information Retrieval", "综述⭐", "中国人民大学 RUC-NLPIR", "ACM TOIS 2025",
     "领域旗舰综述：生成式文档检索 + 可靠响应生成两大分支，docid 设计/增量学习/评测全覆盖", "arxiv.org/abs/2404.14851；github.com/RUC-NLPIR/GenIR-Survey"],
    ["LLM4IR: Large Language Models for Information Retrieval", "综述", "中国人民大学", "arXiv 2023（持续更新）",
     "LLM 时代 IR 全景，含 LLM 作生成式检索器专章", "arxiv.org/abs/2308.07107"],
    ["A Survey of Generative Search and Recommendation in the Era of LLMs", "综述", "HKUST / USTC 等", "arXiv 2024",
     "统一“生成式搜索+生成式推荐”框架", "arxiv.org/abs/2404.16924"],
    ["A Survey on Generative Recommendation: Data, Model, and Tasks", "综述", "—", "arXiv 2025.10",
     "数据-模型-任务三元框架的生成式推荐综述", "arxiv.org/abs/2510.27157"],
    ["A Survey of Item Identifiers in Generative Recommendation", "综述", "HKBU LAGAS", "2026",
     "物品标识符（SID）构建/对齐/生成三阶段专门综述，配套论文列表持续更新", "github.com/HKBU-LAGAS/Awesome-Item-ID-Gen-RecSys"],
    ["Recent Advances in Generative Information Retrieval", "教程⭐", "中科院计算所 + 莱顿 + UvA", "SIGIR 2024",
     "GR 范式、DocID 设计、训练、约束解码、应用，全套 slides", "generative-ir.github.io"],
    ["同系列教程（WWW/ECIR 2024）", "教程", "同上", "WWW/ECIR 2024",
     "392 页 slides + 18 个开源实现链接", "thewebconf2024-generative-ir.github.io"],
    ["Generative Recommendation: Foundations and Frontiers", "教程", "港城大 + 快手", "KDD 2026",
     "三解耦视角（tokenization/架构/优化）", "applied-machine-learning-lab.github.io/KDD2026_GenRec_Tutorial"],
    ["LLMs for Generative Recommendation", "教程", "HKUST 团队", "WWW 2025",
     "306 页 slides，语义树 item ID 等", "large-genrec.github.io"],
    ["awesome-generative-information-retrieval", "开源列表⭐", "UvA（Bénédict）", "2023 至今",
     "最系统的 GenIR 论文/资源汇总（729 stars）", "github.com/gabriben/awesome-generative-information-retrieval"],
    ["GenIR-Survey 仓库", "开源列表", "人大 RUC-NLPIR", "2024 至今", "TOIS 综述配套仓库，按方向分类", "github.com/RUC-NLPIR/GenIR-Survey"],
    ["FlashRAG", "工具包", "人大 RUC-NLPIR", "WWW 2025", "模块化 RAG 工具包（3.6k stars），GenIR 生态基础设施", "github.com/RUC-NLPIR/FlashRAG"],
    ["generative-recommenders（HSTU）", "工业开源", "Meta", "2024", "HSTU 官方实现（2k stars）+ NVIDIA 参考实现", "github.com/meta-recsys/generative-recommenders"],
    ["MiniOneRec / OpenOneRec", "工业开源", "快手", "2025.10 / 2025.12",
     "OneRec 全流程复现框架 / OneRec-Foundation（1.7B/8B）+ RecIF-Bench", "github.com/Kuaishou-OneRec/OpenOneRec"],
    ["GRID", "框架", "Snap Research", "CIKM 2025", "语义 ID 生成式推荐实践手册配套开源框架", "github.com/snap-research/GRID"],
    ["static-constraint-decoding", "工业开源", "YouTube", "2026", "向量化约束解码官方实现", "github.com/youtube/static-constraint-decoding"],
    ["NQ320K / MS MARCO GR setting", "基准", "DSI 开创", "2022 至今",
     "学术标准设定：模型直接生成 DocID，指标 R@K / MRR@10", "—"],
    ["Kwai26", "基准/数据集", "快手", "2026",
     "13.1 亿记录、3185 万 Item-ID 的工业多目标 GR 基准", "huggingface.co/datasets/zhaoliu2517/Kwai26"],
    ["RecIF-Bench", "基准", "快手", "2025.12",
     "首个推荐指令遵循基准（1 亿交互、短视频/广告/电商三域 8 类任务）", "github.com/Kuaishou-OneRec/OpenOneRec"],
    ["AL-GR", "基准/数据集", "浙大 + 淘天", "KDD 2026",
     "140 亿交互、2.5 亿商品的产业级 SID 基准", "github.com/selous123/al_sid"],
    ["Gen-IR Workshop @ SIGIR 2023/2024", "Workshop", "UvA + 中科院 + Google 等", "2023/2024",
     "领域奠基 workshop；2025 后由 GENNEXT@SIGIR'25 等接续", "coda.io/@sigir/gen-ir"],
]

# ---------------------------------------------------------------- build
wb = Workbook()

# Sheet1 总览
ws = wb.active
ws.title = "总览"
info_rows = [
    ["工作表", "内容", "条目数"],
    ["2. 奠基经典（2020-2023）", "生成式检索奠基与经典论文：GENRE→DSI/NCI/SEAL→TIGER→docid 设计精细化→RIPOR/LTRGR", str(len(classic))],
    ["3. 最新进展（2024-2026）", "LLM 时代 docid 生成、语义 ID 端到端化、混合架构、推理效率、推理增强等前沿方向", str(len(recent))],
    ["4. 业界落地", "Google/YouTube、Meta、快手、百度、阿里、腾讯、京东、美团、字节、Spotify、Snap、Pinterest 等生产系统", str(len(industry))],
    ["5. 综述与资源", "综述、教程、开源项目、基准数据集、Workshop", str(len(resources))],
]
write_sheet(ws, "生成式检索（Generative Retrieval）论文清单 · 总览",
            ["工作表", "内容", "条目数"], info_rows[1:],
            widths={"B": 26, "C": 62, "D": 10},
            note_lines=[
                "调研日期：2026-11-07｜配套详细报告：生成式检索调研报告.md（同目录）",
                "信息核实：论文元数据经 arXiv 原始页面、ACM DL、作者主页交叉核实；无法核实处标注“待确认”。",
                "常见勘误：SEAL=NeurIPS 2022（非 ICML）；NOVO=CIKM 2023（Zihan Wang，非 Zhuang/SIGIR）；RIPOR=WWW 2024（非 EMNLP 2023）；",
                "NCI=微软亚洲研究院（非 Google）；GenIR 综述=ACM TOIS 2025（非 FnTIR）；Ultron=arXiv 预印本（非 SIGIR 2022）。",
                "未核实名称（二手转述，谨慎使用）：RepBF/RepLA、IGReD、Taurus/HIR/Mouchak、GBench、MAUI、Hybrid-GenRet、TinyRec。",
                "工业效果数字均引自论文摘要或官方博客（A/B 口径因公司而异，仅供量级参考）。",
            ])

# Sheet2 奠基经典
ws2 = wb.create_sheet("奠基经典2020-2023")
write_sheet(ws2, "一、奠基与经典论文（2020–2023）",
            ["简称", "英文标题", "机构", "年份/会议", "核心贡献（中文）", "arXiv", "代码（GitHub）"],
            classic,
            widths={"B": 13, "C": 40, "D": 16, "E": 18, "F": 52, "G": 13, "H": 26},
            note_lines=["按时间脉络排序；“arXiv=待确认”表示未找到 arXiv 版本（可能仅有会议版）；勘误信息见单元格内“注意”标注。"])

# Sheet3 最新进展
ws3 = wb.create_sheet("最新进展2024-2026")
write_sheet(ws3, "二、最新进展（2024–2026.11）",
            ["简称", "英文标题/说明", "机构", "年份/会议", "方向", "核心贡献（中文）", "arXiv"],
            recent,
            widths={"B": 15, "C": 38, "D": 17, "E": 15, "F": 11, "G": 48, "H": 15})

# Sheet4 业界落地
ws4 = wb.create_sheet("业界落地")
write_sheet(ws4, "三、业界落地全景（按公司）",
            ["公司", "系统/论文", "业务场景", "年份/渠道", "落地效果/规模（论文或官方博客口径）", "arXiv/链接"],
            industry,
            widths={"B": 13, "C": 26, "D": 15, "E": 16, "F": 55, "G": 18},
            note_lines=[
                "效果数字均为各公司内部 A/B 口径，仅供量级参考；“待确认”=未能从公开渠道核实。",
                "最硬规模数字参考：快手 OneRec 承接 25% QPS；天猫生成式召回通道占成交 72.63%；快手 GR4AD 广告收入 +4.2%（4 亿用户全量）。",
            ])

# Sheet5 综述与资源
ws5 = wb.create_sheet("综述与资源")
write_sheet(ws5, "四、综述、教程、开源项目与基准",
            ["名称", "类型", "机构/维护者", "年份", "说明", "链接"],
            resources,
            widths={"B": 34, "C": 10, "D": 20, "E": 15, "F": 44, "G": 40},
            note_lines=["⭐ = 最推荐优先阅读/使用。"])

wb.properties.creator = "Z.ai"
wb.save(OUT)
print("saved:", OUT)
print("sheets:", wb.sheetnames)
print("rows:", len(classic), len(recent), len(industry), len(resources))
