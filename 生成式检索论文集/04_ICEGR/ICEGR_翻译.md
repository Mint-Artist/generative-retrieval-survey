# ICEGR：面向电商搜索的意图一致端到端生成式检索框架

**英文原题：** ICEGR: An Intent-Coherent End-to-End Generative Retrieval Framework for E-commerce Search

**作者与机构：**
Jiayi Tuo\*（拓佳毅，Baidu）、Hehan Li\*（李赫翰，Baidu）、Dongjun Fu\*（傅东军，Baidu）、Xin Lu（Baidu）、Ling Zhuang（Baidu）、Fuwei Zhang（Beihang University）、Meifang Li（Baidu）、Peizhi Xu（Baidu）、Hanmeng Liu（Baidu）、Shuanglong Li（Baidu）、Liwei Qian（Baidu）、Yanbiao Ma✉（Renmin University of China）、Fuzhen Zhuang✉（Beihang University）

（\* 注：前三位作者对本工作贡献相同；✉ 为通讯作者。机构均为 Beijing, China）

**arXiv 编号：** arXiv:2608.29652（v3）

**译注：** 本文发表于 WSDM '27（The 20th ACM International Conference on Web Search and Data Mining, February 2027），是百度电商搜索联合北京航空航天大学、中国人民大学发表的工业界论文。全文术语遵循信息检索领域标准译法：generative retrieval 译作"生成式检索"、semantic ID（SID）译作"语义ID"、intent 译作"意图"、beam search 译作"束搜索"等；模型名、数据集名、指标名（Recall@K、NDCG@K 等）保留英文。

---

## 摘要

生成式检索（Generative Retrieval, GR）对电商搜索颇具前景，但现有方法难以在整个训练流程中保持查询-意图一致性。首先，基于静态商品信息构建的语义ID（semantic ID, SID）限制了 SID 编码商品-意图关联的能力。其次，尽管监督微调（supervised fine-tuning, SFT）能在整个商品目录上学习商品-SID 映射，但由于查询到 SID 的训练完全依赖在线日志，低曝光商品仍然缺乏真实的查询意图监督，导致这些商品的检索性能不佳。第三，面向业务的偏好优化可能偏袒热门或高价值商品，而非最匹配查询意图的商品，从而削弱查询-商品相关性。

为解决这些问题，我们提出 ICEGR——一个面向电商搜索的意图一致端到端生成式检索框架，将查询意图贯穿于 GR 训练的全流程。ICEGR 包含三个组件：(1) 意图感知的 SID 构建（Intent-Aware SID Construction），将查询意图信号融入 SID 构建，使 SID 能够捕捉超越静态商品信息的搜索意图；(2) 合成查询增强的统一 SFT（Synthetic Query-Enhanced Unified SFT），将多个 SFT 任务统一到单一的查询到 SID 目标之下，并用合成查询增强在线日志的稀疏监督，为低曝光商品提供互补的查询意图监督；(3) 相关性校准的偏好优化（Relevance-Calibrated Preference Optimization），将查询-商品相关性与业务信号整合进一个边界自适应的偏好目标，在保留查询意图的同时实现业务偏好学习。离线实验表明，ICEGR 相对基线分别提升 Recall@20 达 21.7%、NDCG@20 达 26.6%。当 ICEGR 作为端到端生成式检索通路部署于百度电商搜索系统时，在线 A/B 测试取得了 CTR 相对提升 3.52%、订单量提升 15.96%、GMV 提升 7.53% 的结果。

**关键词：** 生成式检索、电商搜索、语义ID、端到端、查询意图、偏好优化

图1：传统 MCA 与生成式检索的对比。传统 MCA 对商品候选进行召回、过滤和排序，而 ICEGR 直接生成 top-$k$ 个商品。随后两组候选集合并形成最终结果列表。

## 1. 引言

电商搜索旨在从大规模商品目录中检索出满足用户查询所表达购物意图的商品 [10]。如图1所示，工业界电商搜索系统普遍采用多阶段级联架构（multi-stage cascade architecture, MCA），由查询理解、召回、粗排和精排等阶段构成 [12, 4]。该架构通过逐步收窄候选集来产出最终的商品排序列表。然而，不同阶段往往依赖各自独立的模型、特征和优化目标。一旦某个相关商品在召回阶段被遗漏，下游排序模型挽回它的能力十分有限，从而给系统整体性能设置了上限 [17]。

生成式检索为解决级联架构的局限提供了一个有前景的方向。该范式将每个商品映射为一个生成式语义ID（SID），并采用自回归模型以用户查询为条件直接生成相关商品的 SID，从而缓解多阶段级联造成的性能损失 [20, 5]。然而，尽管现有端到端模型统一了检索架构，它们仍难以在整个学习流程中保持查询-意图一致性 [19, 14]。在不同阶段，这些模型可能偏好商品内容、面向商品的监督信号或流行度信号，而非一致地建模真实的查询-商品相关关系 [27, 29, 25]。这种架构统一与查询意图一致性之间的差距，仍是生成式检索在工业系统中有效落地的关键障碍。

更具体地说，查询意图一致性不足体现在三个方面。第一，在 SID 构建阶段，现有方法通常基于标题、描述等静态商品内容构建 SID [27, 5]。然而，内容相似度并不能完全刻画商品与查询意图的关联方式 [15]。例如，查询"情人节送女朋友的礼物"既可能被一支香奈儿口红满足，也可能被一条宝格丽项链满足。这些商品在内容上差异很大，却满足同一购物意图。第二，在监督学习阶段，以商品为中心的监督主要促使模型记住商品与其 SID 之间的映射，而不是学习从查询意图出发检索相关 SID [20, 5]。尽管真实交互日志能提供查询到 SID 的监督 [22]，但工业电商系统中商品的曝光服从长尾分布，因此许多低曝光商品缺乏充分且可靠的查询-商品关联信号 [11]。最后，在偏好优化阶段，直接依据点击率或商业价值构建偏好对，可能使模型偏向与查询意图不匹配的热门或高价值商品，进一步加剧意图失准 [3, 25]。

为解决这些问题，我们提出 ICEGR——一个面向电商搜索的意图一致端到端生成式检索框架。ICEGR 将查询意图置于整个生成式检索学习流程的中心。首先，我们提出意图感知的 SID 构建（Intent-Aware SID Construction, IA-SID），在商品离散化之前引入查询诱导的商品共点击关系和历史关联查询。这使商品 SID 能够同时编码内容语义与检索意图关联。其次，我们提出合成查询增强的统一 SFT（Synthetic Query-Enhanced Unified SFT, SQE-SFT），从整个商品目录自动构建多粒度合成查询，并以合成查询到 SID 的监督取代商品记忆式训练。它降低了训练与推理输入之间的失配，并为低曝光商品补充了交互日志之外的监督，提升了它们的可检索性。最后，我们提出相关性校准的偏好优化（Relevance-Calibrated Preference Optimization, RCPO），将查询-商品相关性与业务信号整合进偏好优化。RCPO 将偏好比较限制在语义相关的候选之内，用用户行为与商业价值信号精化其排序，并使每次偏好更新的强度自适应于成对综合得分差距（margin）。这一设计在保留查询意图的同时实现了业务偏好学习。

我们在离线环境和生产级在线 A/B 测试中评估了 ICEGR。离线环境下，ICEGR 分别提升 Recall@20 和 NDCG@20 达 21.7% 和 26.6%。我们进一步将 ICEGR 作为端到端生成式检索通路部署到百度电商搜索系统。在线 A/B 测试取得 CTR 相对提升 3.52%、订单量提升 15.96%、GMV 提升 7.53% 的结果。这些结果表明，在 SID 构建、生成式监督和偏好优化全流程中保持查询意图一致性，能够将离线检索的改进转化为切实的在线业务收益。更广泛地说，这种以意图为中心、贯穿全流程的优化范式超越了各阶段的局部优化，为联合提升检索质量与业务效果提供了一条可行路径。

概括而言，我们的主要贡献如下：

（1）我们提出意图感知的 SID 构建（IA-SID），在商品离散化之前整合查询诱导的商品共点击关系和历史关联查询。SID 在保留商品语义的同时，也保留了搜索行为所揭示的查询-商品意图关联。

（2）我们提出合成查询增强的统一 SFT（SQE-SFT），在整个目录上自动构建多粒度合成查询，并将传统的商品记忆式训练重塑为查询到 SID 的监督。这为低曝光商品提供了稀疏交互日志之外的互补性查询级监督。

（3）我们提出相关性校准的偏好优化（RCPO），将语义相关性约束、行为信号与商业价值整合进偏好学习。通过用成对综合得分差距校准偏好更新的强度，RCPO 在保留查询意图的同时实现了业务偏好学习。

（4）我们将 ICEGR 作为端到端生成式检索通路部署于百度电商搜索系统。离线实验与生产级 A/B 测试验证了其在提升检索质量与业务效果方面的有效性。

图2：ICEGR 概览，其在三个阶段强制生成式检索全程保持查询意图对齐：(a) IA-SID 构建与搜索意图对齐的语义ID；(b) SQE-SFT 用多粒度合成查询扩充稀疏的真实查询到 SID 监督，随后在同一目标下对真实查询进行微调；(c) RCPO 通过边界自适应的偏好优化，在语义相关候选之间学习业务感知的偏好。

## 2. 相关工作

### 2.1 电商搜索与商品检索

电商搜索根据用户查询表达的购物意图从大规模商品目录中检索商品 [10]。工业系统通常采用多阶段级联架构，组合查询理解、候选召回、粗排与精排，以平衡检索效率与排序质量 [12, 4, 8]。这些阶段往往使用不同的模型和优化目标 [17]。DSSM 等基于嵌入的模型被广泛用于候选召回 [7]，而 DeepFM、DCN、DIN 等排序模型则面向特征交互建模而发展 [6, 23, 34]。多阶段级联在实践中虽然有效，但召回阶段遗漏的商品无法被下游排序恢复，且各阶段目标不一致使得查询-商品相关性难以跨阶段一致地保持。

### 2.2 面向电商搜索的生成式检索

生成式检索将候选召回表述为条件生成任务，其中商品或文档由离散标识符序列表示，并由查询或用户上下文生成 [24, 20, 18]。在电商搜索中，生成的商品必须满足查询-商品相关性和商品属性约束，同时兼顾实际业务目标 [15, 19, 25, 5]。

近期研究从系统与建模两个视角探讨了生成式检索。LLMGR 将基于 LLM 的生成式检索应用于工业搜索，并引入外部知识以改进 Alipay Search 中的检索 [26]。OneSearch 提出了统一的生成式搜索框架，OneSearch-V2 则进一步研究了潜空间推理与自蒸馏 [1, 2]。OneRetrieval 将多条电商检索分支整合进一个可编辑的生成式模型 [33]。另一类工作专注于构建信息丰富的语义标识符。GSID 为商品学习生成式语义表示，CAT-ID2 和 HierGR 则将类目或层级结构融入标识符学习 [27, 13, 31]。CQ-SID 进一步研究了带专家引导强化学习的受限语义标识符学习 [35]。查询与意图信息也被引入生成式检索：TSGR 研究了面向淘宝搜索的查询感知语义表示 [30]，CaLIR 研究了面向生成式检索的类目引导的潜在意图推理 [32]。这些方法共同用商品、查询或意图信息丰富了标识符空间或生成过程。这类工作引出一个问题：此类相关性信号应如何在后续训练与优化中被保留。

查询级监督、相关性学习与偏好优化为改进生成式检索提供了互补途径。合成查询方法可以增强查询-商品监督，尤其适用于交互数据有限的商品 [11]。强化学习也被用于改进电商搜索中的查询-商品相关性建模 [28]。偏好优化方法利用点击等行为信号，使生成结果与用户偏好对齐 [9]。综合来看，先前工作已从系统统一、语义标识符构建、查询级监督和偏好优化等方面推进了生成式检索。这些组件在检索流程中扮演不同角色：SID 空间定义生成目标，监督训练学习查询到 SID 的映射，偏好优化决定更偏好哪些生成商品。若缺乏共享的相关性准则，后一阶段可能会削弱流程早期建立的查询意图信号。

ICEGR 通过将查询-商品相关性作为 SID 构建、监督训练与偏好优化的共享准则，来解决这一流程级一致性问题。它将查询诱导的商品关系和历史查询语义融入 SID 构建，用多粒度合成查询增强查询到 SID 的监督，并用相关性校准的比较约束业务偏好优化。

## 3. 方法

本节介绍 ICEGR 框架，如图2所示。该框架由三个顺序执行的阶段组成：IA-SID（第3.1节）、SQE-SFT（第3.2节）和 RCPO（第3.3节）。

### 3.1 意图感知的 SID 构建

在生成式检索中，SID 定义了目标空间并塑造查询到 SID 的学习。现有 SID 通常依赖标题、描述、类目等静态商品语义。这些信号虽然保留了商品属性，但并未显式捕捉用户交互中的搜索意图或行为关系。为弥补这一局限，我们提出意图感知的 SID 构建（IA-SID），在将表示量化为层级 SID 之前，整合查询-商品相关性、共点击关系以及每个商品的历史查询意图画像。

#### 3.1.1 查询-商品嵌入对齐

构建与搜索对齐的 SID 需要一个能刻画行为层面查询-商品相关性的表示空间。然而，通用文本编码器面向开放域语义相似度优化，因此并未显式建模电商点击日志中隐含的相关性信号。

我们利用历史查询-商品点击日志，将通用编码器 $f_{\Theta}$ 适配到电商搜索。给定一个被点击的样本对 $(q, i^{+})$ 和一组负样本集合 $\mathcal{N}_{q}$，我们优化 InfoNCE 目标 [21]

$$
\mathcal{L}_{\mathrm{InfoNCE}}=-\sum_{(q,i^{+})\in\mathcal{D}}\log\frac{\exp\!\left(\operatorname{sim}(\mathbf{e}_{q},\mathbf{x}_{i^{+}})/\tau_{\mathrm{emb}}\right)}{\displaystyle\sum_{i\in\{i^{+}\}\cup\mathcal{N}_{q}}\exp\!\left(\operatorname{sim}(\mathbf{e}_{q},\mathbf{x}_{i})/\tau_{\mathrm{emb}}\right)}.
\tag{1}
$$

该目标将每个查询拉近其被点击的商品，并推离负样本。所得嵌入为 $\mathbf{e}_{q}=f_{\Theta}(q)$ 和 $\mathbf{x}_{i}=f_{\Theta}(t_{i})$，其中 $t_{i}$ 表示与商品 $i$ 相关的文本。这些嵌入为后续的关系建模与意图融合模块所共享。

#### 3.1.2 意图引导的商品关系建模

查询-商品对比学习能捕捉查询与商品之间的成对相关性，但并未显式建模由共享查询诱导的商品间关系。为解决这一局限，我们基于共享查询的点击行为构建加权商品关系图 $G$，从而将这些行为关系注入商品表示。

令 $\mathcal{C}_{q}$ 表示在查询 $q$ 下被点击的商品集合，$n_{i,q}$ 表示商品 $i$ 在查询 $q$ 下获得的累计点击次数。我们将商品 $i$ 与 $j$ 之间的边权定义为

$$
w_{ij}=\sum_{q\in\mathcal{Q}:\,i,j\in\mathcal{C}_{q}}\sqrt{n_{i,q}\,n_{j,q}}.
\tag{2}
$$

几何平均平衡了每个商品对的点击证据，而对查询求和则累积了商品通过共享查询相关联的证据。令 $\mathbf{A}_{ij}=w_{ij}$，并定义对称归一化的邻接矩阵为 $\widehat{\mathbf{A}}=\mathbf{D}^{-\frac{1}{2}}\mathbf{A}\mathbf{D}^{-\frac{1}{2}}$，其中 $\mathbf{D}$ 是对应于 $\mathbf{A}$ 的度矩阵。

给定与搜索对齐的商品表示 $\mathbf{X}$，我们执行带语义锚定的多步图传播：

$$
\mathbf{H}^{(0)}=\mathbf{X},\qquad \mathbf{H}^{(t)}=(1-\alpha_{g})\,\widehat{\mathbf{A}}\mathbf{H}^{(t-1)}+\alpha_{g}\mathbf{X},\qquad t=1,\ldots,T.
\tag{3}
$$

邻域传播注入了由共享查询诱导的高阶商品关系，而残差项则保持与原始搜索对齐表示的一致性。因此，$\mathbf{h}_{i}$ 表示被共享查询诱导的共点击结构所增强的商品表示。

#### 3.1.3 意图增强的商品表示融合

查询诱导的行为图通过商品间关系注入搜索信号。作为补充，我们用每个商品的历史被点击查询来刻画其查询意图画像（query-intent profile），因为单个商品可能匹配多种搜索意图。

对于商品 $i$，我们保留有效查询集合 $\mathcal{Q}_{i}=\left\{q\in\mathcal{Q}\mid n_{i,q}\geq\delta_{c}\right\}$，并按经验点击频率聚合相应的查询嵌入：

$$
a_{i,q}=\frac{n_{i,q}}{\displaystyle\sum_{q^{\prime}\in\mathcal{Q}_{i}}n_{i,q^{\prime}}},\qquad \mathbf{z}_{i}=\sum_{q\in\mathcal{Q}_{i}}a_{i,q}\,\mathbf{e}_{q}.
\tag{4}
$$

这里 $\mathbf{e}_{q}$ 表示查询 $q$ 的对齐表示。所得向量 $\mathbf{z}_{i}$ 可视为商品 $i$ 的查询意图画像，概括了与该商品点击相关的主要历史查询表达及其相对频率。

由于历史点击证据在不同商品间差异很大，我们使用置信度门控的残差融合来控制查询意图注入的强度：

$$
\widetilde{\mathbf{x}}_{i}=\frac{\mathbf{h}_{i}+\lambda\eta_{i}\mathbf{z}_{i}}{\left\|\mathbf{h}_{i}+\lambda\eta_{i}\mathbf{z}_{i}\right\|_{2}},\qquad \eta_{i}=\min\!\left(1,\frac{\log\!\left(\lvert\mathcal{Q}_{i}\rvert+1\right)}{\log\!\left(Q_{\max}+1\right)}\right).
\tag{5}
$$

这里 $\lambda$ 控制融合强度，$\eta_{i}$ 是由保留查询数量导出的商品级置信度门，$Q_{\max}$ 是该门的饱和阈值。该门防止稀疏或带噪的查询画像主导商品表示，同时允许拥有充分历史查询-点击证据的商品获得更强的意图信号。

我们使用 $L$ 级残差 K-means 量化器（RQ-KMeans）[16] 对增强表示 $\widetilde{\mathbf{x}}_{i}$ 进行量化，并将各级码字拼接构成商品 $i$ 的层级 SID $\mathbf{s}_{i}=\left(c_{i}^{(1)},c_{i}^{(2)},\ldots,c_{i}^{(L)}\right)$。该过程将每个商品映射为一个编码了搜索意图结构的 SID 序列。SID 作为生成模型的预测目标，使查询到 SID 的学习得以在一个与搜索行为更紧密对齐的目标空间中进行。

### 3.2 合成查询增强的统一 SFT

即使 SID 编码了查询意图，生成模型仍须学会将自然语言查询映射到相应的 SID。SFT 的各种表述将商品-SID 关联学习与查询对齐结合在一起。商品到 SID 与 SID 到商品的任务让模型学会将目录条目与 SID 关联起来，但其输入与推理时遇到的自然语言查询不同。直接在交互日志上进行查询到 SID 的训练与目标任务更对齐，但因为此类查询稀疏，对长尾商品的覆盖有限。因此我们提出合成查询增强的统一 SFT（SQE-SFT），它从商品目录信息生成多粒度合成查询，并在统一的查询到 SID 目标下将其与真实交互查询联合训练。这一设计在保持与观测查询分布对齐的同时扩展了监督覆盖。

#### 3.2.1 多粒度合成查询构建

给定商品 $i$ 的目录文本 $t_{i}$，我们首先使用商品语义解析模型提取其结构化语义：

$$
\mathcal{E}(t_{i})=\left(\mathrm{spu}_{i},\mathcal{B}_{i},\mathcal{P}_{i},\mathcal{R}_{i},\mathcal{M}_{i},\mathcal{A}_{i}\right).
\tag{6}
$$

这里 $\mathrm{spu}_{i}$ 表示归一化的商品名称；$\mathcal{B}_{i}$、$\mathcal{P}_{i}$、$\mathcal{R}_{i}$、$\mathcal{M}_{i}$ 分别表示品牌、类目、系列和型号的集合；$\mathcal{A}_{i}$ 表示属性集合。我们用这些字段来建模两类常见检索行为：通过标识或层级语义指定目标商品，以及用显式属性约束商品查询。

**实体指定查询（Entity-Specifying Queries）。** 用户可能直接通过名称指定目标商品，或利用其类目、品牌、系列或型号收窄候选商品范围。为覆盖这些行为，我们从三个互补视角构建查询：商品标识（identity）、类目范围（category scope）和商品变体（variant），记为 $\mathcal{G}=\{\mathrm{id},\mathrm{cat},\mathrm{var}\}$。对于每个视角 $g\in\mathcal{G}$，我们将商品 $i$ 的相应字段组合为候选语义集合 $\mathcal{V}_{i}^{g}$。所得实体指定查询集合为

$$
\mathcal{Q}_{i}^{\mathrm{ent}}=\left\{\tau_{\mathrm{ent}}(g,v)\mid g\in\mathcal{G},\;v\in\mathcal{V}_{i}^{g}\right\},
\tag{7}
$$

其中 $\tau_{\mathrm{ent}}(g,\cdot)$ 将视角 $g$ 下的结构化语义组合表述（verbalize）为自然语言查询。

**属性约束查询（Attribute-Constraining Queries）。** 为覆盖由商品属性或使用需求驱动的检索意图，我们首先剔除 $\mathcal{A}_{i}$ 中在整个目录中出现频率过低的属性，得到过滤后的属性集 $\mathcal{A}_{i}^{+}$。然后我们以归一化商品名 $\mathrm{spu}_{i}$ 作为查询锚点以保持其与目标商品的关联，并采样属性子集来构建属性约束查询：

$$
\mathcal{Q}_{i}^{\mathrm{attr}}=\left\{\tau_{\mathrm{attr}}\left(\mathrm{spu}_{i},\widetilde{\mathcal{A}}_{i}\right)\;\middle|\;\widetilde{\mathcal{A}}_{i}\in\operatorname{Sample}\!\left(\mathcal{A}_{i}^{+}\right)\right\}.
\tag{8}
$$

这里 $\operatorname{Sample}(\cdot)$ 生成大小不一的非空属性子集，$\tau_{\mathrm{attr}}(\cdot,\cdot)$ 将商品名与所选属性表述为自然语言查询。

最后，我们合并两个查询集合，并经过归一化、去重与有效性过滤，得到商品 $i$ 的合成查询集合 $\mathcal{Q}_{i}^{\mathrm{syn}}$。表1展示了从一个真实目录商品生成的示例。

表1：SQE-SFT 中多粒度查询合成的示例。

| 阶段 / 查询类型 | 示例输出 |
| --- | --- |
| 原始标题（$t_{i}$） | JBL TUNE BUDS2 二代琉璃豆真无线蓝牙耳机，入耳式设计，通话降噪，双耳传输，兼容华为苹果手机，升级款白色 |
| 归一化语义（$\mathcal{E}(t_{i})$） | JBL TUNE BUDS2 二代琉璃豆白色真无线耳机，入耳式设计，通话降噪，双耳传输，兼容华为苹果手机 |
| 商品标识（$\mathcal{Q}_{i}^{\mathrm{id}}$） | (i) JBL TUNE BUDS2；(ii) JBL TUNE BUDS2 真无线耳机；(iii) JBL 二代琉璃豆耳机 |
| 类目范围（$\mathcal{Q}_{i}^{\mathrm{cat}}$） | JBL 真无线耳机 |
| 系列/变体（$\mathcal{Q}_{i}^{\mathrm{var}}$） | JBL TUNE BUDS |
| 属性约束（$\mathcal{Q}_{i}^{\mathrm{attr}}$） | (i) JBL TUNE BUDS2 入耳式耳机；(ii) JBL TUNE BUDS2 带通话降噪和双耳传输的耳机；(iii) 兼容华为苹果手机的白色 JBL TUNE BUDS2 耳机 |

#### 3.2.2 统一两阶段 SFT

令 $\mathcal{Q}_{i}^{\mathrm{real}}$ 表示与商品 $i$ 的点击或转化相关联的真实用户查询集合。对任意 $q\in\mathcal{Q}_{i}^{\mathrm{syn}}\cup\mathcal{Q}_{i}^{\mathrm{real}}$，我们使用如下统一的指令模板：

"给定用户请求'$q$'，推荐最相关的 SID。"

期望的响应是与 $q$ 关联商品的 SID $\mathbf{s}_{i}$。在第一阶段，我们在合成查询到 SID 的样本上训练模型，学习从商品语义到 SID 的全目录映射。该阶段为长尾商品和基于属性的意图提供直接监督。然后我们在真实查询到 SID 的样本对上微调模型，使其与真实用户查询分布及观测交互对齐。两个阶段优化同一个查询到 SID 的目标，所得模型作为后续偏好优化的初始化。

### 3.3 相关性校准的偏好优化

尽管 SFT 让模型学会将查询映射到相关 SID，但它并未显式建模用户行为与商业结果所反映的业务偏好。在没有相关性约束的情况下直接优化这些业务信号，可能偏向与查询意图不匹配但具商业价值的商品。因此我们提出相关性校准的偏好优化（RCPO），将业务偏好学习与相关性感知的偏好构建相结合。RCPO 将偏好比较限制在语义相关的候选之内，用行为与商业价值信号精化其排序，并使每次偏好更新的权重自适应于所得的成对综合得分差距。这一设计在保持查询意图一致的同时实现业务感知的偏好学习。

#### 3.3.1 多信号偏好对构建

给定查询 $q$，对生成式检索模型执行束搜索得到候选 SID $\mathcal{R}_{q}=\{\hat{\mathbf{s}}_{k}\}_{k=1}^{N_{\mathrm{beam}}}$。令 $\mathcal{H}_{q}^{\mathrm{sid}}$ 表示交互日志中在查询 $q$ 下获得正反馈的 SID。我们通过将每个候选 SID $\hat{\mathbf{s}}$ 映射到其自然语言描述 $\mathbf{t}(\hat{\mathbf{s}})$，并用微调过的打分器 $g_{\phi}$ 对查询-描述对打分来度量语义相关性：$\mathrm{SRS}(q,\hat{\mathbf{s}})=g_{\phi}\bigl(q,\mathbf{t}(\hat{\mathbf{s}})\bigr)$。

由于语义相关性本身无法捕捉用户在多个相关商品之间的偏好，我们进一步从交互日志估计平滑业务偏好（Smoothed Business Preference, SBP）。SBP 聚合点击、支付与交易价值信号，并采用类目级均值平滑以缓解稀疏交互带来的估计偏差。其完整定义见附录 A。

基于 SRS 与 SBP，我们用两种互补策略构建偏好对。语义相关性锚定（Semantic Relevance Anchoring, SRA）依据语义相关性确定偏好方向，而业务偏好精化（Business Preference Refinement, BPR）则在语义可比的候选之间精化业务偏好排序。对于每个具有历史正交互的查询，SRA 从 $\mathcal{H}_{q}^{\mathrm{sid}}$ 中选取 SRS 最高的 SID 作为偏好（preferred）SID，并从 $\mathcal{R}_{q}\setminus\mathcal{H}_{q}^{\mathrm{sid}}$ 中选取一个 SRS 相对较低的候选作为拒绝（rejected）SID。BPR 则比较满足相关性约束且 SRS 差距不超过预设边界的同查询候选，偏好其中 SBP 更高者。所得偏好对构成离线数据集 $\mathcal{D}_{\mathrm{pref}}=\{(q,\hat{\mathbf{s}}^{+},\hat{\mathbf{s}}^{-})\}$。

#### 3.3.2 RCPO 目标

给定 $\mathcal{D}_{\mathrm{pref}}$，我们从 SQE-SFT 模型初始化策略模型 $\pi_{\theta}$，并冻结一份副本作为参考模型 $\pi_{\mathrm{ref}}$。尽管相关性约束的偏好构建能产生语义可比的偏好对，它并未量化离线证据对每个对内所指派偏好的支持强度。因此我们对每个查询-SID 对计算离线综合得分，即其 SRS 与 SBP（均归一化到 $[0,1]$）的等权平均。

对每个偏好对，令 $r^{+}$ 与 $r^{-}$ 分别表示偏好与拒绝 SID 的综合得分，并定义带符号的校准差距为 $\Delta_{r}=r^{+}-r^{-}$。我们用该差距确定每个对特有的 RCPO 系数：

$$
\beta_{\mathrm{eff}}=\beta\cdot\operatorname{clip}\!\left(1+\log\frac{2-\Delta_{r}}{2+\Delta_{r}},\,0.2,\,1.8\right).
\tag{9}
$$

这里 $\beta$ 是基础温度。该系数随 $\Delta_{r}$ 增大而减小：较大的正向差距表示离线证据对偏好 SID 有强支持，因此我们降低此类对的权重以避免冗余优化。较小的差距获得较大的系数，从而增大较不确定的偏好对在 RCPO 目标中的影响。若任一得分不可得，我们将该乘性校准因子设为 1。

对偏好对 $(\hat{\mathbf{s}}^{+},\hat{\mathbf{s}}^{-})$，我们定义策略模型与参考模型之间的相对偏好差为

$$
\Delta_{\theta}=\log\frac{\pi_{\theta}(\hat{\mathbf{s}}^{+}\mid x_{q})}{\pi_{\mathrm{ref}}(\hat{\mathbf{s}}^{+}\mid x_{q})}-\log\frac{\pi_{\theta}(\hat{\mathbf{s}}^{-}\mid x_{q})}{\pi_{\mathrm{ref}}(\hat{\mathbf{s}}^{-}\mid x_{q})}.
\tag{10}
$$

然后我们优化如下带长度归一化辅助 SFT 项（作用于偏好 SID）的 RCPO 目标：

$$
\mathcal{L}=-\mathbb{E}_{\mathcal{D}_{\mathrm{pref}}}\Biggl[\log\sigma\!\left(\beta_{\mathrm{eff}}\Delta_{\theta}\right)+\frac{\eta_{\mathrm{sft}}}{\lvert\hat{\mathbf{s}}^{+}\rvert}\log\pi_{\theta}(\hat{\mathbf{s}}^{+}\mid x_{q})\Biggr].
\tag{11}
$$

这里 $\eta_{\mathrm{sft}}$ 控制辅助 SFT 项的权重，$x_{q}$ 表示查询输入。相关性约束的偏好构建与 RCPO 相结合，使模型在学习业务偏好的同时保持与查询意图的对齐。

## 4. 实验

### 4.1. 实验设置

**数据集。** 我们按时间顺序切分近三个月的百度电商搜索日志，保留最后一天用于测试，其余数据用于训练与验证。日志覆盖 12.8M 个不同查询和 14.6M 个不同商品。所有训练产物（包括 SID 构建与样本生成）均仅从测试前数据导出，以防止时间泄露。

**离线对比与消融设置。** 我们以生产中的生成式检索模型 ProdGR 作为基线。由于先前的工业生成式检索系统均使用专有数据与基础设施进行训练和服务，其结果与我们的结果不直接可比。因此我们在相同的 0.5B 骨干、RQ-KMeans SID 构建、数据切分、SFT/RL 训练设置和束搜索配置下比较 ProdGR 与 ICEGR；每项消融仅改变所研究的组件。

**评估指标。** 对每个测试查询，我们以用户交互过的商品集合为真实标注（ground truth），用 Recall@$K$、MRR@$K$ 与 NDCG@$K$ 评估检索列表。相关性为二值，MRR 使用第一个相关商品的排名，NDCG 以截断于 $K$ 的理想排序归一化，所有指标均在查询上做宏平均。

**实现细节。** 我们使用一个 8B 的电商适配嵌入模型，产出 256 维融合表示。在 IA-SID 中，我们执行 $T=3$ 步图传播，$\alpha_{g}=0.2$。商品 SID 由一个三级 RQ-KMeans 量化器生成，每级 1,024 个码字，其中 $Q_{\max}=50$、$\lambda=1.0$。偏好对使用一个 0.6B 的语义相关性模型构建。

### 4.2. 离线性能

在离线评估中，我们从留出的最后一天用户搜索日志中采样 20,000 个不同查询，并以它们的被点击商品作为真实相关商品。ICEGR 与 ProdGR 在同一查询集上评估。ICEGR 在两个截断位置均稳定优于 ProdGR（表2）。在 $K=20$ 处，Recall@20、MRR@20 与 NDCG@20 分别提升 21.7%、26.6% 与 26.6%；在 $K=50$ 处的相应提升为 16.1%、26.0% 与 23.4%。MRR 与 NDCG 的更大提升表明，ICEGR 除提高相关商品的覆盖率外，还改进了检索列表顶部相关商品的排序。

表2：累积式组件叠加下的主要离线检索结果。

| 方法 | Recall@20 | MRR@20 | NDCG@20 | Recall@50 | MRR@50 | NDCG@50 |
| --- | --- | --- | --- | --- | --- | --- |
| ProdGR | 0.4247 | 0.1982 | 0.2318 | 0.5606 | 0.2022 | 0.2623 |
| +SQE-SFT | 0.4954 | 0.2381 | 0.2784 | 0.6255 | 0.2418 | 0.3079 |
| +IA-SID | 0.5006 | 0.2433 | 0.2841 | 0.6312 | 0.2471 | 0.3136 |
| +RCPO（ICEGR） | 0.5169 | 0.2509 | 0.2935 | 0.6506 | 0.2548 | 0.3237 |

### 4.3. 消融研究

**各组件总体贡献。** 我们首先量化 ICEGR 三个保持查询意图阶段的贡献。在表2的组件累加协议下，SQE-SFT 贡献了累计 Recall@20 增益的 76.7%，其次是 RCPO（17.7%）和 IA-SID（5.6%）。这一分解表明，扩展的查询到 SID 监督是所观察到改进的主要来源。SQE-SFT 在不同商品流行度下的进一步分析见第4.4.1节。

**IA-SID。** 将搜索对齐编码器替换为基础编码器会使 Recall@20 从 0.5169 降至 0.4179，低于 ProdGR 的 0.4247（表3），表明通用语义表示本身不足以捕捉面向检索的查询-商品相关性。两个 IA-SID 机制效果各异：移除意图引导的商品关系建模使 Recall@20 下降 4.35%，而移除意图增强的商品表示融合使 MRR@20 下降 7.21%。前者更多贡献相关商品覆盖，后者更强地影响第一个相关商品的排名。同时移除两个机制导致 Recall@20 下降 13.60%，超过二者单独下降之和 9.13%，表明两种搜索信号互补。

表3：截断位置 20 处 IA-SID 的消融研究。

| 方法 | Recall@20 | MRR@20 | NDCG@20 |
| --- | --- | --- | --- |
| ICEGR | 0.5169 | 0.2509 | 0.2935 |
| w/o 查询-商品对齐 | 0.4179 (-19.15%) | 0.2154 (-14.15%) | 0.2449 (-16.56%) |
| w/o 关系建模 | 0.4944 (-4.35%) | 0.2490 (-0.75%) | 0.2857 (-2.67%) |
| w/o 表示融合 | 0.4922 (-4.78%) | 0.2328 (-7.21%) | 0.2750 (-6.30%) |
| w/o 两个模块 | 0.4466 (-13.60%) | 0.2314 (-7.78%) | 0.2605 (-11.24%) |

**融合权重敏感性。** 随着全局融合权重 $\lambda$ 增大，验证集性能呈倒 U 型趋势，在 $\lambda=1.0$ 处达到峰值（图3）。我们在后续所有实验中采用该值。该趋势表明有效的融合必须在查询意图信号与商品语义之间取得平衡。

图3：查询意图融合权重 $\lambda$ 的敏感性分析。图中展示了随融合权重变化时 Recall@20、MRR@20、NDCG@20 相对 $\lambda=0$ 的相对增益（%）变化曲线，峰值位于 $\lambda=1.0$。

**RCPO。** 我们评估 RCPO 能否在学习业务偏好的同时保持查询相关性。仅用偏好 SID 做 SFT 训练使 Recall@20 相对 ICEGR 下降 2.46%，表明改进并非仅仅来自添加高质量的查询到 SID 样本对（表4）。反转偏好标签使 MRR@20 降至 0.1719，低于 ProdGR 的 0.1982，证实了偏好方向的重要性。在偏好对与设置固定的条件下，用固定 $\beta$ 替换逐对 $\beta_{\mathrm{eff}}$ 使 NDCG@20 下降 7.36%，而移除 SBP 对 MRR@20 的影响最大（6.26%）。两个机制角色互补：逐对标定提升头部结果的排序质量，而 SBP 更强地影响第一个相关商品的排名。

表4：截断位置 20 处 RCPO 的消融与对照研究。

| 方法 | Recall@20 | MRR@20 | NDCG@20 | 相关性 |
| --- | --- | --- | --- | --- |
| ICEGR | 0.5169 | 0.2509 | 0.2935 | 0.87 |
| w/o 动态 $\beta_{\mathrm{eff}}$ | 0.4937 (-4.49%) | 0.2372 (-5.46%) | 0.2719 (-7.36%) | 0.58 (-33.3%) |
| 仅偏好 SFT（Preferred-only SFT） | 0.5042 (-2.46%) | 0.2420 (-3.55%) | 0.2858 (-2.62%) | – |
| 偏好标签反转 | 0.4192 (-18.90%) | 0.1719 (-31.49%) | 0.2152 (-26.68%) | – |
| w/o 业务偏好（SBP） | 0.5016 (-2.96%) | 0.2352 (-6.26%) | 0.2798 (-4.67%) | – |

**人工相关性评估。** 我们对随机抽样的 50 个测试查询的匿名化 top-5 结果进行了盲评。相关性比率从 ICEGR 的 0.87 降至固定 $\beta$ 变体的 0.58。附录 B 中的代表性案例表明，固定 $\beta$ 变体会把业务偏好但与查询无关的商品推到头部排名，展示了逐对 DPO 标定在保持查询-商品相关性中的作用。

### 4.4. 进一步分析

我们研究查询到 SID 监督与骨干容量如何影响 ICEGR 的检索质量。

#### 4.4.1. 合成监督使低热度商品受益

为分离 SQE-SFT 的贡献，我们将 ICEGR 与一个用 ProdGR 的 SFT 流程替换 SQE-SFT、其余保持 IA-SID 与 RCPO 不变的变体进行比较。我们按目标商品在训练集中的点击频次将测试查询-商品对划分为 Tail、Low、Mid 与 Hot 四组，阈值见附录 C。

如图4(a) 所示，SQE-SFT 在四组中均提升性能，其中 Low 组的 Recall@20 增益最大（36.2%），其次为 Tail（21.2%），Mid 与 Hot 组增益逐渐变小。MRR@20 与 NDCG@20 呈类似趋势（附录表7）。从 Low 到 Hot 的递减与合成监督的边际收益递减一致：Mid 与 Hot 商品已获得大量基于交互的监督，留给合成查询的提升空间较小。

有趣的是，Low 组尽管拥有比 Tail 组更密集的真实监督，却取得了更大的增益。这一结果表明合成监督与真实监督是互补的。Tail 商品更依赖合成查询的覆盖，但其增益可能受合成查询与真实用户查询分布失配的限制。Low 商品既受益于广泛的合成覆盖，又有足够多的真实交互数据以保持与真实用户查询的对齐，因而获得最大提升。这些发现支持了统一的两阶段 SFT 设计。

#### 4.4.2. 缩放定律验证

我们固定训练数据、SID 码本与解码配置，仅在 0.5B、1.5B、3B 与 7B 之间改变骨干规模。对每个规模，我们报告五个 SFT epoch 中验证性能最佳的检查点。图4(b) 显示，Recall@20 从 0.5B 的 0.5043 单调上升至 7B 的 0.5334。从 0.5B 到 7B，MRR@20 与 NDCG@20 分别提升 5.1% 与 5.8%。这些结果验证了 LLM 缩放定律（Scaling Law）在 ICEGR 上的可用性，且在所评估范围内没有明显的性能平台期。

综合来看，这些结果表明监督覆盖与骨干容量是互补的改进来源。SQE-SFT 对低热度商品尤为有益，而骨干扩容在评估范围内持续提升检索质量。因此我们在线上部署 0.5B 模型，它提供了强力的准确性-效率折中。

图4：ICEGR 的进一步分析：(a) SQE-SFT 在不同商品流行度分组下的增益；(b) Recall@20 随骨干规模的变化。

### 4.5. 在线 A/B 测试

我们将 0.5B 参数的 ICEGR 模型部署到百度电商搜索系统，并在 20% 的线上流量上进行了为期两周的 A/B 测试。对照组使用现有 MCA 系统，实验组使用叠加了 ICEGR 的同一生产流程。ICEGR 的 top-$k$ 结果以固定数量直接插入最终展示列表，形成一条从用户查询到商品展示的端到端生成式检索通路。其余候选进入现有的相关性过滤与排序流程。出于保密原因，我们仅报告相对对照组的相对提升。

如表5所示，ICEGR 使 Show PV、CTR 与 Long-click Rate 分别提升 8.84%、3.52% 与 5.27%。商品落地页 PV（Product Landing Page PV）也提升 8.75%，而订单量（Order Volume）与 GMV 分别增长 15.96% 与 7.53%。这些结果表明 ICEGR 在改进点击效率、点击后参与度与业务结果的同时，扩大了商品曝光。

表5：ICEGR 相对生产 MCA 基线的在线 A/B 测试结果。

| 维度 | 指标 | 相对提升 |
| --- | --- | --- |
| 曝光 | Show PV | +8.84% |
| 用户参与 | CTR | +3.52% |
| 用户参与 | Long-click Rate | +5.27% |
| 用户参与 | Product Landing Page PV | +8.75% |
| 业务转化 | Order Volume | +15.96% |
| 业务转化 | GMV | +7.53% |

注：Show PV 指百度搜索结果页中商品的曝光数量。CTR 为点击率，由商品点击数除以 Show PV 计算。Long-click Rate 基于长按点击行为度量点击后参与度。Product Landing Page PV 指用户从百度搜索结果跳转到商品落地页（可能属于京东、淘宝或其他电商平台）时产生的页面浏览量。

### 4.6. 在线服务效率

自回归解码的计算成本是大规模部署生成式检索系统的主要挑战。ICEGR 运行在由 67 块 NVIDIA L20 GPU 组成的集群上，在束大小为 50 的情况下，可承载 2,200 QPS 的峰值负载，平均延迟 156 ms，满足当前生产环境的吞吐与延迟要求。

## 5. 结论

我们提出了 ICEGR，一个面向电商搜索的意图一致生成式检索框架。其核心设计原则是在整个检索流程中保持查询意图，覆盖语义ID构建、查询到 SID 监督与偏好优化。ICEGR 相对生产生成式检索基线将 Recall@20 提升 21.7%、NDCG@20 提升 26.6%。我们的分析进一步表明，改进查询到 SID 监督对低热度商品尤为重要，而在评估范围内增大骨干容量可带来持续的额外增益。RCPO 通过仅在语义相关候选之间引入业务感知偏好、并使优化强度自适应于偏好差距，对这些改进形成补充。在线 A/B 测试证实，ICEGR 在满足生产延迟与吞吐要求的同时，提升了 CTR、订单量与 GMV。

## 附录 A 平滑业务偏好（SBP）的构建

对于 $(q,s)$ 的第 $i$ 次展示（impression），点击、支付与交易价值分别刻画吸引力、转化与货币价值。为缓解高价值离群点的影响，我们对 $g_{i}$ 做对数缩放并截断：

$$
\bar{g}_{i}=\min\!\left(\frac{\log(1+g_{i})}{\log(1+g_{0.95})},1\right),
\tag{12}
$$

其中 $g_{i}$ 为交易价值（若无支付发生则设为零），$g_{0.95}$ 为训练集中非零交易价值的第 95 百分位数。我们定义展示级业务奖励为

$$
r_{i}=\lambda_{\mathrm{clk}}y_{i}^{\mathrm{clk}}+\lambda_{\mathrm{pay}}y_{i}^{\mathrm{pay}}+\lambda_{\mathrm{gmv}}\bar{g}_{i},
\tag{13}
$$

其中 $y_{i}^{\mathrm{clk}}$ 与 $y_{i}^{\mathrm{pay}}$ 分别为二值的点击与支付指示变量，三个权重之和为 1。

对每个查询-商品对 $(q,s)$，我们用类目级平滑聚合其历史奖励：

$$
\mathrm{SBP}(q,s)=\frac{\displaystyle\sum_{i\in\mathcal{I}_{q,s}}r_{i}+\kappa\,\mu_{c(s)}}{\lvert\mathcal{I}_{q,s}\rvert+\kappa},
\tag{14}
$$

其中 $\mathcal{I}_{q,s}$ 为 $(q,s)$ 的历史展示集合，$c(s)$ 表示 $s$ 的类目，$\mu_{c(s)}$ 为该类目在训练集上的平均奖励。参数 $\kappa$ 控制先验强度，并降低稀疏样本对的方差。我们设 $\lambda_{\mathrm{clk}}=0.2$、$\lambda_{\mathrm{pay}}=0.3$、$\lambda_{\mathrm{gmv}}=0.5$、$\kappa=10$。这些值基于下游验证性能选定。

## 附录 B DPO 标定的定性分析

ICEGR 在 32 个查询上取得了更高的 top-5 相关性比率，在 18 个查询上持平。表6给出两个例子。对"欧米茄维修中心"，固定 $\beta$ 变体给出的是欧米茄手表，而 ICEGR 返回五条维修服务。对"宗申封闭式电动三轮车哪个好"，固定 $\beta$ 变体混合了汽油与电动三轮车，而 ICEGR 只返回封闭式电动三轮车。

表6：两个代表性查询的 top-5 结果。✓ 与 ✗ 表示人工标注的相关性。标题已翻译并缩写。

| 排名 | 固定 $\beta$ DPO | ICEGR |
| --- | --- | --- |
| (a) 查询："欧米茄维修中心" | | |
| 1 | 海马系列 300 手表 ✗ | 维修：清洗与换电池 ✓ |
| 2 | De Ville 手表 ✗ | 维修：机芯上油 ✓ |
| 3 | 维修：清洗与换电池 ✓ | 维修：更换电池 ✓ |
| 4 | De Ville 手表（变体）✗ | 维修：预约服务 ✓ |
| 5 | 星座系列 18K 手表 ✗ | 维修：翻新与抛光 ✓ |
| (b) 查询："宗申封闭式电动三轮车哪个好" | | |
| 1 | 封闭式电动，5 门 ✓ | 封闭式电动，5 门 ✓ |
| 2 | 汽油自卸货车 ✗ | 封闭式电动，4 门 ✓ |
| 3 | 汽油平板座椅三轮车 ✗ | 封闭式电动，60V 全封闭驾驶室 ✓ |
| 4 | 封闭式电动，4 门 ✓ | 封闭式电动，家用 ✓ |
| 5 | 汽油 200 水冷 ✗ | 封闭式电动，60V/1500W ✓ |

## 附录 C SQE-SFT 的流行度分层评估

对每个测试查询-商品对，令 $c_{i}$ 为目标商品在训练集中的点击次数。我们将样本对划分为四个互斥的分桶：Tail（$c_{i}\leq 2$）、Low（$2<c_{i}\leq 9$）、Mid（$9<c_{i}\leq 99$）与 Hot（$c_{i}>99$）。阈值仅使用训练集统计量。表7报告了带 SQE-SFT 的 ICEGR 在这些分桶上的表现。

表7：截断位置 20 处的流行度分层性能。括号内为相对标准 SFT 的相对增益。

| 分桶 | 样本对数 | Recall@20 | MRR@20 | NDCG@20 |
| --- | --- | --- | --- | --- |
| Tail | 6,126 | 0.2088 (+21.22%) | 0.0556 (+38.83%) | 0.0889 (+30.15%) |
| Low | 4,211 | 0.4531 (+36.17%) | 0.1449 (+54.84%) | 0.2125 (+46.01%) |
| Mid | 7,295 | 0.5934 (+11.31%) | 0.2098 (+17.39%) | 0.2947 (+14.79%) |
| Hot | 5,486 | 0.6864 (+4.78%) | 0.2818 (+10.33%) | 0.3727 (+7.89%) |

## 伦理声明

我们使用在机构授权下收集并处理的聚合、去标识化查询-商品交互日志。在分析之前，我们移除了原始用户标识符、设备标识符及其他可直接识别身份的字段。敏感的自由文本查询被排除在研究语料之外，对结果数据的访问仅限于经授权的研究人员。

由于基于点击、支付与 GMV 的信号可能放大流行度偏差与商业偏差，我们使用查询-商品相关性约束偏好构建，并分别针对长尾查询与低曝光商品评估性能。合成查询仅用于改进监督覆盖，并经过语义相关性、属性一致性与去重过滤。

在线评估是依照适用的审批与监控程序进行的受控 A/B 测试。该系统面向电商搜索，不用于对个人产生重大影响的决策。我们监控本文实际报告的指标，并为部署维护了回滚与应急响应程序。

## 参考文献

（以下参考文献保留英文原文，不翻译。正文中引用编号 [1]-[35] 与下列条目对应；HTML 版按作者-年份引用，PDF 版按编号引用。）

[1] Ben Chen, Xian Guo, Siyuan Wang, Zihan Liang, Yufei Ma, Yue Lv, Chenyi Lei, Yuqing DING, Wenwu Ou, Han Li, and Kun Gai. 2026. OneSearch: A Preliminary Exploration of the Unified End-to-End Generative Framework for E-commerce Search. In Forty-third International Conference on Machine Learning. Seoul, South Korea. https://openreview.net/forum?id=JKGgHY9FKa

[2] Ben Chen, Siyuan Wang, Yufei Ma, Zihan Liang, Xuxin Zhang, Yue Lv, Ying Yang, Huangyu Dai, Lingtao Mao, Tong Zhao, Zhipeng Qian, Xinyu Sun, Zhixin Zhai, Yang Zhao, Bochao Liu, Jingshan Lv, Xiao Liang, Hui Kong, Jing Chen, Han Li, Chenyi Lei, Wenwu Ou, and Kun Gai. 2026. OneSearch-V2: The Latent Reasoning Enhanced Self-distillation Generative Search Framework. arXiv:2603.24422 [cs.IR] https://arxiv.org/abs/2603.24422

[3] Zhiguo Chen, Guohao Sun, Yiming Qiu, Xingzhi Yao, Mingming Li, Huimu Wang, Yangqi Zhang, Songlin Wang, and Sulong Xu. 2026. RAD-DPO: Robust Adaptive Denoising Direct Preference Optimization for Generative Retrieval in E-commerce. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (Australia) (SIGIR '26). Association for Computing Machinery, New York, NY, USA, 4507–4512. doi:10.1145/3805712.3808512

[4] Sunhao Dai, Jiakai Tang, Jiahua Wu, Kun Wang, Yuxuan Zhu, Bingjun Chen, Bangyang Hong, Yu Zhao, Cong Fu, Kangle Wu, Yabo Ni, Anxiang Zeng, Wenjie Wang, Xu Chen, Jun Xu, and See-Kiong Ng. 2026. OnePiece: Bringing Context Engineering and Reasoning to Industrial Cascade Ranking System. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (Republic of Korea) (KDD '26). Association for Computing Machinery, New York, NY, USA, 7117–7128. doi:10.1145/3770855.3818456

[5] Songtao Fang, Zihao Xu, Shaowei Wei, Jin Zhang, and Zhuojun Wang. 2026. Generative Retrieval for E-commerce: Jointly Learning Embedding and Codebook with Same Product Cluster. In Proceedings of the ACM Web Conference 2026 (United Arab Emirates) (WWW '26). Association for Computing Machinery, New York, NY, USA, 8369–8372. doi:10.1145/3774904.3792862

[6] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. In Proceedings of the 26th International Joint Conference on Artificial Intelligence (Melbourne, Australia) (IJCAI'17). AAAI Press, Melbourne, Australia, 1725–1731.

[7] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM International Conference on Information & Knowledge Management (San Francisco, California, USA) (CIKM '13). Association for Computing Machinery, New York, NY, USA, 2333–2338. doi:10.1145/2505515.2505665

[8] Shuping Ji, Jianguo Yao, and Dong Zhang. 2026. Effective Offline LLM and DNN based Matching, Filtering and Ranking for Search Ads Retrieval in E-Commerce. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (Australia) (SIGIR '26). Association for Computing Machinery, New York, NY, USA, 4671–4676. doi:10.1145/3805712.3808470

[9] Mingming Li, Huimu Wang, Zuxu Chen, Guangtao Nie, Yiming Qiu, Guoyu Tang, Lin Liu, and Jingwei Zhuo. 2024. Generative Retrieval with Preference Optimization for E-commerce Search. arXiv:2407.19829 [cs.IR] https://arxiv.org/abs/2407.19829

[10] Sen Li, Fuyu Lv, Taiwei Jin, Guli Lin, Keping Yang, Xiaoyi Zeng, Xiao-Ming Wu, and Qianli Ma. 2021. Embedding-based Product Retrieval in Taobao Search. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (Virtual Event, Singapore) (KDD '21). Association for Computing Machinery, New York, NY, USA, 3181–3189. doi:10.1145/3447548.3467101

[11] Gui Ling, Weiyuan Li, Yue Jiang, Wenjun Peng, Xingxian Liu, Dongshuai Li, Fuyu Lv, Dan Ou, and Haihong Tang. 2026. Synthetic Data Powers Product Retrieval for Long-tail Knowledge-Intensive Queries in E-commerce Search. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (Australia) (SIGIR '26). Association for Computing Machinery, New York, NY, USA, 4748–4752. doi:10.1145/3805712.3808409

[12] Shichen Liu, Fei Xiao, Wenwu Ou, and Luo Si. 2017. Cascade Ranking for Operational E-commerce Search. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (Halifax, NS, Canada) (KDD '17). Association for Computing Machinery, New York, NY, USA, 1557–1565. doi:10.1145/3097983.3098011

[13] Xiaoyu Liu, Fuwei Zhang, Yiqing Wu, Xinyu Jia, Zenghua Xia, Fuzhen Zhuang, Zhao Zhang, Fei Jiang, and Wei Lin. 2026. CAT-ID2: Category-Tree Integrated Document Identifier Learning for Generative Retrieval In E-commerce. In Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining (USA) (WSDM '26). Association for Computing Machinery, New York, NY, USA, 426–435. doi:10.1145/3773966.3777925

[14] Zhiding Liu, Ben Chen, Mingyue Cheng, Enhong Chen, Li Li, Chenyi Lei, Wenwu Ou, Han Li, and Kun Gai. 2026. Towards Context-aware Reasoning-enhanced Generative Searching in E-commerce. In Proceedings of the ACM Web Conference 2026 (United Arab Emirates) (WWW '26). Association for Computing Machinery, New York, NY, USA, 6551–6561. doi:10.1145/3774904.3792562

[15] Ming Pang, Chunyuan Yuan, Xiaodong He, Zheng Fang, Donghao Xie, Fanyi Qu, Xue Jiang, Changping Peng, Zhangang Lin, Zheng Luo, and Jingping Shao. 2025. Generative Retrieval and Alignment Model: A New Paradigm for E-commerce Retrieval. In Companion Proceedings of the ACM on Web Conference 2025 (Sydney NSW, Australia) (WWW '25). Association for Computing Machinery, New York, NY, USA, 413–421. doi:10.1145/3701716.3715228

[16] Gustavo Penha, Edoardo D'Amico, Marco De Nadai, Enrico Palumbo, Alexandre Tamborrino, Ali Vardasbi, Max Lefarov, Shawn Lin, Timothy Heath, Francesco Fabbri, and Hugues Bouchard. 2025. Semantic IDs for Joint Generative Search and Recommendation. In Proceedings of the Nineteenth ACM Conference on Recommender Systems (RecSys '25). Association for Computing Machinery, New York, NY, USA, 1296–1301. doi:10.1145/3705328.3759300

[17] Jiarui Qin, Jiachen Zhu, Bo Chen, Zhirong Liu, Weiwen Liu, Ruiming Tang, Rui Zhang, Yong Yu, and Weinan Zhang. 2022. RankFlow: Joint Optimization of Multi-Stage Cascade Ranking Systems as Flows. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (Madrid, Spain) (SIGIR '22). Association for Computing Machinery, New York, NY, USA, 814–824. doi:10.1145/3477495.3532050

[18] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Keshavan, Trung Vu, Lukasz Heidt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, and Maheswaran Sathiamoorthy. 2023. Recommender systems with generative retrieval. In Proceedings of the 37th International Conference on Neural Information Processing Systems (New Orleans, LA, USA) (NIPS '23). Curran Associates Inc., Red Hook, NY, USA, Article 452, 17 pages.

[19] Yedan Shen, Kaixin Wu, Yuechen Ding, Jingyuan Wen, Hong Liu, Mingjie Zhong, Zhouhan Lin, Jia Xu, and Linjian Mo. 2025. Alleviating LLM-based Generative Retrieval Hallucination in Alipay Search. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (Padua, Italy) (SIGIR '25). Association for Computing Machinery, New York, NY, USA, 4294–4298. doi:10.1145/3726302.3731951

[20] Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, and Donald Metzler. 2022. Transformer memory as a differentiable search index. In Proceedings of the 36th International Conference on Neural Information Processing Systems (New Orleans, LA, USA) (NIPS '22). Curran Associates Inc., Red Hook, NY, USA, Article 1587, 13 pages.

[21] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2019. Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748 [cs.LG] https://arxiv.org/abs/1807.03748

[22] Huimu Wang, Yiming Qiu, Xingzhi Yao, Guangtao Nie, Zuxu Chen, Zhenlin He, Songlin Wang, Guoyu Tang, Sulong Xu, Jingwei Zhuo, and Mingming Li. 2026. Bridging the Gap: Generative Retrieval via Query-to-Multi-Span Framework for Effective E-commerce Search. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (Australia) (SIGIR '26). Association for Computing Machinery, New York, NY, USA, 4960–4964. doi:10.1145/3805712.3808479

[23] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & Cross Network for Ad Click Predictions. In Proceedings of the ADKDD'17 (Halifax, NS, Canada) (ADKDD'17). Association for Computing Machinery, New York, NY, USA, Article 12, 7 pages. doi:10.1145/3124749.3124754

[24] Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Hao Sun, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, Xing Xie, Hao Allen Sun, Weiwei Deng, Qi Zhang, and Mao Yang. 2022. A neural corpus indexer for document retrieval. In Proceedings of the 36th International Conference on Neural Information Processing Systems (New Orleans, LA, USA) (NIPS '22). Curran Associates Inc., Red Hook, NY, USA, Article 1856, 15 pages.

[25] Yifan Wang, Yixuan Wang, Yidan Liang, Qiang Liu, and Fei Xiao. 2026. Towards Sustainable Growth: A Multi-Value-Aware Retrieval Framework for E-Commerce Search. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (Republic of Korea) (KDD '26). Association for Computing Machinery, New York, NY, USA, 8179–8189. doi:10.1145/3770855.3818495

[26] Chen Wei, Yixin Ji, Zeyuan Chen, Jia Xu, and Zhongyi Liu. 2024. LLMGR: Large Language Model-based Generative Retrieval in Alipay Search. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (Washington DC, USA) (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2847–2851. doi:10.1145/3626772.3661364

[27] Haiyang Yang, Qinye Xie, Qingheng Zhang, Chen Li Yu, Huike Zou, Chengbao Lian, Shuguang Han, Fei Huang, Jufeng Chen, and Bo Zheng. 2025. GSID: Generative Semantic Indexing for E-Commerce Product Understanding. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, Saloni Potdar, Lina Rojas-Barahona, and Sebastien Montella (Eds.). Association for Computational Linguistics, Suzhou (China), 1113–1121. doi:10.18653/v1/2025.emnlp-industry.78

[28] Jianhui Yang, Yiming Jin, Pengkun Jiao, Chenhe Dong, Zerui Huang, Shaowei Yao, Xiaojiang Zhou, Dan Ou, and Haihong Tang. 2026. TaoSR-AGRL: Adaptive Guided Reinforcement Learning Framework for E-commerce Search Relevance. In Proceedings of the ACM Web Conference 2026 (United Arab Emirates) (WWW '26). Association for Computing Machinery, New York, NY, USA, 7955–7966. doi:10.1145/3774904.3792799

[29] Shaowei Yao, Jiwei Tan, Xi Chen, Keping Yang, Rong Xiao, Hongbo Deng, and Xiaojun Wan. 2021. Learning a Product Relevance Model from Click-Through Data in E-Commerce. In Proceedings of the Web Conference 2021 (Ljubljana, Slovenia) (WWW '21). Association for Computing Machinery, New York, NY, USA, 2890–2899. doi:10.1145/3442381.3450129

[30] Tianyu Zhan, Gui Ling, Tong Xiong, Kunhai Lin, Yang Wang, Kaixuan Zhang, Zhihong Chen, Yuliang Yan, Dan Ou, Shengyu Zhang, Haihong Tang, and Bo Zheng. 2026. TSGR: Taobao Search Generative Retrieval. arXiv:2607.18796 [cs.IR] https://arxiv.org/abs/2607.18796

[31] Fuwei Zhang, Xiaoyu Liu, Xinyu Jia, Yingfei Zhang, Zenghua Xia, Fei Jiang, Fuzhen Zhuang, Wei Lin, and Zhao Zhang. 2025. HierGR: Hierarchical Semantic Representation Enhancement for Generative Retrieval in Food Delivery Search. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track), Georg Rehm and Yunyao Li (Eds.). Association for Computational Linguistics, Vienna, Austria, 444–455. doi:10.18653/v1/2025.acl-industry.31

[32] Fuwei Zhang, Xiaoyu Liu, Jiajie Jin, Jiale Mao, Wei Chen, Dongbo Xi, Yifan Yang, Peng Yan, Zichao Hao, Zhao Zhang, and Fuzhen Zhuang. 2026. Beyond Matching: Category-Guided Latent Intent Reasoning for Generative Retrieval in E-Commerce. arXiv:2606.07075 [cs.IR] https://arxiv.org/abs/2606.07075

[33] Xuxin Zhang, Ben Chen, Yue Lv, Siyuan Wang, Yupeng Li, Yufei Ma, Zihan Liang, Tong Zhao, Ying Yang, Huangyu Dai, Lingtao Mao, Zhipeng Qian, Xinyu Sun, Chenyi Lei, Wenwu Ou, and Kun Gai. 2026. OneRetrieval: Unifying Multi-branch E-commerce Retrieval with an Editable Generative Model. arXiv:2606.13533 [cs.IR] https://arxiv.org/abs/2606.13533

[34] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep Interest Network for Click-through Rate Prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (London, United Kingdom) (KDD '18). Association for Computing Machinery, New York, NY, USA, 1059–1068. doi:10.1145/3219819.3219823

[35] Jianbo Zhu, Xing Fang, Jing Wang, Mingmin Jin, Bokang Wang, Guangxin Song, Zhenyu Xie, and Junjie Bai. 2026. Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL. arXiv:2605.14434 [cs.IR] https://arxiv.org/abs/2605.14434

---

（全文完。参考文献 [1]-[35] 原样保留英文，未作翻译。）
