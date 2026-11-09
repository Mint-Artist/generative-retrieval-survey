# PLUM：将预训练语言模型改造用于工业级生成式推荐

**英文原题**：PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations

**作者与机构**（保留原文；核心贡献者按字母序）：

Ruining He (Google DeepMind), Lukasz Heldt (YouTube), Lichan Hong (Google DeepMind), Raghunandan Keshavan (YouTube), Shifan Mao (YouTube), Nikhil Mehta (Google DeepMind), Zhengyang Su (YouTube), Alicia Tsai (Google DeepMind), Yueqi Wang (YouTube), Shao-Chuan Wang (Google DeepMind), Xinyang Yi (Google DeepMind), Lexi Baugher (YouTube), Baykal Cakici (YouTube), Ed Chi (Google DeepMind), Cristos Goodrow (YouTube), Ningren Han (YouTube), He Ma (YouTube), Romer Rosales (YouTube), Abby Van Soest (YouTube), Devansh Tandon (YouTube), Su-Lin Wu (YouTube), Weilong Yang (YouTube), Yilin Zheng (YouTube)

> 原文脚注：核心贡献者（Core contributors），按字母序。通讯作者：Xinyang Yi (xinyang@google.com)、Lukasz Heldt (heldt@google.com)、Raghunandan Keshavan (hkraghunandan@google.com)、Ruining He (ruininghe@google.com)、Nikhil Mehta (nikhilmehta@google.com)、Alicia Tsai (aliciatsai@google.com)、Lichan Hong (lichan@google.com)。

**arXiv 编号与译注**：arXiv:2510.07784（v1）。原文使用 ACM 会议模板，会议信息在预印本中为占位符（据外部信息，该论文为 WWW'26 投稿/发表）。本译文结构忠实还原原文：章节层级、段落划分与原文一一对应；公式以 LaTeX 保留；图题译为中文并注明"图N"；表格转为 Markdown 表格（文字单元格翻译，数字不变）；参考文献保留英文原文。专有名词（TIGER、PLUM、RQ-VAE、Gemini、YouTube Shorts、Recall@K 等）与指标名保留英文。

---

## 摘要

大型语言模型（LLM）为信息任务的建模与计算带来了新范式。推荐系统是一个关键的应用领域，有望从这些大模型固有的序列建模能力与世界知识中显著受益。本文提出 PLUM——一个将预训练 LLM 适配到工业规模推荐任务的框架。PLUM 由以下部分组成：基于语义ID（Semantic IDs）的物品符号化（item tokenization）、在领域特定数据上的继续预训练（continued pre-training, CPT），以及面向推荐目标的任务特定微调。在微调方面，我们特别关注生成式检索（generative retrieval）：模型被直接训练为基于用户上下文生成被推荐物品的语义ID。我们在大规模内部视频推荐数据集上开展了全面实验。结果表明，与一个基于大型嵌入表构建、经过高度优化的生产模型相比，PLUM 在检索上取得了显著提升。我们还给出了模型检索性能的 scaling 研究、关于 CPT 的经验总结、对语义ID的若干改进，以及支撑该框架在 YouTube 上为数亿至数十亿用户上线的训练与推理方法概览。

**关键词**：推荐系统（Recommender Systems）、生成式检索（Generative Retrieval）、大型语言模型（Large Language Models）

## 1. 引言

推荐系统是现代数字平台的关键组成部分，深刻塑造着用户发现内容并与内容互动的方式。过去几十年，深度学习模型在推荐的检索（retrieval）与排序（ranking）两个阶段都取得了显著成功。尽管已经使用神经网络，工业推荐系统的主流范式仍然依赖大规模嵌入表（embedding tables）来表示诸如物品 ID 之类的高基数类别特征（参见 (Covington et al., 2016; Naumov et al., 2019; Liu et al., 2022; Coleman et al., 2023) 等工作）。在这类大型嵌入模型（Large Embedding Models, LEM）中，绝大多数参数都位于嵌入表内。这种架构选择虽然非常擅长记忆用户—物品交互，却阻碍了更深、更复杂的网络（例如 (Chen et al., 2019b; Gu et al., 2020)）发挥潜在优势。这种以扩大嵌入表为核心的扩展方法论，与 LLM 的做法形成鲜明对比——后者强调扩大神经网络规模，以学习紧凑输入 token 的组合表示。

LLM 的成功激发了构建推荐模型的新范式转变 (Rajput et al., 2023; Zhai et al., 2024; Firooz et al., 2025; Zhou et al., 2025; Pang et al., 2025; Chen et al., 2025)。LLM 固有的序列建模能力与其中编码的海量世界知识，为构建更智能、更个性化的推荐系统创造了机会。然而，将 LLM 适配到推荐并非易事。首要挑战在于弥合领域差距（domain gap）：LLM 的预训练并未包含目标领域的用户行为数据与物品语料，这使其难以从用户活动中理解用户偏好以及物品质量的细微差别。因此，直接将现成的 LLM 应用于推荐任务会表现出持续的性能差距，即便在小规模公开数据集上也是如此 (Hou et al., 2024)。其次，基于大型嵌入表的传统输入表示带来了扩展性挑战：这些表需要海量训练数据，使得训练大型 Transformer 架构代价高昂。

本文朝着教会 LLM 解决推荐任务的方向迈出一步。我们提出 PLUM——一个将预训练 LLM 高效适配到工业规模生成式推荐的框架。PLUM 包含三个关键阶段：

- **物品符号化（Item tokenization）**。语料中的每个物品由一串离散 token 表示，即语义ID（Semantic IDs, SIDs）。在采用 RQ-VAE 的先前工作 (Rajput et al., 2023; Singh et al., 2024) 的基础上，我们引入一组新技术（称为 SID-v2），将用户行为信号与多模态内容嵌入纳入 SID 的生成，并通过多分辨率码本（multi-resolution codebooks）与渐进式掩码（progressive masking）改进其层级一致性。

- **继续预训练（CPT）**。在此阶段，预训练 LLM 的词表被扩展以纳入新的 SID token。模型在领域特定物品数据与用户序列、以及通用领域文本数据构成的混合语料上进一步预训练，使新的 SID 模态与模型的既有知识对齐。

- **任务特定微调（Task-specific fine-tuning）**。最后，模型针对特定的推荐目标进行微调。虽然 PLUM 可以支持多种下游任务，本文聚焦其在生成式检索上的应用：训练一个 decoder-only 模型，自回归地生成用户可能互动的下一个物品的 SID。生成式检索无需为物品语料维护单独的索引，并且绕开了基于嵌入的检索 (Covington et al., 2016; Yi et al., 2019) 中点积相似度的限制 (Weller et al., 2025)。

PLUM 框架正是为解决上述挑战而设计的。继续预训练通过向预训练 LLM 注入领域语料与用户行为模式来弥合领域差距。我们观察到，经过 CPT 的模型能够展示基本的少样本学习能力——基于 SID 输入生成文本 token。基于 SID 的输入表示则绕开了 LEM 的扩展瓶颈。我们的实验表明，通过将模型复杂度从输入嵌入转移到神经网络，与一个使用大型嵌入表的生产级 Transformer 检索模型相比，基于 PLUM 的生成式检索能取得显著更优的样本效率（sample efficiency）。正因如此，虽然 PLUM 检索所用的 Transformer 架构拥有 LEM 约 100 倍的稠密参数量，但由于收敛更快，检索任务的总体训练成本与 LEM 相当。我们还观察到有效的 scaling 特性：直到激活参数超过 9 亿（总参数约 42 亿）的混合专家（Mixture-of-Experts, MoE）模型，检索性能仍在持续提升。

除上述贡献外，我们还将分享在大型生产环境中部署基于 PLUM 的检索的实践经验，特别是 PLUM 检索与基于 LEM 的检索在线上 A/B 实验中的对比。PLUM 框架已在 YouTube 推荐中投入生产，通过在线与离线推理，在多个核心 surface 上同时服务于长视频（long-form）与短视频（short-form）的检索。

### 1.1. 相关工作

受 LLM 近期进展的启发，主要出现了两条研究路线。第一条专注于用 Transformer 架构（或其变体）扩展 LEM 的神经网络部分，我们在 1.1.1 节讨论。第二条如 1.1.2 节与 1.1.3 节所述，专注于基于 token 的生成式推荐，将推荐检索重新表述为 seq2seq 转导任务。

#### 1.1.1. 序列推荐

对用户序列与特征交互的建模一直是改进推荐模型的关键领域：从使用经典的序列神经网络 (Hidasi et al., 2016; Kang and McAuley, 2018)，到设计定制的特征交互组件 (Wang et al., 2021; Zhang et al., 2024)。近来，新兴的趋势是使用类 Transformer 架构，把用户历史与异构特征统一在单一序列中 (Zhai et al., 2024; Huang et al., 2025; Han et al., 2025)，相比传统 MLP 架构展现出更好的 scaling 特性。此外，训练范式本身也在向 LLM 的做法靠拢，一些工作采用自监督预训练目标、随后进行任务特定微调 (Netflix, 2024; Chen et al., 2025)。所有这些进展的一条共同主线是：它们仍然依赖大规模嵌入表来处理类别特征。我们的工作脱离了这一范式，用紧凑的输入 token 取代大型嵌入表。除 SID 之外，我们使用 LLM 来处理异构特征：对数值特征进行符号化，或直接将稠密嵌入特征当作软 token（soft token）使用。

#### 1.1.2. 语义ID与量化

将物品表示为离散 token 序列（即语义ID, Semantic IDs, SIDs）的概念，已成为传统 ID 嵌入的有力替代方案。该方法使物品可以被视为一种能够被序列模型处理的"语言"。早期工作证明了从头训练检索模型来预测 SID 的可行性 (Rajput et al., 2023)；后续研究表明，SID 可以经哈希替换随机的物品 ID 嵌入，从而增强排序模型的泛化能力 (Singh et al., 2024; Zheng et al., 2025)。

大量近期研究专注于通过在量化过程中融入多样化信号来提升 SID 的质量与表达能力，包括融合多模态内容特征 (Xu et al., 2025)，以及从用户行为中注入协同过滤信号 (Penha et al., 2025; Li et al., 2025; Yao et al., 2025; Ye et al., 2025)。PLUM 并行地贡献于这条研究路线。与 MMQ (Xu et al., 2025) 为每个模态生成不同 token 的做法不同，我们只是把多个内容嵌入拼接后用一个 MLP 编码，从而在支持更多嵌入上保持灵活性。在纳入用户行为方面，多数工作在量化时把协同过滤（CF）物品嵌入融合进内容嵌入，但基于 CF 的物品嵌入通常会随物品热度变化而动态变化，导致量化器与下游模型需要频繁重训。我们避免了这一点：将 CF 信号用于对比训练目标，引导量化器从内容信息中捕获有用信息。此外，我们还提出了另外两项关键创新，用于改进 SID 训练与生成式检索的对齐。

#### 1.1.3. 生成式检索与对齐

生成式检索将推荐检索重新表述为一个序列到序列任务：模型以自回归方式直接生成相关物品的 SID。自文档搜索 (Tay et al., 2022) 与推荐 (Rajput et al., 2023) 的开创性工作以来，这一思路已在业界获得显著关注（例如 (Wang et al., 2025; Mei et al., 2025; Ju et al., 2025; Pang et al., 2025)）。例如，OneRec (Zhou et al., 2025) 采用 encoder-decoder 架构生成视频 SID，并利用 RL 训练与奖励模型来增强模型质量。另一些工作 (Yang et al., 2024; Yang et al., 2025) 提出同时生成 SID 与稠密嵌入的混合模型，以缓解量化带来的信息损失。这些方法的一个共同点是聚焦于从头训练生成式模型。我们的工作聚焦于将 SID 与 LLM 对齐，研究 LLM 预训练与 CPT 对生成式检索性能的影响。虽然已有研究 (Cao et al., 2024; Firooz et al., 2025) 探索了 LLM 与推荐任务的对齐，但其重心主要在语言接口上；如何有效地把 SID 作为一种新模态纳入 LLM，仍缺乏深入研究。

## 2. PLUM 框架

### 2.1. 语义ID

> 图1：我们的语义ID模型示意图。它接收两个多模态视频嵌入，对其进行编码，并用残差量化器将结果压缩为一个量化ID。该ID的训练目标是既重建原始输入，又利用对比损失对共现视频进行语义聚类。

语义ID（SID）在概念上是从物品底层内容特征导出的一组离散码字元组（tuple of discrete codewords）。该过程包含两个主要阶段：(1) 将高层内容特征编码为一个稠密的语义嵌入；(2) 将该嵌入量化为一个层级化的码字元组。生成式检索模型的效能从根本上取决于 SID 的语义丰富度与结构完整性。在采用残差量化变分自编码器（Residual-Quantized Variational AutoEncoder, RQ-VAE）的基础框架 TIGER 之上，我们对这一生成过程引入了一系列重要增强。最初的方法受限于两点：依赖单一内容嵌入来源，且 ID 结构中缺失协同信号。我们的方法旨在产生更全面、与用户行为更对齐、层级更连贯的 SID。图1 展示了我们训练 SID 模型的总体设计。

#### 2.1.1. 融合的多模态内容表示

(Rajput et al., 2023) 依赖单一内容表示的一个主要局限，在于其天然无法捕捉复杂媒体物品的多面性。例如，一个视频的完整语义是通过其文本元数据、视觉内容与音频的组合来传达的；单模态表示按定义就忽略了这些丰富且正交的信息来源。为克服这一点，我们将框架架构设计为对输入来源的具体数量与类型不敏感（agnostic），使其能够摄取并融合多个异构表示。该模型被设计为对每个物品接收一组不同的嵌入向量 $\{x_{m}\}_{m=1}^{M}$。对这多个表示的融合通过如下方式实现：用单独的嵌入编码器 $\mathcal{E}_{m}$ 将 $x_{m}$ 编码为隐向量 $z_{m}$，再经拼接 $\tilde{z}=[z_{1},\dots,z_{M}]$ 与一次投影，得到单一、统一的特征编码向量 $z$。通过整合跨模态信息，我们为后续的 RQ-VAE 量化过程构成了更优的输入，确保所得的 SID 建立在对物品内容的全面理解之上。

#### 2.1.2. 量化中的层级化改进

我们进一步改进 RQ-VAE 架构本身，以产生更高效、更有意义的层级。这涉及两项关键创新：

- **多分辨率码本（Multi-Resolution Codebooks）**：先前工作 (Rajput et al., 2023) 采用固定且统一分辨率的码本，这在参数上可能并不高效，导致 SID 空间巨大且稀疏——大多数潜在的码字组合从未被分配。我们用多分辨率码本结构取而代之：初始的 SID 层级具有高分辨率、判别性最强，而后续码字以更低的分辨率编码低熵残差。特别地，码本容量（cardinality）是量化层级的函数：$2048/2^{level-1}$，从而得到更紧凑、更高效的 SID。

- **渐进式掩码（Progressive Masking）**：为了在残差量化过程中强制形成更严格、更可解释的层级，我们引入了渐进式掩码。具体而言，我们定义一个二值掩码标量 $m_{l}\in\{0,1\}$（$l$ 为码本层级），满足 $m_{l}=\mathbf{1}_{l<r}$，其中 $r\in[1,L]$ 是一个随机整数，$L$ 是码本总层级数。该掩码用于在 SID 训练中只选取前 $r$ 个码本层级。

#### 2.1.3. 带共现对比正则的 RQ-VAE

仅从物品内在内容导出的 SID，可能无法完全捕捉推荐场景下用户所感知的相似性。用户行为为视频相关性提供了强有力的外部信号：经常被一起观看的视频往往在语义上相互关联。为弥合基于内容的相似性与基于行为的相似性之间的差距，我们在 ID 生成阶段直接注入强协同信号：向 RQ-VAE 的训练目标引入一个共现对比损失项（$\mathcal{L}_{con}$）。该目标函数旨在依据用户交互序列中物品的共现模式来塑造嵌入空间：该损失鼓励模型为频繁一同出现的物品生成相似的 SID 表示，同时推远不共现物品的表示。特别地，对比损失定义为：

$$\mathcal{L}_{con}=-\sum_{i=1}^{2N_{b}}\frac{\exp(sim(p_{i},p_{i}^{+}))}{\sum_{j=1}^{2N_{b}}\exp(sim(p_{i},p_{j}))},\tag{1}$$

其中 $p_{i}$ 表示一个批次（batch）中 $N_{b}$ 个视频的视频表示，$p_{i}^{+}$ 表示与视频 $i$ 共现的视频的表示，而 $sim(p_{i},p_{j})$ 是视频 $i$ 与 $j$ 之间的点积相似度。

#### 2.1.4. SID 训练损失

除 (Singh et al., 2024) 引入的训练损失之外，我们加入了共现对比损失项。特别地，总体损失定义为：

$$\mathcal{L}=\mathcal{L}_{recon}+\mathcal{L}_{rq}+\mathcal{L}_{con},\tag{2}$$

其中 $\mathcal{L}_{recon}=\sum_{m=1}^{M}\|x_{m}-\hat{x}_{m}\|^{2}$ 是每个多模态嵌入的重建损失，而 $\mathcal{L}_{rq}=\sum_{l=1}^{L}\beta\|r_{l}-sg[e^{l}_{*}]\|^{2}+\|sg[r_{l}]-e^{l}_{*}\|^{2}$，其中 $e^{l}_{*}$ 是第 $l$ 层码本中最近的码字，$r_{l}=r_{l-1}-e^{l}_{*}$ 是从 $r_{0}=z$ 开始逐层递归计算的残差，$sg$ 为停止梯度（stop-gradient）算子。如图1 所示，最终的量化向量 $\hat{z}$ 以渐进式掩码的方式计算，即所选各层码字之和：$\hat{z}=\sum_{l=1}^{L}m_{l}\,e^{l}_{*}$。该量化向量随后作为每个解码器 $\mathcal{D}_{m}$ 的输入，用于重建输入嵌入 $\hat{x}_{m}$。

### 2.2. 继续预训练

在创建 SID 词表之后，继续预训练（CPT）阶段的一个主要目标是得到一个基础模型检查点：其中 SID token 具有语义根基（semantically grounded），并与基座语言模型既有的文本 token 对齐。为此，我们使用下一 token 预测（next-token prediction），在由两大主要来源构成的大规模语料上训练模型：

- **用户行为数据（User behavior data）**：该来源对个性化与捕捉观看历史至关重要。训练期间，每个样本利用用户观看历史以及附加的观看特征来建模用户行为序列。
- **视频元数据语料（Video metadata corpus）**：该大规模语料旨在建立 SID 与其对应文本特征之间的强关联。每个样本包含一个视频的 SID、标题、描述、ASR 字幕、频道名称，以及合成生成的数据。

表1 展示了我们 CPT 训练数据的模式（schema）。

**表1：继续预训练中使用的示例 schema。**

| 数据类型 | 示例内容 |
|---|---|
| 用户行为训练数据示例 | `wh = <sid_1> <channel_name> <watch_ratio> <watch_time> <hours_since_final_watch> <sid_2> <channel_name> … ‖ <sid_n>` |
| SID + 视频标题 | `Video <sid> has title (en): <video_title>` |
| SID + 视频主题 | `The topics in video <sid> are: <topics>` |

（译注：表中为训练数据模板，占位符原样保留。）

**数据混合与训练**。CPT 阶段的训练集由用户行为数据与视频元数据语料混合构成，每个来源各占训练样本的 50%。CPT 阶段进行 100 万步训练，批大小为 16，总计约 2600 亿个 token。我们设置了多种评估来度量：基于用户历史生成 SID 的性能；通过留出（held-out）的视频元数据语料衡量 SID 与语言的联合建模能力；以及通过标准文本基准跟踪通用语言能力的退化情况。

**上下文学习（In-context Learning）**。CPT 模型由于在 SID 与自然语言的混合语料上训练，保留了生成自由格式文本的灵活性，并具备上下文少样本学习能力。我们在附录 A.2 中提供若干示例。

### 2.3. 生成式检索

虽然继续预训练（CPT）阶段使模型能够基于用户历史生成下一个视频的 SID，但随后的监督微调（Supervised Fine-Tuning, SFT）阶段对于将其专门化于检索任务而言必不可少。该 SFT 阶段使模型得以纳入更丰富的特征集合——尤其是实时上下文（real-time context）——并直接针对与用户体验对齐的奖励信号进行优化。

> 图2：面向下一个视频推荐的生成式检索示意图。输入提示（prompt）是 SID token、文本与数值特征自定义 token 交织而成的序列。

此微调采用标准的自回归极大似然目标。模型学习预测真实（ground-truth）视频的 SID token，这些视频定义为：给定用户上下文与历史时，用户日志中被点击的视频。具体而言，模型经训练以最小化以下损失：

$$\mathcal{L}_{\text{SFT}}=-\sum_{t=1}^{L}r(\text{user},v_{click})\cdot\log P(sid_{t}\,|\,\text{Context}_{\text{user}},\text{History}_{\text{user}},sid_{<t}),\tag{3}$$

其中 $[sid_{1},..,sid_{L}]$ 表示被点击视频 $v_{click}$ 的 SID，$r(\text{user},v_{click})$ 是每次点击的人工设计奖励信号。实践中，鉴于训练成本高昂，我们基于该奖励对训练样本进行采样，然后在损失中对采样得到的样本等权处理。如图2 所示，输入提示不仅包含 SID token 与数值特征的自定义 token，还包含可被预训练 LLM 自然编码的其他文本特征。

推理期间，我们使用束搜索（beam search）解码出多条 SID 序列，作为检索到的候选集合。每个生成的 SID 随后被映射回我们数十亿规模语料中的真实视频。虽然这一生成过程有可能产生无效 SID（即幻觉，hallucination）或 SID 到视频的碰撞（collision），但我们观察到 SFT 之后的幻觉率非常低（< 5%），且 SID 到视频映射的唯一性保持在高位（见表4）。

## 3. 实验

在本节中，我们进行一组全面的实验来验证 PLUM 框架。我们首先将基于 PLUM 的生成式检索模型与一个贡献了生产中大多数曝光量的、基于 Transformer 的大型嵌入检索模型进行比较（3.1 节）。随后，3.2 节展示我们对 SID 所提各项增强的有效性。接着，我们对 PLUM 框架中两个关键组件进行消融研究：3.3 节量化继续预训练（CPT）阶段的精确影响，以及从预训练 LLM 初始化的价值。最后，我们给出详细的 scaling 研究，分析模型规模、计算量与检索性能之间的关系（3.4 节）。

### 3.1. 生成式检索的性能

传统推荐系统使用大型嵌入模型来推荐候选。而生成式检索相较传统系统具有多项优势。本节研究生成式检索在 YouTube 生产设置上的有效性。

#### 3.1.1. 实验设置

**模型**。在以下实验中，我们在长视频（Long Form Video, LFV）与 Shorts 上训练了一个来自 Gemini-1.5 混合专家（MoE）家族、激活参数量 9 亿（900M）的 PLUM 模型。该模型从 Gemini 热启动（warm start），并如上文所述在新互动数据上持续微调。训练数据是最新数据与历史数据的混合。serving 时，目标序列使用束搜索解码。在我们的实验中，束搜索的表现优于随机解码，不过以牺牲一定的多样性为代价。

**基线**。我们将上述模型的性能与传统 LEM 检索进行比较。该基线是表现最优的生产模型，自 (Chen et al., 2019a) 等早期工作以来经过了高度优化。该模型同样基于 Transformer 架构，但其大部分参数位于嵌入层，输入与输出物品 ID 的词表规模均为 O(10M)（千万量级）。具体而言，LEM 的神经网络仅占其总参数的 0.4%，而 PLUM 的神经网络占其总参数的 90%。

#### 3.1.2. 实验结果

**推荐质量**。我们从几个不同维度评估推荐质量。首先，我们将推荐系统的有效词表规模（effective vocab size）度量为其覆盖 95% 曝光量所需的唯一视频数量。更高的有效词表规模通常更有利于小众内容的个性化发现，表明模型的泛化能力更强。其次，我们用两个指标比较推荐的有效性：a) 点击率（CTR）与推荐接受度（Recommendation Acceptance）。对推荐接受度，我们同时比较单次观看的时长（WT/View）与单次观看观看完成的比例（WF/View）。在所有情形下，我们报告的都是 900M MoE 模型的指标相对 LEM 模型指标的比值。

**表2：推荐质量比较：每个数字均为比值，即 PLUM 的指标除以 LEM 的指标。**

| 指标 | LFV | Shorts |
|---|---|---|
| 有效词表规模 | 2.60x | 13.24x |
| CTR | 1.42x | 1.33x |
| WT/View | 0.72x | 1.13x |
| WF/View | 1.32x | 1.03x |

我们观察到，PLUM 模型取得了大得多的有效词表规模，同时在用户反应相关指标上与 LEM 相当。

**线上实验**。我们通过将 PLUM 模型的推荐加入候选池来开展线上实验。为公平比较，我们（等量）提高了生产中表现最优的检索模型的配额（quota），并以此作为基线，记为 LEM+。在下表中，我们报告了相对 LEM+ 在 LFV 与 Shorts 上四个关键指标的变动。这表明基于 PLUM 的检索即便不使用大型嵌入表，也能够在现有系统之上贡献独特价值。

**表3：互动（engagement）比较：加入 PLUM 相对 LEM+ 的百分比变化。**

| 指标 | LFV | Shorts |
|---|---|---|
| 互动用户数 | +0.07% | +0.28% |
| Panel CTR | +0.76% | +4.96% |
| 观看数 | +0.80% | +0.39% |
| 满意度 | +0.06% | +0.39% |

**样本效率**。在我们的研究中，PLUM 模型的样本效率极高。900M MoE 模型每天训练约 2.5 亿个样本；相比之下，传统 LEM 每天训练数十亿个样本。这种样本效率也使这类模型在生产中切实可行，尽管其单样本训练成本要大得多。与 LEM 相比，900M MoE 模型的训练 flops 不到 LEM 的 0.55 倍。

### 3.2. 语义ID消融研究

在表4 中，我们讨论针对先前 SID (Rajput et al., 2023) 所引入的各项改动的重要性。表中我们将先前工作称为 SIDv1，将我们提出的方法称为 SIDv2。由于每个 SID 模型会产生不同的索引结构，我们报告索引唯一性（index uniqueness）——刻画语料中有多少视频在 SID 空间中被唯一表示——以及在该 SID 上训练生成式检索模型后的最终 Video Recall@10。Video Recall@K 的计算方式为：给定 PLUM 的 top-K 个 SID 预测，衡量成功检索到特定视频的召回率。当一个 SID 映射到多个视频时，我们在计算召回指标时随机采样一个视频。

我们在一个简单设置下对 SIDv2 的改动进行消融实验：使用 900M MoE 的 PLUM 检索模型、不做 CPT，并移除提示中的观看历史。如表4 所示，这些实验同时评估了语义ID的唯一性与下游生成式检索的 VID Recall@10。结果表明，所有改动都提升了生成式物品检索的召回率。值得注意的是，引入共现对齐任务同时大幅改善了语义ID的唯一性与召回率。

**表4：SIDv2 改动的消融实验**

| SID 模型 | SID 唯一性 | VID Recall@10 |
|---|---|---|
| SIDv1（基线） | 94.0% | 12.3% |
| SIDv2（本文方法） | 96.7% | 14.4% |
| 消融：多分辨率 | 94.8% | 13.2% |
| 消融：多嵌入 | 96.9% | 12.8% |
| 消融：共现 | 91.8% | 12.6% |

### 3.3. 继续预训练的影响

为衡量继续预训练阶段的重要性，我们评估其对生成式检索最终模型的训练效率与检索性能的影响。

#### 3.3.1. 实验设置

我们的训练方法包含两个阶段：继续预训练与生成式检索微调，各自配有特定的学习目标与相应的一组超参数。为评估每一步的独特贡献，我们设计了一个受控的 2x2 消融研究，其中包含四个共享同一 decoder-only Transformer 架构的模型配置。本研究使用 MoE-900M 架构。四个模型如下：

- **R1：检索 SFT（随机初始化）**。Transformer 模型以随机初始化的权重直接在生成式检索任务上训练。
- **R2：检索 SFT（LLM 初始化）**。该模型从预训练 LLM 检查点初始化，但跳过 CPT 阶段。
- **CR1：CPT（随机初始化）+ 检索 SFT**。模型训练遵循 PLUM 的两个训练阶段，但 CPT 阶段使用的模型为随机初始化。
- **CR2：CPT（LLM 初始化）+ 检索 SFT**。这是我们提议框架的完整且符合预期的实现。

#### 3.3.2. 实验结果

> 图3：第 8 天的 Recall@10 与训练损失随检索 SFT 训练步数的变化。(a) Recall@10；(b) 训练损失。

**继续预训练的影响**。主要结果汇总于表5。依照 scaling 研究的做法，四个模型均在大型留出的次日（next-day）数据集上评估，并报告 recall@10 作为检索指标。R1 与 CR1（或 R2 与 CR2）之间存在巨大的性能差距，显示了 CPT 对检索微调的收益。此外，图3 展示了第 8 天的 recall@10 与训练损失随训练步数的变化。可以清楚地看到，经过 CPT 的模型收敛要快得多。因此，CPT 有利于任务特定微调的训练效率，尤其是当多个下游任务共享同一个 CPT 模型时。

**表5：四种模型配置的主生成式检索性能。**

| 模型 | 预训练 LLM | CPT | Recall@10（第 8 天） |
|---|---|---|---|
| R1 | 否 | 否 | 0.19 |
| R2 | 是 | 否 | 0.23 |
| CR1 | 否 | 是 | 0.27 |
| CR2 | 是 | 是 | 0.28 |

**预训练 LLM 的影响**。接下来我们考察从通用 LLM 初始化训练与从随机初始化状态训练相比的收益。表5 的结果呈现出一致的规律：无论有无 CPT，从预训练 LLM 初始化的模型均持续优于其随机初始化的对照模型。我们假设这一优势主要源于预训练 LLM 对自然语言的既有理解，或者说 LLM 通过大规模预训练学到的通用序列处理能力可直接用于推荐任务。对这一观察的进一步研究超出了本文的范围。

### 3.4. Scaling 研究

在本节中，我们使用 YouTube 生产数据对生成式检索模型进行 scaling 研究。我们使用 Gemini-1.5 (Team, 2024) 混合专家（MoE）模型家族的四种不同规模：MoE-110M、MoE-370M、MoE-900M 与 MoE-3B（此处规模指每 token 的激活参数量），其总参数量从不到 10 亿到超过 100 亿不等。特别地，我们旨在回答：(1) 损失如何随不同的模型规模与计算量变化？(2) 检索指标如何随不同的模型规模与计算量变化？(3) 对于不同的计算预算，最佳模型规模是多少？

#### 3.4.1. 实验设置

**数据集**。我们的研究在一个生产数据集上进行，该数据集来自 YouTube 的一个关键 surface：给定当前正在观看的视频以及其他上下文信号（如观看历史），推荐接下来观看什么视频（参见 (Zhao et al., 2019) 中的图4）。此处的推荐系统采用多阶段架构，涵盖候选生成、预排序与排序等阶段。我们的研究聚焦于候选生成阶段，在包含数十亿视频的全面视频语料上评估检索性能。

**训练与评估**。我们的输入提示格式为："watch history | user features | context video features"，其中观看历史是按时间顺序排列的观看序列，每次观看由 SID token 与其他特征 token 的拼接表示。在本研究中，输入序列长度固定为 1,536 个 token，可覆盖约 100 次最近的观看及其他特征。SFT 标签通过一个基于用户互动与满意度信号的下采样机制，选取 YouTube 上的下一次观看。下一次观看的 SID token 被用作预测目标。我们随机打乱连续 7 天（自 2025 年 7 月起）的数据用于训练。评估在随后的一天（第 8 天）进行。我们的实验使用 1,024 块 Google v6e 张量处理单元（TPU），每块配备 32GB 高带宽内存（HBM）。我们并行运行 4 个 trainer，每个使用 256 块 TPU。

**超参数选择**。我们先用 MoE-900M 模型开展了一项 pilot（先导）研究，以确定模型对全局批大小与学习率这两个超参数的敏感性（详见附录 A.1）。基于该 pilot 研究的经验，在每种设置下，我们都将批大小设为约饱和 HBM 的水平，因为结果表明更大的批大小更可取。学习率方面，以 MoE-900M 使用 $5\times 10^{-5}$ 为锚点，基于我们的最佳猜测，对更小的模型使用略大的学习率（反之亦然）。请注意，与自然语言处理中的 scaling law 研究（例如 (Hoffmann et al., 2022)）不同，我们采用恒定学习率。这一做法有两个原因：(1) 生产模型的持续训练并不存在特定的停止步数；(2) 恒定学习率允许我们对训练过程中的所有中间数据点进行完美而精确的插值。不同模型规模的超参数汇总于表6。所有模型均从其对应的继续预训练检查点热启动（这些检查点均为 Iso-FLOPS $1\times 10^{22}$）。继续预训练的训练 FLOPs 不计入本节中的任何 FLOPs 数字。

**表6：Scaling Law 研究配置。**

| 模型规模 | 学习率 | 全局批大小 |
|---|---|---|
| MoE-110M | $1\times 10^{-4}$ | 25,600 |
| MoE-370M | $7\times 10^{-5}$ | 15,360 |
| MoE-900M | $5\times 10^{-5}$ | 7,680 |
| MoE-3B | $2\times 10^{-5}$ | 3,584 |

#### 3.4.2. 实验结果

**损失随计算量与模型规模的变化**。图4 展示了（第 8 天）训练/评估损失与训练 Iso-FLOPS 之间的相关性。图4(a) 揭示了每个模型规模的 Iso-FLOPS 与训练损失之间存在清晰的幂律（power-law）相关。随着 Iso-FLOPS 的增大，代表训练损失前沿的最优模型规模逐步转移：从 MoE-110M（红线）开始，转移到 MoE-370M（绿线），再到 MoE-900M（紫线），并被假设最终会到达 MoE-3B（蓝线）。图4(b) 表明，在最终开始饱和之前，Iso-FLOPS 与每个模型规模的评估损失之间同样存在幂律相关，我们将其归因于我们大量的训练。我们还观察到，评估损失前沿向更大模型转移的时间明显早于训练损失前沿，这表明更大的模型对未来数据分布具有更强的泛化能力。

> 图4：增大训练 Iso-FLOPS 时训练与评估损失的变化。(a) Iso-FLOPS 与训练损失的关系。(b) Iso-FLOPS 与评估损失的关系。

**检索指标随计算量与模型规模的变化**。类似地，图5 报告了（第 8 天）训练/评估 Recall@10 与训练 Iso-FLOPS 之间的关系，两轴均以对数刻度呈现。发现与图4 类似，不同之处在于：图5 中评估 Recall@10 的增长放缓迹象更少，这表明模型在持续改进其生成准确且完整、并能导向正确文档的序列。此外，三个较小的模型已处理了超过一个 epoch 的数据，其中 MoE-110M 最多训练了 4.24 个 epoch，但仍未显示出过拟合迹象（见图5(b)）。

> 图5：增大训练 Iso-FLOPS 时训练与评估 Recall@10 的变化。(a) Iso-FLOPS 与训练 Recall@10 的关系。(b) Iso-FLOPS 与评估 Recall@10 的关系。

**固定 Iso-FLOPS 预算下的最优模型规模**。在图6 中，我们展示了随着给定 Iso-FLOPS 预算的增大，最优模型规模向更大模型的逐渐转移。请注意，我们对每个模型使用恒定学习率，这使我们能够通过单次运行直接评估每个 Iso-FLOPS 预算下的模型性能。与 Chinchilla 论文 (Hoffmann et al., 2022) 类似，图6(b) 中在我们考虑的大多数 Iso-FLOPS 预算下都存在一个峰值。

> 图6：Iso-FLOPS 曲线。我们在多种固定的 Iso-FLOPS 预算下评估训练损失与评估 Recall@10，并为每个模型规模选择适当的训练步数。(a) 固定计算预算下的训练损失。(b) 固定计算预算下的评估 Recall@10。

#### 3.4.3. 局限性讨论

我们注意到，在我们所考虑的计算预算下，MoE-3B 模型并未超越 MoE-900M 模型。我们的 scaling 研究受限于计算资源。次优的超参数设置可能是一个重要的促成因素。例如，我们的 pilot 研究表明，同一模型规模下，更大的批大小与更高的训练效率相关。由于我们为每个模型规模分配了相同的训练资源，且使用饱和 HBM 的批大小（见表6），更大的模型可能因其（相对）更小的批大小而处于劣势。事实上，训练结束时，MoE-3B 模型仅处理了 0.57 个 epoch 的训练数据（约 50 亿样本），与第二大的模型相差 2 倍多。我们在生成式检索任务上的 scaling 研究表明，compute-optimal 训练需要训练样本数与模型规模的同步扩展。

## 4. 结论与未来工作

我们提出了 PLUM——一个将预训练 LLM 适配到大规模推荐任务的框架。我们的方法通过 SIDv2 增强物品符号化；使用大规模继续预训练阶段使 LLM 与用户行为对齐、并将 SID 落地于文本；随后通过监督微调阶段，使模型在生成式检索上表现出色。

作为已在 YouTube 上线的平台，PLUM 能够为当前系统贡献独特价值，同时也是我们构建的第一个不使用大型嵌入表的神经模型。我们的消融实验验证了 CPT 阶段的价值、从预训练 LLM 初始化的收益，以及我们提出的 SID 技术。本文是将 LLM 与真实世界推荐系统对齐的一项初步研究，开启了若干未来研究方向，例如将 PLUM 框架应用于排序与个性化搜索等其他任务、为候选多样性开发新的解码策略，以及无缝地同时生成 SID 与自然语言。

## 5. 致谢

我们感谢以下人员在将本文所述技术应用于多种 surface 与用例的过程中所付出的努力与反馈（按字母序）：Saksham Agarwal, Sriraj Badam, Sourabh Bansod, Xilun Chen, Yi Chen, Bernardo Cunha, Onkar Dalal, Mingyan Gao, Elad Ganmor, Danfeng Guo, Pooja Gupta, Ralf Gutsche, Yuan Hao, Chuan He, Lily He, Yan Huang, Kate Jones, Jiewen Lang, Pengfei Li, Wen Li, Li Wei, Sean Liew, Qiguang Liu, Sunny Liu, Yang Liu, Jianan Lu, Yilong Luan, Shifan Mao, Tomer Margolin, Radwa Metwali, Aniruddh Nath, Hardik Patel, Amy Pu, Vasin Punyakanok, Murphy Ren, Yuji Roh, Yuan Shao, Ajay Shekhawat, Fabio Soldo, Yanwei Song, Qian Sun, Jiaxi Tang, Aahan Tyagi, Diego Uribe, Jacky Wang, Ting Wang, Siqi Wu, Bo Yan, Shuhao Ye, Likang Yin, Qiao Zhang, Rein Zhang, Sai Zhang, Vivian Zhang, Yaping Zhang, Gwendolyn Zhao, Zelong Zhao, Zhilei Zou, Dav Zimak。我们同时感谢 Mahesh Sathiamoorthy 与 Anima Singh 对本项目概念化与初期开发所作的深远贡献。

## 参考文献

（参考文献保留英文原文，原样收录，不作翻译。）

- Cao et al. (2024) Yuwei Cao, Nikhil Mehta, Xinyang Yi, Raghunandan Keshavan, Lukasz Heldt, Lichan Hong, Ed H. Chi, and Maheswaran Sathiamoorthy. 2024. Aligning Large Language Models with Recommendation Knowledge. (2024). arXiv:2404.00245 [cs.IR] https://arxiv.org/abs/2404.00245
- Chen et al. (2019a) Minmin Chen, Alex Beutel, Paul Covington, Sagar Jain, Francois Belletti, and Ed H. Chi. 2019a. Top-K Off-Policy Correction for a REINFORCE Recommender System. In *Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining* (Melbourne VIC, Australia) (*WSDM '19*). Association for Computing Machinery, New York, NY, USA, 456–464. doi:10.1145/3289600.3290999
- Chen et al. (2019b) Qiwei Chen, Huan Zhao, Wei Li, Pipei Huang, and Wenwu Ou. 2019b. Behavior sequence transformer for e-commerce recommendation in Alibaba. In *Proceedings of the 1st International Workshop on Deep Learning Practice for High-Dimensional Sparse Data* (Anchorage, Alaska) (*DLP-KDD '19*). Association for Computing Machinery, New York, NY, USA, Article 12, 4 pages. doi:10.1145/3326937.3341261
- Chen et al. (2025) Xiangyi Chen, Kousik Rajesh, Matthew Lawhon, Zelun Wang, Hanyu Li, Haomiao Li, Saurabh Vishwas Joshi, Pong Eksombatchai, Jaewon Yang, Yi-Ping Hsu, Jiajing Xu, and Charles Rosenberg. 2025. PinFM: Foundation Model for User Activity Sequences at a Billion-scale Visual Discovery Platform. (2025). arXiv:2507.12704 [cs.LG] https://arxiv.org/abs/2507.12704
- Coleman et al. (2023) Benjamin Coleman, Wang-Cheng Kang, Matthew Fahrbach, Ruoxi Wang, Lichan Hong, Ed H. Chi, and Derek Zhiyuan Cheng. 2023. Unified embedding: battle-tested feature representations for web-scale ML systems. In *Proceedings of the 37th International Conference on Neural Information Processing Systems* (New Orleans, LA, USA) (*NIPS '23*). Curran Associates Inc., Red Hook, NY, USA, Article 2453, 22 pages.
- Covington et al. (2016) Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep Neural Networks for YouTube Recommendations. In *Proceedings of the 10th ACM Conference on Recommender Systems* (Boston, Massachusetts, USA) (*RecSys '16*). Association for Computing Machinery, New York, NY, USA, 191–198. doi:10.1145/2959100.2959190
- Firooz et al. (2025) Hamed Firooz, Maziar Sanjabi, Adrian Englhardt, Aman Gupta, Ben Levine, Dre Olgiati, Gungor Polatkan, Iuliia Melnychuk, Karthik Ramgopal, Kirill Talanine, Kutta Srinivasan, Luke Simon, Natesh Sivasubramoniapillai, Necip Fazil Ayan, Qingquan Song, Samira Sriram, Souvik Ghosh, Tao Song, Vignesh Kothapalli, Xiaoling Zhai, Ya Xu, Yu Wang, and Yun Dai. 2025. 360Brew: A Decoder-only Foundation Model for Personalized Ranking and Recommendation. (2025). arXiv:2501.16450 [cs.IR] https://arxiv.org/abs/2501.16450
- Gu et al. (2020) Yulong Gu, Zhuoye Ding, Shuaiqiang Wang, Lixin Zou, Yiding Liu, and Dawei Yin. 2020. Deep Multifaceted Transformers for Multi-objective Ranking in Large-Scale E-commerce Recommender Systems. In *Proceedings of the 29th ACM International Conference on Information & Knowledge Management* (Virtual Event, Ireland) (*CIKM '20*). Association for Computing Machinery, New York, NY, USA, 2493–2500. doi:10.1145/3340531.3412697
- Han et al. (2025) Ruidong Han, Bin Yin, Shangyu Chen, He Jiang, Fei Jiang, Xiang Li, Chi Ma, Mincong Huang, Xiaoguang Li, Chunzhen Jing, Yueming Han, Menglei Zhou, Lei Yu, Chuan Liu, and Wei Lin. 2025. MTGR: Industrial-Scale Generative Recommendation Framework in Meituan. (05 2025). doi:10.48550/arXiv.2505.18654
- Hidasi et al. (2016) Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based Recommendations with Recurrent Neural Networks. arXiv:1511.06939 [cs.LG] https://arxiv.org/abs/1511.06939
- Hoffmann et al. (2022) Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. 2022. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556* (2022).
- Hou et al. (2024) Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, and Wayne Xin Zhao. 2024. Large Language Models are Zero-Shot Rankers for Recommender Systems. Springer-Verlag, Berlin, Heidelberg, 364–381. doi:10.1007/978-3-031-56060-6_24
- Huang et al. (2025) Yanhua Huang, Yuqi Chen, Xiong Cao, Rui Yang, Mingliang Qi, Yinghao Zhu, Qingchang Han, Yaowei Liu, Zhaoyu Liu, Xuefeng Yao, Yuting Jia, Leilei Ma, Yinqi Zhang, Taoyu Zhu, Liujie Zhang, Lei Chen, Weihang Chen, Min Zhu, Ruiwen Xu, and Lei Zhang. 2025. Towards Large-scale Generative Ranking. (2025). arXiv:2505.04180 [cs.IR] https://arxiv.org/abs/2505.04180
- Ju et al. (2025) Clark Mingxuan Ju, Liam Collins, Leonardo Neves, Bhuvesh Kumar, Louis Yufeng Wang, Tong Zhao, and Neil Shah. 2025. Generative Recommendation with Semantic IDs: A Practitioner's Handbook. (2025). arXiv:2507.22224 [cs.IR] https://arxiv.org/abs/2507.22224
- Kang and McAuley (2018) Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential Recommendation. arXiv:1808.09781 [cs.IR] https://arxiv.org/abs/1808.09781
- Li et al. (2025) Kaiyuan Li, Rui Xiang, Yong Bai, Yongxiang Tang, Yanhua Cheng, Xialong Liu, Peng Jiang, and Kun Gai. 2025. BBQRec: Behavior-Bind Quantization for Multi-Modal Sequential Recommendation. (2025). arXiv:2504.06636 [cs.IR] https://arxiv.org/abs/2504.06636
- Liu et al. (2022) Zhuoran Liu, Leqi Zou, Xuan Zou, Caihua Wang, Biao Zhang, Da Tang, Bolin Zhu, Yijie Zhu, Peng Wu, Ke Wang, and Youlong Cheng. 2022. Monolith: Real Time Recommendation System With Collisionless Embedding Table. arXiv:2209.07663 [cs.IR] https://arxiv.org/abs/2209.07663
- Mei et al. (2025) M. Jeffrey Mei, Florian Henkel, Samuel E. Sandberg, Oliver Bembom, and Andreas F. Ehmann. 2025. Semantic IDs for Music Recommendation. *arXiv preprint arXiv:2507.18800* (2025). arXiv:2507.18800 [cs.IR]
- Naumov et al. (2019) Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu, Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii, Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko, Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong, and Misha Smelyanskiy. 2019. Deep Learning Recommendation Model for Personalization and Recommendation Systems. (2019). arXiv:1906.00091 [cs.IR] https://arxiv.org/abs/1906.00091
- Netflix (2024) Netflix. 2024. Foundation Model for Personalized Recommendation. https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39.
- Pang et al. (2025) Ming Pang, Chunyuan Yuan, Xiaoyu He, Zheng Fang, Donghao Xie, Fanyi Qu, Xue Jiang, Changping Peng, Zhangang Lin, Zheng Luo, and Jingping Shao. 2025. Generative Retrieval and Alignment Model: A New Paradigm for E-commerce Retrieval. In *Companion Proceedings of the ACM on Web Conference 2025* (Sydney NSW, Australia) (*WWW '25*). Association for Computing Machinery, New York, NY, USA, 413–421. doi:10.1145/3701716.3715228
- Penha et al. (2025) Gustavo Penha, Edoardo D'Amico, Marco De Nadai, Enrico Palumbo, Alexandre Tamborrino, Ali Vardasbi, Max Lefarov, Shawn Lin, Timothy Heath, Francesco Fabbri, and Hugues Bouchard. 2025. Semantic IDs for Joint Generative Search and Recommendation. (2025). arXiv:2508.10478 [cs.IR] https://arxiv.org/abs/2508.10478
- Rajput et al. (2023) Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Keshavan, Trung Vu, Lukasz Heidt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, and Maheswaran Sathiamoorthy. 2023. Recommender systems with generative retrieval. In *Proceedings of the 37th International Conference on Neural Information Processing Systems* (New Orleans, LA, USA) (*NIPS '23*). Curran Associates Inc., Red Hook, NY, USA, Article 452, 17 pages.
- Singh et al. (2024) Anima Singh, Trung Vu, Nikhil Mehta, Raghunandan Keshavan, Maheswaran Sathiamoorthy, Yilin Zheng, Lichan Hong, Lukasz Heldt, Li Wei, Devansh Tandon, Ed Chi, and Xinyang Yi. 2024. Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations. In *Proceedings of the 18th ACM Conference on Recommender Systems* (Bari, Italy) (*RecSys '24*). Association for Computing Machinery, New York, NY, USA.
- Tay et al. (2022) Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, and Donald Metzler. 2022. Transformer memory as a differentiable search index. In *Proceedings of the 36th International Conference on Neural Information Processing Systems* (New Orleans, LA, USA) (*NIPS '22*). Curran Associates Inc., Red Hook, NY, USA, Article 1587, 13 pages.
- Team (2024) Gemini Team. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. *arXiv preprint arXiv:2403.05530* (2024).
- Wang et al. (2025) Dongsheng Wang, Yuxi Huang, Shen Gao, Yifan Wang, Chengrui Huang, and Shuo Shang. 2025. Generative Next POI Recommendation with Semantic ID. (2025). arXiv:2506.01375 [cs.IR] https://arxiv.org/abs/2506.01375
- Wang et al. (2021) Ruoxi Wang, Rakesh Shivanna, Derek Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed Chi. 2021. DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems. In *Proceedings of the Web Conference 2021* (Ljubljana, Slovenia) (*WWW '21*). Association for Computing Machinery, New York, NY, USA, 1785–1797. doi:10.1145/3442381.3450078
- Weller et al. (2025) Orion Weller, Michael Boratko, Iftekhar Naim, and Jinhyuk Lee. 2025. On the Theoretical Limitations of Embedding-Based Retrieval. arXiv:2508.21038 [cs.IR] https://arxiv.org/abs/2508.21038
- Xu et al. (2025) Yi Xu, Moyu Zhang, Chenxuan Li, Zhihao Liao, Haibo Xing, Hao Deng, Jinxin Hu, Yu Zhang, Xiaoyi Zeng, and Jing Zhang. 2025. MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation. (2025). arXiv:2508.15281 [cs.IR] https://arxiv.org/abs/2508.15281
- Yang et al. (2024) Liu Yang, Fabian Paischer, Kaveh Hassani, Jiacheng Li, Shuai Shao, Zhang Gabriel Li, Yun He, Xue Feng, Nima Noorshams, Sem Park, Bo Long, Robert D Nowak, Xiaoli Gao, and Hamid Eghbalzadeh. 2024. Unifying Generative and Dense Retrieval for Sequential Recommendation. (2024). arXiv:2411.18814 [cs.IR] https://arxiv.org/abs/2411.18814
- Yang et al. (2025) Yuhao Yang, Zhi Ji, Zhaopeng Li, Yi Li, Zhonglin Mo, Yue Ding, Kai Chen, Zijian Zhang, Jie Li, Shuanglong Li, and Lin Liu. 2025. Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations. (2025). arXiv:2503.02453 [cs.IR] https://arxiv.org/abs/2503.02453
- Yao et al. (2025) Yining Yao, Ziwei Li, Shuwen Xiao, Boya Du, Jialin Zhu, Junjun Zheng, Xiangheng Kong, and Yuning Jiang. 2025. SaviorRec: Semantic-Behavior Alignment for Cold-Start Recommendation. (2025). arXiv:2508.01375 [cs.IR] https://arxiv.org/abs/2508.01375
- Ye et al. (2025) Wencai Ye, Mingjie Sun, Shaoyun Shi, Peng Wang, Wenjin Wu, and Peng Jiang. 2025. DAS: Dual-Aligned Semantic IDs Empowered Industrial Recommender System. (2025). arXiv:2508.10584 [cs.IR] https://arxiv.org/abs/2508.10584
- Yi et al. (2019) Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, and Ed Chi. 2019. Sampling-bias-corrected neural modeling for large corpus item recommendations. In *Proceedings of the 13th ACM Conference on Recommender Systems* (Copenhagen, Denmark) (*RecSys '19*). Association for Computing Machinery, New York, NY, USA, 269–277. doi:10.1145/3298689.3346996
- Zhai et al. (2024) Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Jiayuan He, Yinghai Lu, and Yu Shi. 2024. Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations. In *Proceedings of the 41st International Conference on Machine Learning* (*Proceedings of Machine Learning Research, Vol. 235*), Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (Eds.). PMLR, 58484–58509. https://proceedings.mlr.press/v235/zhai24a.html
- Zhang et al. (2024) Buyun Zhang, Liang Luo, Yuxin Chen, Jade Nie, Xi Liu, Daifeng Guo, Yanli Zhao, Shen Li, Yuchen Hao, Yantao Yao, Guna Lakshminarayanan, Ellie Dingqiao Wen, Jongsoo Park, Maxim Naumov, and Wenlin Chen. 2024. Wukong: Towards a Scaling Law for Large-Scale Recommendation. (2024). arXiv:2403.02545 [cs.LG] https://arxiv.org/abs/2403.02545
- Zhao et al. (2019) Zhe Zhao, Lichan Hong, Li Wei, Jilin Chen, Aniruddh Nath, Shawn Andrews, Aditee Kumthekar, Maheswaran Sathiamoorthy, Xinyang Yi, and Ed Chi. 2019. Recommending what video to watch next: a multitask ranking system. In *Proceedings of the 13th ACM Conference on Recommender Systems* (Copenhagen, Denmark) (*RecSys '19*).
- Zheng et al. (2025) Carolina Zheng, Minhui Huang, Dmitrii Pedchenko, Kaushik Rangadurai, Siyu Wang, Gaby Nahum, Jie Lei, Yang Yang, Tao Liu, Zutian Luo, Xiaohan Wei, Dinesh Ramasamy, Jiyan Yang, Yiping Han, Lin Yang, Hangjun Xu, Rong Jin, and Shuang Yang. 2025. Enhancing Embedding Representation Stability in Recommendation Systems with Semantic ID. (2025). arXiv:2504.02137 [cs.IR] https://arxiv.org/abs/2504.02137
- Zhou et al. (2025) Guorui Zhou, Jiaxin Deng, Jinghao Zhang, Kuo Cai, Lejian Ren, Qiang Luo, Qianqian Wang, Qigen Hu, Rui Huang, Shiyao Wang, Weifeng Ding, Wuchao Li, Xinchen Luo, Xingmei Wang, Zexuan Cheng, Zixing Zhang, Bin Zhang, Boxuan Wang, Chaoyi Ma, Chengru Song, Chenhui Wang, Di Wang, Dongxue Meng, Fan Yang, Fangyu Zhang, Feng Jiang, Fuxing Zhang, Gang Wang, Guowang Zhang, Han Li, Hengrui Hu, Hezheng Lin, Hongtao Cheng, Hongyang Cao, Huanjie Wang, Jiaming Huang, Jiapeng Chen, Jiaqiang Liu, Jinghui Jia, Kun Gai, Lantao Hu, Liang Zeng, Liao Yu, Qiang Wang, Qidong Zhou, Shengzhe Wang, Shihui He, Shuang Yang, Shujie Yang, Sui Huang, Tao Wu, Tiantian He, Tingting Gao, Wei Yuan, Xiao Liang, Xiaoxiao Xu, Xugang Liu, Yan Wang, Yi Wang, Yiwu Liu, Yue Song, Yufei Zhang, Yunfan Wu, Yunfeng Zhao, and Zhanyu Liu. 2025. OneRec Technical Report. arXiv:2506.13695 [cs.IR] https://arxiv.org/abs/2506.13695

## 附录 A

### A.1. Scaling 先导（Pilot）研究

作为 scaling 研究的前奏，我们使用 MoE-900M 模型开展了一项 pilot 研究。该 pilot 的目标是确定模型对两个超参数的敏感性：全局批大小与学习率。该 pilot 的结论将为后续更大规模的 scaling 分析提供超参数选择依据，确保计算资源得到有效利用。

#### A.1.1. 实验设置

我们运行了 4 个 trainer，各使用 256 块 TPU，并从（相同的）预训练检查点初始化。该 pilot 运行了约两周，这一时长足以确立清晰的性能趋势。"基础设置"（Trainer A）使用恒定学习率 $1\times 10^{-4}$ 以及饱和可用 HBM 的批大小。其余配置则系统性地改变这两个超参数之一。

**表7：使用 MoE-900M 的 Pilot 研究配置。**

| Trainer | 学习率 | 批大小 |
|---|---|---|
| A（基础） | $1\times 10^{-4}$ | 饱和 HBM |
| B | $1\times 10^{-4}$ | 0.5 倍饱和 HBM |
| C | $2\times 10^{-4}$ | 饱和 HBM |
| D | $5\times 10^{-5}$ | 饱和 HBM |

#### A.1.2. 结果

**学习率敏感性**。结果表明，训练过程对合理范围内的不同学习率具有容忍度。具体而言，在 pilot 研究期间，基础学习率 $1\times 10^{-4}$（Trainer A）与较低学习率 $5\times 10^{-5}$（Trainer D）之间的性能差异不具统计显著性。但是，将学习率翻倍至 $2\times 10^{-4}$（Trainer C）会导致性能明显变差，表明该学习率可能超出了该模型与数据集的最优范围。

**批大小敏感性**。使用更小的批大小会损害训练效率。批大小减半需要超过两倍的训练步数才能达到可比的性能水平。这凸显了为 compute-optimal 训练将批大小最大化至硬件内存极限的重要性。

### A.2. 上下文学习示例

表8 给出了若干示例，用于展示经过 CPT（包含语义ID）之后模型的上下文学习能力。基于预训练 LLM 的 CPT 模型能够遵循少样本任务描述，以语义上恰当的短语正确补全句子。相比之下，若将同样的推荐预训练应用于随机初始化的模型，该模型则难以形成连贯的短语，且无法区分 SID token 与文本 token。

**表8：少样本上下文学习（in-context learning）的定性示例。每个子表展示基于文本的提示与两种不同初始化方式下模型的输出。（上方的缩略图展示提示中每个语义ID所对应的视觉内容。）**

**(a) 示例 1**

| 项目 | 内容 |
|---|---|
| 视频缩略图 | （见原文，略） |
| 少样本输入 | The video A1991 … H10 is about health and pain management. / The video A364 … H37 is about the basics of quantum physics. / The video A37 … H25 is about |
| LLM 初始化 CPT 输出 | psychology and mind |
| 随机初始化 CPT 输出 | 100% video A252 H7 |

**(b) 示例 2**

| 项目 | 内容 |
|---|---|
| 视频缩略图 | （见原文，略） |
| 少样本输入 | Video A37 … H41 answers the question: do you really need 8 hours of sleep? / Video A1926 … H8 answers the question: how buy-now pay-later loan business makes profit? / Video A1882 … H32 answers the question: |
| LLM 初始化 CPT 输出 | the impact of language in your life. |
| 随机初始化 CPT 输出 | - - - 100% -1999-11-16 |

（译注：表8 中少样本输入与模型输出为原文英文示例，原样保留；缩略图图片无法在 Markdown 中呈现，请参照原文 PDF。）

---

*（全文完。本翻译依据 arXiv:2510.07784v1 官方 HTML 版本完成；公式、图表编号与原文一一对应。）*
