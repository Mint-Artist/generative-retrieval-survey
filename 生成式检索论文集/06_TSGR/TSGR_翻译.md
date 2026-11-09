# TSGR：淘宝搜索生成式检索

**英文原题：** TSGR: Taobao Search Generative Retrieval

**作者与机构（保留原文）：**

Tianyu Zhan, Gui Ling, Tong Xiong, Kunhai Lin, Yang Wang, Kaixuan Zhang, Zhihong Chen, Yuliang Yan, Dan Ou, Shengyu Zhang, Haihong Tang, Bo Zheng

- Taobao & Tmall Group of Alibaba, Hangzhou, China（除 Shengyu Zhang 外全体作者）
- Zhejiang University, Hangzhou, China（Shengyu Zhang）

**arXiv 编号与译注：**

- arXiv:2607.18796 [cs.IR]，最新版本 v2（v1 提交于 2026-07-21，v2 更新于 2026-07-22），本文依据 v2 的官方 HTML 版翻译。
- 关键词（原文）：Generative Retrieval, Semantic ID, Business Value Modeling（生成式检索、语义ID、商业价值建模）。
- 译注 1：原文采用 ACM 会议论文模板，版权信息块（会议名、日期、ISBN 等）为模板占位符且未填写，译文从略。CCS 概念为 Information systems → Retrieval models and ranking（信息系统 → 检索模型与排序）。
- 译注 2：论文中的在线 A/B 收益为 IPV +0.43%、交易笔数（Transaction Count）+1.12%、GMV +1.64%；IPV 指商品详情页浏览量（Item Page View）。离线主指标 HR@1000 较 FORGE 提升 9.16%（0.7735 → 0.8651）。
- 译注 3：文中 Pre-SFT、SFT、RL、HR@K、CTR、CVR、GMV、PV、IPV、SID、GR 等指标名/缩写按惯例保留英文；query 译作"查询"（淘宝场景即"搜索词"），item 译作"商品"。

---

## 摘要

生成式检索（Generative Retrieval, GR）通过训练单个自回归模型直接生成目标商品的语义ID（Semantic ID, SID），已在工业电商搜索中展现出强大潜力。然而，现有 GR 系统主要面向语义匹配进行优化，对商品商业价值并不敏感：SID 的构建不感知价值，候选的排序也无法获取商品边信息（side-info）。其结果是，高价值商品在检索阶段经常被漏检或被降权，限制了下游的业务收益。这一局限在淘宝搜索这类以业务目标为系统设计核心的工业场景中尤为关键。为此，我们提出淘宝搜索生成式检索（Taobao Search Generative Retrieval, TSGR），一个将价值感知同时融入商品表示与候选排序的统一生成式检索框架。1）在商品表示方面，TSGR 提出查询感知并行语义ID（Query-aware Parallel SID, QP-SID），通过构建源自查询-商品统计的并行码本，把查询条件下的价值排序编入 SID 构建，使价值更高且与查询相关的商品获得更优的 token 索引。2）在候选排序方面，我们提出价值感知排序模块（Value-aware Ranking Module, VRM），该模块构建于 GR 之上并与之联合优化，使单个模型能够同时充当检索器与预排序器（pre-ranker），无需专门的预排序阶段。渐进式训练流水线进一步使模型与语义相关性、用户偏好和业务目标对齐。离线实验表明，TSGR 在 HR@1000 上取得 9.16% 的提升；在线 A/B 测试进一步验证了其有效性，带来 IPV +0.43%、交易笔数 +1.12%、GMV +1.64% 的收益。TSGR 已全量部署于生产环境。

**关键词：** 生成式检索、语义ID、商业价值建模

---

## 1. 引言

生成式检索（GR）已成为工业电商搜索检索阶段一种富有前景的范式，它利用大语言模型（LLM）的序列生成能力，训练单个自回归模型直接生成目标商品的语义ID（SID）(Tay et al., 2022; Rajput et al., 2023)。这一建模方式实现了端到端优化，同时大幅降低了维护大规模索引的基础设施开销。基于这些优势，GR 已在推荐 (Zhou et al., 2025)、搜索 (Chen et al., 2025) 和广告 (Zhang et al., 2025a, 2026) 等一系列工业应用中快速地大规模部署。

在这些实际收益之外，GR 在概念上与信息检索的本质高度契合：生成模型将整个商品集合的知识编码进其参数，并针对查询重构相关商品；而 SID 编码将庞大的商品空间压缩为紧凑的 token 词表，使全空间优化成为可能，并缓解了传统双塔（dual-encoder）模型的负采样偏差。这些性质共同使 GR 成为工业搜索中大规模语义匹配的坚实基础。

然而，工业电商搜索最终由业务目标驱动（如 CTR、CVR 和 GMV）。下游排序阶段之所以高度依赖这些信号为候选打分，正是因为两个与查询同等相关的商品，其转化概率可能截然不同。当检索阶段不感知价值信号时，它可能检索出语义上合理但商业上逊色的候选，从而限制下游排序器的业务收益 (Zhang et al., 2026)。这种对商业价值的不敏感源于现有电商 GR 框架的运作方式：价值信号很少被融入商品表示或模型输出，由此产生两大关键局限。

**价值不敏感的 SID 构建。** 现有基于 RQ 的 SID 流水线主要设计用于保持多模态语义相似性，普遍缺乏反映商品商业价值的显式机制 (Rajput et al., 2023; Zhou et al., 2025)。结果，语义相近但价值悬殊的商品往往被映射到同一 SID 邻域，形成一个语义上组织良好但价值上几乎无差异的检索空间。此外，SID 通常全局固定且与查询无关，每个商品无论查询为何都携带完全相同的标识，忽视了其相关性与价值会随不同搜索意图而变化。

**价值无感知的候选排序。** 标准 GR 依据束搜索（beam search）下 SID token 的概率对候选排序，反映的更多是语义相关性而非商业相关性。此外，由于候选是在推理时动态生成的，模型无法在输入端预取丰富的商品边信息（如特征与价值信号）。因此，按束搜索分数排序得到的候选，往往偏离最可能被点击或购买的商品。

为解决这些局限，我们提出淘宝搜索生成式检索（TSGR），一个通过两项核心创新将价值感知同时融入商品表示与候选排序的统一生成式框架。在商品表示方面，我们观察到共享相同第三级 SID token 的商品会相互强化训练信号，这促使我们将高价值商品分配到更靠前的 token 索引。基于这一洞察，我们提出查询感知并行语义ID（QP-SID），构建多种并行的商品排序，包括基于整体点击量的默认排序，以及基于查询特定点击统计的查询条件排序。通过将价值更高且与查询相关的商品分配到更靠前的索引，QP-SID 自然地将生成似然与商业价值及查询相关性对齐。在候选排序方面，我们提出价值感知排序模块（VRM），从生成主干（backbone）的隐状态中提取用户表示，并通过交叉注意力（cross-attention）将其与丰富的商品边信息融合，产生与业务目标对齐的分数，用于对束搜索候选重排序。至关重要的是，该模块与生成目标在同一模型内联合优化，天然地将 GR 扩展为一个无需专门预排序阶段的统一检索与预排序框架。为进一步使 TSGR 与下游业务目标对齐，我们采用由 Pre-SFT 多任务预训练与带权多正例（multi-positive）监督的 SFT 组成的渐进式训练流水线。我们还探索了强化学习（RL）的作用，并发现应用 VRM 能取得更优的性能。

我们的主要贡献总结如下：

- 我们提出 TSGR，一个在单个模型内集成检索与预排序、并使整个流水线与业务目标对齐的统一生成式框架。
- 我们设计了 QP-SID，一种将价值信号编码进 SID、并从查询-商品点击统计中导出查询感知排序的商品表示，解决了传统 SID 的价值不敏感与查询无关问题。
- 我们提出 VRM，通过交叉注意力将主干导出的用户表示与商品边信息融合，并与生成主干联合优化，使单个模型同时充当检索器与预排序器。
- 我们在真实淘宝搜索数据上开展了大量离线实验与大规模在线 A/B 测试，取得 IPV +0.43%、交易笔数 +1.12%、GMV +1.64% 的收益。TSGR 已全量部署于生产环境。

---

## 2. 相关工作

### 2.1 生成式检索与推荐

生成式检索与推荐将商品检索重新表述为一个序列生成问题：模型以自回归方式直接产生商品的离散 SID。DSI (Tay et al., 2022) 首次通过将文档语料记忆于模型参数之中，论证了这一范式。TIGER (Rajput et al., 2023) 采用 RQ-VAE 将商品嵌入压缩为层次化的 token 序列，通过共享的量化词表统一语义信号与协同信号；IDGenRec (Tan et al., 2024) 则进一步通过文本化 ID 生成，将生成模型与商品 ID 空间相连。

随着生成模型扩展到十亿参数级的 LLM，这一范式已从概念验证演进到工业部署。在架构方面，HSTU (Zhai and others, 2024) 与 HLLM (Chen et al., 2024) 通过定制的注意力机制与层次化 LLM 设计提升效率，而 OneRec (Zhou et al., 2025) 将推荐表述为带偏好对齐的通用序列预测任务。在电商搜索中，OneSearch (Chen et al., 2025) 提出了统一的端到端生成式框架，OneSearch-V2 (Chen et al., 2026) 进一步引入潜在推理与自蒸馏对其进行扩展。类似趋势也出现在广告 (Zhang et al., 2025a; Zheng et al., 2025; Zhang et al., 2026) 与通用搜索 (Han et al., 2025; Hao et al., 2025) 中。

### 2.2 语义ID构建

语义ID将商品转换为紧凑的离散 token 序列，作为生成式检索的生成目标。TIGER (Rajput et al., 2023) 率先将 RQ-VAE 用于层次量化，后续工作通过 RQ-KMeans (Zhou et al., 2025)、OPQ (Hou et al., 2025) 与联合乘积量化 (Petrov and Macdonald, 2024) 简化了这一过程。为丰富 SID 的语义，LETTER (Wang et al., 2024) 与 LC-Rec (Zheng et al., 2024) 将多模态特征与协同信号对齐；GSID (Yang et al., 2025)、CAT-ID2 (Liu et al., 2026) 与 Hi-Gen (Wu et al., 2024) 则注入属性层级、类目树等领域特定结构。CQ-SID (Zhu et al., 2026) 将商品编码为语义聚类而非唯一标识，以降低束搜索开销。在工业规模上，FORGE (Fu et al., 2025) 在拥有 140 亿次交互的淘宝数据上对 SID 构建进行了基准评测，UniSID (Jiang et al., 2026) 则以端到端方式联合优化 SID 构建与推荐目标。

所有现有方法都为每个商品分配单一的全局 SID，使其无论查询上下文如何都被固定在单一表示中，并在码本构建期间忽视商品商业价值。TSGR 通过查询感知、价值驱动的码本为每个商品构建并行 SID，同时解决上述两个局限：同一商品可在不同查询下关联不同表示，且每个聚类内优先安排高价值商品。

### 2.3 价值感知排序

工业搜索系统通常采用级联架构，检索与排序由相互独立的模型承担 (Pei et al., 2019)，这容易引入目标错位，并阻碍商品价值信号参与候选生成 (Wang et al., 2026)。为缩小这一差距，一系列工作探索直接使用生成模型进行排序。GenRank (Huang et al., 2025) 展示了工业规模的生成式排序，QGS (Song et al., 2026b) 通过 HFG-Attention 将异构特征融入生成式排序。UniVA (Zhang et al., 2026) 提出了 Generation-as-Ranking 建模与 eCPM 感知的强化学习；GoalRank (Zhang et al., 2025b)、GReF (Lin et al., 2025) 与 DeGRe (Song et al., 2026a) 则通过奖励模型与稠密监督引入排序信号。

然而，这些方法大多依赖生成概率作为排序分数，其反映的是序列似然而非真实商品价值，且对工业系统中价值估计至关重要的丰富商品边信息支持有限。相比之下，TSGR 在生成模型内部直接将商品边信息纳入价值感知打分，在单一联合优化目标下统一检索与排序。

---

## 3. 方法

图 1：TSGR 总览：一个在单个模型内联合处理检索与预排序的统一生成式框架。(a) 并行 SID 构建在第 1–2 层使用类目与聚类先验构建三层码本，在第 3 层使用查询感知码本。(b) 生成主干在第 1–2 层进行带前缀约束解码的层次化束搜索，在第 3 层进行标准束搜索。VRM 通过融合 SID 嵌入与商品边信息对候选重排序。

TSGR 是面向工业电商搜索的统一生成式框架，在单个模型内集成检索与预排序。如图 1 所示，查询感知并行 SID 构建（第 3.1 节）为每个商品构建多条不同的 SID 路径，并依据查询选择最合适的一条。价值感知排序模块（第 3.2 节）构建于生成主干之上，对束搜索候选进行重排序。第 3.3 节给出训练流水线，它通过 Pre-SFT 多任务预训练与带权多正例训练的 SFT 渐进式地发展模型能力。此外，我们还探索了 RL 的作用。

### 3.1 查询感知并行 SID 构建

我们提出一种并行 SID 构建方法，基于查询-商品统计为每个商品构建不同的 SID 表示。具体而言，我们利用两类先验信息构建两个码本：一个配置为 32768×8192 的语义码本，用于捕捉商品的语义结构；以及一个总大小为 8192、由 N 个并行分支组成的效率码本，以支持多样化的检索路径，同时将解码预算集中于高概率分支，避免在低概率路径上消耗不必要的搜索容量 (Ling et al., 2026)。

**语义码本构建。** 前两层构成语义码本，它建立在同款商品聚类（same-item cluster）与商品类目等现有商品特征之上，构成自然的商品层次结构：商品 ∈ 聚类 ∈ 类目。我们首先对每个组内所有商品的嵌入取平均，得到聚类级与类目级的嵌入：

$$
\mathbf{e}_{\text{cat}}(g)=\frac{1}{|\mathcal{G}|}\sum_{i\in\mathcal{G}}\mathbf{e}_{i}\qquad \mathbf{e}_{\text{cluster}}(g)=\frac{1}{|\mathcal{C}|}\sum_{i\in\mathcal{C}}\mathbf{e}_{i} \tag{1}
$$

其中 $\mathcal{G}$ 与 $\mathcal{C}$ 分别表示同类目与同聚类中的商品集合，$\mathbf{e}_{i}\in\mathbb{R}^{d}$ 为商品 $i$ 的嵌入。第一层码本直接用类目嵌入初始化，因为类目数量（8100+）与 8192 的码本大小非常接近。随后对聚类嵌入与类目嵌入之间的残差应用 RQ-KMeans，学习第二层码本：

$$
\mathbf{r}(g)=\mathbf{e}_{\text{cluster}}(g)-\mathbf{e}_{\text{cat}}(g) \tag{2}
$$

这产生了一个三层的 8192×4×8192 中间码本。由于总共四层的码本在生产环境中会带来难以承受的部署成本，我们将前两层合并为单层，得到最终大小为 32768×8192 的语义码本。

这一构建向语义码本注入了两个结构性先验：

- **类目先验**：领域分类体系被直接嵌入最粗粒度的划分中，使 SID 空间继承有意义的类目结构。
- **聚类先验**：通过对聚类中心（而非单个商品嵌入）进行量化，保证同一聚类内的所有商品共享完全相同的语义 SID 前缀。

这些先验共同使 SID 空间在语义上组织良好，并为效率码本的构建提供了干净的聚类划分。

**效率码本构建。** 第三层构成效率码本，在每个聚类内独立构建，依据基于点击的价值对商品排序。价值更高的商品占据更靠前的 token 索引，而这些索引在训练中被访问得更频繁，从而将解码预算集中于最有希望的商品，并使生成似然与商品价值对齐。

一种直接的设计是按商品过去 30 天的点击量进行排序，得到默认排序：

$$
\pi_{0}=\operatorname{argsort}_{i\in\mathcal{C}}\,s_{0}(i) \tag{3}
$$

其中 $\mathcal{C}$ 表示聚类内的商品集合，$s_{0}(i)$ 为商品 $i$ 的点击量。然而，这一排序仍与查询无关，忽视了同一商品可能与某个查询高度相关而与另一个查询无关的事实。为克服这一点，我们提出 QP-SID，其依据如下观察：每个查询都可以切分为若干关键词项（term），而每个商品都属于某个聚类，因此"词项-聚类"关系可以由查询-商品统计推断出来。具体地，对每个聚类，我们识别其最具代表性的前 3 个词项，并过滤掉覆盖率超过 80% 的词项，其中覆盖率定义为在该词项下被点击过的聚类内商品所占比例：

$$
\text{cov}(t,\mathcal{C})=\frac{|\{i\in\mathcal{C}:s(i,t)>0\}|}{|\mathcal{C}|} \tag{4}
$$

一个几乎覆盖整个聚类的词项几乎不携带判别信息，因为无论查询意图如何，它都与几乎所有商品相关联。对每个保留的代表词项 $t$，令 $s(i,t)$ 表示商品 $i$ 在词项 $t$ 下过去 30 天的点击量。按 $s(i,t)$ 对商品排序即得到词项特定的排序：

$$
\pi_{t}=\operatorname{argsort}_{i\in\mathcal{C}}\,s(i,t) \tag{5}
$$

由此产生 3 个词项条件排序，连同默认排序 $\pi_{0}$，每个聚类共得到 4 个并行排序。案例分析见附录 A.4。

表 1：Pre-SFT 任务设计。每个任务通过变换输入上下文激活特定的能力维度。多数任务共享统一的 SID 输出格式；标题生成任务使用自然语言输出，以在文本层面激活意图理解。

| 任务类别 | 输入 | 输出 | 能力 |
| --- | --- | --- | --- |
| 编码（Encoding） | 商品标题 | SID | 学习商品语义到 SID 空间的映射 |
| 个性化（Personalization） | 用户画像 + 查询 + 行为序列 | 标题 | 在文本层面激活用户意图理解 |
| 检索（Retrieval） | 搜索查询 | SID | 为域内检索建立查询-商品相关性 |
| 协同（Collaboration） | 触发商品（标题 + SID） | SID | 将共购协同信号注入 SID 空间 |
| 推荐（Recommendation） | 用户画像 + 行为序列 | SID | 支持个性化与跨域泛化 |

### 3.2 价值感知排序模块

在 GR 中，候选按生成概率排序，反映的是序列似然而非商业价值。由于候选商品只有在解码之后才可知，其边信息无法在生成过程中引入。一种自然的补救手段是使用一个独立的下游预排序模型，但这会把两个目标错位的模型耦合在一起：GR 模型在其自身的检索分布上优化 SID 生成似然，而预排序模型则对来自多源候选池的完整候选打分，两者之间的分布差异（distribution gap）会损害其组合效果。

为此，我们在生成主干之上引入 VRM。如图 1(b) 所示，该模块以生成模型的用户表示与商品边信息为输入，为每个候选输出一个价值分数。下面依次描述其四个组成部分：商品表示、用户表示、用户-商品交互与候选打分。

**商品表示。** 对于来自束搜索的每个候选 $(c_{1},c_{2},c_{3})\in\mathcal{S}$，我们从三个互补的来源构建商品表示。第一是从离线特征表取回的商品边信息向量 $\mathbf{x}_{k}\in\mathbb{R}^{d_{f}}$，它聚合了四类特征：(1) 多个时间窗口上的行为统计（如浏览量与支付件数）；(2) 由商品类目层级导出的类目特征；(3) 刻画跨场景行为模式的全局卖家统计；以及 (4) 查询-商品共现特征。第二是从离线特征表取回的商品嵌入 $\mathbf{e}_{k}$，提供可能在 SID 构建过程中丢失的信息。第三是 SID 嵌入，通过主干嵌入层映射候选 SID $(c_{1},c_{2},c_{3})$ 并用 MLP $\phi(\cdot)$ 压缩得到：

$$
\mathbf{z}_{k}=\phi\big(\text{Emb}(c_{1}),\ \text{Emb}(c_{2}),\ \text{Emb}(c_{3})\big) \tag{6}
$$

三者拼接构成最终的商品表示：

$$
\mathbf{v}_{k}=[\mathbf{x}_{k};\ \mathbf{e}_{k};\ \mathbf{z}_{k}] \tag{7}
$$

**用户表示。** 用户表示复用生成主干的隐状态，因此不产生额外的编码开销。给定长度为 $L$ 的输入提示（prompt），主干产生上下文化表示 $\mathbf{H}_{u}\in\mathbb{R}^{L\times D}$，其中 $D$ 为隐层维度。对全部 token 位置做均值池化（mean pooling）并接一个 MLP $\psi(\cdot)$，得到全局用户表示：

$$
\mathbf{u}=\psi\!\left(\frac{1}{L}\sum_{i=1}^{L}\mathbf{H}_{u}[i]\right) \tag{8}
$$

**用户-商品交互。** 为捕捉全局摘要之外的细粒度相关性，我们引入一个注意力模块，让商品表示 $\mathbf{v}_{k}$ 对用户上下文表示 $\mathbf{H}_{u}$ 做注意力：

$$
\mathbf{a}_{k}=\text{CrossAttn}(\mathbf{v}_{k},\ \mathbf{H}_{u},\ \mathbf{H}_{u}) \tag{9}
$$

其中 $\mathbf{v}_{k}$ 作为查询（query），$\mathbf{H}_{u}$ 同时作为键（key）与值（value）。这使得每个候选都能选择性地关注用户上下文中与其最相关的部分，得到编码用户-商品相关性的交互表示 $\mathbf{a}_{k}$。

**特征融合与打分。** 交互表示、用户表示与商品表示拼接后送入三个并行打分头，产生相应的预测分数：

$$
\hat{y}_{k}^{t}=\sigma\big(\text{MLP}_{t}\big([\mathbf{a}_{k};\ \mathbf{u};\ \mathbf{v}_{k}]\big)\big)\quad t\in\{\text{pv},\ \text{ctr},\ \text{cvr}\} \tag{10}
$$

其中 $\sigma(\cdot)$ 为 sigmoid 函数。推理时，候选按复合分数 $\hat{y}_{k}^{\text{pv}}\times\hat{y}_{k}^{\text{ctr}}$ 重排序，产生最终输出。由于该模块共享主干嵌入并复用生成过程中已计算的隐状态，它以可忽略的额外代价，在单个模型内统一了检索与价值感知排序。

表 2：所提方法在淘宝搜索数据集上的性能对比。每个类别内的方法互斥，† 标记 TSGR 所采用的方法。最优结果加粗。

| 类别 | 方法 | HR@20 | HR@100 | HR@500 | HR@1000 | HR@5000 |
| --- | --- | --- | --- | --- | --- | --- |
| SID | FORGE | 0.4328 | 0.5863 | 0.7207 | 0.7735 | 0.8604 |
| SID | QP-SID† | 0.4635 | 0.6235 | 0.7669 | **0.8177** | 0.8961 |
| 预处理 | Pre-SFT† | **0.4694** | 0.6340 | 0.7740 | 0.8222 | 0.8987 |
| 后处理 | 预排序模型（Pre-Ranking Model） | 0.4248 | 0.6264 | 0.8112 | 0.8552 | 0.8771 |
| 后处理 | RL | 0.4659 | 0.6420 | 0.7857 | 0.8277 | **0.9027** |
| 后处理 | VRM (TSGR)† | 0.4586 | **0.6677** | **0.8250** | **0.8651** | 0.8953 |

### 3.3 训练流水线

图 2：训练流水线。模型能力通过 Pre-SFT 多任务预训练与带权多正例损失的 SFT 渐进式地发展。

训练 TSGR 需要三种能力协同发展：商品内容与 SID 空间的语义对齐、个性化的查询到 SID 生成、以及价值感知的候选排序。在单个监督阶段内同时习得这三者十分困难，因此我们采用图 2 所示的流水线逐步构建这些能力，详细的提示模板见附录 A.5。

#### 3.3.1 Pre-SFT

预训练的 LLM 并不具备将商品或查询映射到离散 SID token 的能力。与其强迫模型一次性掌握该能力以及所有其他能力，Pre-SFT 将目标分解为更简单的子任务，每个子任务只聚焦单一能力维度。在这些子任务的大规模样本上训练可以提前激活所需能力，从而降低后续 SFT 阶段的学习难度与数据需求。

我们据此设计了五类 Pre-SFT 任务，总结于表 1。多数任务共享统一的 SID 输出格式，但输入上下文各不相同。训练时，所有任务被打乱并以标准交叉熵损失联合优化。

#### 3.3.2 SFT

SFT 阶段将经 Pre-SFT 初始化的模型适配到个性化检索任务上。

**提示生成框架。** 为支持灵活的特征迭代，我们将 SFT 流水线构建于 Prompt Generation（PG）(Ou et al., 2026) 之上，它提供带可配置特征槽位的统一提示模板。这既支持快速的特征实验，又保证了训练-服务一致性。

**数据构建。** 为用 QP-SID 构建训练数据，第三级 token 由查询 $q$ 与商品 $i$ 所在聚类共同决定。给定查询 $q$，我们抽取其关键词项，并与商品 $i$ 所在聚类的代表词项集 $\mathcal{T}_{\text{cluster}}$ 进行匹配。当存在匹配时，选择最具代表性的匹配词项：

$$
t^{*}=\operatorname{arg\,max}_{t\,\in\,\mathcal{T}_{\text{cluster}}\cap\operatorname{terms}(q)}\operatorname{rel}(t,\mathcal{C}) \tag{11}
$$

其中 $\operatorname{rel}(t,\mathcal{C})$ 为词项 $t$ 与聚类 $\mathcal{C}$ 之间的共现点击量。第三级 token 随之被确定为：

$$
c_{3}(i,q)=\begin{cases}\pi_{t^{*}}(i)&\text{if }\mathcal{T}_{\text{cluster}}\cap\operatorname{terms}(q)\neq\emptyset\\ \pi_{0}(i)&\text{otherwise}\end{cases} \tag{12}
$$

这使得第三级 token 成为商品与查询的共同动态函数：当查询与商品的主要使用场景高度吻合时，该商品在其聚类内被分配到更靠前的索引，从而产生反映细粒度商品-查询相关性的查询感知训练标签。

**会话级带权多正例训练。** 对每个搜索会话，来自四个互动层级（支付、点击、pv、unpv）的商品被打包进单个训练样本，其中互动过的商品（支付、点击、pv）作为多个标签。标签之间施加注意力掩码，以防止跨标签干扰。每个标签被赋予满足 $w_{\text{pay}}>w_{\text{clk}}>w_{\text{pv}}$ 的优先级权重。生成损失是所有有效标签上的加权交叉熵：

$$
\mathcal{L}_{\text{gen}}=\sum_{k\in\mathcal{Y}}w_{k}\cdot\text{CE}\big(\mathbf{o}_{k},\ t_{k}\big) \tag{13}
$$

其中 $\mathcal{Y}$ 为会话中互动商品集合，$\mathbf{o}_{k}$ 为商品 $k$ 的 SID token 解码位置上的 logits，$t_{k}$ 为真实（ground-truth）SID token，$w_{k}$ 为相应权重。

VRM 在该阶段联合训练。其排序损失由 PV、CTR、CVR 预测分数上的三个二元交叉熵项组成，定义在同一 $\mathcal{Y}$ 上：

$$
\mathcal{L}_{\text{rank}}=\sum_{k\in\mathcal{Y}}\sum_{s\in\{\text{pv, ctr, cvr}\}}\text{BCE}\big(\hat{y}_{k}^{s},\ y_{k}^{s}\big) \tag{14}
$$

其中 $\hat{y}_{k}^{t}$ 为预测分数，$y_{k}^{t}$ 为任务 $t$ 的对应标签。总目标由两项组合而成：

$$
\mathcal{L}=\mathcal{L}_{\text{gen}}+\lambda\cdot\mathcal{L}_{\text{rank}} \tag{15}
$$

普通 SFT 使模型与观测到的用户行为对齐，但难以直接优化下游用户满意度信号。为进一步提升性能，我们基于 GRPO (Shao et al., 2024) 构建了完整的 RL 流水线，以在线排序模型作为奖励来源，提供直接的业务目标反馈。然而，与通过排序模块引入商品边信息（第 3.2 节）相比，基于 RL 的方法带来的额外增益有限。因此，我们采用排序模块作为主方案，完整的 RL 实验细节见附录 A.6。

---

## 4. 实验

为更全面地评估 TSGR 的有效性，本节开展了全面的离线与在线 A/B 评估。此外，还进行了大量消融实验以证明各模块的可行性。

### 4.1 实验设置

**数据集与基线。** 我们收集淘宝电商搜索平台两周内超过 2 亿次真实用户互动作为训练数据，测试集从其后一天采样得到。为评估 VRM 的泛化能力，我们还使用 TencentGR (Pan et al., 2026) 数据集对其进行了消融实验（见附录 A.7）。我们与两个基线进行比较：FORGE (Fu et al., 2025)，遵循其做法，将最后一级 token 随机初始化作为 SID 基线，其中每个 token 对应最多 5 个商品；以及预排序模型（Pre-Ranking Model），即在检索之后应用的传统预排序模型。

**评估指标。** 为检验检索性能，我们采用 HitRate@K（HR@K）作为评估指标。遵循工业生成式检索的标准做法 (Zhu et al., 2026; Chen et al., 2025)，我们在相同束宽度下报告 HR@K，并以 HR@1000 作为主要指标，因为检索的目标是确保相关商品出现在传递给下游排序器的候选池中。

**实现细节。** 我们采用 Qwen2.5-0.5B (Yang et al., 2024) 作为生成主干，三层 SID 码本（32768×8192×8192），束宽为 [400, 400, 10000]，以及来自 QP-SID 的每个商品 4 个并行第三级排序。SFT 使用 AdamW（$\beta_{1}=0.9$，$\beta_{2}=0.999$）、余弦学习率 $1\times10^{-4}$、批大小 256 进行训练，$\lambda=1$，价值分数权重 $(w_{\text{pv}}, w_{\text{clk}}, w_{\text{pay}})=(1,2,3)$。更多细节见附录 A.3。

### 4.2 总体对比

表 2 报告了所有对比方法在 HR@K 上的结果。我们围绕三个维度组织实验：SID 构建、预处理与后处理。

QP-SID 通过将商品价值信号与查询条件排序融入效率码本，在所有指标上全面超越 FORGE，HR@1000 提升 +4.42%。在每个聚类内，价值更高的商品被分配到更靠前的 token 索引，而这些索引跨聚类聚合了更多商品，因而获得更多训练信号。在全局价值排序之外，查询条件排序进一步将与查询相关的商品提升至更靠前的索引，帮助模型更有效地呈现相关候选。这两种机制共同确保模型同时优先考虑全局高价值商品与查询特定相关商品。

Pre-SFT 多任务预训练阶段进一步提升了性能，表明该阶段贡献了可观的收益。

作为后处理阶段的基线，传统预排序模型在 HR@500 与 HR@1000 上带来显著提升（+3.30%），但在其他指标上出现性能损失。相比之下，RL 训练在几乎所有指标上都取得了一致的能力提升（HR@1000 +0.55%）。然而，其提升幅度仍然有限，因此 RL 未在实际部署中采用。而 VRM 通过在生成目标联合优化下融合主干导出的用户表示与商品边信息，取得了更强的性能：既避免了跨阶段目标错位，又不引入额外的服务延迟。该模块在 HR@1000 上的提升与预排序模型相当（+3.73% vs. +3.30%），同时在几乎所有其他指标上优于后者。这些结果共同证实，TSGR 对检索与预排序的统一处理方式，比传统级联设计更有效、也更利于部署。

### 4.3 消融研究

为更好地考察每项设计的贡献，我们评估：1) SID 构建策略的影响；2) VRM 的有效性；3) 各训练阶段内的关键设计选择。出于效率考虑，部分消融实验在全量训练数据的采样子集上进行。

#### 4.3.1 SID 设计

图 3：不同 QP-SID 配置与训练策略的对比。每组输入/标签配置（行）与效率 ID 策略（列）的组合均报告 HR@1000 与 HR@100。

我们从 FORGE 逐步改进 SID 设计直至 QP-SID。用效率分数对 FORGE 候选进行重排序可以带来明显增益，但通过专用码本将效率直接编码进 SID 要有效得多。我们进一步用受类目与聚类先验约束的 RQ-KMeans 重构语义码本（第 3.1 节），提升可解释性与可扩展性。将四层码本整合为三层提高了对误差累积的鲁棒性，因为更浅的 SID 不易受到级联预测错误的影响。最后，QP-SID 引入查询感知的并行排序，取代单一的全局效率排序，并在服务时为每个查询选择最佳匹配的排序。

表 3：不同标签策略下 top-100 输出商品的层次化 SID 预测准确率。

| 标签策略 | 聚类命中率 | 第三层命中率 | 总命中率 |
| --- | --- | --- | --- |
| QP-SID（第 1 行） | 78.71 | 72.83 | 57.32 |
| 多路径（第 3 行） | 77.20 | 71.90 | 55.50 |

表 4：训练集与评估集以及模型输出中 QP-SID 类别的分布。

| 类别 | 训练序列 | 标签 | 评估序列 | 模型输出 |
| --- | --- | --- | --- | --- |
| 无命中（默认） | 55.37% | 23.20% | 61.42% | 32.05% |
| 词项排序 = 1 | 25.46% | 54.54% | 22.99% | 44.91% |
| 词项排序 = 2 | 11.63% | 15.00% | 9.71% | 14.53% |
| 词项排序 = 3 | 7.54% | 7.26% | 5.88% | 8.51% |

图 3 考察了不同 QP-SID 配置与训练策略的影响，揭示出两个关键发现。第一，输入与输出 SID 排序之间的一致性至关重要：使用 QP-SID 排序作为训练标签、却保留默认排序作为输入，会削弱模型复制输入 SID 序列的能力，导致 HR@100 大幅下降。第二，引入辅助的查询匹配标签会使第三层 SID 更难学习，放大其梯度并间接损害前两层的准确率，表 3 中的层次化预测准确率进一步印证了这一点。此外，我们观察到 99.8% 的模型输出塌缩到默认路径，表明模型未能利用 QP-SID 排序的多样性。这些观察促成了我们最终的 QP-SID 设计。

表 4 进一步从分布角度验证了这一点。行为序列中的多数商品属于默认类别，说明多数历史点击商品与当前查询关系不大。相比之下，多达 76.8% 的训练标签与查询相关（词项排序 = 1、2 或 3），这符合预期，因为用户点击时带着当前查询的意图。模型输出分布进一步确认，模型已成功学会从多个排序通道检索，而非仅依赖默认通道。

#### 4.3.2 价值感知排序模块

图 4：VRM 中不同分数组合效果的消融研究。

图 5：VRM 中不同架构变体效果的消融研究。

所有实验都在 5000 个商品的候选池上进行。对于排序分数（图 4），我们消融了推理时用于重排序的不同分数组合。所有组合在 HR@1000 上均较基线取得一致增益，其中 PV×CTR 取得最佳整体性能。鉴于 PV×CTR 与 HitRate 目标直接对应，这一结果符合预期。受此观察启发，我们进一步检验从训练中移除 CVR 损失是否会影响性能，结果发现在我们的设置中影响可以忽略。对于模块架构（图 5），完全移除商品嵌入会导致 HR@20 与 HR@1000 同时严重退化，证实商品表示不可或缺。原因在于 SID 构建过程不可避免地丢弃了大量商品信息，使商品嵌入成为恢复丢失语义细节的关键补充。剔除商品边信息会导致 HR@20 大幅下降，而对 HR@1000 的影响适中。值得注意的是，无论是把交叉注意力模块替换为均值池化，还是用直接送入 MLP 的 EOS token 表示替代它，都会使 HR@20 显著下降，而 HR@1000 仅出现轻微下降。这一模式表明，这两种设计可以作为延迟敏感场景下可行的轻量替代方案。更全面的消融结果见附录 A.8。

#### 4.3.3 训练策略

图 6：Pre-SFT 任务设计的消融研究。每个变体在 Encoding 基础上增加单个任务，而 Multi-task Fusion 联合训练所有任务。

图 6 考察了各 Pre-SFT 任务的贡献。每个任务都带来正向增益，其中检索任务因与下游目标最为贴合而贡献最大。个性化任务贡献最小，因为其标题生成格式相对 SID 目标引入了输出空间差异。联合训练全部任务取得 +5.76%，超过任何单个任务，证实这些任务激活了互补的能力维度。

表 5：SFT 训练目标的消融研究。

| 方法 | HR@20 | HR@100 | HR@500 | HR@1k | HR@5k |
| --- | --- | --- | --- | --- | --- |
| Pointwise | 0.3851 | 0.5653 | 0.7188 | 0.7714 | 0.8577 |
| Listwise | **0.4042** | **0.5890** | **0.7405** | **0.7922** | **0.8718** |

如表 5 所示，所提出的会话级带权多正例目标一致优于 pointwise 基线，证实显式建模多个正例之间的相对优先级能带来更好的序列生成质量。

### 4.4 在线 A/B 测试

TSGR 作为现有检索源之外的附加通道部署，其输出直接传递给排序阶段，从而绕过传统的预排序阶段。系统部署在 NVIDIA H20 GPU 上，延迟预算为 65ms。给定用户查询与上下文特征，模型执行约束束搜索：前两层 SID 在预构建的前缀树（prefix tree）所施加的有效前缀约束下解码，以剪除无效 token 序列；第三层使用标准束搜索，产生一组完整的 SID 序列。每个候选生成时，VRM 即刻计算其价值分数，随后对候选重排序并作为最终输出返回。

我们在淘宝搜索平台上将 TSGR 部署于生产 A/B 测试，连续 38 天服务 1% 的线上流量。在实验组中，TSGR 以单个模型取代传统的检索与预排序阶段，将其排序后的 top-1,500 候选直接传递给排序阶段。与生产基线相比，TSGR 取得了 IPV +0.43%、交易笔数 +1.12%、GMV +1.64% 的提升。这些改进具有统计显著性，并且在淘宝搜索的体量下转化为可观的业务价值。A/B 测试成功之后，TSGR 已全量部署于生产环境，在淘宝搜索平台上服务数亿用户。

---

## 5. 结论

我们提出了 TSGR，一个统一的生成式检索框架，解决了现有 GR 系统在工业电商搜索中的价值不敏感问题。我们的核心洞察是将价值感知同时嵌入商品表示与模型输出：QP-SID 将商品效率直接编码进语义标识符，而联合优化的价值感知排序模块使单个模型能够同时充当检索器与预排序器，消除了独立预排序阶段带来的目标错位。为进一步使模型与业务目标对齐，我们采用了 Pre-SFT、带权多正例 SFT 的训练流水线，并探索了 RL 的作用。在淘宝搜索上的大量离线实验与在线 A/B 测试表明，相较强基线取得了一致且显著的提升，TSGR 已成功部署于生产环境，规模化地服务真实流量。

---

## 参考文献（原样保留英文）

1. B. Chen, X. Guo, S. Wang, Z. Liang, Y. Lv, Y. Ma, X. Xiao, B. Xue, X. Zhang, Y. Yang, et al. (2025) Onesearch: a preliminary exploration of the unified end-to-end generative framework for e-commerce search. arXiv preprint arXiv:2509.03236.
2. B. Chen, S. Wang, Y. Ma, Z. Liang, X. Zhang, Y. Lv, Y. Yang, H. Dai, L. Mao, T. Zhao, et al. (2026) OneSearch-v2: the latent reasoning enhanced self-distillation generative search framework. arXiv preprint arXiv:2603.24422.
3. J. Chen, L. Chi, B. Peng, and Z. Yuan (2024) Hllm: enhancing sequential recommendations via hierarchical large language models for item and user modeling. arXiv preprint arXiv:2409.12740.
4. K. Fu, T. Zhang, S. Xiao, Z. Wang, X. Zhang, C. Zhang, Y. Yan, J. Zheng, Y. Li, Z. Chen, et al. (2025) Forge: forming semantic identifiers for generative retrieval in industrial datasets. arXiv preprint arXiv:2509.20904.
5. R. Han, B. Yin, S. Chen, H. Jiang, F. Jiang, X. Li, C. Ma, M. Huang, X. Li, C. Jing, et al. (2025) Mtgr: industrial-scale generative recommendation framework in meituan. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management, pp. 5731–5738.
6. X. Hao, M. Zhang, A. Li, X. Qian, Z. Ma, Y. Zang, S. Yang, Z. Han, X. Ma, J. Liu, et al. (2025) OxygenREC: an instruction-following generative framework for e-commerce recommendation. arXiv preprint arXiv:2512.22386.
7. Y. Hou, J. Li, A. Shin, J. Jeon, A. Santhanam, W. Shao, K. Hassani, N. Yao, and J. McAuley (2025) Generating long semantic ids in parallel for recommendation. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 956–966.
8. Y. Huang, Y. Chen, X. Cao, R. Yang, M. Qi, Y. Zhu, Q. Han, Y. Liu, Z. Liu, X. Yao, et al. (2025) Towards large-scale generative ranking. arXiv preprint arXiv:2505.04180.
9. J. Jiang, X. Zhang, E. Zhang, Y. Xiong, J. Zhang, J. Wang, H. Yu, Y. Wang, H. Wang, X. Yan, et al. (2026) End-to-end semantic id generation for generative advertisement recommendation. arXiv preprint arXiv:2602.10445.
10. Z. Lin, Z. Li, C. Dai, W. Bao, S. Lin, E. Yu, H. Zhang, and L. Zhao (2025) GReF: a unified generative framework for efficient reranking via ordered multi-token prediction. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management, pp. 5879–5887.
11. G. Ling, Z. Chen, Y. Li, T. Xiong, K. Lin, K. Zhang, Y. Yan, D. Ou, H. Tang, and B. Zheng (2026) Beyond semantic ids: encoding business-value ranking into document identifiers for generative retrieval. arXiv:2607.11392.
12. X. Liu, F. Zhang, Y. Wu, X. Jia, Z. Xia, F. Zhuang, Z. Zhang, F. Jiang, and W. Lin (2026) CAT-id2: category-tree integrated document identifier learning for generative retrieval in e-commerce. In Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining, pp. 426–435.
13. D. Ou, G. Ling, H. Wan, H. Zhou, J. Cheng, J. Pang, S. Zhou, W. Shi, W. Ye, W. Zhang, Y. Wang, Y. Li, Y. Yan, Z. Fa, Z. Chen, Z. Wu, B. Zheng, C. Wu, D. Huang, H. Tang, J. Guo, K. Zhang, K. Ma, L. Qu, L. Zhong, T. Lan, T. Xiong, and Z. Wu (2026) Prompt generation technical report. arXiv:2607.11326.
14. J. Pan, W. Xue, C. Zhou, X. Zhou, L. Fan, Y. Wang, H. Xin, Z. Hu, Y. Wang, F. Xu, et al. (2026) Tencent advertising algorithm challenge 2025: all-modality generative recommendation. arXiv preprint arXiv:2604.04976.
15. C. Pei, X. Yang, Q. Cui, X. Lin, F. Sun, P. Jiang, W. Ou, and Y. Zhang (2019) Value-aware recommendation based on reinforcement profit maximization. In The World Wide Web Conference, pp. 3123–3129.
16. A. V. Petrov and C. Macdonald (2024) RecJPQ: training large-catalogue sequential recommenders. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining, pp. 538–547.
17. S. Rajput, N. Mehta, A. Singh, R. Hulikal Keshavan, T. Vu, L. Heldt, L. Hong, Y. Tay, V. Tran, J. Samost, et al. (2023) Recommender systems with generative retrieval. Advances in Neural Information Processing Systems 36, pp. 10299–10315.
18. Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. (2024) Deepseekmath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.
19. C. Song, J. Zhang, C. Chen, Z. Sang, D. Zhao, G. Cao, B. Wu, D. Cai, and J. Jia (2026a) DeGRe: dense-supervised generative reranking for recommendation. arXiv preprint arXiv:2605.25749.
20. Y. Song, Z. Yang, S. Meng, R. Guo, J. Zhang, B. Wang, S. Liu, X. Wang, and G. Jiang (2026b) From item-only to query-item: query-conditioned generative search with qgs in quark. arXiv preprint arXiv:2605.25514.
21. J. Tan, S. Xu, W. Hua, Y. Ge, Z. Li, and Y. Zhang (2024) Idgenrec: llm-recsys alignment with textual id learning. In Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval, pp. 355–364.
22. Y. Tay, V. Tran, M. Dehghani, J. Ni, D. Bahri, H. Mehta, Z. Qin, K. Hui, Z. Zhao, J. Gupta, et al. (2022) Transformer memory as a differentiable search index. Advances in neural information processing systems 35, pp. 21831–21843.
23. W. Wang, H. Bao, X. Lin, J. Zhang, Y. Li, F. Feng, S. Ng, and T. Chua (2024) Learnable item tokenization for generative recommendation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pp. 2400–2409.
24. Y. Wang, Y. Wang, Y. Liang, Q. Liu, and F. Xiao (2026) Towards sustainable growth: a multi-value-aware retrieval framework for e-commerce search. arXiv preprint arXiv:2605.17994.
25. Y. Wu, Y. Feng, J. Wang, W. Zhou, Y. Ye, R. Xiao, and J. Xiao (2024) Hi-gen: generative retrieval for large-scale personalized e-commerce search. arXiv preprint arXiv:2404.15675.
26. A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, et al. (2024) Qwen2. 5 technical report. arXiv e-prints, pp. arXiv–2412.
27. H. Yang, Q. Xie, Q. Zhang, C. L. Yu, H. Zou, C. Lian, S. Han, F. Huang, J. Chen, and B. Zheng (2025) GSID: generative semantic indexing for e-commerce product understanding. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 1113–1121.
28. J. Zhai et al. (2024) Actions speak louder than words: trillion-parameter sequential transducers for generative recommendations.
29. J. Zhang, Y. Li, Y. Liu, C. Wang, Y. Wang, Y. Xiong, X. Liu, H. Wu, Q. Li, E. Zhang, et al. (2025a) GPR: towards a generative pre-trained one-model paradigm for large-scale advertising recommendation. arXiv preprint arXiv:2511.10138.
30. K. Zhang, X. Wang, S. Liu, H. Yang, X. Li, L. Hu, H. Li, Q. Cao, F. Sun, and K. Gai (2025b) Goalrank: group-relative optimization for a large ranking model. arXiv preprint arXiv:2509.22046.
31. X. Zhang, Y. Xiong, J. Zhou, Z. Guo, Z. Pang, J. Huo, J. Wang, X. Sun, E. Zhang, J. Jin, et al. (2026) Unified value alignment for generative recommendation in industrial advertising. arXiv preprint arXiv:2605.05803.
32. B. Zheng, Y. Hou, H. Lu, Y. Chen, W. X. Zhao, M. Chen, and J. Wen (2024) Adapting large language models by integrating collaborative semantics for recommendation. In 2024 IEEE 40th International Conference on Data Engineering (ICDE), pp. 1435–1448.
33. Z. Zheng, Z. Wang, F. Yang, J. Fan, T. Zhang, Y. Wang, and X. Wang (2025) Ega-v2: an end-to-end generative framework for industrial advertising. arXiv preprint arXiv:2505.17549.
34. G. Zhou, J. Deng, J. Zhang, K. Cai, L. Ren, Q. Luo, Q. Wang, Q. Hu, R. Huang, S. Wang, et al. (2025) OneRec technical report. arXiv preprint arXiv:2506.13695.
35. J. Zhu, X. Fang, J. Wang, M. Jin, B. Wang, G. Song, Z. Xie, and J. Bai (2026) Efficient generative retrieval for e-commerce search with semantic cluster ids and expert-guided rl. arXiv preprint arXiv:2605.14434.

（以上参考文献按原样保留英文，未作翻译。）

---

## 附录 A

### A.1 大语言模型（LLM）的使用

本文使用公开可用的大语言模型 Qwen 作为实验的主干模型。在写作方面，我们仅使用 Grammarly 等生成式 AI 工具来提升文本的清晰性、连贯性与整体质量。所有图表、初稿与内容均由作者独立完成，不依赖 LLM。

### A.2 在线部署

图 7：TSGR 的在线部署架构。TSGR 生成的商品绕过预排序，直接进入排序阶段。

如图 7 所示，TSGR 作为淘宝搜索流水线中多路检索之外的独立检索通道部署，关键区别在于其输出绕过预排序、直接进入排序阶段，因为生成的商品已经按价值分数排好序，该分数同时反映了相关性与商业质量。

### A.3 实现细节

SFT 与 GRPO 的批大小分别为 256 和 16（每设备 2 × 8 步梯度累积）。SFT 采用 AdamW（$\beta_{1}=0.9$，$\beta_{2}=0.999$）、余弦学习率 $1\times10^{-4}$ 与 10 步预热；GRPO 采用线性学习率 $4\times10^{-7}$、20 步预热，在 10 万条查询上训练 2 个 epoch。推理在三个 SID 层上应用动态束搜索，束宽为 [400, 400, 10000]，$\lambda$ 设为 1。对于 GRPO，每个查询在约束解码、温度 1.0 的条件下采样 32 个候选；按排序 CTR×CVR 分数排名 top-10 的候选获得 +1 奖励，bottom-10 获得 −1 奖励，每个提示额外附加 16 个离线偏好对。KL 系数设为 0.5，并在样本级 KL 与策略梯度方向冲突时缩放为 0.6×。熵正则（0.005）应用于第 1–2 层，当重叠度超过 20% 时禁用。

### A.4 QP-SID 案例分析

图 8：两个聚类的 QP-SID 案例：手机壳（聚类 (140, 3504)）与餐桌（聚类 (10278, 7631)）。对每个聚类，我们展示默认价值排序（红色）与特定词项下的查询条件排序（蓝色），匹配的词项以红色标出。每个商品旁的分数为相应的价值分数。

图 9：服务时 QP-SID 中词项选择的案例。左：不同查询各自匹配单个词项，该词项决定相应的第三级 SID。右：当查询匹配多个词项时，QP-SID 选择排序最高的词项（实线箭头）并丢弃其余词项（虚线箭头）。匹配的词项以红色标出。

为直观说明 QP-SID 如何组织商品，图 8 给出了两个代表性聚类的案例：手机壳（聚类 (140, 3504)）与餐桌（聚类 (10278, 7631)）。对每个聚类，我们比较默认价值排序与代表性词项下的查询条件排序，词项分别为手机壳的"磁吸"与餐桌的"实木"。每行内的商品按其价值分数降序排列。注意，默认分数与查询条件分数基于不同准则计算，因此不可直接比较；只有每行内的相对次序是有意义的。

对比揭示了 QP-SID 的两种互补行为。默认排序反映聚类内商品的整体商业价值，将全局热门与高价值商品排在前面。查询条件排序则依据查询意图重新排定商品优先级：在"磁吸"或"实木"词项下，标题中明确匹配该词项的商品（红色标出）被提升到顶部位置，即便它们在默认排序中位次较低。这表明 QP-SID 为同一聚类维护多个查询感知的价值排序，使模型能够选择与给定查询主导意图最匹配的排序，从而将意图契合的高价值商品置于更有利的 token 位置。

图 9 进一步说明了服务时 QP-SID 如何在这些排序之间进行选择，每个排序都由一个源自词项-商品点击统计（第 3.1 节）的词项进行索引。我们考虑两种代表性情形。在单匹配情形（图 9 左）中，每个查询恰好匹配一个词项，直接决定分配给目标商品的第三级 SID。例如，查询"分体独立电话机"与"座机电话家用无绳"分别匹配词项"电话机"与"座机"，表明指向同一商品的语义不同的查询可以经由不同词项路由，每个词项都反映查询的具体意图。在多匹配情形（图 9 右）中，一个查询可能同时匹配多个词项，此时 QP-SID 选择排序最高的词项来主导价值排序，并丢弃其余词项。例如，查询"百乐capless钢笔十代"匹配三个词项（"capless"、"百乐"与"钢笔"），排序最高的词项"capless"（实线箭头）被选中，其余被丢弃（虚线箭头）。综上，这些案例表明 QP-SID 能灵活地使价值排序适配查询意图，确保服务时最具意图相关性的排序引导 SID 生成。

### A.5 流水线的提示模板

表 6：五类 Pre-SFT 任务的提示模板。

| 任务类型 | 任务定义 |
| --- | --- |
| Encoding（编码） | 商品的标题为：${title}，请预测它的语义id |
| Personalization（个性化） | 你是一个电商搜索引擎，你的任务是根据用户的个人信息、当前的搜索词、用户历史交互的商品序列，为用户检索下一个商品。用户的性别为：${sex}，年龄为：${age}，购买力级别为：${level}。用户历史交互的商品序列为：${sids}。你检索的商品标题是 |
| Retrieval（检索） | 你是一个电商搜索引擎，你的任务是根据当前的搜索词，为用户检索商品。搜索词为：${query}。你检索的商品是 |
| Collaboration（协同） | 某商品的标题为：${trigger_title}，它的语义id为：${trigger_sid}，购买过它的用户通常还购买过相似商品，标题为：${target_title}，请预测这一相似商品的语义id |
| Recommendation（推荐） | 你是一个电商推荐引擎，你的任务是根据用户的个人信息、用户历史交互的商品序列，为用户推荐下一个商品。用户的性别为：${sex}，年龄为：${age}，购买力级别为：${level}。用户历史交互的商品序列为：${sids}。你推荐的商品是 |

表 7：个性化检索的 SFT 与 RL 提示模板。

| 角色 | 内容 |
| --- | --- |
| System | 请根据用户信息、搜索词、用户历史交互商品序列，为用户检索下一个商品。 |
| User | 用户性别为：${gender}，年龄为：${age}，购买力级别为：${level}。搜索词为：${query}。历史交互商品序列为：${opt_seq}。你检索的商品是 |

表 6 给出五类 Pre-SFT 任务的模板，表 7 给出 SFT 与 RL 阶段共享的统一模板。每个模板都指定了自然语言指令以及以 ${\cdot}$ 表示的占位符，训练时这些占位符会被填充为相应的商品属性（如标题、语义ID）或用户属性（如性别、年龄、购买力、历史交互序列）。所有提示原文均为中文，与淘宝搜索场景保持一致，此处按其原始形式呈现，以忠实反映模型的真实输入。

### A.6 RL 细节

表 8：RL 策略消融研究。每行在前一配置的基础上累积增加一个组件。

| 方法 | HR@20 | HR@100 | HR@500 | HR@1k | HR@5k |
| --- | --- | --- | --- | --- | --- |
| Baseline（基线） | 0.4965 | 0.6580 | 0.7746 | 0.8111 | 0.8747 |
| + Basic RL（基础 RL） | 0.4985 | 0.6584 | 0.7761 | 0.8113 | 0.8737 |
| + DPO Augmentation（DPO 增强） | 0.4975 | 0.6658 | 0.7803 | 0.8152 | 0.8772 |
| + Dynamic Entropy Reg（动态熵正则） | **0.4989** | **0.6674** | **0.7808** | **0.8163** | **0.8788** |

**奖励设计。** 奖励引导生成朝向既相关又具有高预测商业价值的候选，其定义为排序模型 CTR 与 CVR 的乘积：

$$
r(s)=\mathrm{CTR}(q,\,i_{s})\times\mathrm{CVR}(q,\,i_{s}). \tag{16}
$$

**离线 DPO 增强。** 标准 GRPO 依赖在线 rollout 形成组相对奖励，但这些 rollout 通常集中于头部候选。结果，奖励归一化会系统性地抑制低奖励样本，尽管其中一些仍可能落入目标关注的 top-1000 区域。这种偏差可能有利于头部指标占重的评估，但对于我们的检索目标并非最优。为缓解该问题，我们为每个提示附加由排序 CTR×CVR 分数构建的离线偏好对：chosen 样本取自排序 top-B 候选与真实购买标签，rejected 样本则从排名 1000 附近的候选中采样，以降低假负例风险。此外，将在线 rollout 与离线偏好样本混合后，我们在合并集中仅按奖励保留 top-k 与 bottom-k 样本（k=10），并掩蔽其余所有样本的奖励。这一设计将优化集中于高置信度的偏好信号，同时防止 GRPO 的组相对归一化错误抑制那些可能仍属于 top-1000 集合的模糊候选。

**动态层次熵正则。** 为鼓励探索多样的 SID 路径，我们以动态的分层策略应用熵正则。对每个提示，我们测量 rollout SID 与真实 SID 在第 1–2 层的匹配率 $m$，并用一个二元系数对正则进行门控：

$$
\beta(m)=\begin{cases}\beta_{0}&\text{if }m<\tau_{\text{low}},\\ 0&\text{if }m>\tau_{\text{high}},\end{cases} \tag{17}
$$

从而将探索限制在尚未收敛的提示上。我们仅在第 1–2 层应用该正则，因为第三层的熵会扰动由 SFT 校准好的细粒度商品分布。相应的项被加入 GRPO 目标：

$$
\mathcal{L}_{\text{RL}}=\mathcal{L}_{\text{GRPO}}-\beta(m)\sum_{l\in\{1,2\}}\mathcal{H}(\pi_{\theta}^{(l)}), \tag{18}
$$

其中 $\mathcal{H}(\pi_{\theta}^{(l)})$ 表示第 $l$ 层策略分布的熵。

表 8 表明，各 RL 组件均有增量贡献。仅基础 RL 只带来边际增益（+0.02%）。DPO 增强带来更显著的提升（+0.39%），表明基于偏好的监督有助于模型更好地区分正负候选。动态熵正则进一步提升性能（+0.11%），证实 RL 训练期间保持输出多样性是有益的。累计 +0.52% 的温和增益解释了 RL 为何未在实际部署中采用。

### A.7 TencentGR 数据集

表 9：TencentGR-10M 数据集上的消融实验。

| 数据集 | 方法 | HR@500 |
| --- | --- | --- |
| TencentGR-10M | Base（基线） | 0.0786 |
| TencentGR-10M | Value-aware Ranking Module（价值感知排序模块） | 0.0794 |

TencentGR 数据集 (Pan et al., 2026) 带来若干实际挑战，需要专门的预处理才能用于评估我们的框架。具体而言，该数据集没有查询字段、没有预构建的语义ID、用户与商品属性完全匿名化，且没有带真实标签的公开测试集。下面说明各挑战的应对方式。

**SID 构建。** 由于该数据集不提供语义ID，我们直接从 1024 维多模态商品嵌入构建。应用层次聚类产生两级 SID 前缀，第三层则通过在每个聚类内按交易、点击与曝光次数降序排列商品来确定。每个商品由此被映射到一个三层的 SID。

**商品边信息。** 由于原始商品内容不公开，商品边信息由三个来源组装：(1) 构建好的 SID；(2) 匿名化的商品特征，拼接从行为序列聚合的每商品曝光、点击与交易计数（缺失值以 −1 填充）；(3) $\ell_{2}$ 归一化的 1024 维多模态嵌入。

**行为序列预处理。** 原始行为序列约含 1010 万条记录，其中许多仅由曝光构成、没有任何点击或交易，因而对训练与评估无信息量。过滤掉这些纯曝光记录后，保留约 771 万条记录。每条剩余记录按多标签训练方式处理：最近的曝光、点击与交易各被选为一个标签；为防止标签泄漏，仅使用早于最早标签时间戳的行为作为输入；输入再按曝光、点击、交易分别切分为三个独立子序列。

**提示构建。** 由于没有可解释的用户属性，所有用户特征以 "Feature N: value" 的形式纳入提示，不做语义区分。提示指示模型根据用户特征与以 SID 字符串编码的三个历史子序列推荐下一个商品。输出为按优先级排序的最多三个 SID 字符串，分别对应交易、点击与曝光标签，当高优先级标签缺失时以低优先级标签兜底。每个训练样本还额外携带标签商品的多模态嵌入与边信息、标签优先级指示符，以及用于多标签对齐的组标识符。

**训练/测试划分。** 由于没有公开测试集，我们按 MD5(user_id) 对全部记录排序，指定前 20,000 条记录为测试集，其余用于训练，以此构建可复现的划分。该流程确保测试集中的用户不会出现在训练集中。

### A.8 其他实验

我们在表 10 中提供价值感知排序模块的全面消融研究，评估了不同排序策略（包括 PV、CTR、CVR 及其组合）在多种模型变体上的表现。结果表明，采用 PV×CTR 排序的基于注意力（Attention）的模型取得最佳整体性能。

表 10：价值感知排序模块的消融研究。

| 模型设计 | 排序策略 | HR@20 | HR@100 | HR@500 | HR@1000 | HR@5000 |
| --- | --- | --- | --- | --- | --- | --- |
| Attention（交叉注意力） | Base | 0.4754 | 0.6431 | 0.7807 | 0.8252 | 0.8953 |
| Attention | PV | 0.3387 | 0.5803 | 0.7889 | 0.8482 | 0.8953 |
| Attention | CTR | 0.3744 | 0.5401 | 0.7088 | 0.7824 | 0.8953 |
| Attention | CVR | 0.2149 | 0.3097 | 0.4717 | 0.5802 | 0.8953 |
| Attention | PV×CTR | **0.4586** | **0.6677** | **0.8250** | **0.8651** | 0.8953 |
| Attention | PV×CTR×CVR | 0.4443 | 0.6626 | 0.8239 | 0.8650 | 0.8953 |
| Mean Pooling（均值池化） | Base | 0.4774 | 0.6492 | 0.7822 | 0.8269 | 0.8985 |
| Mean Pooling | PV | 0.2726 | 0.5273 | 0.7599 | 0.8303 | 0.8985 |
| Mean Pooling | CTR | 0.3148 | 0.4995 | 0.7020 | 0.7865 | 0.8985 |
| Mean Pooling | CVR | 0.1863 | 0.3002 | 0.4825 | 0.6008 | 0.8985 |
| Mean Pooling | PV×CTR | 0.3920 | 0.6161 | 0.8003 | 0.8488 | 0.8985 |
| Mean Pooling | PV×CTR×CVR | 0.3849 | 0.6146 | 0.7999 | 0.8503 | 0.8985 |
| EOS Token（EOS 表示） | Base | 0.4763 | 0.6493 | 0.7840 | 0.8272 | 0.8948 |
| EOS Token | PV | 0.3587 | 0.5975 | 0.7942 | 0.8483 | 0.8948 |
| EOS Token | CTR | 0.3455 | 0.5278 | 0.7227 | 0.7977 | 0.8948 |
| EOS Token | CVR | 0.2418 | 0.3662 | 0.5448 | 0.6514 | 0.8948 |
| EOS Token | PV×CTR | 0.3862 | 0.6058 | 0.8016 | 0.8539 | 0.8948 |
| EOS Token | PV×CTR×CVR | 0.3727 | 0.5916 | 0.7953 | 0.8512 | 0.8948 |
| No Embedding（无商品嵌入） | Base | 0.4775 | 0.6432 | 0.7790 | 0.8233 | 0.8954 |
| No Embedding | PV | 0.2236 | 0.4016 | 0.6336 | 0.7390 | 0.8954 |
| No Embedding | CTR | 0.2123 | 0.3342 | 0.5092 | 0.6053 | 0.8954 |
| No Embedding | CVR | 0.1548 | 0.2355 | 0.3656 | 0.4613 | 0.8954 |
| No Embedding | PV×CTR | 0.2600 | 0.4356 | 0.6448 | 0.7487 | 0.8954 |
| No Embedding | PV×CTR×CVR | 0.2621 | 0.4384 | 0.6483 | 0.7502 | 0.8954 |
| No Side-Info（无边信息） | Base | 0.4763 | 0.6469 | 0.7803 | 0.8269 | 0.8963 |
| No Side-Info | PV | 0.3868 | 0.6097 | 0.7922 | 0.8429 | 0.8963 |
| No Side-Info | CTR | 0.3580 | 0.5382 | 0.7194 | 0.7913 | 0.8963 |
| No Side-Info | CVR | 0.1645 | 0.2404 | 0.4096 | 0.5432 | 0.8963 |
| No Side-Info | PV×CTR | 0.4116 | 0.6412 | 0.8139 | 0.8580 | 0.8963 |
| No Side-Info | PV×CTR×CVR | 0.4013 | 0.6341 | 0.8125 | 0.8570 | 0.8963 |

（全文完。译文中公式、图表编号与原文一一对应；参考文献未翻译，原样保留。）
