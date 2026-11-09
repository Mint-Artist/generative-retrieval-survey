# UniGD：面向工业检索的统一生成-判别框架

**英文原题：** UniGD: A Unified Generative-Discriminative Framework for Industrial Retrieval

**作者与机构：** Shujie Ji\*, Yawei Kong\*, Yilin Zhao, Li Wang†, Xialong Liu, Peng Jiang
Kuaishou Technology, Beijing, China
{jishujie03, kongyawei, zhaoyilin05, wangli18, zhaolei16, jiangpeng}@kuaishou.com
（\* 共同第一作者；† 通讯作者）

**arXiv 编号：** arXiv:2608.03150v1 [cs.AI] 4 Aug 2026

**译注：** 本翻译忠实还原原文结构，章节与原文一一对应。专有名词（模型名、数据集名、指标名如 Recall@K、NDCG@K、MRR@K）保留英文；数学公式以 LaTeX 保留。原文无编号式章节标题（AAAI 风格），本译文相应不加编号。图 1—图 3 的图题与表 1—表 4 的表题均按原文翻译，表格中数字保持不变；浮动表格按其在正文中被讨论的位置就近放置。

## 摘要（Abstract）

生成式检索（Generative Retrieval, GR）是工业搜索广告中一种富有前景的范式，但其部署受到严格的相关性与延迟要求的制约。现有系统将 GR 与一个独立的相关性模型级联，使生成似然目标与 query–ad 相关性判别相互解耦，这损害了有效性并增加了服务成本。我们提出统一生成-判别框架（Unified Generative-Discriminative framework, UniGD），将检索与相关性打分整合进单一模型。为缓解联合优化中的梯度干扰，UniGD 引入冲突感知梯度增强（Conflict-Aware Gradient Enhancement, CAGE）来自适应地协调这两个目标。UniGD 进一步设计了码本锚定表示模块（Codebook-Anchored Representation Module, CAM），将物品表示锚定到从多模态预训练模型蒸馏得到的冻结层次化码本上，从而赋予其丰富且可泛化的语义先验。针对异构的短视频、商品与直播广告，UniGD 提出异构广告物料建模（Heterogeneous Ad-material Modeling, HAM），在共享骨干网络上捕获跨类型语义共性，同时保留各类型专属的建模能力。在快手搜索广告平台上的在线 A/B 测试表明，UniGD 使广告收入提升 5.78%，推理延迟降低 33.1%，并改善了判别式相关性估计。在 NQ320K 和 MS300K 上，UniGD 相对最强复现 GR 基线的 Recall@10 分别提升 8.44% 和 3.19%。

**关键词：** 生成式检索，搜索广告

## 引言（Introduction）

搜索广告系统必须在严格的延迟约束下，从数亿候选广告中检索出语义相关的广告。传统的多阶段检索流水线通过渐进式的候选过滤来应对这一挑战，但每个阶段通常被独立优化，导致目标不一致、整体检索效果受限。生成式检索（Generative Retrieval, GR）(Tay et al. 2022; Bevilacqua et al. 2022) 近期作为一种面向大规模候选检索的富有前景的范式而受到越来越多的关注。通过自回归解码，GR 将候选检索转化为语义标识符（Semantic Identifier, SID）(van den Oord, Vinyals, and Kavukcuoglu 2017; Rajput et al. 2023; Deng et al. 2025) 的条件生成。这一生成式范式超越了传统的基于匹配的检索，实现了 query 与候选之间语义交互的统一建模。

在真实的工业搜索广告部署中，GR 通常与一个下游相关性模型级联，以确保最终结果的相关性。具体而言，GR 首先通过自回归生成似然检索 top-$K$ 候选广告，随后相关性模型利用 query-ad 匹配信号给出判别式相关性分数。尽管这种级联范式已在实践中被采用，但它在工业部署中仍面临若干关键挑战。

**(1) 目标差异与服务开销。**
生成式检索与判别式相关性打分之间的差异，可能使那些生成概率较低但实际相关的广告在 top-$K$ 截断阶段被过早丢弃，根本来不及到达下游相关性模型进行重打分 (Tang et al. 2024; Cheng et al. 2025)。此外，级联式 GR 范式引入了额外的在线推理延迟。尽管已有研究引入排序监督来重塑生成似然的相对次序 (Zhou et al. 2024; Li et al. 2024b)，这类偏好信号并不能直接产生显式的点式（point-wise）相关性分数。因此，它们不适用于基于阈值的过滤，且作为供下游模块使用的相关性特征，其可靠性有限。

**(2) 长尾与新广告的表示。**
可靠的相关性打分需要有判别力的 ad 侧表示，但长尾广告和新广告往往缺乏足够的交互信号来进行表示学习。这一问题在我们平台上尤为突出：新广告占每日流量高达 36.9%。现有的级联或辅助判别模块 (Tang et al. 2024; Dai et al. 2025) 通常依赖单独训练的 ad 编码器或周期性更新的 ad embedding，难以在广告刚被索引后立即提供可靠的表示。

**(3) 异构广告物料的统一建模。**
工业搜索广告覆盖多种广告物料类型，包括短视频、商品和直播广告，它们表现出显著的语义异构性。现有 GR 方法缺乏对跨物料差异的显式建模，因而难以在共享的 SID 语义空间中容纳不同物料类型的语义特性，限制了 GR 在工业广告部署中的泛化能力。

为应对这些挑战，我们提出 UniGD——一个将生成式检索与相关性判别整合于单一模型的统一生成-判别框架。与级联式 GR 不同，UniGD 在共享骨干网络上联合建模自回归生成与 query-ad 相关性判别，实现两者相互促进与优化协同，同时消除额外相关性模型的服务开销。具体而言，UniGD 引入了三项关键设计。

**(1) 冲突感知梯度增强（Conflict-Aware Gradient Enhancement, CAGE）。**
由于在共享骨干网络上联合优化生成与判别目标可能产生不一致的梯度更新方向，我们引入 CAGE 在反向传播过程中协调二者的梯度，使两个目标都能从联合优化中获益。

**(2) 码本锚定表示模块（Codebook-Anchored Representation Module, CAM）。**
为解决行为监督稀疏下广告检索的冷启动瓶颈，我们设计了 CAM，将 item 侧表示锚定到由大规模多模态预训练模型导出的冻结层次化码本上，使长尾广告和新广告能够继承其语义先验，并在被索引后即刻获得可靠的表示。

**(3) 异构广告物料建模（Heterogeneous Ad-material Modeling, HAM）。**
考虑到广告物料之间的语义异构性，我们提出 HAM，为每种物料类型分配专属的语义空间，并将物料感知信号注入统一骨干网络。如此一来，SID 生成与相关性判别都能针对每种物料类型聚焦于恰当的信号源，从而在避免过度简化的同质化建模的同时增强统一的生成-判别过程。

我们通过工业级在线部署与公开基准评测两方面验证 UniGD。在快手搜索广告平台上的在线 A/B 测试表明，UniGD 将广告收入提升 5.78%，同时将推理延迟从 13.00 ms 降至 8.70 ms（降低 33.1%）。在 NQ320K 与 MS300K 基准上，UniGD 的 Recall@10 分别比最强复现 GR 基线提升 8.44% 和 3.19%。

本文的主要贡献总结如下：

- 我们提出 UniGD，一个统一生成-判别检索框架，在单一共享模型内同时执行自回归 SID 生成与显式判别式相关性打分，从而无需级联的相关性模型。
- 我们为有效的工业部署开发了三个关键组件：CAGE 缓解生成与判别目标之间的优化冲突；CAM 为长尾及新索引广告提供可靠的 item 侧表示；HAM 捕获短视频、商品与直播广告之间的语义异构性。
- 我们开展了广泛的在线与离线评测，证明 UniGD 在显著提升检索效果与广告收入的同时降低了在线推理延迟。

## 相关工作（Related Work）

### 生成式方法（Generative Methods）

与传统的基于词项的检索 (Robertson et al. 1995; Nogueira and Lin 2019; Raffel et al. 2020) 和稠密检索器 (Karpukhin et al. 2020; Xiong et al. 2021; Zhan et al. 2020; Ni et al. 2022; Qu et al. 2021) 不同，生成式检索将大规模候选检索重构为序列生成。早期研究通过文档标识符生成、query 生成增强、前缀感知解码和可学习 tokenization 奠定了技术基础 (Tay et al. 2022; Zhuang et al. 2022; Wang et al. 2022; Sun et al. 2023)。后续工作通过可扩展的多阶段训练、高效的同时解码（simultaneous decoding）与语义 docid 设计推进了该范式 (Zeng et al. 2024; Zeng, Luo, and Zamani 2024; Zhou et al. 2022; Li et al. 2024a; Guo et al. 2025)，并将基于 SID 的生成扩展到序列推荐以及工业推荐与广告流水线 (Rajput et al. 2023; Deng et al. 2025; Zheng et al. 2025; Xue et al. 2026)。尽管取得了这些进展，现有 GR 方法大多经由 MLE 优化生成概率，并在推理时依赖束搜索，这会放大高频 SID，并低估语义相关的长尾与新物品。

### 生成-判别方法（Generative-Discriminative Methods）

已有若干研究将面向排序或判别的信号整合进生成式检索与推荐 (Zhang et al. 2025; Sun et al. 2026; Fu et al. 2026; Chen et al. 2025)。代表性工作包括：将面向排序的目标或文档级相关性优化引入基于 MLE 的 GR (Zhou et al. 2024; Li et al. 2024b; Mekonnen, Tang, and de Rijke 2025)；通过端到端训练使生成与 query-item 相关性对齐 (Pang et al. 2025)；以及设计判别式标识符以锐化候选区分度 (Cheng et al. 2025)。这些工作在弥合生成与面向相关性的排序方面取得了有价值的进展，但搜索广告快速的广告更替对检索与判别精度提出了更高要求，需要一个在保证可扩展性的同时将候选生成与相关性估计紧耦合的统一框架。

## 方法（Methodology）

### 总览（Overview）

本文提出 UniGD，一个统一的 decoder-only 框架，整合了基于 SID 的候选生成与 query-ad 相关性打分。如图 1 所示，HAM 为生成与判别两类学习捕获物料专属语义，CAM 从冻结的 SID 码本中导出码本锚定的 ad 表示用于相关性打分，CAGE 在联合优化期间协调这两个目标。图 2 展示了相应的工业服务架构：候选生成与相关性打分在同一模型内完成，从而去除外置在线相关性模型，降低推理延迟与服务开销。

图1：UniGD 的训练框架。HAM-G 与 HAM-D 联合优化生成式目标与基于 CAM 的判别，CAGE 则通过三阶段训练方案协调二者的梯度。

### 统一生成-判别建模（Unified Generative-Discriminative Modeling）

给定 query $q$ 与广告语料库 $\mathcal{C}$，UniGD 在共享的 decoder-only 骨干网络中统一自回归 SID 生成与判别式相关性估计，联合建模条件生成概率 $p(\mathbf{s}(a)|q)$ 与显式相关性分数 $r(q,a)$。

**SID tokenization。**
每个广告 $a\in\mathcal{C}$ 由层次化语义标识符（SID）
$$\mathbf{s}(a)=(s_{1}(a),\ldots,s_{T}(a))$$
索引，采用 RQ-KMeans 构造。为进行自回归生成，我们在模型词表中增补层级专属的 SID token，并将 $\mathbf{s}(a)$ 序列化为 token 序列，其中 $s_{t}(a)\in\mathcal{V}_{t}$ 表示第 $t$ 层的 SID token。同一套 SID 索引也被 CAM 用于检索相应的码字以进行相关性估计。

**自回归 SID 生成。**
在给定 query $q$ 的条件下，UniGD 将 SID 生成概率分解为

$$p(\mathbf{s}(a)\mid q)=\prod_{t=1}^{T}p\left(s_{t}(a)\mid q,\mathbf{s}_{<t}(a)\right),\tag{1}$$

其中 $\mathbf{s}_{<t}(a)$ 表示先前的 SID 前缀。在解码步 $t$，共享 decoder 产生最后一层隐状态 $\mathbf{h}^{\mathrm{sid}}_{t}\in\mathbb{R}^{d}$，并在 $\mathcal{V}_{t}$ 上预测下一个 SID token：

$$p\!\left(s_{t}(a)\mid q,\mathbf{s}_{<t}(a)\right)=\operatorname{Softmax}\!\left(\mathbf{W}_{t}\mathbf{h}^{\mathrm{sid}}_{t}+\mathbf{b}_{t}\right)_{s_{t}(a)},\tag{2}$$

其中 $\mathbf{W}_{t}\in\mathbb{R}^{|\mathcal{V}_{t}|\times d}$ 与 $\mathbf{b}_{t}\in\mathbb{R}^{|\mathcal{V}_{t}|}$ 是与 SID 层级 $t$ 关联的预测参数，它们进一步由 HAM 适配以捕获物料专属语义。对于训练对 $(q,a)$，生成目标为

$$\mathcal{L}_{\mathrm{gen}}=-\sum_{t=1}^{T}\log p\left(s_{t}(a)\mid q,\mathbf{s}_{<t}(a)\right).\tag{3}$$

**判别式相关性估计。**
UniGD 从共享 decoder 的 query-token 隐状态中导出用于相关性估计的 query 表示。令 $\mathbf{H}_{q}\in\mathbb{R}^{N_{q}\times d}$ 表示 $N_{q}$ 个有效 query-token 位置上最后一层隐状态矩阵。我们将这些状态聚合为 query 级表示 $\mathbf{u}_{q}=\operatorname{Agg}(\mathbf{H}_{q})$，其中 $\operatorname{Agg}(\cdot)$ 在我们的实现中实例化为平均池化（mean pooling）。对于候选 $a$，令 $\mathbf{u}^{CAM}_{a}\in\mathbb{R}^{d_{s}}$ 表示由 CAM 构造的码本锚定表示。为将 $\mathbf{u}_{q}$ 与 $\mathbf{u}^{CAM}_{a}$ 对齐到共享语义空间，我们分别采用 query 侧适配器 $g_{q}$ 与 ad 侧适配器 $g_{a}$。相关性 logit 计算为

$$r(q,a)=\psi\!\left(\left[g_{q}(\mathbf{u}_{q});g_{a}(\mathbf{u}^{CAM}_{a})\right]\right),\tag{4}$$

其中 $[\cdot\,;\cdot]$ 表示向量拼接，$\psi$ 是输出 $r(q,a)\in[0,1]$ 的轻量打分网络。

给定 query $q$ 的有监督候选集 $\mathcal{D}_{q}=\{(a_{i},y_{i})\}_{i=1}^{K}$，其中 $y_{i}\in[0,1]$ 是由人工标注或离线相关性模型导出的归一化相关性目标，点式损失为

$$\mathcal{L}_{\mathrm{point}}=\frac{1}{K}\sum_{i=1}^{K}\left(r(q,a_{i})-y_{i}\right)^{2}.\tag{5}$$

为进一步强化相对相关性次序，我们引入辅助的成对排序损失。给定 query $q$ 的一个正例广告 $a^{+}$ 和 $M$ 个负例广告 $\{a^{-}_{j}\}_{j=1}^{M}$，其定义为

$$\mathcal{L}_{\mathrm{pair}}=\frac{1}{M}\sum_{j=1}^{M}\max\left(0,\gamma-r(q,a^{+})+r(q,a^{-}_{j})\right),\tag{6}$$

其中 $\gamma\in(0,1)$ 为排序间隔（margin）。完整的判别目标组合了点式分数监督与成对排序：

$$\mathcal{L}_{\mathrm{disc}}=\mathcal{L}_{\mathrm{point}}+\lambda\mathcal{L}_{\mathrm{pair}},\tag{7}$$

其中 $\lambda$ 控制成对监督的贡献。

为充分发展生成与判别两方面的能力，UniGD 采用三阶段训练方案。阶段 1（Stage 1）与阶段 2（Stage 2）分别优化 $\mathcal{L}_{\mathrm{gen}}$ 与 $\mathcal{L}_{\mathrm{disc}}$ 以初始化相应任务组件。阶段 3（Stage 3）在 CAGE 协调共享参数上梯度的情况下联合优化两个目标。

### 码本锚定表示模块（Codebook-Anchored Representation Module）

判别式相关性估计要求在严格的在线延迟约束下拥有信息丰富的 ad 侧表示，这对于交互信号稀疏的新广告与长尾广告尤其具有挑战性。为解决这一问题，我们引入码本锚定表示模块（CAM）。给定广告的 SID，CAM 通过查找并组合相应码字来构造其多模态表示的量化近似。

**多模态残差量化。**
我们采用在大规模广告创意语料上预训练的大规模多模态编码器，将每个广告的异构内容编码为信息丰富的语义表示：

$$\mathbf{v}_{a}=\operatorname{Enc}_{\mathrm{MM}}(a)\in\mathbb{R}^{d_{s}}.\tag{8}$$

RQ-KMeans 在索引语料库的 ad 表示上离线训练，以学习 $T$ 个残差码本。令 $\mathbf{C}_{t}\in\mathbb{R}^{|\mathcal{V}_{t}|\times d_{s}}$ 表示第 $t$ 层的码本。给定前序各层留下的残差 $\boldsymbol{\rho}_{t-1}(a)$，第 $t$ 层的 SID token 被指派到其最近的码字：

$$s_{t}(a)=\arg\min_{k\in\mathcal{V}_{t}}\left\|\boldsymbol{\rho}_{t-1}(a)-\mathbf{C}_{t}[k]\right\|_{2}^{2},\tag{9}$$

其中 $\boldsymbol{\rho}_{0}(a)=\mathbf{v}_{a}$。靠前的层捕获多模态表示的主导成分，而后续的层对剩余残差进行量化，逐步编码更细粒度的信息。

**码本锚定重构。**
CAM 检索由每层 SID token 索引的码字 $\mathbf{c}_{t}(a)=\mathbf{C}_{t}[s_{t}(a)]$，并递归地更新残差 $\boldsymbol{\rho}_{t}(a)=\boldsymbol{\rho}_{t-1}(a)-\mathbf{c}_{t}(a)$。ad 侧表示通过求和所选码字重构：

$$\mathbf{u}^{CAM}_{a}=\sum_{t=1}^{T}\mathbf{c}_{t}(a).\tag{10}$$

于是 $\mathbf{v}_{a}=\mathbf{u}^{CAM}_{a}+\boldsymbol{\rho}_{T}(a)$，其中 $\boldsymbol{\rho}_{T}(a)$ 为最终量化残差。因此，$\mathbf{u}^{CAM}_{a}$ 提供了对原始多模态表示的量化重构，并作为判别式相关性估计的 ad 侧输入。在整个 UniGD 训练过程中码本保持冻结，从而维持 SID token 与其离线学得码字之间的对应关系，提供稳定的语义锚点。一旦广告被赋予 SID，CAM 只需 $T$ 次码本查找和一次轻量向量求和，即可在索引完成后立即进行相关性估计，无需在线的 ad 侧编码器推理或复杂的特征构建。

### 异构广告物料建模（Heterogeneous Ad-material Modeling）

工业搜索广告包含异构的广告物料，包括短视频、商品和直播广告。它们在内容模态、结构、呈现形态和信息密度上差异显著，导致不同的语义分布与相关性模式。因此，同质化的建模方案可能掩盖物料专属语义。我们引入异构广告物料建模（HAM），在保留共享骨干网络的同时，按物料类型特化码本构造、SID 生成与相关性估计。

令 $m\in\mathcal{M}=\{\mathrm{video},\mathrm{product},\mathrm{live}\}$ 表示物料类型。HAM 对每种类型分别执行残差量化，得到物料专属码本 $\{\mathbf{C}_{t}^{(m)}\}_{t=1}^{T_{m}}$ 与 SID 词表 $\{\mathcal{V}_{t}^{(m)}\}_{t=1}^{T_{m}}$。在生成侧，它将式 (2) 中的输出参数替换为 $\mathbf{W}_{t}^{(m)}$ 与 $\mathbf{b}_{t}^{(m)}$，在 $\mathcal{V}_{t}^{(m)}$ 上进行预测。训练时，$m$ 取自目标广告的物料类型元数据。推理时，所有物料专属输出头利用共享 decoder 隐状态并行执行类型专属的束搜索，所得的带类型 SID 候选被合并送入相关性打分，无需额外的骨干网络前向。在相关性估计侧，CAM 从类型 $m$ 的码本中检索码字，重构物料感知的 ad 表示

$$\mathbf{u}_{a}^{CAM,(m)}=\sum_{t=1}^{T_{m}}\mathbf{C}_{t}^{(m)}[s_{t}(a)].\tag{11}$$

将 $\mathbf{u}_{a}^{CAM,(m)}$ 代入相关性打分函数即得到物料感知的相关性分数 $r(q,a,m)$。如此，生成与判别在一致的物料专属 SID 语义上进行，同时在各物料之间共享骨干网络与打分架构。

### 冲突感知梯度增强（Conflict-Aware Gradient Enhancement）

联合优化可能在共享 decoder 参数上产生来自生成与判别目标的冲突梯度，导致优化干扰。为此我们引入冲突感知梯度增强（CAGE），在优先保障自回归 SID 生成的同时，自适应地协调二者的梯度。

图2：UniGD 在工业搜索广告中的在线服务流水线。

令 $\theta_{s}$ 表示共享 decoder 参数。我们计算任务梯度 $\mathbf{g}_{\mathrm{gen}}=\nabla_{\theta_{s}}\mathcal{L}_{\mathrm{gen}}$ 与 $\mathbf{g}_{\mathrm{disc}}=\nabla_{\theta_{s}}(\eta\mathcal{L}_{\mathrm{disc}})$，其中 $\eta>0$ 用于重标度判别梯度，以缓解两个目标之间的量级失衡。二者的方向一致性由余弦相似度 $c=\operatorname{cos}(\mathbf{g}_{\mathrm{gen}},\mathbf{g}_{\mathrm{disc}})$ 度量。当 $c\geq 0$ 时，两个梯度相互兼容，直接相加；当 $c<0$ 时，CAGE 通过将判别梯度投影到生成方向的正交补空间来移除其冲突分量：

$$\mathbf{g}_{\mathrm{disc}}^{\perp}=\mathbf{g}_{\mathrm{disc}}-\frac{\mathbf{g}_{\mathrm{gen}}^{\top}\mathbf{g}_{\mathrm{disc}}}{\|\mathbf{g}_{\mathrm{gen}}\|_{2}^{2}}\mathbf{g}_{\mathrm{gen}}.\tag{12}$$

用于更新共享 decoder 的梯度为

$$\mathbf{g}_{\mathrm{CAGE}}=\begin{cases}\mathbf{g}_{\mathrm{gen}}+\mathbf{g}_{\mathrm{disc}},&c\geq 0,\\[3pt] \alpha(c)\,\mathbf{g}_{\mathrm{gen}}+\beta(c)\,\mathbf{g}_{\mathrm{disc}}^{\perp},&c<0,\end{cases}\tag{13}$$

其中 $\alpha(c)=1+\kappa(-c)$，$\beta(c)=1-\kappa(-c)$，$\kappa\in[0,1]$ 控制自适应强度。随着 $c$ 减小，CAGE 越来越偏向生成方向，在保留不冲突的判别监督的同时保障 SID 生成。所得梯度用于更新共享 decoder，而任务专属参数则由各自的目标分别优化。

### 工业在线服务（Industrial Online Serving）

图 2 展示了部署在我们搜索广告系统中的 UniGD 持续学习、增量索引与实时服务流水线。请求上下文与用户反馈被收集构成 query-ad 训练样本，细粒度相关性目标由离线教师模型（teacher model）提供。训练得到的参数周期性地更新到推理服务器上。与此同时，实时 SID 索引维护物料专属的 SID-ad 映射。新增或更新的广告被量化为 SID 并增量索引。由于这些映射与模型更新相互独立，新索引的广告无需重建依赖模型的 embedding 索引即可进入检索。

服务期间，UniGD 通过束搜索生成候选 SID，并从共享 decoder 导出 query 表示。CAM 从物料专属码本中重构每个候选的 ad 侧表示，轻量打分模块以显式相关性分数对候选重排。保留的 SID 随后通过实时索引解析为实际广告。这一统一流水线既去除了单独部署的在线相关性模型，也免除了对每个候选进行 ad 侧编码器推理的需要。

## 实验（Experiments）

### 实验设置（Experimental Settings）

**数据集。**
我们在两个公开生成式检索基准与一个工业搜索广告数据集上评估 UniGD。NQ320K 源自 Natural Questions (Kwiatkowski et al. 2019)，包含 320K 个训练 query 与 7,830 个测试 query，语料为维基百科文章。MS300K (Zhou et al. 2022) 整理自 MS MARCO (Bajaj et al. 2016)，包含 320K 篇文档、360K 个训练 query 和 772 个测试 query。我们遵循 DDRO (Mekonnen, Tang, and de Rijke 2025) 使用的数据划分以便与先前工作比较。在公开基准上，query-文档对被标注为相关（正例）或不相关（负例）。我们的工业数据集覆盖从生产搜索广告日志采集的商品、短视频与直播广告，包含用于生成训练的 153M 正例对与用于判别训练的 248M 对。测试集分别包含 1K 个 query 上的 277K 对，以及超过 10K 人工标注对。

**评测指标。**
对于生成式检索，我们遵循先前工作 (Wang et al. 2022; Zhou, Dou, and Wen 2023; Zhou et al. 2024; Zhou et al. 2022) 报告 Recall@1/5/10 与 MRR@10。在公开基准上，显著性检验针对最强复现基线、基于逐 query 指标值的双侧配对 t 检验（$p<0.05$）。在工业数据集的相关性估计上，我们报告 AUC、Spearman 与 Pearson，分别评估二分类判别能力、次序一致性与分数相关性。相关性标注取值从 0（不相关）到 3（高度相关）；计算 AUC 时，标注 0–1 视为负例、2–3 视为正例。

**基线。**
工业评测中，我们将 UniGD 与生产级联系统比较——后者先做生成式检索，再经过一个独立部署的相关性模型。在公开基准上，我们将 UniGD 与代表性的生成式检索方法比较：DSI (Tay et al. 2022) 将检索形式化为文档标识符生成；NCI (Wang et al. 2022) 使用语义结构化的数字标识符；Ultron (Zhou et al. 2022) 结合关键词与语义文档标识符并采用多阶段训练；LTRGR (Li et al. 2024b) 将排序监督引入生成式检索；GenRRL (Zhou, Dou, and Wen 2023) 利用相关性反馈优化生成；DDRO (Mekonnen, Tang, and de Rijke 2025) 将基于 PQ 的标识符与直接偏好优化 (Rafailov et al. 2023) 相结合。

**实现细节。**
在公开基准上，我们以 T5-base（220M 参数）实例化 UniGD，并采用 DDRO 的基于 PQ 的 SID（$T=24$），复用 DDRO 冻结的 PQ 质心作为 CAM 码本。在工业设置中，我们使用约 60M 参数的六层 decoder-only Transformer，判别组件仅引入 1.12M 额外可训练参数。工业 SID 对商品广告使用 3 个量化层，对短视频与直播广告使用 4 层。两种设置下，UniGD 各阶段均以 AdamW 训练两个 epoch。阶段 1 以 $1\times 10^{-4}$ 的学习率、256 的批大小优化 $\mathcal{L}_{\mathrm{gen}}$。阶段 2 冻结共享 decoder，仅以 $\mathcal{L}_{\mathrm{disc}}$ 优化 query 侧与 item 侧适配器，学习率为 $1\times 10^{-3}$，$\lambda=0.5$，$\gamma=0.2$。阶段 3 使用 CAGE 联合优化两个目标，$\eta=100$、$\kappa=0.5$，其中 $\eta$ 平衡二者梯度量级；共享 decoder 学习率为 $5\times 10^{-5}$，任务专属模块学习率为 $1\times 10^{-4}$。工业训练在 8 张 A800 GPU 上耗时三天；公开与工业推理的束宽分别为 100 和 32。

### 公开基准评测（Public Benchmark Evaluation）

表1：在 NQ320K 与 MS300K 上与代表性生成式检索基线的生成式检索性能（%）。加粗：最佳；下划线：次佳。∗ 表示在我们的统一协议下复现的结果。† 表示在逐 query 指标值上经双侧配对 t 检验（$p<0.05$）相对各指标最强复现基线的统计显著提升。SI、TU、PQ 分别表示 Semantic ID、Title and URL 与 Product Quantization。

| 模型 | NQ320K Recall@1 | NQ320K Recall@5 | NQ320K Recall@10 | NQ320K MRR@10 | MS300K Recall@1 | MS300K Recall@5 | MS300K Recall@10 | MS300K MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DSI (SI) | 27.42 | 47.26 | 56.58 | 34.31 | 25.74 | 43.58 | 53.84 | 33.92 |
| NCI (SI) | 32.69 | 55.82 | 69.20 | 42.84 | 29.54 | 57.99 | 67.28 | 40.46 |
| Ultron (TU) | 33.78 | 54.20 | 67.05 | 42.51 | 29.82 | 60.39 | 68.31 | 42.53 |
| LTRGR (SI) | 32.80 | 56.20 | 68.74 | 44.80 | 32.69 | 64.37 | 72.43 | 47.85 |
| GenRRL (Sum)∗ | 35.82 | 57.04 | 71.49 | 45.93 | 32.78 | 63.91 | 75.80 | 45.98 |
| DDRO (TU)∗ | 40.52 | 52.81 | 55.63 | 45.68 | 37.92 | 66.13 | 73.72 | 49.74 |
| DDRO (PQ)∗ | 48.57 | 63.78 | 66.94 | 55.18 | 32.61 | 64.05 | 72.71 | 45.43 |
| **UniGD（本文）** | **72.33**† | **75.84**† | **77.52**† | **74.21**† | **60.64**† | **71.85**† | **78.22**† | **65.12**† |

表 1 在 NQ320K 与 MS300K 上将 UniGD 与代表性生成式检索方法进行比较。UniGD 在所有指标上持续取得最佳性能，并相对最强复现基线取得统计显著的提升。UniGD 在两个数据集上均将 Recall@1 提升超过 22 个绝对点，同时 Recall@10 相对增益在 NQ320K 上为 8.44%、在 MS300K 上为 3.19%。Recall@1 与 MRR@10 的提升表明 UniGD 能更可靠地将相关候选置于排序列表顶部，凸显了显式判别监督对学习 query 与候选之间细粒度相关性信号的价值。与此同时，Recall@10 的持续增益说明这一收益不止于头部排序的精细化，还通过共享生成骨干网络改善了候选覆盖。两个基准上的结果证明了统一生成与判别建模的有效性与可泛化性。

### 工业离线评测（Industrial Offline Evaluation）

我们在工业数据集上从生成式检索与判别式相关性估计两个互补视角评估 UniGD。

表2：在我们的工业测试集与 NQ320K 上的比较与消融结果。

| 变体 | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| **我们的工业测试集** | | | | |
| UniGD | 19.57 | 32.83 | 53.37 | 27.20 |
| w/o CAM | 13.24 | 26.67 | 46.33 | 20.21 |
| w/o HAM | 18.86 | 32.13 | 52.43 | 26.39 |
| w/o CAGE | 18.36 | 31.23 | 51.57 | 25.62 |
| Baseline | 12.53 | 25.82 | 43.98 | 19.14 |
| **NQ320K** | | | | |
| UniGD | 72.33 | 75.84 | 77.52 | 74.21 |
| w/o CAM | 45.52 | 58.96 | 65.26 | 52.03 |
| w/o CAGE | 67.22 | 71.45 | 74.38 | 69.18 |
| Baseline | 42.50 | 56.63 | 61.18 | 48.62 |

**生成式检索。**
表 2 将 UniGD 与一个使用相同骨干网络与 SID 配置、但仅训练自回归 SID 生成的 GR 基线进行比较。在我们的工业测试集上，UniGD 的 Recall@1 与 MRR@10 分别提升 7.04 与 8.06 个点，证明了联合建模候选生成与显式判别式相关性估计的价值。通过共享 decoder，判别目标以相关性监督补充 token 级生成，促使 query 表示捕获有信息量的 query–候选匹配信号。

**相关性估计。**
我们在人工标注的工业测试集上评估相关性预测，对比对象包括作为离线教师模型的 Qwen2-VL-7B、部署为在线学生模型的六层 Transformer，以及 GR + RelToken（通过 MLP 从 SID 后的相关性 token 状态预测相关性）。如表 4 所示，UniGD 在所有指标上均持续优于两种在线替代方案，展示了显式 query–ad 表示匹配的优势。凭借 60M 骨干网络与仅 1.1M 的额外参数，UniGD 与离线教师模型表现相当，且 Pearson 相关性略高。这些结果表明，UniGD 在生产环境中的生成式检索器内部保留了强相关性估计能力，避免了单独部署在线相关性模型。

表4：在我们工业人工标注测试集上的相关性估计性能。离线教师模型仅作参考，不用于在线推理。

| 模型设置 | AUC | Spearman | Pearson |
| --- | --- | --- | --- |
| Offline Teacher（离线教师） | 0.907 | 0.805 | 0.810 |
| Online Student（在线学生） | 0.881 | 0.783 | 0.784 |
| GR+RelToken | 0.890 | 0.795 | 0.796 |
| UniGD | 0.905 | 0.798 | 0.811 |

### 消融研究（Ablation Study）

我们在 NQ320K 与工业广告数据集上对 UniGD 的组件进行消融。如表 2 所示，我们在两个数据集上移除 CAM 并将 CAGE 替换为标准联合优化；在工业数据集上，我们还把 HAM 替换为所有物料类型共享的码本、SID 空间与生成头。移除 CAM 导致 Recall@1 与 MRR@10 的最大退化，说明码本锚定表示与显式相关性估计对候选排序的重要性。移除 CAGE 使所有指标一致退化，而将 HAM 替换为共享建模则降低工业性能，证实了冲突感知优化与物料专属建模各自的收益。图 3 考察了联合优化期间 CAGE 的效果：有 CAGE 时，生成与判别的验证准确率均稳步提升；没有 CAGE 时，二者在达到中间峰值后便开始下降。这些结果表明 CAGE 缓解了优化干扰，稳定了联合训练。

图3：UniGD 的训练动态。(a) 三个阶段的训练损失。(b) 有无 CAGE 时联合训练期间的生成与判别准确率。

### 工业在线评测（Industrial Online Evaluation）

我们在快手搜索广告系统中开展为期一周的受控在线 A/B 测试，系统服务数亿广告候选，其中新索引广告占每日流量的 36.9%。各组共享同一服务环境，仅检索与相关性估计流水线不同。我们将 UniGD 与生产级联系统（GR + 外置相关性模型）、GR + RelToken 以及 UniGD-Gen 进行比较。UniGD-Gen 保留 UniGD 的生成组件，但以生产相关性模型替换其内部判别打分。

表3：快手搜索广告系统中的在线 A/B 结果。Δ 表示相对生产基线的变化。Imp. (LT) 与 Imp. (New) 分别表示长尾广告与新广告的曝光变化。

| 模型设置 | Δ 收入 | Δ 曝光 | Δ Imp. (LT) | Δ Imp. (New) | Δ CTR | CVR | 不相关比率 | 延迟 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline（基线） | – | – | – | – | – | – | 6.79% | 13.0 ms |
| GR+RelToken | −2.08% | +1.84% | −3.42% | −5.71% | −0.25% | −0.24% | 17.30% | 10.1 ms |
| UniGD-Gen | +3.54% | +2.61% | +10.40% | +15.30% | +0.56% | +0.41% | 6.15% | 12.9 ms |
| UniGD | +5.78% | +3.12% | +16.80% | +24.60% | +0.94% | +0.68% | 5.81% | 8.7 ms |

如表 3 所示，相对生产基线，UniGD 将收入与新广告曝光分别提升 5.78% 与 24.60%，同时降低不相关比率。它还通过以 CAM 查找与轻量内部打分取代外置相关性模型，将检索延迟从 13.00 ms 降至 8.70 ms。GR + RelToken 使不相关比率大幅升至 17.30%，表明 SID 后的 token 状态所提供的相关性信号不如显式 query–ad 表示匹配可靠。UniGD 在所有主要指标上相对 GR + RelToken 与 UniGD-Gen 的一致优势表明，将判别式相关性估计直接整合进生成式检索能同时提升效果与服务效率。

## 结论（Conclusion）

我们提出了面向工业搜索广告的统一生成-判别框架 UniGD。UniGD 在共享模型中将自回归候选生成与显式相关性估计耦合：CAM 从语义码本重构 ad 表示，HAM 适配异构广告物料，CAGE 在联合优化期间协调两个目标。公开基准与大规模工业数据上的实验证明了其在生成式检索与相关性估计两方面的提升，快手搜索广告中的在线 A/B 测试进一步验证了其生产有效性。Beyond 搜索广告，UniGD 还有望通过将生成式候选检索与任务专属判别目标（如相关性估计、偏好预测与排序）耦合，应用于推荐与通用信息检索。在这些更广领域评估 UniGD 是未来工作的一个有前景方向。

## 参考文献（References，原样保留）

- Bajaj et al. (2016)
  Bajaj, P.; Campos, D.; Craswell, N.; Deng, L.; Gao, J.; Liu, X.; Majumder, R.; McNamara, A.; Mitra, B.; Nguyen, T.; et al. 2016. MS MARCO: A Human Generated Machine Reading Comprehension Dataset. arXiv:1611.09268.
- Bevilacqua et al. (2022)
  Bevilacqua, M.; Ottaviano, G.; Lewis, P.; Yih, W.-t.; Riedel, S.; and Petroni, F. 2022. Autoregressive Search Engines: Generating Substrings as Document Identifiers. In *Advances in Neural Information Processing Systems*, volume 35, 31668–31683.
- Chen et al. (2025)
  Chen, B.; Guo, X.; Wang, S.; Liang, Z.; Lv, Y.; Ma, Y.; Xiao, X.; Xue, B.; Zhang, X.; Yang, Y.; et al. 2025. OneSearch: A Preliminary Exploration of the Unified End-to-End Generative Framework for E-commerce Search. arXiv:2509.03236.
- Cheng et al. (2025)
  Cheng, J.; Dou, Z.; Zhu, Y.; and Li, X. 2025. Descriptive and Discriminative Document Identifiers for Generative Retrieval. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 39.
- Dai et al. (2025)
  Dai, S.; Tang, J.; Wu, J.; Wang, K.; Zhu, Y.; Chen, B.; Hong, B.; Zhao, Y.; Fu, C.; Wu, K.; et al. 2025. OnePiece: Bringing Context Engineering and Reasoning to Industrial Cascade Ranking System. arXiv:2509.18091.
- Deng et al. (2025)
  Deng, J.; Wang, S.; Cai, K.; Ren, L.; Hu, Q.; Ding, W.; Luo, Q.; and Zhou, G. 2025. OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment. arXiv:2502.18965.
- Fu et al. (2026)
  Fu, K.; Wu, C.; Yuan, K.; Cao, B.; Huang, D.; Yan, Y.; Zheng, J.; Zhang, J.; Zhou, S.; Wu, J.; et al. 2026. RankGR: Rank-Enhanced Generative Retrieval with Listwise Direct Preference Optimization in Recommendation. arXiv:2602.08575.
- Guo et al. (2025)
  Guo, X.; Chen, B.; Wang, S.; Yang, Y.; Lei, C.; Ding, Y.; and Li, H. 2025. OneSug: The Unified End-to-End Generative Framework for E-commerce Query Suggestion. arXiv:2506.06913.
- Karpukhin et al. (2020)
  Karpukhin, V.; Oğuz, B.; Min, S.; Lewis, P.; Wu, L.; Edunov, S.; Chen, D.; and Yih, W.-t. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*, 6769–6781.
- Kwiatkowski et al. (2019)
  Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.; Parikh, A.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin, J.; Lee, K.; et al. 2019. Natural Questions: A Benchmark for Question Answering Research. *Transactions of the Association for Computational Linguistics*, 7: 452–466.
- Li et al. (2024a)
  Li, M.; Wang, H.; Chen, Z.; Nie, G.; Qiu, Y.; Tang, G.; Liu, L.; and Zhuo, J. 2024a. Generative Retrieval with Preference Optimization for E-commerce Search. arXiv:2407.19829.
- Li et al. (2024b)
  Li, Y.; Yang, N.; Wang, L.; Wei, F.; and Li, W. 2024b. Learning to Rank in Generative Retrieval. In *Proceedings of the AAAI Conference on Artificial Intelligence*. ArXiv:2306.15222.
- Mekonnen, Tang, and de Rijke (2025)
  Mekonnen, K. A.; Tang, Y.; and de Rijke, M. 2025. Lightweight and Direct Document Relevance Optimization for Generative Information Retrieval. In *Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)*.
- Ni et al. (2022)
  Ni, J.; Abrego, G. H.; Constant, N.; Ma, J.; Hall, K.; Cer, D.; and Yang, Y. 2022. Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text Models. In *Findings of the Association for Computational Linguistics: ACL 2022*, 1864–1874.
- Nogueira and Lin (2019)
  Nogueira, R.; and Lin, J. 2019. From doc2query to docTTTTTquery. *Online preprint*.
- Pang et al. (2025)
  Pang, M.; Yuan, C.; He, X.; Fang, Z.; Xie, D.; Qu, F.; Jiang, X.; Peng, C.; Lin, Z.; Law, C.; et al. 2025. Generative Retrieval and Alignment Model: A New Paradigm for E-commerce Retrieval. In *Companion Proceedings of the ACM Web Conference 2025*, 413–421.
- Qu et al. (2021)
  Qu, Y.; Ding, Y.; Liu, J.; Liu, K.; Ren, R.; Zhao, W. X.; Dong, D.; Wu, H.; and Wang, H. 2021. RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. In *Proceedings of NAACL-HLT*, 5835–5847.
- Rafailov et al. (2023)
  Rafailov, R.; Sharma, A.; Mitchell, E.; Ermon, S.; Manning, C. D.; and Finn, C. 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv:2305.18290.
- Raffel et al. (2020)
  Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; and Liu, P. J. 2020. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *Journal of Machine Learning Research*, 21(140): 1–67.
- Rajput et al. (2023)
  Rajput, S.; Mehta, N.; Singh, A.; Keshavan, R. H.; Heldt, L.; Hong, L.; Tay, Y.; Tran, V. Q.; Samost, J.; Kula, M.; et al. 2023. Recommender Systems with Generative Retrieval. In *Advances in Neural Information Processing Systems*, volume 36.
- Robertson et al. (1995)
  Robertson, S. E.; Walker, S.; Jones, S.; Hancock-Beaulieu, M. M.; and Gatford, M. 1995. Okapi at TREC-3. In *Proceedings of the Third Text REtrieval Conference (TREC-3)*, 109–126.
- Sun et al. (2023)
  Sun, W.; Yan, L.; Chen, Z.; Wang, S.; Zhu, H.; Ren, P.; Chen, Z.; Yin, D.; de Rijke, M.; and Ren, Z. 2023. Learning to Tokenize for Generative Retrieval. In *Advances in Neural Information Processing Systems (NeurIPS)*.
- Sun et al. (2026)
  Sun, Y.; Huang, S.; Guan, Z.; Luo, Q.; Tang, R.; Gai, K.; and Zhou, G. 2026. GRank: Towards Target-Aware and Streamlined Industrial Retrieval with a Generate-Rank Framework. In *Proceedings of the ACM Web Conference 2026*, 1–11.
- Tang et al. (2024)
  Tang, Y.; Zhang, R.; Guo, J.; de Rijke, M.; Chen, W.; and Cheng, X. 2024. Generative Retrieval Meets Multi-Graded Relevance. In *Advances in Neural Information Processing Systems*, volume 37.
- Tay et al. (2022)
  Tay, Y.; Tran, V.; Dehghani, M.; Ni, J.; Bahri, D.; Mehta, H.; Qin, Z.; Hui, K.; Zhao, Z.; Gupta, J.; et al. 2022. Transformer Memory as a Differentiable Search Index. In *Advances in Neural Information Processing Systems*, volume 35, 21831–21843.
- van den Oord, Vinyals, and Kavukcuoglu (2017)
  van den Oord, A.; Vinyals, O.; and Kavukcuoglu, K. 2017. Neural Discrete Representation Learning. In *Advances in Neural Information Processing Systems*, volume 30.
- Wang et al. (2022)
  Wang, Y.; Hou, Y.; Wang, H.; Miao, Z.; Wu, S.; Chen, Q.; Xia, Y.; Chi, C.; Zhao, G.; Liu, Z.; et al. 2022. A Neural Corpus Indexer for Document Retrieval. In *Advances in Neural Information Processing Systems*, volume 35, 25600–25614.
- Xiong et al. (2021)
  Xiong, L.; Xiong, C.; Li, Y.; Tang, K.-F.; Liu, J.; Bennett, P. N.; Ahmed, J.; and Overwijk, A. 2021. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. In *International Conference on Learning Representations (ICLR)*.
- Xue et al. (2026)
  Xue, B.; Liu, D.; Wang, L.; Sun, M.; Wang, P.; Zhang, P.; Shi, S.; Xu, T.; Sha, Y.; Liu, Z.; et al. 2026. Generative Recommendation for Large-Scale Advertising. arXiv:2602.22732.
- Zeng et al. (2024)
  Zeng, H.; Luo, C.; Jin, B.; Sarwar, S. M.; Wei, T.; and Zamani, H. 2024. Scalable and Effective Generative Information Retrieval. In *Proceedings of the ACM Web Conference (WWW)*.
- Zeng, Luo, and Zamani (2024)
  Zeng, H.; Luo, C.; and Zamani, H. 2024. Planning Ahead in Generative Retrieval: Guiding Autoregressive Generation through Simultaneous Decoding. In *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval*.
- Zhan et al. (2020)
  Zhan, J.; Mao, J.; Liu, Y.; Zhang, M.; and Ma, S. 2020. RepBERT: Contextualized Text Embeddings for First-Stage Retrieval. arXiv:2006.15498.
- Zhang et al. (2025)
  Zhang, L.; Song, K.; Lee, Y. Q.; Guo, W.; Wang, H.; Li, Y.; Guo, H.; Liu, Y.; Lian, D.; and Chen, E. 2025. Killing Two Birds with One Stone: Unifying Retrieval and Ranking with a Single Generative Recommendation Model. arXiv:2504.16454.
- Zheng et al. (2025)
  Zheng, Z.; Wang, Z.; Yang, F.; Fan, J.; Zhang, T.; Wang, Y.; and Wang, X. 2025. EGA-V2: An End-to-end Generative Framework for Industrial Advertising. arXiv:2505.17549.
- Zhou, Dou, and Wen (2023)
  Zhou, Y.; Dou, Z.; and Wen, J.-R. 2023. Enhancing Generative Retrieval with Reinforcement Learning from Relevance Feedback. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 12481–12490.
- Zhou et al. (2024)
  Zhou, Y.; Yao, J.; Dou, Z.; Tu, Y.; Wu, L.; Chua, T.-S.; and Wen, J.-R. 2024. ROGER: Ranking-Oriented Generative Retrieval. *ACM Transactions on Information Systems*, 43(1).
- Zhou et al. (2022)
  Zhou, Y.; Yao, J.; Dou, Z.; Wu, L.; Zhang, P.; and Wen, J.-R. 2022. Ultron: An Ultimate Retriever on Corpus with a Model-based Indexer. arXiv:2208.09257.
- Zhuang et al. (2022)
  Zhuang, S.; Ren, H.; Shou, L.; Pei, J.; Gong, M.; Zuccon, G.; and Jiang, D. 2022. Bridging the Gap Between Indexing and Retrieval for Differentiable Search Index with Query Generation. arXiv:2206.10128.

（参考文献共 37 条，按原文顺序原样保留，未作翻译。）
