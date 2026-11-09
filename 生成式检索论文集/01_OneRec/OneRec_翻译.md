# OneRec：用生成式推荐器与迭代偏好对齐统一检索与排序

**英文原题**：OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment

**作者与机构**：
Jiaxin Deng\*、Shiyao Wang\*（\*同等贡献）、Kuo Cai、Lejian Ren、Qigen Hu、Weifeng Ding、Qiang Luo、Guorui Zhou（通讯作者）
KuaiShou Inc., Beijing, China（快手，中国北京）

**arXiv 编号与译注**：arXiv:2502.18965v1 [cs.IR]，2025 年 2 月 26 日提交。
译注：(1) arXiv 当前元数据标题为 "...Generative Recommender and **Iterative** Preference Alignment"，而 v1 版 PDF/HTML 正文标题为 "...Generative Recommender and Preference Alignment"（无 "Iterative"），本文以 arXiv 当前标题为准。(2) 本翻译忠实还原原文结构与内容，包括原文中的少量笔误（如 "remantic IDs" 应为 "semantic IDs"，译文径按正确含义译出）。(3) 文内引用采用原文的"作者-年份"格式，原样保留；参考文献表保留英文原文，未做翻译。

---

###### 摘要

近来，基于生成式检索的推荐系统（GR）以自回归方式直接生成候选视频，已成为一种颇具前景的范式。然而，大多数现代推荐系统仍采用"检索+排序"（retrieve-and-rank）策略，生成式模型仅在检索阶段充当一个选择器。本文提出 OneRec，用一个统一的生成式模型取代级联式学习框架。据我们所知，这是首个在真实场景中显著超越当前复杂且精心设计的推荐系统的端到端生成式模型。具体而言，OneRec 包含：1）一个编码器-解码器结构，对用户历史行为序列进行编码，并逐步解码出用户可能感兴趣的视频。我们采用稀疏混合专家（Mixture-of-Experts, MoE），在不按比例增加计算 FLOPs 的情况下扩展模型容量。2）一种会话级（session-wise）生成方法。与传统的下一物品（next-item）预测不同，我们提出会话级生成；相比依赖手工规则来恰当组合生成结果的逐点（point-by-point）生成，它更为优雅且上下文连贯。3）一个结合直接偏好优化（Direct Preference Optimization, DPO）的迭代偏好对齐（Iterative Preference Alignment）模块，用于提升生成结果的质量。与 NLP 中的 DPO 不同，推荐系统对每个用户的每次浏览请求通常只有一次展示结果的机会，因而无法同时获得正、负样本。为突破这一限制，我们设计了奖励模型来模拟用户生成，并根据推荐系统在线学习的特性定制采样策略。大量实验表明，有限数量的 DPO 样本即可对齐用户兴趣偏好并显著提升生成结果的质量。我们将 OneRec 部署于拥有数亿日活跃用户的短视频推荐平台快手的主场景，实现了 1.6% 的观看时长增长，这是一项相当可观的提升。

###### 关键词：

生成式推荐、自回归生成、语义分词、直接偏好优化

## 1. 引言

为了在效率与有效性之间取得平衡，大多数现代推荐系统采用级联排序（cascade ranking）策略（Covington et al., 2016; Liu et al., 2017; Wang et al., 2011; Qin et al., 2022）。如图 1(b) 所示，一个典型的级联排序系统采用三阶段流水线：召回（recall）（Covington et al., 2016; Huang et al., 2013; Zhu et al., 2018）、粗排（pre-ranking）（Ma et al., 2021; Wang et al., 2020）与精排（ranking）（Burges, 2010; Guo et al., 2017; Hidasi, 2015; Zhou et al., 2019; Zhou et al., 2018; Pi et al., 2020; Chang et al., 2023）。每个阶段负责从接收到的物品中选出 top-$k$ 物品并将结果传给下一阶段，共同平衡系统响应时间与排序精度之间的折中。

图1：(a) 我们提出的用于端到端生成的统一架构。(b) 一个典型的级联排序系统，自底向上包括三个阶段：检索、粗排与精排。

尽管在实践中颇为高效，现有方法通常将各个排序器相互独立地对待，每个孤立阶段的有效性构成了后续排序阶段的性能上界，从而限制了整体排序系统的性能。尽管已有诸多努力（Gallagher et al., 2019; Fei et al., 2021; Hron et al., 2021; Qin et al., 2022; Huang et al., 2023; Wang et al., 2024a）通过让排序器之间产生交互来提升整体推荐性能，它们仍然保持着传统的级联排序范式。近来，基于生成式检索的推荐系统（GR）（Rajput et al., 2023; Wang et al., 2024b; Zheng et al., 2024）以自回归序列生成的方式直接生成候选物品的标识符，已成为一种颇具前景的范式。通过用量化后的语义ID（semantic ID）索引物品以编码物品语义（Lee et al., 2022），推荐器得以利用物品内部蕴含的丰富语义信息。GR 的生成特性使其适合通过束搜索（beam search）解码直接选取候选物品，并产生更加多样的推荐结果。然而，当前的生成式模型仅充当检索阶段的选择器，因为其推荐精度尚无法与精心设计的多层级联排序器相匹敌。

为应对上述挑战，我们提出名为 OneRec 的单阶段推荐统一端到端生成式框架。首先，我们提出一个编码器-解码器架构。受训练大语言模型时观察到的缩放定律（scaling laws）的启发，我们发现扩展推荐模型的容量同样能持续提升性能。因此我们基于 MoE 结构（Zoph et al., 2022; Du et al., 2022; Dai et al., 2024）扩大模型参数，显著增强了模型刻画用户兴趣的能力。其次，不同于传统的逐点下一物品预测，我们提出一种会话级列表生成方法，考虑每个会话内物品的相对内容与顺序。逐点生成方法需要手工设计的策略来保证生成结果的连贯性与多样性；相比之下，会话级学习过程通过向模型喂入经过挑选的偏好数据，使模型能够自主学习最优的会话结构。最后同样重要的是，我们探索利用直接偏好优化（DPO）（Rafailov et al., 2024）进行偏好学习，以进一步提升生成结果的质量。在构建偏好对（preference pairs）时，我们受困难负采样（hard negative sampling）（Shi et al., 2023）启发，从束搜索结果中构造"自困难"（self-hard）被拒样本，而非随机采样。我们提出迭代偏好对齐（Iterative Preference Alignment, IPA）策略，基于预训练奖励模型（reward model, RM）给出的分数对采样得到的响应进行排序，识别出最优的选中（chosen）样本与最差的被拒（rejected）样本。我们在大规模工业数据集上的实验表明了所提方法的优越性；我们还进行了一系列消融实验，详细论证了每个模块的有效性。本文的主要贡献总结如下：

- • 为克服级联排序的局限，我们引入 OneRec，一个单阶段生成式推荐框架。据我们所知，这是最早能够以统一生成模型处理物品推荐、并显著超越传统多阶段排序流水线的工业方案之一。

- • 我们通过会话级生成方式，凸显了模型容量与目标物品上下文信息的必要性，从而实现更准确的预测并增强生成物品的多样性。

- • 我们提出了一种基于个性化奖励模型的新型自困难负样本选择策略。借助直接偏好优化，我们增强了 OneRec 在更广泛用户偏好上的泛化能力。大量离线实验与在线 A/B 测试验证了其有效性与效率。

图2：OneRec 的整体框架，包含两个阶段：(i) 会话训练阶段，使用会话级数据训练 OneRec；(ii) IPA 阶段，利用带自困难负样本的迭代直接偏好优化。

## 2. 相关工作

### 2.1. 生成式推荐

近年来，随着生成式模型的显著进展，生成式推荐受到越来越多的关注。与传统的基于嵌入的检索方法不同——后者主要依赖双塔模型为每个候选物品计算排序分，并借助高效的 MIPS 或 ANN（Houle and Nett, 2014; Muja and Lowe, 2014; Shrivastava and Li, 2014; Ge et al., 2013; Jegou et al., 2010）搜索系统检索 top-$k$ 相关物品——生成式检索（Generative Retrieval, GR）（Tang et al., 2023）方法将从数据库中检索相关文档的问题形式化为一个序列生成任务，逐个生成相关文档的词元（token）。文档词元可以是文档标题、文档 ID 或预训练得到的语义ID（Tay et al., 2022）。GENRE（De Cao et al., 2020）首次采用 Transformer 架构进行实体检索，基于条件上下文以自回归方式生成实体名称。DSI（Tay et al., 2022）首次提出为文档赋予结构化语义ID、并训练编码器-解码器模型进行生成式文档检索的概念。沿循这一范式，TIGER（Rajput et al., 2023）引入了面向推荐系统的生成式物品检索模型的形式化定义。

除生成框架本身之外，如何索引物品也引起了越来越多的关注。近期研究聚焦于语义索引技术（Rajput et al., 2023; Tay et al., 2022; Feng et al., 2022），其目标是基于内容信息来索引物品。具体而言，TIGER（Rajput et al., 2023）与 LC-Rec（Zheng et al., 2024）对从物品标题和描述导出的文本嵌入应用残差量化（RQ-VAE）以完成分词。Recforest（Feng et al., 2022）对物品文本嵌入利用层次化 k-means 聚类，得到聚类索引作为词元。此外，近期研究如 EAGER（Wang et al., 2024b）则探索将语义信息与协同信息同时整合进分词过程。

### 2.2. 语言模型的偏好对齐

在大语言模型（LLM）的后训练（Dubey et al., 2024）阶段，基于人类反馈的强化学习（RLHF）（Stiennon et al., 2020; Ouyang et al., 2022）是让 LLM 与人类价值观对齐的主流方法，它以代表人类反馈的奖励模型为指导、运用强化学习技术完成对齐。然而，RLHF 存在不稳定与低效的问题。直接偏好优化（DPO）（Rafailov et al., 2024）由此被提出：它以闭式推导出最优策略，使得可以直接利用偏好数据进行优化。除此之外，学界还提出了若干变体以进一步改进原始 DPO。例如，IPO（Azar et al., 2024）以一个通用目标绕过了 DPO 中的两处近似；cDPO（Rafailov et al., 2024）通过引入超参数 $\epsilon$ 来缓解噪声标签的影响；rDPO（Chowdhury et al., 2024）为原始二值交叉熵损失设计了无偏估计。其他变体还包括 CPO（Xu et al., 2024b）、simDPO（Chowdhury et al., 2024）等，它们也从不同方面增强或扩展了 DPO。然而，与偏好数据由人类显式标注的传统 NLP 场景不同，推荐系统中的偏好学习因用户-物品交互数据的稀疏性而面临独特挑战。这一挑战导致"将 DPO 适配于推荐"的研究在很大程度上仍未被充分探索。与 S-DPO 侧重于为基于 LM 的推荐器在用户偏好数据中引入多个负样本不同，我们训练了一个奖励模型，并基于奖励模型给出的分数为不同用户挑选个性化的偏好数据。

## 3. 方法

在本节中，我们提出 OneRec，一个以单阶段检索方式生成目标物品的端到端框架。在 3.1 节中，我们首先从特征工程的角度介绍工业应用中单阶段生成式推荐流水线的构建。随后在 3.2 节中，我们形式化地定义会话级生成任务，并给出我们所提出的 OneRec 模型的架构。最后，我们在 3.3 节中详述模型结合个性化奖励模型进行自困难负采样的能力，并展示我们如何通过直接偏好优化迭代地提升模型性能。OneRec 的整体框架如图 2 所示。

### 3.1. 预备知识

本节中，我们从特征工程的角度介绍单阶段生成式推荐流水线的构建。对于用户侧特征，OneRec 以正样本历史行为序列 $\mathcal{H}_{u}=\{\bm{v}^{h}_{1},\bm{v}^{h}_{2},\ldots,\bm{v}^{h}_{n}\}$ 作为输入，其中 $\bm{v}$ 表示用户有效观看过或与之交互过（点赞、关注、分享）的视频，$n$ 为行为序列的长度。OneRec 的输出是一个视频列表，构成一个会话 $\mathcal{S}=\{\bm{v}_{1},\bm{v}_{2},...,\bm{v}_{m}\}$，其中 $m$ 为会话内视频的数量（"会话"的详细定义见 3.2 节）。

对于每个视频 $\bm{v}_{i}$，我们用多模态嵌入 $\textbf{{e}}_{i}\in\mathbb{R}^{d}$ 来描述它，这些嵌入已与真实的用户-物品行为分布对齐（Luo et al., 2024）。基于这一预训练多模态表示，现有生成式推荐框架（Liu et al., 2024; Rajput et al., 2023）使用 RQ-VAE（Zeghidour et al., 2021）将嵌入编码成语义词元。然而，此类方法因码字分布不均衡而并非最优，这一现象被称为"沙漏现象"（hourglass phenomenon）（Kuai et al., 2024）。我们应用一种多层次均衡量化机制，以残差 K-Means 量化算法（Luo et al., 2024）来变换 $\textbf{{e}}_{i}$。在第一层（$l=1$），初始残差定义为 $\bm{r}_{i}^{1}=\bm{e}_{i}$。在每一层 $l$，我们有一个码本 $\mathcal{C}_{l}=\{\bm{c}_{1}^{l},...,\bm{c}_{K}^{l}\}$，其中 $K$ 为码本大小。距离最近的质心节点嵌入的索引通过 $\bm{s}_{i}^{l}=\arg\min_{k}\|\textbf{{r}}_{i}^{l}-\bm{c}_{k}^{l}\|_{2}^{2}$ 生成，而下一层 $l+1$ 的残差定义为 $\bm{r}_{i}^{l+1}=\bm{r}_{i}^{l}-\bm{c}_{\bm{s}_{i}^{l}}^{l}$。于是，相应的码本词元通过如下层次化索引生成：

$$\bm{s}_{i}^{1}=\arg\min_{k}\|\bm{r}_{i}^{1}-\bm{c}_{k}^{1}\|_{2}^{2},\quad\bm{r}_{i}^{2}=\bm{r}_{i}^{1}-\bm{c}_{\bm{s}_{i}^{1}}^{1}$$

$$\bm{s}_{i}^{2}=\arg\min_{k}\|\bm{r}_{i}^{2}-\bm{c}_{k}^{2}\|_{2}^{2},\quad\bm{r}_{i}^{3}=\bm{r}_{i}^{2}-\bm{c}_{\bm{s}_{i}^{2}}^{2}$$

$$\vdots$$

$$\bm{s}_{i}^{L}=\arg\min_{k}\|\bm{r}_{i}^{L}-\bm{c}_{k}^{L}\|_{2}^{2}$$

其中 $L$ 为语义ID 的总层数。

为构建均衡的码本 $\mathcal{C}_{l}=\{\bm{c}_{1}^{l},\ldots,\bm{c}_{K}^{l}\}$，我们按算法 1 所述应用 Balanced K-means 对物品集合进行划分。给定视频全集 $\mathcal{V}$，该算法将集合划分为 $K$ 个簇，每个簇恰好包含 $w=|\mathcal{V}|/K$ 个视频。在迭代计算过程中，每个质心依据欧氏距离依次被分配 $w$ 个最近的尚未分配的视频，随后用被分配视频的均值向量对质心进行重新校准。当簇分配达到收敛时即满足终止准则。

**算法 1 Balanced K-means 聚类**

输入：物品集合 $\mathcal{V}$，簇数 $K$

1. 计算 $w\leftarrow|\mathcal{V}|/K$
2. 以随机选择初始化质心 $\mathcal{C}_{l}=\{\bm{c}_{1}^{l},\ldots,\bm{c}_{K}^{l}\}$；
3. **repeat**
4.  &emsp;初始化未分配集合 $\mathcal{U}\leftarrow\mathcal{V}$
5.  &emsp;**for** 每个簇 $k\in\{1,\ldots,K\}$ **do**
6.  &emsp;&emsp;按到质心 $\bm{c}_{k}^{l}$ 的距离升序排序 $\mathcal{U}$；
7.  &emsp;&emsp;分配 $\mathcal{V}_{k}\leftarrow\mathcal{U}[0:w-1]$；
8.  &emsp;&emsp;更新质心 $\bm{c}_{k}^{l}\leftarrow\frac{1}{w}\sum_{\bm{r}^{l}\in\mathcal{V}_{k}}\bm{r}^{l}$；
9.  &emsp;&emsp;移除已分配物品 $\mathcal{U}\leftarrow\mathcal{U}\setminus\mathcal{V}_{k}$；
10. &emsp;**end for**
11. **until** 分配收敛；

输出：优化后的码本 $\mathcal{C}_{l}=\{\bm{c}_{1}^{l},\ldots,\bm{c}_{K}^{l}\}$

### 3.2. 会话级列表生成

与只预测下一个视频的传统逐点（point-wise）推荐方法不同，会话级（session-wise）生成旨在基于用户的历史交互序列生成高价值会话的列表，这使得推荐模型能够捕捉推荐列表中各视频之间的依赖关系。具体而言，一个会话（session）指响应用户一次请求所返回的一批短视频，通常由 5 到 10 个视频组成。会话内的视频一般需要综合考虑用户兴趣、连贯性与多样性等因素。我们制定了若干识别高质量会话的准则，包括：

- • 用户在会话内实际观看的短视频数量大于等于 5；

- • 用户观看该会话的总时长超过某个阈值；

- • 用户表现出交互行为，例如对视频进行点赞、收藏或分享。

这一做法确保我们的会话级模型从真实的用户参与模式中学习，并捕捉会话列表内部更准确的上下文信息。于是，我们的会话级模型 $\mathcal{M}$ 的目标可以形式化为：

$$\mathcal{S}:=\mathcal{M}(\mathcal{H}_{u})\tag{1}$$

其中 $\mathcal{H}_{u}$ 由语义ID 表示：$\mathcal{H}_{u}=\{(\bm{s}_{1}^{1},\bm{s}_{1}^{2},\cdots,\bm{s}_{1}^{L}),(\bm{s}_{2}^{1},\bm{s}_{2}^{2},\cdots,\bm{s}_{2}^{L}),\cdots,(\bm{s}_{n}^{1},\bm{s}_{n}^{2},\cdots,\bm{s}_{n}^{L})\}$，而 $\mathcal{S}=\{(\bm{s}_{1}^{1},\bm{s}_{1}^{2},\cdots,\bm{s}_{1}^{L}),(\bm{s}_{2}^{1},\bm{s}_{2}^{2},\cdots,\bm{s}_{2}^{L}),\cdots,(\bm{s}_{m}^{1},\bm{s}_{m}^{2},\cdots,\bm{s}_{m}^{L})\}$。

如图 2(a) 所示，与 T5（Xu et al., 2024a）架构一致，我们的模型采用基于 Transformer 的框架，由两个主要部件组成：用于建模用户历史交互的编码器，以及用于会话列表生成的解码器。具体而言，编码器利用堆叠的多头自注意力层与前馈层来处理输入序列 $\mathcal{H}_{u}$。我们将编码后的历史交互特征记为 $\textbf{{H}}=Encoder(\mathcal{H}_{u})$。

解码器以目标会话的语义ID 作为输入，并以自回归方式生成目标。为了以合理的经济成本训练更大的模型，对于解码器中的前馈神经网络（FNN），我们采用基于 Transformer 的语言模型常用的 MoE 架构（Zoph et al., 2022; Du et al., 2022; Dai et al., 2024），将第 $l$ 个 FFN 替换为：

$$\begin{split}\mathbf{H}_{t}^{l+1}&=\sum_{i=1}^{N_{\rm MoE}}\left({g_{i,t}\operatorname{FFN}_{i}\left(\mathbf{H}_{t}^{l}\right)}\right)+\mathbf{H}_{t}^{l},\\
g_{i,t}&=\begin{cases}s_{i,t},&s_{i,t}\in\operatorname{Topk}(\{s_{j,t}|1\leq j\leq N\},K_{\rm MoE}),\\
0,&\text{otherwise},\end{cases}\\
s_{i,t}&=\operatorname{Softmax}_{i}\left({\mathbf{H}_{t}^{l}}^{T}\mathbf{e}_{i}^{l}\right),\end{split}\tag{2}$$

其中 $N_{\rm MoE}$ 表示专家总数，$\operatorname{FFN}_{i}(\cdot)$ 是第 $i$ 个专家 FFN，$g_{i,t}$ 表示第 $i$ 个专家的门控值。门控值 $g_{i,t}$ 是稀疏的，也就是说 $N_{\rm MoE}$ 个门控值中只有 $K_{\rm MoE}$ 个非零。这一稀疏性保证了 MoE 层内部的计算效率，每个词元只会被分配到 $K_{\rm MoE}$ 个专家中并在其中完成计算。

训练时，我们在词元序列的开头加入起始词元 $\bm{s}_{[\rm BOS]}$，以构造解码器输入：

$$\begin{gathered}{\mathcal{\bar{S}}}=\{\bm{s}_{[\rm BOS]},\bm{s}_{1}^{1},\bm{s}_{1}^{2},\cdots,\bm{s}_{1}^{L},\bm{s}_{[\rm BOS]},\bm{s}_{2}^{1},\bm{s}_{2}^{2},\cdots,\bm{s}_{2}^{L},\\
\cdots,\bm{s}_{[\rm BOS]},\bm{s}_{m}^{1},\bm{s}_{m}^{2},\cdots,\bm{s}_{m}^{L}\}\end{gathered}\tag{3}$$

我们在目标会话的语义ID 上使用交叉熵损失进行下一词元预测（next-token prediction）。NTP 损失 $\mathcal{L}_{\rm NTP}$ 形式化为：

$$\begin{gathered}\mathcal{L}_{\rm NTP}=-\sum_{i=1}^{m}\sum_{j=1}^{L}\log P(\bm{s}_{i}^{j+1}\mid[\bm{s}_{[\rm BOS]},\bm{s}_{1}^{1},\bm{s}_{1}^{2},\cdots,\bm{s}_{1}^{L},\cdots,\\
\bm{s}_{[\rm BOS]},\bm{s}_{i}^{1},\cdots,\bm{s}_{i}^{j}];\Theta).\end{gathered}\tag{4}$$

在会话级列表生成任务上经过一定量的训练后，我们得到种子模型 $\mathcal{M}_{t}$。

### 3.3. 结合奖励模型的迭代偏好对齐

3.2 节所定义的高质量会话提供了宝贵的训练数据，使模型能够学习"什么构成一个好的会话"，从而保证所生成视频的质量。在此基础之上，我们希望通过直接偏好优化（DPO）进一步增强模型的能力。在传统自然语言处理（NLP）场景中，偏好数据由人类显式标注；然而，推荐系统中的偏好学习因用户-物品交互数据的稀疏性而面临独特挑战，这就需要一个奖励模型（RM）。为此，我们在 3.3.1 节引入一个会话级奖励模型。此外，我们改进了传统 DPO，提出能让模型实现自我提升（self-improvement）的迭代直接偏好优化，见 3.3.2 节的描述。

**算法 2 迭代偏好对齐（IPA）**

输入：响应数量 $N$，预训练 RM $R(\bm{u},\mathcal{S})$，种子模型 $\mathcal{M}_{t}$，DPO 比例 $r_{\mathrm{DPO}}$，总轮数 $T$，每轮样本数 $N_{\mathrm{sample}}$

1. **for** $\mathit{epoch}\leftarrow t$ **to** $T$ **do**
2.  &emsp;**for** $\mathit{sample}\leftarrow 1$ **to** $N_{\mathrm{sample}}$ **do**
3.  &emsp;&emsp;**if** $\mathit{rand}()<r_{\mathrm{DPO}}$ **then**
4.  &emsp;&emsp;&emsp;通过 $\mathcal{M}_{t}$ 生成 $N$ 个响应：
5.  &emsp;&emsp;&emsp;**for** $i\leftarrow 1$ **to** $N$ **do**
6.  &emsp;&emsp;&emsp;&emsp;$\mathcal{S}_{u}^{i}\sim\mathcal{M}_{t}(\mathcal{H}_{u})$；
7.  &emsp;&emsp;&emsp;&emsp;$r_{u}^{i}\leftarrow R(\bm{u},\mathcal{S}_{u}^{i})$；
8.  &emsp;&emsp;&emsp;**end for**
9.  &emsp;&emsp;&emsp;选出最优与最差响应：
10. &emsp;&emsp;&emsp;$\mathcal{S}_{u}^{w}\leftarrow\mathcal{S}_{u}^{\arg\max_{i}r_{u}^{i}}$；
11. &emsp;&emsp;&emsp;$\mathcal{S}_{u}^{l}\leftarrow\mathcal{S}_{u}^{\arg\min_{i}r_{u}^{i}}$；
12. &emsp;&emsp;&emsp;计算 NTP 与 DPO 损失：
13. &emsp;&emsp;&emsp;$\mathcal{L}\leftarrow\mathcal{L}_{\mathrm{NTP}}+\lambda\mathcal{L}_{\mathrm{DPO}}$；
14. &emsp;&emsp;**else**
15. &emsp;&emsp;&emsp;计算 NTP 损失：
16. &emsp;&emsp;&emsp;$\mathcal{L}\leftarrow\mathcal{L}_{\mathrm{NTP}}$；
17. &emsp;&emsp;**end if**
18. &emsp;&emsp;更新参数：
19. &emsp;&emsp;$\Theta\leftarrow\Theta-\alpha\nabla_{\Theta}\mathcal{L}$；
20. &emsp;**end for**
21. &emsp;更新模型快照：$\mathcal{M}_{t+1}\leftarrow\mathcal{M}_{t}$；
22. **end for**

输出：优化后的参数 $\Theta$

#### 3.3.1. 奖励模型训练

我们用 $R(\bm{u},\mathcal{S})$ 表示奖励模型，它负责为不同用户挑选偏好数据。这里，输出 $r$ 表示用户 $u$（通常由用户行为来表征）对会话 $\mathcal{S}=\{\bm{v}_{1},\bm{v}_{2},\ldots,\bm{v}_{m}\}$ 的偏好所对应的奖励。

为了让 RM 具备对会话进行排序的能力，我们首先提取 $\mathcal{S}$ 中每个物品 $\bm{v}_{i}$ 的目标感知（target-aware）表示 $\bm{e}_{i}=\bm{v}_{i}\odot\bm{u}$，其中 $\odot$ 表示目标感知操作（例如面向用户行为的目标注意力）。由此我们得到会话 $\mathcal{S}$ 的目标感知表示 $\bm{h}=\{\bm{e}_{1},\bm{e}_{2},\cdots,\bm{e}_{m}\}$。随后，会话内的物品通过自注意力层彼此交互，融合不同物品之间必要的信息：

$$\bm{h}_{f}=\mathrm{SelfAttention}(\bm{h}\bm{W}^{Q}_{s},\bm{h}\ \bm{W}^{K}_{s},\bm{h}\bm{W}^{V}_{s})\tag{5}$$

接下来，我们利用不同的塔（tower）对多目标奖励进行预测，且 RM 已用海量推荐数据完成预训练：

$$\small\begin{split}\hat{r}^{swt}&=\texttt{Tower}^{swt}\big(\texttt{Sum}\big(\bm{h}_{f}\big)\big),\hat{r}^{vtr}=\texttt{Tower}^{vtr}\big(\texttt{Sum}\big(\bm{h}_{f}\big)\big),\\
\hat{r}^{wtr}&=\texttt{Tower}^{wtr}\big(\texttt{Sum}\big(\bm{h}_{f}\big)\big),\hat{r}^{ltr}=\texttt{Tower}^{ltr}\big(\texttt{Sum}\big(\bm{h}_{f}\big)\big),\\
&\texttt{where}\ \texttt{Tower}(\cdot)=\texttt{Sigmoid}\big(\texttt{MLP}(\cdot)\big)\end{split}\tag{6}$$

在得到每个会话的所有估计奖励 $\hat{r}^{swt},\dots$ 与真实标签 $y^{swt},\dots$ 之后，我们直接最小化二值交叉熵损失来训练 RM。损失函数 $\mathcal{L}_{\rm RM}$ 定义如下：

$$\begin{split}\mathcal{L}_{\rm RM}=-\sum_{{swt,\dots}}^{xtr}\left(y^{xtr}\log{(\hat{r}^{xtr})}+(1-y^{xtr})\log{(1-\hat{r}^{xtr})}\right)\end{split}\tag{7}$$

#### 3.3.2. 迭代偏好对齐

基于预训练 RM $R(\bm{u},\mathcal{S})$ 与当前的 OneRec $\mathcal{M}_{t}$，我们通过束搜索为每个用户生成 $N$ 个不同的响应：

$$\mathcal{S}_{u}^{n}\sim M_{t}(\mathcal{H}_{u})\quad\text{for all}\ u\in\mathcal{U}\ \text{and}\ n\in[N],\tag{8}$$

其中我们用 $[N]$ 表示 $\{1,2,\dots,N\}$。

然后，我们基于 RM $R(\bm{u},\mathcal{S})$ 计算上述每个响应的奖励 $r_{u}^{n}$：

$$r_{u}^{n}=R(\bm{u},\mathcal{S}_{u}^{n})\tag{9}$$

接下来，我们通过选取奖励值最高的获胜响应 $(\mathcal{S}_{u}^{w},\mathcal{H}_{u})$ 和奖励值最低的失败响应 $(\mathcal{S}_{u}^{l},\mathcal{H}_{u})$，构建偏好对 $D_{t}^{\text{pairs}}=(\mathcal{S}_{u}^{w},\mathcal{S}_{u}^{l},\mathcal{H}_{u})$。给定偏好对，我们便可以训练一个新模型 $M_{t+1}$：它从模型 $M_{t}$ 初始化而来，并使用结合了 DPO 损失（Rafailov et al., 2024）的损失函数进行更新，以从偏好对中学习。每个偏好对对应的损失如下：

$$\begin{split}\mathcal{L}_{\text{DPO}}&=\mathcal{L}_{\text{DPO}}(\mathcal{S}_{u}^{w},\mathcal{S}_{u}^{l}|\mathcal{H}_{u})\\
&=-\log\sigma\left(\beta\log\frac{M_{t+1}(\mathcal{S}_{u}^{w}|\mathcal{H}_{u})}{M_{t}(\mathcal{S}_{u}^{w}|\mathcal{H}_{u})}-\beta\log\frac{M_{t+1}(\mathcal{S}_{u}^{l}|\mathcal{H}_{u})}{M_{t}(\mathcal{S}_{u}^{l}|\mathcal{H}_{u})}\right).\end{split}\tag{10}$$

如算法 2 与图 2(b) 所示，整体流程涉及训练一系列模型 $M_{t},\dots,M_{T}$。为缓解束搜索推理阶段的计算负担，我们仅随机采样 $r_{\rm DPO}=1\%$ 的数据用于偏好对齐。对每个后续模型 $M_{t+1}$ 而言，它从上一个模型 $M_{t}$ 初始化，并使用由 $M_{t}$ 生成的偏好数据 $D_{t}^{\text{pairs}}$ 进行训练。

表1：我们所提出的 OneRec（绿色）与逐点方法（棕色）、列表方法（蓝色）及偏好对齐方法（黄色）的离线性能对比。最优结果以粗体标出，次优结果加下划线。带 $\uparrow$ 的指标表示越高越好，带 $\downarrow$ 的指标表示越低越好。

| 模型 | 观看时长类指标 | 交互类指标 |
|---|---|---|
| | swt$\uparrow$ | vtr$\uparrow$ | wtr$\uparrow$ | ltr$\uparrow$ |
| | mean | max | mean | max | mean | max | mean | max |
| **逐点判别式方法** |
| SASRec | 0.0375±0.002 | 0.0803±0.005 | 0.4313±0.013 | 0.5801±0.013 | 0.00294±0.001 | 0.00978±0.001 | 0.0314±0.002 | 0.0604±0.004 |
| BERT4Rec | 0.0336±0.002 | 0.0706±0.004 | 0.4192±0.014 | 0.5633±0.013 | 0.00281±0.001 | 0.00932±0.001 | 0.0316±0.002 | 0.0606±0.004 |
| FDSA | 0.0325±0.002 | 0.0683±0.005 | 0.4145±0.015 | 0.5588±0.014 | 0.00271±0.001 | 0.00921±0.001 | 0.0313±0.002 | 0.0604±0.003 |
| **逐点生成式方法** |
| TIGER-0.1B | 0.0879±0.007 | 0.1286±0.010 | 0.5826±0.016 | 0.6625±0.017 | 0.00277±0.001 | 0.00671±0.001 | 0.0316±0.004 | 0.0541±0.007 |
| TIGER-1B | 0.0873±0.006 | 0.1368±0.010 | 0.5827±0.015 | 0.6776±0.015 | 0.00292±0.001 | 0.00758±0.001 | 0.0323±0.004 | 0.0579±0.008 |
| **列表生成式方法** |
| OneRec-0.1B | 0.0973±0.010 | 0.1501±0.015 | 0.6001±0.021 | 0.6981±0.021 | 0.00326±0.001 | 0.00870±0.001 | 0.0349±0.009 | 0.0642±0.015 |
| OneRec-1B | 0.0991±0.008 | 0.1529±0.012 | 0.6039±0.020 | 0.7013±0.020 | 0.00349±0.001 | 0.00919±0.002 | 0.0360±0.005 | 0.0660±0.008 |
| **带偏好对齐的列表生成式方法** |
| OneRec-1B+DPO | 0.1014±0.010 | 0.1595±0.015 | 0.6127±0.017 | 0.7116±0.016 | 0.00339±0.001 | 0.00896±0.001 | 0.0351±0.004 | 0.0644±0.008 |
| OneRec-1B+IPO | 0.0979±0.003 | 0.1528±0.005 | 0.6000±0.007 | 0.7012±0.007 | 0.00335±0.001 | 0.00905±0.001 | 0.0350±0.003 | 0.0654±0.004 |
| OneRec-1B+cDPO | 0.0993±0.006 | 0.1547±0.008 | 0.6030±0.011 | 0.7030±0.009 | 0.00339±0.001 | 0.00901±0.001 | 0.0355±0.006 | 0.0652±0.009 |
| OneRec-1B+rDPO | 0.1005±0.006 | 0.1555±0.008 | 0.6071±0.014 | 0.7059±0.011 | 0.00339±0.001 | 0.00899±0.001 | 0.0357±0.004 | 0.0657±0.006 |
| OneRec-1B+CPO | 0.0993±0.008 | 0.1538±0.012 | 0.6045±0.021 | 0.7029±0.018 | 0.00343±0.001 | 0.00911±0.002 | 0.0357±0.008 | 0.0659±0.014 |
| OneRec-1B+simPO | 0.0995±0.008 | 0.1536±0.013 | 0.6047±0.016 | 0.7022±0.015 | 0.00349±0.001 | 0.00918±0.001 | 0.0360±0.005 | 0.0659±0.008 |
| OneRec-1B+S-DPO | 0.1021±0.008 | 0.1575±0.013 | 0.6096±0.016 | 0.7070±0.015 | 0.00345±0.001 | 0.00909±0.001 | 0.0361±0.004 | 0.0659±0.008 |
| **OneRec-1B+IPA** | **0.1025±0.009** | **0.1933±0.017** | **0.6141±0.020** | **0.7646±0.021** | **0.00354±0.001** | **0.00992±0.001** | **0.0397±0.004** | **0.1203±0.010** |

## 4. 系统部署

OneRec 已成功落地于真实的工业场景。在稳定性与性能之间取得平衡后，我们部署 OneRec-1B 用于在线服务。如图 3 所示，我们的部署架构由三个核心组件构成：1) 训练系统，2) 在线服务系统，3) DPO 样本服务器。该系统将收集到的交互日志作为训练数据，最初采用下一词元预测目标 $\mathcal{L}_{\rm NTP}$ 训练种子模型；收敛后，我们加入 DPO 损失 $\mathcal{L}_{\rm DPO}$ 进行偏好对齐，并利用 XLA 与 bfloat16 混合精度训练来优化计算效率和内存占用。训练完成的参数会同步至在线推理模块与 DPO 采样服务器，分别用于实时服务和基于偏好的数据选择。为提升推理性能，我们实施了两项关键优化：一是结合 float16 量化的键值（key-value）缓存解码机制，以降低 GPU 显存开销；二是采用束大小为 128 的束搜索配置，以平衡生成质量与延迟。此外，得益于 MoE 架构，推理期间仅激活 13% 的参数。

图3：OneRec 的在线部署框架。

## 5. 实验

本节中，我们首先在离线设置下将 OneRec 与逐点方法及若干 DPO 变体进行比较。然后，我们针对所提出的模块开展若干消融实验，以验证 OneRec 的有效性。最后，我们将 OneRec 部署上线，并在快手进行 A/B 测试，进一步验证其线上性能。

### 5.1. 实验设置

#### 5.1.1. 实现细节

我们的模型使用 Adam 优化器训练，初始学习率为 $2\times 10^{-4}$。我们使用 NVIDIA A800 GPU 来优化 OneRec。DPO 采样比例 $r_{\text{DPO}}$ 在整个训练过程中设为 1%，并通过束搜索为每个用户生成 $N=128$ 个不同的响应；语义标识符聚类过程对每个码本层采用 $K=8192$ 个簇，码本层数设为 $L=3$；混合专家架构包含 $N_{\text{MoE}}=24$ 个专家，每次前向传播通过 top-$k$ 选择激活 $K_{\text{MoE}}=2$ 个专家；对于会话建模，我们考虑 $m=5$ 个目标会话物品，并采用 $n=256$ 条历史行为作为上下文。

#### 5.1.2. 基线方法

我们采用以下具有代表性的推荐模型、DPO 及其变体作为额外的对比基线。基线方法包括：

- • SASRec（Kang and McAuley, 2018）采用单向 Transformer 架构捕捉用户-物品交互中的序列依赖，用于下一物品预测。

- • BERT4Rec（Sun et al., 2019）利用双向 Transformer 与掩码语言建模，通过序列重构学习上下文化的物品表示。

- • FDSA（Zhang et al., 2019）实现双自注意力通路，在异构推荐场景中联合建模物品级转移与特征级变换模式。

- • TIGER（Rajput et al., 2023）利用层次化语义标识符与生成式检索技术，通过自回归序列生成进行序列推荐。

- • DPO（Rafailov et al., 2024）通过隐式奖励建模，从人类反馈数据中导出闭式奖励函数，将偏好优化形式化。

- • IPO（Azar et al., 2024）提出一个理论上有根据的偏好优化框架，绕过标准 DPO 中固有的近似。

- • cDPO（Mitchell, [n. d.]）引入了一种结合标签翻转率参数 $\epsilon$ 的鲁棒性感知变体，以应对带噪声的偏好标注。

- • rDPO（Chowdhury et al., 2024）利用重要性采样开发无偏损失估计器，以降低偏好优化中的方差。

- • CPO（Xu et al., 2024b）通过对序列似然奖励与监督微调目标的联合训练，将对比学习与偏好优化统一起来。

- • simPO（Meng et al., 2024）通过采用序列级奖励边际并以归一化概率平均消除对参考模型的依赖，进行偏好优化。

- • S-DPO（Chen et al., 2024）通过困难负采样与多物品对比学习将 DPO 适配于推荐系统，以增强排序精度。

#### 5.1.3. 评估指标

我们采用若干关键指标来评估模型性能。每项指标在评估模型输出不同方面的作用各有侧重，我们在每次迭代中随机采样的测试案例集合上进行评估。为了估计每个特定用户-会话对的各种交互概率，我们采用预训练的奖励模型来评估推荐会话的价值。我们计算不同目标指标的平均奖励，包括会话观看时长（session watch time, swt）、观看概率（view probability, vtr）、关注概率（follow probability, wtr）与点赞概率（like probability, ltr）。在这些目标中，swt 与 vtr 属于观看时长类指标，而 wtr 与 ltr 属于交互类指标。

图4：关于 DPO 采样比例 $r_{\rm DPO}$ 的消融研究。结果表明，1% 的 DPO 训练比例即可带来显著收益，而进一步提高采样比例仅带来有限的改进。

### 5.2. 离线性能

表 1 给出了 OneRec 与各种基线的全面对比。对于观看时长类指标，我们主要关注会话观看时长（swt），以及交互类指标中的点赞概率（ltr）。我们的结果揭示了三个关键观察：

第一，所提出的会话级生成方法显著优于传统的基于点积的方法以及 TIGER 这类逐点生成方法。与 TIGER-1B 相比，OneRec-1B 的最大 swt 高出 1.78%，最大 ltr 高出 3.36%。这证明了会话级建模在保持推荐上下文连贯性方面的优势，而逐点方法难以在生成输出中兼顾连贯性与多样性。

第二，小比例的 DPO 训练即可带来可观收益。仅使用 1% 的 DPO 训练比例（$r_{\rm DPO}$），OneRec-1B+IPA 便在最大 swt 上超越基础版 OneRec-1B 达 4.04%，在最大 ltr 上超出 5.43%。这表明有限的 DPO 训练即可有效地将模型与期望的生成模式对齐。

第三，所提出的 IPA 策略优于各种现有 DPO 变体。如表 1 所示，IPA 相比其他 DPO 实现取得了更优的性能。值得注意的是，一些 DPO 基线甚至表现不如未对齐的 OneRec-1B 模型，这表明对自生成输出进行迭代挖掘以完成偏好挑选，比其他方法更为有效。

图5：语义ID 各层 softmax 输出概率分布的可视化。红星代表奖励值最高的物品的语义ID。

图6：OneRec 在模型缩放方面的可扩展性。结果表明，随着参数规模的增大，OneRec 持续受益于性能提升。

### 5.3. 消融研究

#### 5.3.1. DPO 采样比例消融

为了研究 DPO 训练中采样比例 $r_{\rm DPO}$ 的影响，我们在受控条件下将 DPO 采样比例从 1% 变化到 5%。如图 4 所示，消融结果表明，在多个评估目标上增大采样比例仅带来边际性能提升。值得注意的是，超过 1% 基线之后的性能增益并不显著，尽管计算开销有所增加。值得指出的是，DPO 样本服务器推理期间的 GPU 资源利用率与采样比例之间存在线性关系：5% 的采样比例所需的 GPU 资源是 1% 基线的 5 倍。这一扩展特性在计算效率与模型性能之间建立了明确的权衡。因此，在计算效率与性能之间取得最佳折中后，我们采用 1% 的 DPO 采样比例进行训练——该设置仅需更高采样比例所需计算资源的 20%，即可平均达到所观测最大性能的 95%。

#### 5.3.2. 模型缩放消融

我们评估了 OneRec 在模型规模增大时的表现。如图 6 所示，将 OneRec 从 0.05B 扩展到 1B 取得了一致的精度增益，展现出一致的缩放特性。具体而言，与 OneRec-0.05B 相比，OneRec-0.1B 取得了最高 14.45% 的显著精度增益；扩展到 0.2B、0.5B 和 1B 时，还可分别额外取得 5.09%、5.70% 和 5.69% 的精度增益。

### 5.4. OneRec 的预测动态

如图 5 所示，我们展示了不同层上 8192 个码字的预测概率分布，其中红星表示奖励值最高的物品的语义ID。与 OneRec 基线相比，OneRec+IPA 的预测分布表现出显著的置信度偏移，这表明我们提出的偏好对齐策略有效地促使基础模型产生偏好的生成模式。此外，我们观察到第一层的概率分布相比后续层表现出更大的发散性（熵 = 6.00），而后续层的分布则逐渐集中（第二层平均熵 = 3.71，第三层熵 = 0.048）。这种层次化的不确定性递减可归因于自回归解码机制：初始层的预测从前序解码步骤继承了更高的不确定性，而较后的层则受益于不断累积的上下文，从而约束了决策空间。

### 5.5. 在线 A/B 测试

为评估 OneRec 的在线性能，我们在快手主页面（main page）的视频推荐场景中进行了严格的在线 A/B 测试，用 1% 的主流量开展实验，比较 OneRec 与当前多阶段推荐系统的性能。我们使用总观看时长（Total Watch Time）来衡量用户观看视频的总时间，用平均观看时长（Average View Duration）计算用户在被推荐系统曝光的某次请求会话中每条视频的平均观看时间。在线评估显示，OneRec 在总观看时长上取得了 1.68% 的提升，在平均观看时长上取得了 6.56% 的提升，这表明 OneRec 取得了明显更好的推荐效果，并为平台带来了可观的收益增量。

表2：在线 A/B 测试设置下，OneRec 相比当前多阶段系统的绝对提升。

| 模型 | 总观看时长 | 平均观看时长 |
|---|---|---|
| OneRec-0.1B | +0.57% | +4.26% |
| OneRec-1B | +1.21% | +5.01% |
| OneRec-1B+IPA | +1.68% | +6.56% |

## 6. 结论

本文聚焦于介绍单阶段生成式推荐的一种工业解决方案。我们的方案确立了三项关键贡献：首先，我们通过应用 MoE 架构，以高计算效率有效地扩展了模型参数，为大规模工业推荐提供了可扩展的蓝图。其次，我们发现了以会话级生成方式建模目标物品上下文信息的必要性，证明上下文化的序列建模本质上比孤立的逐点方式能更好地捕捉用户偏好动态。此外，我们提出了迭代偏好对齐（IPA）策略，以提升 OneRec 在多样化用户偏好模式上的泛化能力。大量离线实验与在线 A/B 测试验证了 OneRec 的有效性与效率。另外，我们对在线结果的分析也揭示出，除用户观看时长之外，我们的模型在点赞等交互类指标上仍存在局限。在未来的研究中，我们旨在增强端到端生成式推荐在多目标建模方面的能力，以提供更好的用户体验。

## 参考文献（原样保留英文）

- Azar et al. (2024) Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. 2024. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics. PMLR, 4447–4455.

- Burges (2010) Christopher JC Burges. 2010. From ranknet to lambdarank to lambdamart: An overview. Learning 11, 23-581 (2010), 81.

- Chang et al. (2023) Jianxin Chang, Chenbin Zhang, Zhiyi Fu, Xiaoxue Zang, Lin Guan, Jing Lu, Yiqun Hui, Dewei Leng, Yanan Niu, Yang Song, et al. 2023. TWIN: TWo-stage interest network for lifelong user behavior modeling in CTR prediction at kuaishou. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3785–3794.

- Chen et al. (2024) Yuxin Chen, Junfei Tan, An Zhang, Zhengyi Yang, Leheng Sheng, Enzhi Zhang, Xiang Wang, and Tat-Seng Chua. 2024. On Softmax Direct Preference Optimization for Recommendation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems. https://openreview.net/forum?id=qp5VbGTaM0

- Chowdhury et al. (2024) Sayak Ray Chowdhury, Anush Kini, and Nagarajan Natarajan. 2024. Provably Robust DPO: Aligning Language Models with Noisy Feedback. In ICML 2024.

- Covington et al. (2016) Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems. 191–198.

- Dai et al. (2024) Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. 2024. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. arXiv preprint arXiv:2401.06066 (2024).

- De Cao et al. (2020) Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2020. Autoregressive entity retrieval. arXiv preprint arXiv:2010.00904 (2020).

- Du et al. (2022) Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. 2022. Glam: Efficient scaling of language models with mixture-of-experts. In International Conference on Machine Learning. PMLR, 5547–5569.

- Dubey et al. (2024) Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024).

- Fei et al. (2021) Hongliang Fei, Jingyuan Zhang, Xingxuan Zhou, Junhao Zhao, Xinyang Qi, and Ping Li. 2021. GemNN: gating-enhanced multi-task neural networks with feature interaction learning for CTR prediction. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval. 2166–2171.

- Feng et al. (2022) Chao Feng, Wuchao Li, Defu Lian, Zheng Liu, and Enhong Chen. 2022. Recommender forest for efficient retrieval. Advances in Neural Information Processing Systems 35 (2022), 38912–38924.

- Gallagher et al. (2019) Luke Gallagher, Ruey-Cheng Chen, Roi Blanco, and J Shane Culpepper. 2019. Joint optimization of cascade ranking models. In Proceedings of the twelfth ACM international conference on web search and data mining. 15–23.

- Ge et al. (2013) Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized product quantization. IEEE transactions on pattern analysis and machine intelligence 36, 4 (2013), 744–755.

- Guo et al. (2017) Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. arXiv preprint arXiv:1703.04247 (2017).

- Hidasi (2015) B Hidasi. 2015. Session-based Recommendations with Recurrent Neural Networks. arXiv preprint arXiv:1511.06939 (2015).

- Houle and Nett (2014) Michael E Houle and Michael Nett. 2014. Rank-based similarity search: Reducing the dimensional dependence. IEEE transactions on pattern analysis and machine intelligence 37, 1 (2014), 136–150.

- Hron et al. (2021) Jiri Hron, Karl Krauth, Michael Jordan, and Niki Kilbertus. 2021. On component interactions in two-stage recommender systems. Advances in neural information processing systems 34 (2021), 2744–2757.

- Huang et al. (2013) Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2333–2338.

- Huang et al. (2023) Xu Huang, Defu Lian, Jin Chen, Liu Zheng, Xing Xie, and Enhong Chen. 2023. Cooperative Retriever and Ranker in Deep Recommenders. In Proceedings of the ACM Web Conference 2023. 1150–1161.

- Jegou et al. (2010) Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence 33, 1 (2010), 117–128.

- Kang and McAuley (2018) Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM). IEEE, 197–206.

- Kuai et al. (2024) Zhirui Kuai, Zuxu Chen, Huimu Wang, Mingming Li, Dadong Miao, Wang Binbin, Xusong Chen, Li Kuang, Yuxing Han, Jiaxing Wang, et al. 2024. Breaking the Hourglass Phenomenon of Residual Quantization: Enhancing the Upper Bound of Generative Retrieval. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track. 677–685.

- Lee et al. (2022) Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. 2022. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11523–11532.

- Liu et al. (2024) Han Liu, Yinwei Wei, Xuemeng Song, Weili Guan, Yuan-Fang Li, and Liqiang Nie. 2024. MMGRec: Multimodal Generative Recommendation with Transformer Model. arXiv preprint arXiv:2404.16555 (2024).

- Liu et al. (2017) Shichen Liu, Fei Xiao, Wenwu Ou, and Luo Si. 2017. Cascade ranking for operational e-commerce search. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 1557–1565.

- Luo et al. (2024) Xinchen Luo, Jiangxia Cao, Tianyu Sun, Jinkai Yu, Rui Huang, Wei Yuan, Hezheng Lin, Yichen Zheng, Shiyao Wang, Qigen Hu, et al. 2024. QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou. arXiv preprint arXiv:2411.11739 (2024).

- Ma et al. (2021) Xu Ma, Pengjie Wang, Hui Zhao, Shaoguo Liu, Chuhan Zhao, Wei Lin, Kuang-Chih Lee, Jian Xu, and Bo Zheng. 2021. Towards a better tradeoff between effectiveness and efficiency in pre-ranking: A learnable feature selection based approach. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2036–2040.

- Meng et al. (2024) Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. SimPO: Simple Preference Optimization with a Reference-Free Reward. In Advances in Neural Information Processing Systems (NeurIPS).

- Mitchell ([n. d.]) Eric Mitchell. [n. d.]. A note on dpo with noisy preferences and relationship to ipo, 2023. URL https://ericmitchell. ai/cdpo. pdf ([n. d.]).

- Muja and Lowe (2014) Marius Muja and David G Lowe. 2014. Scalable nearest neighbor algorithms for high dimensional data. IEEE transactions on pattern analysis and machine intelligence 36, 11 (2014), 2227–2240.

- Ouyang et al. (2022) Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems 35 (2022), 27730–27744.

- Pi et al. (2020) Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang Zhu, and Kun Gai. 2020. Search-based user interest modeling with lifelong sequential behavior data for click-through rate prediction. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2685–2692.

- Qin et al. (2022) Jiarui Qin, Jiachen Zhu, Bo Chen, Zhirong Liu, Weiwen Liu, Ruiming Tang, Rui Zhang, Yong Yu, and Weinan Zhang. 2022. Rankflow: Joint optimization of multi-stage cascade ranking systems as flows. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 814–824.

- Rafailov et al. (2024) Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2024. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems 36 (2024).

- Rajput et al. (2023) Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. 2023. Recommender systems with generative retrieval. Advances in Neural Information Processing Systems 36 (2023), 10299–10315.

- Shi et al. (2023) Wentao Shi, Jiawei Chen, Fuli Feng, Jizhi Zhang, Junkang Wu, Chongming Gao, and Xiangnan He. 2023. On the theories behind hard negative sampling for recommendation. In Proceedings of the ACM Web Conference 2023. 812–822.

- Shrivastava and Li (2014) Anshumali Shrivastava and Ping Li. 2014. Asymmetric LSH (ALSH) for sublinear time maximum inner product search (MIPS). Advances in neural information processing systems 27 (2014).

- Stiennon et al. (2020) Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. 2020. Learning to summarize with human feedback. Advances in Neural Information Processing Systems 33 (2020), 3008–3021.

- Sun et al. (2019) Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management. 1441–1450.

- Tang et al. (2023) Yubao Tang, Ruqing Zhang, Jiafeng Guo, and Maarten de Rijke. 2023. Recent advances in generative information retrieval. In Proceedings of the Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region. 294–297.

- Tay et al. (2022) Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, et al. 2022. Transformer memory as a differentiable search index. Advances in Neural Information Processing Systems 35 (2022), 21831–21843.

- Wang et al. (2011) Lidan Wang, Jimmy Lin, and Donald Metzler. 2011. A cascade ranking model for efficient ranked retrieval. In Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval. 105–114.

- Wang et al. (2024a) Yunli Wang, Zhiqiang Wang, Jian Yang, Shiyang Wen, Dongying Kong, Han Li, and Kun Gai. 2024a. Adaptive Neural Ranking Framework: Toward Maximized Business Goal for Cascade Ranking Systems. In Proceedings of the ACM on Web Conference 2024. 3798–3809.

- Wang et al. (2024b) Ye Wang, Jiahao Xun, Minjie Hong, Jieming Zhu, Tao Jin, Wang Lin, Haoyuan Li, Linjun Li, Yan Xia, Zhou Zhao, et al. 2024b. EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3245–3254.

- Wang et al. (2020) Zhe Wang, Liqin Zhao, Biye Jiang, Guorui Zhou, Xiaoqiang Zhu, and Kun Gai. 2020. Cold: Towards the next generation of pre-ranking system. arXiv preprint arXiv:2007.16122 (2020).

- Xu et al. (2024b) Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim. 2024b. Contrastive preference optimization: Pushing the boundaries of llm performance in machine translation. arXiv preprint arXiv:2401.08417 (2024).

- Xu et al. (2024a) Shuyuan Xu, Wenyue Hua, and Yongfeng Zhang. 2024a. Openp5: An open-source platform for developing, training, and evaluating llm-based recommender systems. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 386–394.

- Zeghidour et al. (2021) Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. 2021. Soundstream: An end-to-end neural audio codec. IEEE/ACM Transactions on Audio, Speech, and Language Processing 30 (2021), 495–507.

- Zhang et al. (2019) Tingting Zhang, Pengpeng Zhao, Yanchi Liu, Victor S Sheng, Jiajie Xu, Deqing Wang, Guanfeng Liu, Xiaofang Zhou, et al. 2019. Feature-level deeper self-attention network for sequential recommendation.. In IJCAI. 4320–4326.

- Zheng et al. (2024) Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, and Ji-Rong Wen. 2024. Adapting large language models by integrating collaborative semantics for recommendation. In 2024 IEEE 40th International Conference on Data Engineering (ICDE). IEEE, 1435–1448.

- Zhou et al. (2019) Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep interest evolution network for click-through rate prediction. In Proceedings of the AAAI conference on artificial intelligence, Vol. 33. 5941–5948.

- Zhou et al. (2018) Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1059–1068.

- Zhu et al. (2018) Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. 2018. Learning tree-based deep model for recommender systems. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1079–1088.

- Zoph et al. (2022) Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. 2022. Designing effective sparse expert models. arXiv preprint arXiv:2202.08906 2, 3 (2022), 17.

---

*译文完。原文参考文献表共 55 条，以上按原样收录（未翻译）。*
