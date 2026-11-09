# 面向生成式信息检索中新文档融入的模型编辑（DOME）

**英文原题：** Model Editing for New Document Integration in Generative Information Retrieval

**作者与机构：**
- Zhen Zhang（Shandong University, Qingdao, China；zhen.zhang.sdu@gmail.com）
- Zihan Wang（University of Amsterdam, Amsterdam, The Netherlands；通讯作者；zhw.cypher@gmail.com）
- Xinyu Ma（Baidu Inc., Beijing, China）
- Shuaiqiang Wang（Baidu Inc., Beijing, China）
- Dawei Yin（Baidu Inc., Beijing, China）
- Xin Xin（Shandong University, Qingdao, China）
- Pengjie Ren（Shandong University, Qingdao, China）
- Maarten de Rijke（University of Amsterdam, Amsterdam, The Netherlands）
- Zhaochun Ren（Leiden University, Leiden, The Netherlands）

**arXiv 编号与译注：** arXiv:2603.02773v2 [cs.IR]，2026 年 5 月 11 日。发表于 Proceedings of the ACM Web Conference 2026（WWW '26），2026 年 4 月 13–17 日，阿联酋迪拜。DOI: 10.1145/3774904.3792178；ISBN: 979-8-4007-2307-0/2026/04；CCS: Information systems → Retrieval models and ranking；许可证：CC BY 4.0。代码地址：https://github.com/zhangzhen-research/DOME 。
译注：docID（document identifier，文档标识符）、GR（Generative Retrieval，生成式检索）、FFN（feed-forward network，前馈网络）等术语在译文中保留英文缩写；模型名（DSI、Ultron、ROME、MEMIT、AlphaEdit 等）、数据集名（NQ、MS MARCO 等）、指标名保留英文。

---

## 摘要

生成式检索（generative retrieval, GR）将信息检索（Information Retrieval, IR）任务重新表述为文档标识符（docID）的生成。尽管前景可观，现有 GR 模型对新增文档的泛化能力很差，常常无法生成正确的 docID。增量训练（incremental training）虽提供了一种直接的补救手段，但其计算开销高昂、资源消耗大，且容易发生灾难性遗忘（catastrophic forgetting），从而限制了 GR 的可扩展性与实用性。

本文中，我们识别出核心瓶颈在于解码器将隐藏状态映射到新增文档正确 docID 的能力。模型编辑（model editing）能够对参数进行针对性的修改以实现 docID 映射，代表了一种有前景的解决方案。然而，将模型编辑应用于当前的 GR 模型并非易事：由于检索结果中共享 docID 的高度重叠，不同查询间的编辑向量（edit vector）难以区分，这严重阻碍了编辑的效果。为解决这一问题，我们提出 DOME（docID-oriented model editing，面向 docID 的模型编辑），一种高效且有效地使 GR 模型适应未见文档的新方法。DOME 包含三个阶段：(1) 关键层识别；(2) 编辑向量优化；(3) 更新的构建与应用。其核心是采用一种**混合标签自适应训练（hybrid-label adaptive training）**策略来学习具有区分性的编辑向量：软标签（soft label）保留查询特有的语义以获得可区分的更新，硬标签（hard label）则强制实施精确的映射修改。在包括 NQ 与 MS MARCO 在内的广泛使用的基准上的实验表明，我们的方法显著提升了新文档上的检索性能，同时在原始语料库上保持了有效性。此外，DOME 仅需增量训练约 60% 的训练时间即可达成上述效果，大幅降低了计算成本，使高效、频繁的模型更新成为可能。^1

> ^1 我们的代码可在 https://github.com/zhangzhen-research/DOME 获取。

**关键词：** 模型编辑（Model editing）、生成式检索（Generative retrieval）

## 1 引言

生成式检索（GR）标志着信息检索的一场范式转变。GR 模型不再搜索预构建的索引，而是直接针对查询生成相关文档的文档标识符（docID）（Tay et al., 2022; Wang et al., 2022; Bevilacqua et al., 2022; Tang et al., 2023; Zhou et al., 2024）。由于支持端到端训练并展现出优异的检索性能，GR 在 IR 社区正获得越来越多的关注（Cao et al., 2021b; Zeng et al., 2024a; Zhang et al., 2024; Tang et al., 2024）。然而，一个关键缺陷阻碍了 GR 的实际部署：对新文档的泛化能力差，模型常常无法检索到初始训练阶段之后加入的条目（Mehta et al., 2023; Kim et al., 2024; Zhang et al., 2025a; Zhang et al., 2025b; Wu et al., 2025）。解决这一问题的一种普遍策略是增量训练：当有文档加入语料库时，GR 模型在新增内容上重训（Mehta et al., 2023; Guo et al., 2024）。这虽然使模型得以学习新条目的 docID，但计算代价高昂，需要大量数据与资源。此外，它容易发生灾难性遗忘（Mehta et al., 2023; Kim et al., 2024; Liu et al., 2025）——适应新文档会削弱模型在初始文档集上的性能。这些缺点显著限制了当前 GR 模型的可扩展性与实用性。

图1：基于层次化 k-means docID 的 GR 模型（Tay et al., 2022）在 NQ 数据集（Kwiatkowski et al., 2019）上对初始文档与新文档的行为分析。模型在初始文档集上训练，新文档的 docID 使用相同的 k-means 聚类流程分配。(a) 中间表示的 T-SNE（Maaten and Hinton, 2008）可视化，不同颜色表示不同的 docID 前缀（即 10、30、50），每个前缀对应一种特定的文档类型；方块表示初始文档，圆圈表示新文档。(b) 初始文档集与新文档集上的检索性能（Recall@1 与 Recall@100）。

在本工作中，我们首先进行了细致的分析以识别 GR 模型在新增文档上性能不佳的根本原因。我们的发现揭示，关键瓶颈在于解码器将隐藏状态映射到这些文档正确 docID 的能力。如图1(a) 所示，编码器能够在表示空间中将新增文档放置得与初始训练集中语义相关的文档相近。这表明模型已成功捕获新文档的语义内容。相比之下，图1(b) 显示新文档的检索准确率显著下降。这一差异表明，语义表示与特定 docID 之间的映射仅在初始训练中建立，无法泛化到新文档。因此，全量重训并非必要：关键在于为新文档更新这一映射，同时保持在初始文档集上的检索效果。

这一洞见自然地引出模型编辑这一有前景的解决方案，它能对 docID 映射进行针对性的参数修改，同时避免增量训练的高计算成本与灾难性遗忘（Mitchell et al., 2022; Dong et al., 2022）。然而，将模型编辑应用于 GR 受到一个根本性问题的破坏：不同查询的编辑向量缺乏可区分性。在典型的"主体–关系–客体"三元组（记作 $\langle s,r,o\rangle$）编辑中，目标客体往往是唯一的。相比之下，GR 以自回归方式将 docID 生成为离散词元序列。在每一步，模型以查询和已生成的词元（docID 前缀）为条件预测下一个词元（目标 docID）。GR 中的编辑请求采用 $\langle query, prefix\ docID, target\ docID\rangle$ 映射的形式。一个关键挑战来自有限的 docID 词表，这不可避免地导致许多不同的查询与 docID 前缀被映射到同一个目标 docID。因此，编辑向量对不同的 GR 上下文失去了可区分性。这使模型无法为每个查询学到精确映射，严重降低编辑准确率。

为应对上述挑战，我们提出 DOME（docID-oriented model editing），一种面向 GR 的专用编辑方法，旨在无需全量重训的情况下使模型高效适应新文档。DOME 分三个阶段运行：第一，通过平均补丁分析（average patching analysis）（Reusch and Belinkov, 2025）识别编码 docID 映射知识的解码器 FFN 模块，确保更新仅限于最相关的参数。第二，使用**混合标签自适应训练**策略优化新 docID 的编辑向量，该策略将模型输出分布中的软标签（保留查询特有的语义多样性）与强制精确生成正确 docID 词元的硬标签（真实标签）相结合。训练过程中，重心逐渐从软标签转向硬标签，在初期鼓励具有区分性的更新，并在末期确保精确的 docID 生成。最后，将优化后的编辑向量组装为受约束的更新，并应用于选定的 FFN 模块，使模型得以纳入新文档的映射，同时在原始语料库上保持检索性能。在两个标准基准 Natural Questions（NQ）与 MS-MARCO 上的大量实验表明，我们的方法显著提升了新文档的检索准确率，同时对灾难性遗忘表现出很强的韧性。此外，与传统的增量训练相比，DOME 显著缩短了适应时间，从而在真实检索场景中实现快得多的部署。

我们的贡献总结如下：

- 我们识别出 GR 适应新文档的核心瓶颈在于模型无法生成新分配的 docID，即便它能正确理解这些文档的内容。
- 我们提出混合标签自适应训练策略，以解决将模型编辑应用于 GR 时出现的编辑向量可区分性不足的问题。
- 我们开发了 DOME，一个面向 GR 的专用高效模型编辑框架，在新文档上取得强检索性能的同时，显著缓解灾难性遗忘并降低适应成本。
- 在 NQ 与 MS-MARCO 上的大量实验表明，DOME 始终优于增量训练，同时将适应时间降低超过 40%。

## 2 相关工作

### 2.1 生成式检索

生成式检索（GR）将文档搜索表述为序列生成任务：模型针对给定查询直接输出 _docID_，从而取代基于索引的查找。docID 可以是**文本式**标识符，如标题或伪查询（Zhou et al., 2022; Bevilacqua et al., 2022; Li et al., 2023），也可以是由离散词元组成的**数字式**编码（Sun et al., 2023; Yang et al., 2023; Zhou et al., 2023; Zhao et al., 2025; Mekonnen et al., 2025）。文本式 ID 灵活但字符串匹配代价高昂（Wang et al., 2024; Wang et al., 2023c）。数字式 ID 通常通过聚类或量化得到，例如 k-means（Tay et al., 2022）、PQ（Zhou et al., 2022）、RQ（Zeng et al., 2024b），或原子式单数字 ID（Kishore et al., 2023）。我们关注结构化数字 docID（如 PQ 或 RQ），它们相较原子式 ID 提供了语义分组能力与可扩展性。

GR 的一个关键挑战是对未见文档的泛化（Kim et al., 2024）。DSI++（Mehta et al., 2023）将 DSI 扩展到结构化数字 docID，并通过混合训练（mixed training）与平坦化损失（flattened loss）改善未见文档的泛化。incDSI（Kishore et al., 2023）通过受约束的参数更新实现实时插入，但仅支持原子式 ID 且需要扩展码本（codebook），限制了其对结构化 docID 的适用性。

### 2.2 模型编辑

模型编辑通过对参数进行小规模、有针对性的修改来更新内部知识，而无需全量重训（Mazzia et al., 2024）。元学习方法如 KE（Cao et al., 2021a）与 MEND（Mitchell et al., 2022）使用超网络（hypernetwork）生成基于梯度的编辑，实现快速更新但影响的参数众多。"先定位后编辑"（locate-then-edit）类方法则识别承载知识的组件并只更新选定的参数（Santurkar et al., 2021; Xu et al., 2022）。ROME（Meng et al., 2022）定位编码特定事实的 FFN 神经元并施加闭式解更新，MEMIT（Meng et al., 2023）支持批量编辑，AlphaEdit（Fang et al., 2025）进一步引入正交投影以减少终身编辑中的遗忘。

在本工作中，我们提出 DOME，一个利用模型编辑在初始训练之后融入新文档的 GR 适应框架。与增量重训（Mehta et al., 2023; Kim et al., 2024; Lyu et al., 2024; Lyu et al., 2025）不同，DOME 通过选择性参数编辑避免了昂贵的更新并缓解遗忘。与现有编辑方法（Meng et al., 2022; Meng et al., 2023; Fang et al., 2025）相比——这些方法因编辑向量不可区分而在 GR 上表现不佳——DOME 引入混合标签自适应训练策略，将软标签的语义多样性与硬标签的精确性相结合，并逐渐过渡到纯硬标签以实现精确的 docID 映射。这一面向 GR 的专门设计使新标识符得以高效融入，同时在原始文档与新增文档上均保持检索性能。

## 3 预备知识

### 3.1 生成式检索

生成式检索（GR）将文档检索表述为一个序列生成任务。给定查询 $q$，GR 模型直接生成最相关文档的文档标识符（docID）（Cao et al., 2021b）。docID 表示为离散词元序列 $\bm{y}=(y_{1},\dots,y_{T})$。这些词元通常通过结构化方案分配，例如层次化 k-means 聚类（Ahmed et al., 2020）。

解码器自回归地生成 docID 序列 $\bm{y}$，以查询与 docID 前缀 $y_{<t}$ 为条件逐个预测词元 $y_{t}$，如下所示：

$$p(\bm{y}\mid q)=\prod_{t=1}^{T}p(y_{t}\mid q,y_{<t}). \tag{1}$$

训练目标是最大化该条件似然，等价于对查询–文档对 $(q,d)$ 最小化负对数似然损失：

$$\mathcal{L}_{\text{GR}}=-\sum_{t=1}^{T}\log p(y_{t}\mid q,y_{<t};\theta), \tag{2}$$

其中 $\theta$ 表示 GR 模型的参数（Sutskever et al., 2014）。

**适应新文档。** 我们的目标是使一个最初在语料库 $\mathcal{C}_{\text{init}}$ 上训练的预训练 GR 模型，能够有效检索新增语料库 $\mathcal{C}_{\text{new}}$ 中的文档（Mehta et al., 2023）。模型参数使其能够为 $\mathcal{C}_{\text{init}}$ 中的文档正确生成 docID，但当查询 $q$ 的相关文档位于 $\mathcal{C}_{\text{new}}$ 中时，模型往往无法产生正确的 docID 序列。我们不依赖增量训练，而是将这一适应过程表述为一个模型编辑问题：在为 $\mathcal{C}_{\text{new}}$ 注入新 docID 映射的同时，保持模型在 $\mathcal{C}_{\text{init}}$ 上的性能。

### 3.2 模型编辑

模型编辑旨在无需全量重训的情况下高效更新语言模型中的特定知识，同时保留其原有知识与性能（Jiang et al., 2025; Khandelwal et al., 2024）。在典型设定中，存储的知识表示为"主体–关系–客体"三元组 $\langle s,r,o\rangle$（Cheng et al., 2024; Wang et al., 2023b; Wang et al., 2019; He et al., 2020; Wang et al., 2023a），一个编辑请求指定一种行为修改，使模型对给定的 $(s,r)$ 对输出更新后的客体 $o$（Zheng et al., 2023）。

在 Transformer 架构中，前馈网络（FFN）模块可以被解释为一个**键值记忆（key–value memory）**（Geva et al., 2021）：**键向量（key vector）** $k$ 编码"主体–关系"对 $(s,r)$，而**值向量（value vector）** $v$ 编码客体 $o$。具体而言，给定隐藏输入状态 $h$，输入投影 $W_{\mathrm{in}}$ 加上非线性激活 $\sigma$ 产生**键**向量：

$$k=\sigma(W_{\mathrm{in}}h)\in\mathbb{R}^{d_{0}}, \tag{3}$$

其中 $d_{0}$ 表示 FFN 中间层的维度。输出投影 $W_{\mathrm{out}}\in\mathbb{R}^{d_{1}\times d_{0}}$ 随后将 $k$ 映射为**值**向量：

$$m=W_{\mathrm{out}}k\in\mathbb{R}^{d_{1}}, \tag{4}$$

其中 $d_{1}$ 表示 FFN 输出层的维度。

模型编辑通过添加更新矩阵 $\Delta$ 来修改输出投影 $W_{\mathrm{out}}$。假设我们要执行 $u$ 次编辑以注入新知识。对每个编辑 $\langle s_{i},r_{i},o_{i}\rangle$，我们获得一个表示所需知识修改的新的键值对 $(k_{i},v_{i})$：每个 $k_{i}$ 通过将上下文 $(s_{i},r_{i})$ 送入 $W_{\mathrm{in}}$ 并经非线性激活 $\sigma$ 计算得到，而 $v_{i}$ 是期望的值，即模型应产生的 $o_{i}$ 对应的目标输出向量。将所有 $u$ 个键与值堆叠起来，得到 $K_{1}=[k_{1}\mid k_{2}\mid\dots\mid k_{u}]\in\mathbb{R}^{d_{0}\times u}$ 与 $V_{1}=[v_{1}\mid v_{2}\mid\dots\mid v_{u}]\in\mathbb{R}^{d_{1}\times u}$，其中 $K_{1}$ 表示键，$V_{1}$ 表示它们的期望值（Vaswani et al., 2017）。

为获得 $v_{i}$，我们首先计算模型的原始输出

$$m_{i}=W_{\mathrm{out}}k_{i}, \tag{5}$$

然后确定一个最小的编辑向量 $\delta_{i}$，将预测移向正确的客体 $o_{i}$：

$$\delta_{i}=\arg\min_{\delta}-\log p_{\theta,\,m_{i}+\delta}(o_{i}\mid s_{i},r_{i}), \tag{6}$$

其中 $p_{\theta,m_{i}+\delta}$ 表示将 FFN 输出替换为 $m_{i}+\delta$ 时模型的输出分布。更新后的值为 $v_{i}=m_{i}+\delta_{i}$，堆叠所有更新后的键与值即得完整的 $K_{1}$ 与 $V_{1}$。

为缓解遗忘（Dai et al., 2022），我们额外构建一组应当保持不变的保留键值对 $(K_{0},V_{0})$。其中 $K_{0}$ 的获取方式与 $K_{1}$ 相同但来自原有知识，而 $V_{0}=W_{\mathrm{out}}K_{0}$ 存储它们的原始输出。更新矩阵 $\Delta$ 的计算需对齐 $(K_{1},V_{1})$，同时最小程度地扰动 $(K_{0},V_{0})$：

$$\Delta=\arg\min_{\tilde{\Delta}}\;\|(W_{\mathrm{out}}+\tilde{\Delta})K_{1}-V_{1}\|_{F}^{2}+\|\tilde{\Delta}K_{0}\|_{F}^{2}, \tag{7}$$

其中 $\|\cdot\|_{F}$ 表示 Frobenius 范数（Böttcher and Wenzel, 2008）。该目标具有如下闭式解：

$$\Delta=(V_{1}-W_{\mathrm{out}}K_{1})\left(K_{1}K_{1}^{\top}+K_{0}K_{0}^{\top}\right)^{-1}K_{1}^{\top}. \tag{8}$$

实践中，$K_{0}$ 可从原始语料中大量辅助输入估计得到，以近似被保留知识的子空间。

**面向 GR 的编辑。** 在生成式检索中，编辑过程针对文档标识符的自回归生成（Tay et al., 2022）。每个编辑请求可表示为 $\langle q,y_{<t},y_{t}\rangle$，其中 $q$ 是输入查询，$y_{<t}$ 是截至第 $t{-}1$ 步已生成的 docID 前缀，$y_{t}$ 是期望的下一个 docID 词元。在该设定下，FFN 键 $k$ 由以 $(q,y_{<t})$ 为条件的解码器隐藏状态计算得到，相应的值 $v$ 编码目标 docID 词元 $y_{t}$（Vaswani et al., 2017）。通过将多个这样的键与值堆叠为矩阵 $K_{1}\in\mathbb{R}^{d_{0}\times u}$ 与 $V_{1}\in\mathbb{R}^{d_{1}\times u}$，该编辑框架可以直接调整模型内部的键值表示，使其对给定的"查询–前缀"上下文产生预期输出。

## 4 讨论

在展示我们提出的方法之前，我们先进行全面分析，以识别限制当前 GR 模型适应新文档的 docID 映射瓶颈（见 §4.1）。我们进一步指出，现有模型编辑方法无法完全解决这一问题，因为其有效性受到跨查询编辑向量不可区分的严重阻碍（见 §4.2）。

### 4.1 docID 映射瓶颈

近期对生成式检索（GR）的机制分析（Reusch and Belinkov, 2025; Lee et al., 2024）表明：编码器捕获输入的语义内容，而解码器生成相应的 docID。解码过程在概念上可分为三个阶段：(1) 初始化任务表示的启动（priming）阶段；(2) 交叉注意力从编码器输出中提取语义的桥接（bridging）阶段；(3) FFN 模块将语义特征转换为最终 docID 词元的交互（interaction）阶段（Reusch and Belinkov, 2025）。

表1：GR 模型（Tay et al., 2022）在 NQ（Kwiatkowski et al., 2019）上对初始文档与新增文档的 Recall@10 与准确率对比。

| 文档集 | Recall@10 | 第 i 个 docID 的准确率：DocID 1 | DocID 2 | DocID 3 |
| --- | --- | --- | --- | --- |
| 初始文档 | 0.862 | 0.774 | 0.772 | 0.757 |
| 新文档 | 0.173 | 0.423 | 0.385 | 0.420 |

如图1(a) 所示，尽管编码器为新文档产生了有意义的表示，解码器却无法生成它们正确的 docID。为分析这一失败，我们考察了 docID 级别的预测准确率。如表1 所示，解码器在单个 docID 词元上达到 42.3% 的准确率，但在完整 docID 序列上仅有 17.3%。这一差距表明，模型学到的是从语义到 docID 的近似映射而非精确映射：它能预测部分词元，却无法给出检索所需的精确序列。因此，GR 模型适应新文档的核心挑战在于其无法调整这一已学到的 docID 映射。故而，与其重训整个模型，更有效的做法是选择性地更新负责 docID 映射的解码器参数，从而克服该瓶颈并改善新增文档的检索。

图2：在 NQ（Kwiatkowski et al., 2019）上对 GR 方法（Tay et al., 2022）应用 GR 编辑请求时，编辑向量 $\delta$ 的逐对余弦相似度占比。黄色表示目标 docID 不同的请求对，蓝色表示目标 docID 相同的请求对。

图3：所提出的 DOME 框架总览。(a) 使用补丁（patching）技术诊断模型，定位负责 docID 映射的关键层。(b) 采用混合标签自适应训练策略为新映射计算多样化且有效的编辑向量（$\delta$）。为清晰起见，图中仅保留 FFN 层中关键的 $W_{\mathrm{out}}$ 组件。(c) 随后这些编辑向量被用于构建更新矩阵（$\Delta$），并应用于关键层参数（$W_{\text{out}}$）以注入新的 docID 映射。

### 4.2 GR 中不可区分的编辑向量

如 §4.1 所确立的，GR 模型因编码语义与离散 docID 词元之间的映射错位，而难以对新文档生成精确的 docID。模型编辑通过无需全量重训的针对性参数更新提供了一种有前景的解决方案，但其在 GR 中的有效性受限于跨查询的不可区分编辑向量。具体而言，如 §3.2 所述，每个 GR 编辑请求可表示为三元组 $\langle q,y_{<t},y_{t}\rangle$，其中 $(q,y_{<t})$ 是输入上下文，$y_{t}$ 是目标 docID 词元。由于 docID 词表有限，许多不同的上下文 $(q,y_{<t})$ 会映射到同一目标词元 $y_{t}$。当在独热（one-hot）标签目标（式 10）下优化时，这些请求共享相同的监督信号，导致即便查询在语义上截然不同，其编辑向量 $\delta_{i}$ 也几乎相同。这种可区分性的缺失使模型无法学到上下文相关的查询–docID 映射。

为研究这一点，我们为新文档构造 GR 编辑请求，从目标 FFN 层计算其编辑向量 $\delta$，并测量逐对余弦相似度（Xia et al., 2015）。如图2 所示，共享同一目标词元的请求，其编辑向量之间表现出很高的相似度。这证实了独热硬标签优化会产生几乎不可区分的更新，使模型难以学到精确的 docID 映射。

## 5 DOME：面向生成式检索的 docID 导向模型编辑

DOME 是一个面向 GR 的专用模型编辑框架，旨在将新的 docID 映射注入预训练 GR 模型，同时保留其原有的检索能力。如图3 所示，DOME 由三个阶段组成：(a) **关键层识别**：通过平均补丁（average patching）技术定位解码器中负责 docID 映射的关键 FFN 层；(b) **编辑向量优化**：使用混合标签自适应训练策略计算多样化且有效的编辑向量，从而缓解编辑向量不可区分的问题；(c) **更新的构建与应用**：将这些编辑向量组装为一个受约束的更新矩阵，并按顺序应用于目标层，在保留既有知识的同时注入新映射。

### 5.1 关键层识别

图3(a) 展示了我们工作流程的第一阶段，其目标是定位存储 docID 映射知识的关键层。我们采用**平均补丁（average patching）**策略来识别这些关键层（Reusch and Belinkov, 2025）。具体而言，我们首先在一个文档集上收集解码器中间层所有模块的输出表示；然后，对另一组文档，将每一层的原始输出替换为从训练数据计算得到的平均表示。通过测量每次替换后检索性能的下降幅度，我们确定哪五层在被补丁时影响最大。我们将从 $l-4$ 到 $l$ 的这五层视为 docID 映射的关键层，因为改变它们的输出会显著降低检索准确率。后续的模型编辑操作仅在这些被识别的层上进行，以确保适应的针对性与高效性。详细分析见附录 A.1。

### 5.2 编辑向量优化

图3(b) 展示了 DOME 的第二阶段：为在 §5.1 中识别出的解码器关键层计算用于注入新 docID 映射的编辑向量。我们从新文档构造一系列 GR 编辑请求。对每个新文档，我们生成以其为相关目标的伪查询（pseudo-queries），并将其与分配给该文档的 docID 序列 $(y_{1},\dots,y_{T})$ 配对（Nogueira et al., 2019）。按照 §3.2 定义的请求格式，每个请求表示为 $\langle query,\,docID\ prefix,\,target\ docID\rangle$。

DOME 的核心是混合标签自适应训练策略，旨在解决 §4.2 中识别出的 GR 编辑向量低可区分性问题。该策略通过将独热硬标签替换为混合标签——把模型的原始输出分布（软标签）与真实标签词元（硬标签）相混合（Müller et al., 2019）——直接应对 $\delta$ 的低可区分性问题：

$$p_{\mathrm{target}}^{(s)}(v\mid q,y_{<t})=(1-\lambda_{s})\,p_{\text{orig}}(v\mid q,y_{<t})+\lambda_{s}\,\mathbf{1}[v=y_{t}], \tag{9}$$

其中 $v$ 是 docID 词表中的词元，$p_{\text{orig}}$ 是模型的原始输出分布，$\mathbf{1}[v=y_{t}]$ 是正确词元的独热分布，$\lambda_{s}\in[0,1]$ 随训练进行逐渐从小初值增大到 $1.0$。软标签分量保留编辑向量间查询特有的多样性，硬标签分量则强制精确预测。

对每个词元级编辑请求，我们在最后一个关键层 $l$ 处计算编辑向量 $\delta_{i}$。然后使用混合标签，通过将式 (10) 中的优化目标修改如下，来优化编辑向量 $\delta_{i}$：

$$\delta_{i}=\arg\min_{\delta_{i}}\left(-\sum_{v\in\mathcal{V}}p_{\mathrm{target}}^{(s)}(v\mid q_{i},y_{<t,i})\log p_{\theta,m_{i}+\delta_{i}}(v\mid q_{i},y_{<t,i})\right), \tag{10}$$

其中 $p_{\theta,m_{i}+\delta_{i}}$ 表示将 FFN 输出替换为 $m_{i}+\delta_{i}$ 时模型的输出分布。由于 $p_{\text{orig}}$ 随查询而变，所得的 $p_{\mathrm{target}}^{(s)}$ 与最优 $\delta_{i}$ 也各不相同，从而在早期鼓励多样化的更新方向。随着 $\lambda_{s}$ 趋近 $1.0$，训练日益聚焦于硬标签，确保学到的映射能为新文档生成精确的 docID 词元。

### 5.3 更新的构建与应用

图3(c) 展示了 DOME 的最后阶段：将优化后的编辑向量整合进目标解码器层，同时避免灾难性遗忘。对每个被识别的 FFN 层 $l$，我们收集两组键值对：$K_{1}^{(l)}$ 包含来自新 GR 编辑请求的键激活，$V_{1}^{(l)}$ 存储相应的目标值——通过将优化后的编辑向量 $\delta$ 加到 FFN 的原始输出上得到。为保留既有知识，我们还从原始语料中的查询–docID 对收集 $K_{0}^{(l)}$ 与 $V_{0}^{(l)}=W_{\mathrm{out}}^{(l)}K_{0}^{(l)}$。

我们的目标是使最后一个被识别层 $l$ 的输出与 $V_{1}^{(l)}$ 对齐。遵循式 (8) 中的受约束闭式解，我们首先直接使用 $K_{0}^{(l)}$ 与 $K_{1}^{(l)}$ 为最后一个层计算更新矩阵 $\Delta$。

为进一步降低知识遗忘的风险，DOME 参照先前工作中的策略，将这一更新分散到多个前置层。具体而言，对最终目标层之前的五层，第 $j$ 层的更新矩阵按其与 $l_{\mathrm{f}}$ 距离的倒数成比例缩放：

$$\Delta^{(j)}=\frac{1}{\mathrm{dist}(j,l)}\left(V_{1}^{(l)}-W_{\mathrm{out}}^{(l)}K_{1}^{(l)}\right)\left(K_{0}^{(j)}{K_{0}^{(j)}}^{\top}+K_{1}^{(j)}{K_{1}^{(j)}}^{\top}\right)^{-1}{K_{1}^{(j)}}^{\top}, \tag{11}$$

其中 $\mathrm{dist}(j,l)=|\,l-j+1\,|$ 表示层的绝对距离（Meng et al., 2023）。这些更新从最早的目标层开始，按顺序依次应用于直至最终层 $l$ 的各层。在某一层应用 $\Delta_{j}$ 之后，我们重新运行前向传播以刷新后续层的键激活 $K_{1}^{(l)}$，确保每次更新都基于最新的中间表示。该阶段将新的 docID 映射精确注入解码器，同时保留已有的检索能力，完成 DOME 对新增语料的适应。

## 6 实验

我们旨在回答以下研究问题：
（RQ1）DOME 能否有效且高效地使生成式检索（GR）模型适应新增文档？
（RQ2）DOME 能否在原始文档集上保持检索性能，从而缓解灾难性遗忘？
（RQ3）DOME 的每个核心组件是否都对其整体性能有贡献？
（RQ4）随着新增文档数量的增加，DOME 能否保持稳定的性能？^2

> ^2 由于篇幅限制，RQ4 的详细分析见附录 A.2。

### 6.1 实验设置

**基线方法。** 我们将 DOME 与三类基线进行比较：
(1) **无训练。** 这类方法在添加新文档后保持所有模型参数冻结，仅依赖模型固有的泛化能力。包括稀疏检索器 BM25（Robertson et al., 2009）、稠密检索器 DPR（Karpukhin et al., 2020），以及生成式检索器 DSI（Tay et al., 2022）与 Ultron（Zhou et al., 2022），均不做参数更新。
(2) **增量训练。** 这类方法通过额外训练更新模型参数以纳入新文档知识。包括 From Scratch（在合并后的语料上从头训练 GR 模型）、New-Doc FT（Mehta et al., 2023，使用新文档的查询与伪查询微调初始 GR 模型）、DSI++（Mehta et al., 2023，在初始与新查询上共同微调，并促进更平坦的损失盆地以缓解灾难性遗忘）。
(3) **模型编辑。** 这类方法无需全量重训，直接向目标参数注入新知识。我们纳入 ROME（Meng et al., 2022）、MEMIT（Meng et al., 2023）与 AlphaEdit（Fang et al., 2025），它们最初为知识编辑设计，此处通过将编辑请求重新定义为 docID 映射并将更新限制在关键 FFN 模块，改造用于 GR。
为保证公平，所有增量训练与模型编辑基线都从同一个在原始语料上训练的 GR 模型出发，并适应同一组新增文档。

**数据集。** 我们在两个标准基准上开展实验：Natural Questions（NQ）（Kwiatkowski et al., 2019）与 MS-MARCO（Nguyen et al., 2016）。为模拟真实场景中新文档的加入，每个语料按 90% 用于训练基础 GR 模型、10% 作为新文档集进行划分。该设置使我们能够评估模型适应新文档的能力，同时保持在原始语料上的检索性能。

**评测指标。** 检索性能使用 Recall@K 与平均倒数排名（Mean Reciprocal Rank, MRR）评估。遵循（Mehta et al., 2023），我们进一步基于 R@10 计算遗忘分数（$F_{n}$），以量化在初始语料上的性能退化。效率以每文档更新时间（TPD）衡量，定义为总训练或编辑时间除以新增文档数量。

表2：NQ（Kwiatkowski et al., 2019）与 MS-MARCO（Nguyen et al., 2016）数据集上的检索性能。每个单元格报告**初始文档 / 新文档**的分数。粗体与下划线分别表示各指标下的最佳与次佳结果。"From Scratch" 仅作为理论上界报告，因其重训成本极高，不参与排名。* 表示相对每个数据集上表现最好的基线有显著提升（t 检验，$p<0.05$）。

| 方法 | NQ R@1 | NQ R@10 | NQ R@100 | NQ MRR@100 | MARCO R@10 | MARCO MRR@10 | TPD (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **无训练** | | | | | | | |
| BM25 | 0.374 / 0.365 | 0.761 / 0.770 | 0.906 / 0.860 | 0.476 / 0.471 | 0.691 / 0.641 | 0.486 / 0.471 | – |
| DPR | 0.651 / 0.647 | 0.874 / 0.871 | 0.926 / 0.921 | 0.706 / 0.710 | 0.767 / 0.754 | 0.526 / 0.515 | – |
| DSI | 0.655 / 0.071 | 0.866 / 0.193 | 0.901 / 0.250 | 0.705 / 0.105 | 0.541 / 0.056 | 0.392 / 0.032 | – |
| Ultron | 0.671 / 0.064 | 0.867 / 0.134 | 0.911 / 0.208 | 0.722 / 0.168 | 0.731 / 0.069 | 0.454 / 0.030 | – |
| **增量训练** | | | | | | | |
| From Scratch | 0.696 / 0.693 | 0.886 / 0.883 | 0.931 / 0.930 | 0.744 / 0.742 | 0.775 / 0.772 | 0.532 / 0.529 | – |
| New-Doc FT | 0.696 / 0.660 | 0.886 / 0.848 | 0.931 / 0.917 | 0.744 / 0.732 | 0.775 / 0.751 | 0.532 / 0.511 | 3.42 |
| DSI++ | 0.696 / 0.674 | 0.886 / 0.878 | 0.931 / 0.920 | 0.744 / 0.735 | 0.775 / 0.759 | 0.532 / 0.517 | 3.54 |
| **模型编辑** | | | | | | | |
| ROME | 0.696 / 0.201 | 0.886 / 0.276 | 0.931 / 0.356 | 0.744 / 0.326 | 0.775 / 0.351 | 0.532 / 0.251 | 20.23 |
| MEMIT | 0.696 / 0.609 | 0.886 / 0.687 | 0.931 / 0.756 | 0.744 / 0.665 | 0.775 / 0.673 | 0.532 / 0.495 | 12.62 |
| AlphaEdit | 0.696 / 0.616 | 0.886 / 0.677 | 0.931 / 0.745 | 0.744 / 0.664 | 0.775 / 0.661 | 0.532 / 0.490 | 12.12 |
| **DOME** | **0.696 / 0.686\*** | **0.886 / 0.880\*** | **0.931 / 0.927\*** | **0.744 / 0.740\*** | **0.775 / 0.764\*** | **0.532 / 0.524\*** | **2.14\*** |

**实现细节。** 实验在四块 NVIDIA A100 GPU 上进行。我们采用 T5-large^3 作为基础 GR 模型，联合优化"查询到 docID"、"文档到 docID"与"伪查询到 docID"生成任务。新文档的伪查询使用 docTTTTTquery 模型（Nogueira et al., 2019）生成。对 DOME 而言，基于我们的层重要性分析，编辑应用于解码器第 14 至 18 层的 FFN 模块。在混合标签自适应训练策略中，混合系数 $\lambda_{s}$ 在 50 个优化步内从 0.3 线性增长到 1.0。主实验中的 docID 分配采用残差量化（residual quantization, RQ），替代方案（包括基于 BM25 的标识符与乘积量化 PQ）在消融研究中考察。基线方面，稀疏检索器 BM25 使用 _bm25s_^4 库实现，稠密检索器 DPR 使用 _pyserini_^5 工具包实现。其他 GR 基线均从其官方代码库复现以保证公平比较。所有模型编辑方法都在同一个于原始语料上预训练的 GR 模型上操作，并将编辑限制在我们分析所识别的解码器层内。

> ^3 https://huggingface.co/google-t5/t5-large
> ^4 https://github.com/xhluca/bm25s
> ^5 https://github.com/castorini/pyserini

### 6.2 新文档上的结果（RQ1）

为回答 RQ1，我们评估 DOME 的整体性能与适应效率。

**性能。** 我们首先评估 DOME 在使 GR 模型适应新增文档方面的有效性，结果见表2。我们观察到：
(1) **不做适应时存在严重的泛化失败。** DSI 与 Ultron 等模型在新文档上性能急剧退化，在 NQ 上的 Recall@1 分别跌至 0.071 与 0.064。这些结果清楚地证实了 GR 模型的根本性泛化瓶颈：解码器无法生成初始训练中未见过的 docID。
(2) **DOME 展示了有效的适应能力。** 在 NQ 上，DOME 在新文档上取得 0.686 的 Recall@1，几乎匹敌全量重训（0.693），并超过最强的增量训练基线 DSI++（0.674）。这一结果表明 DOME 能在保持高检索准确率的同时，使 GR 模型有效适应新语料。
(3) **DOME 大幅超越现有模型编辑方法。** 与 MEMIT 和 AlphaEdit 相比，DOME 取得了显著更高的检索准确率。现有编辑方法在目标 docID 重叠时难以产生有区分性的编辑向量，而 DOME 的混合标签自适应训练实现了精确且稳定的 docID 映射。

**适应效率。** 我们进一步通过测量每文档更新时间（TPD）来考察 RQ1 的效率侧面。结果见表2，主要结论如下：
(1) **DOME 提供了极为快速且可扩展的适应。** DOME 的 TPD 为 2.14 秒，显著快于 DSI++（3.54s）等增量训练方法，证实了其更新 GR 模型的高效率。
(2) **DOME 相对编辑基线取得大幅加速。** 它比 MEMIT（12.62s）与 AlphaEdit（12.12s）快五倍以上，同时保持更优的适应性能。效率的提升源于我们精心设计的框架与训练策略，它们专门为使生成式检索模型有效适应新文档而打造。

### 6.3 初始文档上的结果（RQ2）

为回答 RQ2，我们通过测量适应之后在初始文档语料上的检索性能，评估 DOME 对既有知识的保留程度。结果见表3，主要发现如下：
(1) **DOME 有效保留先前知识。** 其在初始语料上的性能几乎不变，Recall@1 仅从 0.696 轻微下降至 0.692。遗忘分数（$F_{n}$）为 0.003，是所有方法中最好的，进一步证明了其强大的知识保留能力。
(2) **知识保留源自受约束的更新。** 如式 (8) 所定义，DOME 引入了基于初始语料键值对 $(K_{0},V_{0})$ 的约束。$K_{0}K_{0}^{\top}$ 项确保最终更新矩阵 $\Delta$ 对已学文档的输出影响最小。
(3) **其他模型编辑方法表现出严重遗忘。** ROME 崩溃至 Recall@1 仅 0.285，MEMIT 与 AlphaEdit 也都明显差于 DOME。这些编辑策略产生的编辑向量在不同查询间可区分性差，导致与既有知识相互干扰，最终降低模型整体性能。
(4) **DOME 通过混合标签自适应训练缓解干扰。** 通过生成查询特定的更新向量，DOME 避免了相互冲突的更新，取得了远超对比方法的知识保留效果。

表3：NQ 上仅针对初始文档报告的检索性能。粗体表示各指标最佳结果，* 表示显著提升（t 检验，$p<0.05$）。

| 方法 | R@1 | R@10 | R@100 | MRR@100 | $F_{n}$↓ |
| --- | --- | --- | --- | --- | --- |
| Base Model（基础模型） | 0.696 | 0.886 | 0.931 | 0.744 | – |
| New-Doc FT | 0.544 | 0.741 | 0.805 | 0.612 | 0.125 |
| DSI++ | 0.674 | 0.879 | 0.926 | 0.721 | 0.007 |
| ROME | 0.285 | 0.569 | 0.774 | 0.355 | 0.317 |
| MEMIT | 0.654 | 0.866 | 0.904 | 0.705 | 0.020 |
| AlphaEdit | 0.647 | 0.852 | 0.891 | 0.696 | 0.034 |
| **DOME** | **0.692\*** | **0.883\*** | **0.928\*** | **0.736\*** | **0.003\*** |

### 6.4 消融研究（RQ3）

为回答 RQ3，我们进行消融研究，分析 DOME 关键组件的贡献：伪查询数量、docID 分配方式、被编辑的解码器层范围，以及混合标签训练策略。NQ 新文档集上的结果见表4。

表4：NQ 上的消融研究。

| 变体 | R@1 | R@10 | R@100 | MRR@100 |
| --- | --- | --- | --- | --- |
| DOME（完整版） | 0.686 | 0.880 | 0.927 | 0.740 |
| **每个新文档的伪查询数量** | | | | |
| Pseudo = 1 | 0.506 | 0.760 | 0.875 | 0.592 |
| Pseudo = 4 | 0.615 | 0.832 | 0.898 | 0.678 |
| Pseudo = 7 | 0.652 | 0.866 | 0.918 | 0.698 |
| **docID 类型** | | | | |
| 基于 BM25 的 docID | 0.683 | 0.860 | 0.921 | 0.722 |
| 基于 PQ 的 docID | 0.675 | 0.877 | 0.933 | 0.719 |
| **被编辑的解码器层范围** | | | | |
| 第 11–15 层 | 0.659 | 0.857 | 0.904 | 0.704 |
| 第 18–22 层 | 0.536 | 0.717 | 0.836 | 0.624 |
| **软硬标签策略** | | | | |
| 去除软标签（w/o soft label） | 0.646 | 0.805 | 0.901 | 0.697 |
| 去除硬标签（w/o hard label） | 0.325 | 0.556 | 0.711 | 0.436 |

**伪查询数量的影响。** 每个文档的伪查询数量强烈影响适应效果。使用单个查询时性能较差（R@1 = 0.506），增加到四个与七个查询时性能逐步提升至 0.652。更多查询提供了更丰富的上下文线索，增强了从语义表示到 docID 的映射。

**docID 分配方式的影响。** DOME 在不同 docID 方案下保持稳健。将默认的基于 RQ 的标识符替换为基于 BM25 或 PQ 的 docID，性能相当，表明 DOME 的有效性不依赖特定的 docID 结构，可泛化到不同的 GR 系统。

**被编辑层范围的影响。** 性能随所选择的解码器编辑层而变化。编辑第 14–18 层取得最佳结果，而移到更早（11–15 层）或更晚（18–22 层）的层则分别导致中等程度（R@1 = 0.659）与急剧（R@1 = 0.536）的下降。这证实 docID 映射知识集中于解码器的中后部层，凸显了我们层定位策略的重要性。

图4：NQ 上 GR 编辑请求间编辑向量 $\delta$ 的逐对余弦相似度占比。黄色表示目标 docID 不同的请求对，蓝色表示目标 docID 相同的请求对。

**混合标签训练策略的影响。** 混合标签训练对 DOME 的成功至关重要，去除软或硬任一组件都会显著降低性能。去除软标签时，独热监督使同一 docID 的编辑向量坍缩到相似方向，大幅降低适应灵活性并损害检索性能。去除硬标签时，模型缺乏精确监督，导致严重退化。如图4 所示，混合策略降低了共享同一目标 docID 的编辑向量之间的过度相似性，产生更具区分性、查询特定的更新，从而增强适应能力与检索质量。

## 7 结论

在本工作中，我们识别出解码器无法学到精确 docID 映射，是生成式检索模型适应新文档的关键障碍。为此，我们提出了 DOME，一个面向 GR 的专用模型编辑框架，其混合标签自适应训练策略能对关键解码器层产生有区分性且精确的更新。在 NQ 与 MS-MARCO 上的实验表明，DOME 显著改善了新文档上的检索，缓解了灾难性遗忘，并相对增量训练将适应时间降低超过 40%，确立了其作为高效、可扩展解决方案的地位。

尽管前景可观，我们当前的方法仅更新解码器的 FFN 层。在未来工作中，我们将专注于进一步提升编辑过程的效率，并探索对更多模块（包括注意力层与编码器表示）的适应策略。我们预期纳入更多组件将提升模型的适应能力，并在原始与新文档集上增强检索性能。

## 致谢

本研究（部分）受到以下资助：国家自然科学基金（62272274、62202271、62472261、62372275、62522210、T2293773）；国家重点研发计划（项目号 2024YFC3307300 与 2022YFC3303004）；山东省重点研发计划（项目号 2024CXGC010108）；山东省自然科学基金（项目号 ZR2024QF203）；山东省技术创新引导计划（项目号 YDZX2024088）；荷兰研究理事会（NWO）（项目号 024.004.022、NWA.1389.20.183 与 KICH3.LTP.20.006）；以及欧盟（协议号 101070212（FINDHR）与 101201510（UNITE））。所有内容仅代表作者观点，不一定为其各自雇主和/或资助方所认同或支持。

## 参考文献（原样保留英文）

- Ahmed et al. (2020) Mohiuddin Ahmed, Raihan Seraj, and Syed Mohammed Shamsul Islam. 2020. The k-means algorithm: A comprehensive survey and performance evaluation. _Electronics_ 9, 8 (2020), 1295.
- Bevilacqua et al. (2022) Michele Bevilacqua, Giuseppe Ottaviano, Patrick Lewis, Scott Yih, Sebastian Riedel, and Fabio Petroni. 2022. Autoregressive Search Engines: Generating Substrings as Document Identifiers. In _NeurIPS_.
- Böttcher and Wenzel (2008) Albrecht Böttcher and David Wenzel. 2008. The Frobenius norm and the commutator. _Linear algebra and its applications_ 429, 8-9 (2008), 1864–1885.
- Cao et al. (2021a) Nicola De Cao, Wilker Aziz, and Ivan Titov. 2021a. Editing Factual Knowledge in Language Models. In _EMNLP_. ACL, 6491–6506.
- Cao et al. (2021b) Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2021b. Autoregressive Entity Retrieval. In _ICLR_. OpenReview.net.
- Cheng et al. (2024) Siyuan Cheng, Ningyu Zhang, Bozhong Tian, Xi Chen, Qingbin Liu, and Huajun Chen. 2024. Editing language model-based knowledge graph embeddings. In _AAAI_, Vol. 38. 17835–17843.
- Dai et al. (2022) Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. 2022. Knowledge Neurons in Pretrained Transformers. In _ACL_. ACL, 8493–8502.
- Dong et al. (2022) Qingxiu Dong, Damai Dai, Yifan Song, Jingjing Xu, Zhifang Sui, and Lei Li. 2022. Calibrating Factual Knowledge in Pretrained Language Models. In _EMNLP_. ACL, 5937–5947.
- Fang et al. (2025) Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Jie Shi, Xiang Wang, Xiangnan He, and Tat-Seng Chua. 2025. AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models. In _ICLR_. OpenReview.net.
- Geva et al. (2021) Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. 2021. Transformer Feed-Forward Layers Are Key-Value Memories. In _EMNLP_. ACL, 5484–5495.
- Guo et al. (2024) Jiafeng Guo, Changjiang Zhou, Ruqing Zhang, Jiangui Chen, Maarten de Rijke, Yixing Fan, and Xueqi Cheng. 2024. Corpusbrain++: A continual generative pre-training framework for knowledge-intensive language tasks. _ACM Transactions on Information Systems_ (2024).
- He et al. (2020) Yongquan He, Zihan Wang, Peng Zhang, Zhaopeng Tu, and Zhaochun Ren. 2020. VN Network: Embedding Newly Emerging Entities with Virtual Neighbors. In _CIKM_. 505–514.
- Jiang et al. (2025) Houcheng Jiang, Junfeng Fang, Ningyu Zhang, Guojun Ma, Mingyang Wan, Xiang Wang, Xiangnan He, and Tat-seng Chua. 2025. Anyedit: Edit any knowledge encoded in language models. _arXiv preprint arXiv:2502.05628_ (2025).
- Karpukhin et al. (2020) Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In _EMNLP_. ACL, 6769–6781.
- Khandelwal et al. (2024) Aditi Khandelwal, Harman Singh, Hengrui Gu, Tianlong Chen, and Kaixiong Zhou. 2024. Cross-Lingual Multi-Hop Knowledge Editing. In _EMNLP_. ACL, 11995–12015.
- Kim et al. (2024) Chaeeun Kim, Soyoung Yoon, Hyunji Lee, Joel Jang, Sohee Yang, and Minjoon Seo. 2024. Exploring the Practicality of Generative Retrieval on Dynamic Corpora. In _EMNLP_. ACL, 13616–13633.
- Kishore et al. (2023) Varsha Kishore, Chao Wan, Justin Lovelace, Yoav Artzi, and Kilian Q. Weinberger. 2023. IncDSI: Incrementally Updatable Document Retrieval. In _ICML_, Vol. 202. PMLR, 17122–17134.
- Kwiatkowski et al. (2019) Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. 2019. Natural questions: a benchmark for question answering research. _Transactions of the Association for Computational Linguistics_ 7 (2019), 452–466.
- Lee et al. (2024) Sangam Lee, Ryang Heo, SeongKu Kang, Susik Yoon, Jinyoung Yeo, and Dongha Lee. 2024. Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths. _arXiv preprint arXiv:2411.05572_ (2024).
- Li et al. (2023) Yongqi Li, Nan Yang, Liang Wang, Furu Wei, and Wenjie Li. 2023. Multiview Identifiers Enhanced Generative Retrieval. In _ACL_. ACL, 6636–6648.
- Liu et al. (2025) Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Changjiang Zhou, Maarten de Rijke, Yixing Fan, and Xueqi Cheng. 2025. On the Robustness of Generative Information Retrieval Models: An Out-of-Distribution Perspective. In _ECIR_, Vol. 15573. Springer, 407–423.
- Lyu et al. (2024) Yougang Lyu, Lingyong Yan, Shuaiqiang Wang, Haibo Shi, Dawei Yin, Pengjie Ren, Zhumin Chen, Maarten de Rijke, and Zhaochun Ren. 2024. KnowTuning: Knowledge-aware Fine-tuning for Large Language Models. In _EMNLP_. 14535–14556.
- Lyu et al. (2025) Yougang Lyu, Lingyong Yan, Zihan Wang, Dawei Yin, Pengjie Ren, Maarten de Rijke, and Zhaochun Ren. 2025. MACPO: Weak-to-Strong Alignment via Multi-Agent Contrastive Preference Optimization. In _ICLR_.
- Maaten and Hinton (2008) Laurens van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE. _Journal of machine learning research_ 9, Nov (2008), 2579–2605.
- Mazzia et al. (2024) Vittorio Mazzia, Alessandro Pedrani, Andrea Caciolai, Kay Rottmann, and Davide Bernardi. 2024. A survey on knowledge editing of neural networks. _IEEE Transactions on Neural Networks and Learning Systems_ (2024).
- Mehta et al. (2023) Sanket Vaibhav Mehta, Jai Gupta, Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Jinfeng Rao, Marc Najork, Emma Strubell, and Donald Metzler. 2023. DSI++: Updating Transformer Memory with New Documents. In _EMNLP_. ACL, 8198–8213.
- Mekonnen et al. (2025) Kidist Amde Mekonnen, Yubao Tang, and Maarten de Rijke. 2025. Lightweight and Direct Document Relevance Optimization for Generative Information Retrieval. In _SIGIR_. 1327–1338.
- Meng et al. (2022) Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and Editing Factual Associations in GPT. In _NeurIPS_.
- Meng et al. (2023) Kevin Meng, Arnab Sen Sharma, Alex J. Andonian, Yonatan Belinkov, and David Bau. 2023. Mass-Editing Memory in a Transformer. In _ICLR_. OpenReview.net.
- Mitchell et al. (2022) Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D Manning, and Chelsea Finn. 2022. Memory-based model editing at scale. In _ICML_. PMLR, 15817–15831.
- Müller et al. (2019) Rafael Müller, Simon Kornblith, and Geoffrey E. Hinton. 2019. When does label smoothing help?. In _NeurIPS_. 4696–4705.
- Nguyen et al. (2016) Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. In _COCO@NeurIPS_, Vol. 1773. CEUR-WS.org.
- Nogueira et al. (2019) Rodrigo Nogueira, Jimmy Lin, and AI Epistemic. 2019. From doc2query to docTTTTTquery. _Online preprint_ 6, 2 (2019).
- Reusch and Belinkov (2025) Anja Reusch and Yonkat Belinkov. [sic] 2025. Reverse-Engineering the Retrieval Process in GenIR Models. In _SIGIR_. ACM, 668–677.
- Robertson et al. (2009) Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: BM25 and beyond. _Foundations and Trends in Information Retrieval_ 3, 4 (2009), 333–389.
- Santurkar et al. (2021) Shibani Santurkar, Dimitris Tsipras, Mahalaxmi Elango, David Bau, Antonio Torralba, and Aleksander Madry. 2021. Editing a classifier by rewriting its prediction rules. In _NeurIPS_. 23359–23373.
- Sun et al. (2023) Weiwei Sun, Lingyong Yan, Zheng Chen, Shuaiqiang Wang, Haichao Zhu, Pengjie Ren, Zhumin Chen, Dawei Yin, Maarten de Rijke, and Zhaochun Ren. 2023. Learning to Tokenize for Generative Retrieval. In _NeurIPS_.
- Sutskever et al. (2014) Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. Sequence to Sequence Learning with Neural Networks. In _NeurIPS_. 3104–3112.
- Tang et al. (2023) Yubao Tang, Ruqing Zhang, Jiafeng Guo, Jiangui Chen, Zuowei Zhu, Shuaiqiang Wang, Dawei Yin, and Xueqi Cheng. 2023. Semantic-enhanced differentiable search index inspired by learning strategies. In _KDD_. 4904–4913.
- Tang et al. (2024) Yubao Tang, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, and Xueqi Cheng. 2024. Generative retrieval meets multi-graded relevance. _NeurIPS_ 37 (2024), 72790–72817.
- Tay et al. (2022) Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Prakash Gupta, Tal Schuster, William W. Cohen, and Donald Metzler. 2022. Transformer Memory as a Differentiable Search Index. In _NeurIPS_.
- Vaswani et al. (2017) Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In _NeurIPS_. 5998–6008.
- Wang et al. (2022) Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, Xing Xie, Hao Sun, Weiwei Deng, Qi Zhang, and Mao Yang. 2022. A Neural Corpus Indexer for Document Retrieval. In _NeurIPS_.
- Wang et al. (2024) Ye Wang, Xinrun Xu, Rui Xie, Wenxin Hu, and Wei Ye. 2024. Generative retrieval with large language models. _arXiv preprint arXiv:2402.17010_ (2024).
- Wang et al. (2019) Zihan Wang, Zhaochun Ren, Chunyu He, Peng Zhang, and Yue Hu. 2019. Robust Embedding with Multi-Level Structures for Link Prediction. In _IJCAI_. 5240–5246.
- Wang et al. (2023b) Zihan Wang, Kai Zhao, Yongquan He, Zhumin Chen, Pengjie Ren, Maarten de Rijke, and Zhaochun Ren. 2023b. Iteratively Learning Representations for Unseen Entities with Inter-Rule Correlations. In _CIKM_. 2534–2543.
- Wang et al. (2023a) Zihan Wang, Ziqi Zhao, Zhumin Chen, Pengjie Ren, Maarten de Rijke, and Zhaochun Ren. 2023a. Generalizing Few-Shot Named Entity Recognizers to Unseen Domains with Type-Related Features. In _EMNLP_. 2228–2240.
- Wang et al. (2023c) Zihan Wang, Yujia Zhou, Yiteng Tu, and Zhicheng Dou. 2023c. NOVO: learnable and interpretable document identifiers for model-based IR. In _CIKM_. 2656–2665.
- Wu et al. (2025) Shiguang Wu, Zhaochun Ren, Xin Xin, Jiyuan Yang, Mengqi Zhang, Zhumin Chen, Maarten de Rijke, and Pengjie Ren. 2025. Constrained Auto-Regressive Decoding Constrains Generative Retrieval. In _SIGIR_. 2429–2440.
- Xia et al. (2015) Peipei Xia, Li Zhang, and Fanzhang Li. 2015. Learning similarity with cosine similarity ensemble. _Information sciences_ 307 (2015), 39–52.
- Xu et al. (2022) Yang Xu, Yutai Hou, Wanxiang Che, and Min Zhang. 2022. Language anisotropic cross-lingual model editing. _arXiv preprint arXiv:2205.12677_ (2022).
- Yang et al. (2023) Tianchi Yang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, and Qi Zhang. 2023. Auto Search Indexer for End-to-End Document Retrieval. In _EMNLP_. ACL, 6955–6970.
- Zeng et al. (2024b) Hansi Zeng, Chen Luo, Bowen Jin, Sheikh Muhammad Sarwar, Tianxin Wei, and Hamed Zamani. 2024b. Scalable and Effective Generative Information Retrieval. In _WWW_. ACM, 1441–1452.
- Zeng et al. (2024a) Hansi Zeng, Chen Luo, and Hamed Zamani. 2024a. Planning ahead in generative retrieval: Guiding autoregressive generation through simultaneous decoding. In _SIGIR_. ACM, 469–480.
- Zhang et al. (2024) Peitian Zhang, Zheng Liu, Yujia Zhou, Zhicheng Dou, Fangchao Liu, and Zhao Cao. 2024. Generative retrieval via term set generation. In _Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval_. 458–468.
- Zhang et al. (2025b) Yingchen Zhang, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, and Xueqi Cheng. 2025b. Does Generative Retrieval Overcome the Limitations of Dense Retrieval? _arXiv preprint arXiv:2509.22116_ (2025).
- Zhang et al. (2025a) Zhen Zhang, Xinyu Ma, Weiwei Sun, Pengjie Ren, Zhumin Chen, Shuaiqiang Wang, Dawei Yin, Maarten de Rijke, and Zhaochun Ren. 2025a. Replication and Exploration of Generative Retrieval over Dynamic Corpora. In _SIGIR_. ACM, 3325–3334.
- Zhao et al. (2025) Xinpeng Zhao, Zhaochun Ren, Yukun Zhao, Zhenyang Li, Mengqi Zhang, Jun Feng, Ran Chen, Ying Zhou, Zhumin Chen, Shuaiqiang Wang, et al. 2025. DiffuGR: Generative Document Retrieval with Diffusion Language Models. _arXiv preprint arXiv:2511.08150_ (2025).
- Zheng et al. (2023) Ce Zheng, Lei Li, Qingxiu Dong, Yuxuan Fan, Zhiyong Wu, Jingjing Xu, and Baobao Chang. 2023. Can We Edit Factual Knowledge by In-Context Learning?. In _EMNLP_. ACL, 4862–4876.
- Zhou et al. (2024) Yujia Zhou, Jing Yao, Zhicheng Dou, Yiteng Tu, Ledell Wu, Tat-Seng Chua, and Ji-Rong Wen. 2024. ROGER: Ranking-oriented generative retrieval. _TOIS_ 42, 6 (2024), 1–25.
- Zhou et al. (2022) Yujia Zhou, Jing Yao, Zhicheng Dou, Ledell Wu, Peitian Zhang, and Ji-Rong Wen. 2022. Ultron: An ultimate retriever on corpus with a model-based indexer. _arXiv preprint arXiv:2208.09257_ (2022).
- Zhou et al. (2023) Yu-Jia Zhou, Jing Yao, Zhi-Cheng Dou, Ledell Wu, and Ji-Rong Wen. 2023. Dynamicretriever: A pre-trained model-based ir system without an explicit index. _Machine Intelligence Research_ 20, 2 (2023), 276–288.

（以上参考文献原样保留英文，未作翻译。其中 Reusch and Belinkov (2025) 作者名 "Yonkat" 为 arXiv HTML 原文所示，正确拼写应为 Yonatan。）

## 附录 A

### A.1 用于定位关键层的补丁分析

图5：在 NQ 上预测位置 0–3 处 docID 词元时，解码器各层平均补丁（average patching）的准确率。

为研究哪些解码器层存储了 docID 映射的关键知识，我们分析了在生成过程中预测每个 docID 词元时，替换不同模块（FFN、自注意力、交叉注意力）中间输出所造成的影响。我们在四种独立设置下评估，目标词元分别位于 docID 序列中的位置 $t=0$、$t=1$、$t=2$ 与 $t=3$。图5 的横轴表示解码器层索引，纵轴表示在该位置正确预测目标 docID 的概率。

**补丁技术。** 遵循平均补丁方法（Reusch and Belinkov, 2025），对特定的解码器层 $l$ 与模块类型 $M$（FFN / Self-Attn / Cross-Attn），我们首先在训练集上计算平均输出表示 $\bar{h}_{l,M}$。给定一个输入（查询，docID 前缀）上下文 $x$，我们将层 $l$ 中模块 $M$ 的原始输出 $h_{l,M}(x)$ 替换为该平均表示：

$$h^{\prime}_{l,M}(x)=\bar{h}_{l,M}. \tag{12}$$

被补丁后的表示经剩余各层传播，我们测量概率

$$P_{\text{correct}}(t,l,M)=p_{\theta}\left(y_{t}^{\ast}\mid q,y_{<t};h_{l,M}\leftarrow\bar{h}_{l,M}\right), \tag{13}$$

其中 $y_{t}^{\ast}$ 表示位置 $t$ 处的真实 docID 词元。依次对每一层的每个模块执行该操作，我们得到图5 所示的性能曲线。

**观察结果。** 在全部四个目标位置（$t=0,1,2,3$）上，对解码器中段与末段的 FFN 层做补丁会导致 $P_{\text{correct}}$ 最为显著的下降，表明这些层存储了从语义表示到 docID 词元的关键映射知识。对自注意力模块做补丁在 $t=0$ 时几乎没有影响，而交叉注意力补丁会在中间层造成中等程度的下降。对 $t=1,2,3$，自注意力与交叉注意力的补丁都会在解码器早期层引起明显下降。这些一致的趋势证实了 §5.1 的主要发现：中段与末段的 FFN 模块对 docID 映射最为关键，应当作为模型编辑的主要目标。

### A.2 对不同数量文档的泛化（RQ4）

图6：在 NQ 上随新增文档数量变化时，初始文档（左）与新增文档（右）上的性能表现。

为回答 RQ4，我们将新增文档的数量从 100 变化到 1,000，并评估在初始集合与新集合上的检索性能，以分析 DOME 的泛化能力。图6 所示的结果表明，DOME 在不同规模的文档更新下均保持稳健的性能。我们得到以下结论：
(1) **DOME 有效保留既有知识。** 如图6 左图所示，初始集合上的检索性能高度稳定，即便融入 1,000 篇新文档也几乎没有退化。这表明受约束的更新机制无论更新规模如何，都能有效防止灾难性遗忘。
(2) **DOME 对新增文档表现出稳健的泛化。** 如图6 右图所示，新文档上的性能保持强劲，仅随编辑任务变得更复杂而出现轻微且在预期之内的下降。这些结果表明，DOME 能够在不同数量的新文档间有效泛化，在融入更新的同时确保检索性能保持强劲。

---

*译文完。原文：arXiv:2603.02773v2 [cs.IR]，WWW '26。*
