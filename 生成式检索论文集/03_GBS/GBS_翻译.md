# 面向图书搜索的生成式检索（GBS）

**英文原题：** Generative Retrieval for Book Search

**作者与机构（保留原文）：**
Yubao Tang（CAS Key Lab of Network Data Science and Technology, ICT, CAS, University of Chinese Academy of Sciences; University of Amsterdam, Amsterdam, The Netherlands）、Ruqing Zhang（CAS Key Lab of Network Data Science and Technology, ICT, CAS, University of Chinese Academy of Sciences, Beijing, China）、Jiafeng Guo（通讯作者，同 CAS ICT）、Maarten de Rijke（University of Amsterdam, Amsterdam, The Netherlands）、Shihao Liu（Baidu Inc., Beijing, China）、Shuaiqiang Wang（Baidu Inc., Beijing, China）、Dawei Yin（Baidu Inc., Beijing, China）、Xueqi Cheng（CAS Key Lab of Network Data Science and Technology, ICT, CAS, Beijing, China）

**arXiv 编号与译注：** arXiv:2501.11034v1 [cs.IR]，2025 年 1 月 19 日提交。正式发表于 KDD '25（Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1, 2025 年 8 月 3–7 日，加拿大多伦多），DOI: 10.1145/3690624.3709435，全文 12 页。译注：(1) 部分资料称本文为"百度论文（SIGIR 2025）"，但按 arXiv 与 ACM 正式出版信息，本文发表 venue 为 KDD 2025，作者团队为中科院计算所、阿姆斯特丹大学与百度公司合作；(2) 原文引言末段将本方法误写作 "GBR"，应为 "GBS"，译文中已按原文保留并于此说明；(3) 文内引用标记（如 [36]）对应文末参考文献，原样保留。

---

## 摘要

在图书搜索中，系统应当针对查询返回相关的图书信息。图书包含复杂的多面信息，如元数据、大纲和正文，其中大纲提供了章节与小节之间的层级信息。生成式检索是一种新兴检索范式，它将语料库信息整合进单一模型，直接生成与给定查询相关的文档标识符。那么，如何将生成式检索应用于图书搜索？由于图书搜索的独特特性，直接将生成式检索应用于图书搜索是一项挑战：(i) 模型需要保留图书复杂的多面信息，这增加了对标注数据的需求；(ii) 将图书信息切分并当作相互独立的片段集合来学习，可能导致层级信息丢失。

我们提出了一种有效的面向图书搜索的生成式检索框架，包含两大核心组件：(i) 数据增强；(ii) 大纲导向的书籍编码。在数据增强方面，GBS 构建多个查询-图书对用于训练；它基于大纲与多种形式的图书内容构建多个图书标识符，并通过多样化的伪查询模拟真实图书检索场景。这包括促进覆盖的图书标识符增强（使模型学会有效索引），以及增强多样性的查询增强（使模型学会有效检索）。大纲导向的书籍编码通过双层位置编码和保持式注意力机制改善长度外推，以在长序列上维持上下文。在百度私有数据集上的实验表明，GBS 优于强基线，在 MRR@20 上较最先进的 RIPOR 方法提升 9.8%。在公开数据集上的实验证实了 GBS 的鲁棒性与泛化能力，凸显其提升图书检索的潜力。

**CCS 概念：** 信息系统 → 检索模型与排序

**关键词：** 图书检索、生成式检索、生成式模型

**ACM 引用格式：**
Yubao Tang, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Shihao Liu, Shuaiqiang Wang, Dawei Yin, and Xueqi Cheng. 2025. Generative Retrieval for Book Search. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (KDD '25), August 3–7, 2025, Toronto, ON, Canada. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3690624.3709435

（脚注：Ruqing Zhang 的工作在作者访问阿姆斯特丹大学期间完成；Jiafeng Guo 为通讯作者。本文采用知识共享署名 4.0 国际许可协议授权。）

## 1 引言

搜索引擎已成为我们日常生活中获取信息的基础工具。作为通用搜索引擎提供的一项服务，图书搜索 [37, 41, 42] 常常为各类下游任务提供关键资源，包括问答 [14, 22] 和实体检索 [5, 9]，因为图书往往包含高质量、权威且全面的信息。图书搜索需求的持续增长，凸显了有效满足用户需求的重要性 [11]。

图书搜索在两个方面与通用网页搜索存在显著差异 [16]：(i) 图书比网页包含更复杂的信息，包括元数据、大纲和正文；要有效利用图书信息，需要大量标注数据。(ii) 图书呈现内容冗长且结构关系错综复杂。大纲揭示了章与节之间的层级关系，凸显出文本片段之间复杂的相互作用，而有效检索必须理解这些关系。

生成式检索 [19, 36] 是一种新兴的检索范式，它将全部语料库信息整合进一个统一的模型，该模型能够直接为查询生成相关的文档标识符。为此，生成式检索涉及两项基本学习任务：(i) 索引，包括学习图书与 docID 之间的关联；(ii) 检索，即将查询映射到对应的 docID。凭借在文档、段落和实体检索任务上的稳健表现，生成式检索受到越来越多的关注 [6, 9, 34, 35, 46, 47]。

由于摘要中所述的图书独特特性，直接将生成式检索应用于图书搜索是一项挑战。简单地将图书内容切分为多个相互独立的片段用于生成式检索学习，可能造成书籍结构中固有层级关系的丢失，进而影响检索质量。将生成式检索用于图书搜索，仍是一个复杂且悬而未决的问题。

我们的目标是构建一个有效的面向图书搜索的生成式检索框架，称为 GBS，它能针对给定查询准确返回相关图书标识符。为此，我们需要解决两个关键挑战。

**第一，如何在生成式检索框架下实现数据增强？** 考虑到生成式检索的两项学习任务（索引与检索），我们提出一种数据增强方法，为索引与检索构建多种训练数据对。该方法包括：(i) 面向索引的促进覆盖的图书标识符增强，其目标是利用大纲信息设计层级化图书标识符以及多种形式的图书内容，从而提升图书标识符的覆盖度，即书级、章级和节级标识符。我们进一步训练模型学习这些多样内容形式与其对应层级化图书标识符之间的映射，确保模型全面理解图书的结构与内容。(ii) 面向检索的增强多样性的查询增强，它考虑不同难度的查询——从单章即可回答的查询，到需要跨多章内容才能回答的查询。为生成满足这些标准的高质量查询，我们利用具有强大文本生成能力的大语言模型（LLM），设计了不同类型的提示。这些提示有助于创建丰富多样的伪查询集合，提升训练数据的多样性与质量。这些查询随后与书级标识符配对，用于检索任务的学习。

**第二，如何改善长图书输入的长度外推？** 如果将图书内容按顺序输入，很可能会丢失大纲所提供的章节之间的层级关系。此外，模型能够处理的输入长度存在限制。为解决这些问题，融入大纲信息至关重要。大纲不仅提供层级化上下文，还能对内容进行结构化组织，这有助于管理输入信息的长度。因此，我们提出大纲导向的书籍编码，依据书籍的大纲结构对书的章、节内容进行层级化区分。该方法包括：(i) 大纲导向的双层位置编码，基于书籍大纲对章级和节级文本施加层级化位置编码。该方法能更好地捕捉不同章节之间的关系，反映其结构层级。(ii) 大纲导向的保持式注意力，它在传统的多头注意力机制中引入一个额外的记忆模块。该模块存储输入文本中的重要历史信息，帮助注意力机制更有效地从长文本中过滤与整合关键信息。这一增强使模型能够在更长的序列上保持上下文，而这对于准确处理书籍中的大量内容至关重要。

图 1：图书主要由三类信息组成：(1) 元数据，包括书名、作者、出版社等细节；(2) 正文，构成图书的核心内容；(3) 大纲，展示章节之间的层级结构与关系。

图 2：GBS 基于编码器-解码器架构，包含两个组件：(1) 数据增强（橙色虚线框），包括面向索引的促进覆盖的图书标识符增强和面向检索的增强多样性的查询增强，生成多个数据对；(2) 大纲导向的书籍编码，包括大纲导向的双层位置编码（绿色虚线框）和大纲导向的保持式注意力（蓝色虚线框），基于层级信息对长篇书籍内容进行编码。（本图宜以彩色查看。）

在百度私有数据集上的实验表明，我们提出的方法 GBR 显著优于最先进的生成式检索基线 RIPOR，在 MRR@20 上提升 9.8%（译注：原文此处写作 GBR，应为 GBS 之笔误）。此外，我们在公开数据集 WhatsThatBook [18] 上验证了方法的有效性，证实了其鲁棒性与泛化能力。这些结果凸显了 GBS 通过应对图书搜索的独特挑战并充分利用所有可用信息，革新图书搜索的潜力。

## 2 方法论

### 2.1 问题定义

图书是复杂而丰富的信息源，由多种要素构成。我们将图书集合记为 $\mathcal{B}=\{b_{1},\ldots,b_{|\mathcal{B}|}\}$，其中 $b_{i}$ 是 $\mathcal{B}$ 中的第 $i$ 本书。如图 1 所示，一本书包括：

- **元数据**：由图书的基本信息构成，如书名、作者、出版社及其他书目细节。元数据为识别图书提供快速参照。
- **大纲**：表示图书的结构信息，详述内容如何组织为章与节。它提供了图书内容的层级视图，帮助理解信息的组织结构与脉络。
- **正文**：包含分布于各章节之中的详细文本，是实质性信息之所在，提供图书所涵盖的全面材料。

我们旨在构建一个自回归的生成式检索模型 $M$，针对查询集合 $\mathcal{Q}=\{q_{1},\ldots,q_{|\mathcal{Q}|}\}$ 中的给定查询 $q_{i}$，返回相关的图书标识符。模型 $M$ 需要准确理解并解释查询，利用图书的元数据、正文与大纲，识别并返回最相关的图书标识符。

### 2.2 模型架构

与以往的生成式检索研究 [36, 39, 46, 9] 类似，我们采用基于 Transformer 的模型，包含两个主要组件：

(i) **编码器**：一个双向编码器，配备我们专门设计的多头保持式注意力机制，用于编码图书内容或伪查询。我们将在 2.4 节详细解释这一多头保持式注意力机制。

(ii) **标识符解码器**：该组件通过序列生成过程产生图书标识符。

### 2.3 数据增强

其核心思想是模拟真实图书检索场景中的多样化查询，并确保图书标识符全面表征图书信息。这样我们便可以构建多个训练数据对，从而充分利用图书信息。该方法包括面向索引的促进覆盖的图书标识符增强，以及面向检索的增强多样性的查询增强。

#### 2.3.1 面向索引的促进覆盖的图书标识符增强

鉴于图书内容的巨大长度，我们设计多种形式的图书信息作为输入、层级化标识符作为输出，以帮助生成式检索模型更好地记住图书内容与标识符之间的映射。

按照信息压缩程度，我们使用五种类型的图书内容：

- **关键词**：一组包含关键信息的词，直接帮助模型学习图书最关键的信息。考虑到图书开头部分通常包含重要信息，这些词是利用现有关键词抽取器从 leading 章节（前部章节）的文本中抽取的。
- **摘要**：与关键词类似，但提供更连贯、信息更丰富的概要。它由一个摘要生成器从 leading 章节文本生成。
- **节文本**：一个小节的内容相对完整、包含更多细节。为尽量减少信息损失，模型需要学习细粒度的节文本。
- **章文本**：与节文本相比，章包含多个小节，能捕捉小节之间的关系以及更全面的细节。
- **全文**：图书的完整文本，提供章节之间更高层次的关联。

对于图书 $b_{i}$ 的图书标识符 $\mathcal{U}_{i}$，我们基于图书的大纲结构设计三个层级：

- **书级标识符**：由书名、作者和出版社组成，以 "#" 分隔。
- **章级标识符**：在 book-id 的基础上，添加章标题和章级语义结构化编号，以 "#" 分隔，构成 chapter-id。受 [36] 启发，该章级语义结构化编号是通过对 $b_{i}$ 中所有章的文本应用层级 K-means 聚类算法获得的。
- **节级标识符**：在该节所属章的 chapter-id 基础上，添加节标题和节级语义结构化编号，以 "#" 分隔，构成 section-id。类似地，该节级语义结构化编号是通过对当前章内所有节的文本应用层级 K-means 聚类算法获得的。

有些图书可能包含更细粒度的小节，但与节相比，小节中的信息过于碎片化，因此我们不予考虑。此外，有些图书可能只包含最高层级的章，这种情况下我们只考虑章级标识符。

利用上述方法，我们可以构建多个"图书内容-标识符"数据对用于索引任务的学习，其中图书内容作为输入，标识符作为输出。具体而言，全文、关键词和摘要对应 book-id；节文本对应 section-id；章文本对应 chapter-id。训练方法在 2.5 节中描述。

#### 2.3.2 面向检索的增强多样性的查询增强

根据实际检索需求，我们将查询分为两类：(i) 单章可答查询，只关注单章的细节；(ii) 多章可答查询，需要多章的上下文才能全面回答。为生成满足这些标准的高质量查询，我们使用具有强大文本生成能力的大语言模型，设计特定的提示来引导生成过程。两类提示如下：

- 单章可答查询的提示："Given the following chapter from a book, generate {X} pseudo queries that can be answered using the information contained within this single chapter. The queries should focus on key themes, events, characters, and any specific details provided in the chapter. A single chapter content: {chapter texts}."（译：给定书中的如下一章，生成 {X} 个仅用本章所含信息即可回答的伪查询。查询应聚焦于本章的关键主题、事件、人物及任何具体细节。单章内容：{chapter texts}。）
- 多章可答查询的提示："Given the following chapters from a book, where they are separated by a token "#", generate {X} complex pseudo queries that require synthesizing information from multiple chapters to answer. Each query should be clear, specific, and necessitate the integration of information across different chapters. Multiple chapter contents: {chapter texts}."（译：给定书中如下多章，章与章之间以记号 "#" 分隔，生成 {X} 个需要综合多章信息才能回答的复杂伪查询。每个查询应当清晰、具体，并要求整合跨章信息。多章内容：{chapter texts}。）

{X} 与 {chapter texts} 分别表示伪查询的数量和章文本。基于该策略，我们可以为检索任务构建多个"伪查询-相关图书标识符"数据对。

### 2.4 大纲导向的书籍编码

其核心思想是利用图书大纲提供的结构信息，将长篇书籍内容（尤其是全文）作为一个完整单元进行编码。我们通过设计大纲导向的双层位置编码和大纲导向的保持式注意力，来增强 Transformer 编码器中的位置编码和注意力机制。

#### 2.4.1 大纲导向的双层位置编码

关键思想是依据大纲将全文输入编码进模型，通过双层位置编码（包括节级和章级位置编码）凸显不同章节之间的关系。

**节级位置编码。** 在输入图书的完整文本序列 $L=[w_{1},\ldots,w_{|L|}]$ 中，每一节被视为一个独立的片段单元。节级位置编码用于定位每个词元在其所属节内的位置，便于捕捉语义信息。形式化地，在每个节 $L_{l}=[w_{a_{l}},w_{a_{l}+1},\ldots,w_{b_{l}}]$（其中 $l\in|L|$，$a_{l}$ 与 $b_{l}$ 为起始与结束索引）内，我们为词元 $w_{a_{l}+j}$ 编码（局部）位置 $j$，其中 $1\leq j\leq b_{l}-a_{l}+1$。通常，一节之内的词元数量有限，因此原始的绝对位置编码即可满足需求 [38]。$L_{l}$ 中的每个词元 $w_{a_{l}+j}$ 被赋予一个实值嵌入 $e_{j}$，随后加到输入词元嵌入上。该嵌入 $e_{j}$ 在不同节 $L_{l}$ 中处于相同局部位置 $j$ 的词元之间保持一致。节级位置编码与输入嵌入相结合，有助于维持图书内部的结构与上下文完整性。

**章级位置编码。** 尽管节级位置编码能定位词元在单个节内的位置，但它无法区分不同节之间的位置，无法捕捉节间上下文关系。为此，我们进一步引入章级位置编码，它标明每个词元所属的节，增强对训练中未出现的更长序列的处理能力。该章级编码使用基于节索引之间距离的相对位置编码 [30, 32]。我们采用旋转位置编码 [32] 作为章级位置编码。对于分别位于第 $n$ 节和第 $m$ 节的一对词元 $(w_{l_{1}},w_{l_{2}})$，我们赋予两个旋转矩阵 $R_{f,n}$ 和 $R_{f,m}$，其中 $f$ 表示旋转矩阵的预定义参数 [32]。给定一对注意力查询-键 $\hat{q}_{l_{1}}$ 和 $k_{l_{2}}\in\mathbb{R}^{d}$，注意力得分计算为 $(\hat{q}_{l_{1}}R_{f,n}(k_{l_{2}}R_{f,m})^{T})\cdot(\sqrt{d})^{-1}=(\hat{q}_{l_{1}}R_{f,n-m}k_{l_{2}}^{T})\cdot(\sqrt{d})^{-1}$。这里 $\hat{q}$ 是注意力查询，区别于作为模型输入构造的伪查询；$d$ 是 $k$ 的维度和缩放因子。该章级位置编码被整合进 Transformer 编码器原始的标准多头注意力（MHA）模块 [38] 之中。

#### 2.4.2 大纲导向的保持式注意力

我们在基于 Transformer 的骨干网络的编码器 [25, 38] 中引入一个额外的保持式记忆模块。该模块存储输入文本中的关键信息，随后与原始标准 MHA 聚合。这有助于注意力机制更有效地从长文本中过滤与整合重要信息。

**标准多头注意力。** 对于标准 MHA 中的单个头，注意力上下文 $C\in\mathbb{R}^{N\times d}$ 由缩放点积注意力计算，它从输入文本序列 $L\in\mathbb{R}^{N\times d}$ 通过公式 $C=\mathrm{softmax}((\hat{Q}K^{T})\cdot(\sqrt{d})^{-1})V$ 得到。其中 $\hat{Q}=L\xi_{Q}$、$K=L\xi_{K}$、$V=L\xi_{V}$，$\xi_{Q}$、$\xi_{K}$、$\xi_{V}$ 为可训练的投影矩阵。在 MHA 中，我们同时为序列中的每个元素生成 $H$ 个注意力上下文向量，沿第二维拼接这些向量，并将拼接后的向量投影回模型空间，得到最终的注意力输出。

**保持式记忆。** 在保持式注意力中，我们不再生成新的记忆条目，而是复用点积注意力过程中得到的查询、键、值状态（$\hat{Q}$、$K$、$V$）。点积注意力与保持式记忆之间的状态复用，有利于高效适配长上下文，并提升训练与推理速度。其目标是将键值对存入保持式记忆并使用查询向量进行检索，参见 [21]。保持式记忆以关联矩阵参数化 [29]，使更新与检索过程可以表述为一种线性注意力机制 [31]。该方法受益于类似方法中所采用的稳定训练技术。具体而言，我们采用 [15] 中的更新与检索方式，因其简单有效。

**保持式记忆检索。** 使用 $\hat{Q}$ 从保持式记忆 $MM_{s-1}$ 中计算新内容 $C_{new}$：

$$C_{new}=(\sigma(\hat{Q})\cdot MM_{s-1})\cdot(\sigma(\hat{Q})\cdot z_{s-1})^{-1}$$

其中 $\sigma$ 表示逐元素的 ELU + 1 激活函数 [7]，$z_{s-1}$ 表示对所有键求和得到的归一化项，参见 [15]。

**保持式记忆更新。** 检索之后，我们用新的键值条目更新记忆和归一化项。更新后的状态计算如下：

$$MM_{s}\leftarrow MM_{s-1}+\sigma(K)^{T}V, \tag{1}$$

$$z_{s}\leftarrow z_{s-1}+\sum_{t=1}^{N}\sigma(K_{t}). \tag{2}$$

新计算出的记忆状态 $MM_{s}$ 与 $z_{s}$ 随后被传递给下一个片段单元 $s+1$，在每层注意力内部建立起递推关系。

**历史上下文注入。** 我们使用一个可学习的门控标量 $\alpha$ 将局部注意力状态 $C$ 与检索到的记忆内容 $C_{new}$ 相结合：

$$C_{total}=\text{sigmoid}(\alpha)\odot C_{new}+(1-\text{sigmoid}(\alpha))\odot C. \tag{3}$$

在多头保持式注意力中，我们并行计算 $H$ 个上下文状态，随后将其拼接并投影，得到最终的注意力输出：$A=[C^{1}_{total};\ldots;C^{H}_{total}]\xi_{A}$，其中 $\xi_{A}$ 表示可训练权重。

### 2.5 训练

遵循 [36, 39]，对于 2.3 节构建的训练数据对，我们采用极大似然估计（MLE）[43] 来学习索引和检索任务。对于索引任务，给定多种形式的图书作为输入，模型最大化对应标识符作为输出的似然。关键词、摘要和全文对应 book-id；节文本对应 section-id；章文本对应 chapter-id。该任务可形式化为：

$$\mathcal{L}_{ind}(\mathcal{B},\mathcal{Y};\theta)=-\sum^{|B|}_{b_{i}\in B,y_{i,j}\in\mathcal{Y}_{i},o_{i,j}\in\mathcal{O}_{i}}\log P(o_{i,j}|y_{i,j}), \tag{4}$$

其中 $y_{i,j}\in\mathcal{Y}_{i}$ 可以是图书 $b_{i}$ 的任意形式的输入，$o_{i,j}\in\mathcal{O}_{i}$ 是 $y_{i,j}$ 对应的标识符。

对于检索任务，给定伪查询作为输入，模型最大化对应图书标识符作为输出的似然，形式化为：

$$\mathcal{L}_{rel}(\mathcal{B},\mathcal{Q};\theta)=-\sum^{|B|\times|\mathcal{Q}|}_{b_{i}\in B,q_{i,j}\in\mathcal{Q}}\log P(u_{i}^{t}|q_{i,j}), \tag{5}$$

其中 $\theta$ 表示模型参数，$u_{i}^{t}$ 是查询 $q_{i,j}$ 对应的图书 $b_{i}$ 的相关 book-id。

我们采用多任务学习的方式联合训练这两个任务，参见 [36, 52]。整体优化目标可形式化为：

$$\mathcal{L}(\mathcal{B},\mathcal{Y},\mathcal{Q};\theta)=\mathcal{L}_{ind}(\mathcal{B},\mathcal{Y};\theta)+\mathcal{L}_{rel}(\mathcal{B},\mathcal{Q};\theta). \tag{6}$$

### 2.6 推理

模型训练完成后，当输入一个查询时我们进行推理。遵循 [9]，为确保生成的标识符有效，我们使用前缀树约束模型的生成。为整合不同层级的标识符，我们考虑书级和章级标识符。鉴于单本书的节级标识符数量过多，可能在推理时干扰模型，我们暂时排除该层级。我们：(i) 首先使用标识符构建前缀树；(ii) 然后使用并行或串行方法解码，并聚合两个层级的推理结果，得到图书与输入查询之间的相关性得分。

**前缀树构建。** 对于前缀树，节点用预定义候选集合中的词元标注。每个节点的子节点表示从根遍历到该节点所确定前缀之后所有允许的延续 [9]。我们构建三种类型的树：

- **书级前缀树**：包含 $B$ 中所有 book-id。
- **单本书前缀树**：我们为每本书的 chapter-id 构建一棵单独的树。
- **章级前缀树**：我们将所有书的 chapter-id 汇总在一起构建一棵树。

**并行解码与聚合。** 首先，我们分别使用书级前缀树和章级前缀树进行解码，得到书级标识符列表和章级标识符列表。每本书的书级相关性得分是其对应的概率似然值，记为 $s_{b}(q,b_{i})$。每本书的章级相关性得分是生成的章级标识符列表中覆盖该书 chapter-id 的总数，记为 $s_{c}(q,b_{i})$。

其次，遵循 [17]，章级相关性得分作为书级相关性得分的权重。我们将查询 $q$ 与图书 $b_{i}$ 之间的总体相关性得分聚合为两个得分的乘积，形式化为：

$$score(q,b_{i})=s_{b}(q,b_{i})\times s_{c}(q,b_{i})$$

**串行解码与聚合。** 首先，我们使用书级前缀树进行约束解码，得到相关 book-id 列表。每本书的书级相关性得分与并行解码相同，记为 $s_{b}(q,b_{i})$。

接下来，在相关 book-id 列表范围内，我们使用每本书的单本书前缀树对每本书进行更细粒度的推理，得到相关 chapter-id 列表。每本书的章级相关性得分是列表中其所有 chapter-id 的似然值之和，记为 $s_{c}(q,b_{i})$。

最后，我们将 $q$ 与 $b_{i}$ 之间的总体相关性得分聚合为两个得分的加权和，形式化为：

$$score(q,b_{i})=\beta s_{b}(q,b_{i})+\gamma s_{c}(q,b_{i})$$

其中 $\beta$ 与 $\gamma$ 为权重。

### 2.7 GBS

GBS 的定义如下：首先，我们使用 2.3 节所述的数据增强方法为图书构建数据对。对于每个数据对，我们按照 2.2 节所述的模型结构，使用 2.4 节所述的大纲导向的书籍编码对输入进行编码，然后通过解码器预测标识符。该过程使用 2.5 节所述的训练目标进行优化。训练完成后，模型使用 2.6 节所述的任意解码方法（并行或串行）进行推理。我们考虑 GBS 的两个变体：GBS^P（使用并行解码）和 GBS^S（使用串行解码）。

## 3 实验设置

**数据集。** 我们同时使用一个私有数据集和一个公开数据集进行实验。(i) 百度图书搜索数据集：该数据集来自百度真实场景，包含一个带有元数据、大纲和正文的图书库。数据集同时包含中文和英文图书。我们采样了三个不同规模的数据集，即 10K、20K 和 40K。平均而言，每本书约含 225K 词。我们使用 2.3 节所述方法为每本书构造伪查询用于训练和评估。具体地，我们为每本书生成五个单章可答查询和五个多章可答查询，即 $X=5$。(ii) WhatsThatBook [18]：该数据集由图书搜索的"话到嘴边"（tip-of-the-tongue）查询构成，采集自 GoodReads 社区论坛的用户互动。该数据集仅包含英文图书和查询。查询包括论坛讨论内容，文档是带有对应元数据的图书。平均而言，每本书约含 131K 词。注意，该数据集原始训练查询共 11.6K 条，平均每本书一条标注查询；此外，我们为每本书生成每类各 4 条伪查询。

数据集统计信息见表 1。

表 1：数据集统计信息。#Book 表示图书数量；#Train 表示训练集中查询的数量；#Test 表示测试查询的数量。

| 数据集 | #Book | #Train | #Test |
| --- | --- | --- | --- |
| BBS 10K | 10K | 1M | 1K |
| BBS 20K | 20K | 2M | 1K |
| BBS 40K | 40K | 4M | 1.5K |
| WhatsThatBook | 14K | 1.5M | 1.45K |

**评估指标。** 与生成式检索工作 [17, 34, 36] 一致，我们采用命中率与 $K=\{10\}$ 的平均倒数排名（MRR@$K$，$K=\{20\}$）作为评估指标。

**基线。** 遵循现有生成式检索研究 [36, 40, 46]，我们考虑三类基线：(i) 稀疏检索基线：BM25 [27] 和 DocT5Query [23]；(ii) 稠密检索基线：RepBERT [48] 和 DPR [14]；(iii) 生成式检索基线：DSI [36]、GENRE [9]、SEAL [3]、DSI-QG [52]、NCI [39]、CorpusBrain [6]、Ultron [50]、GenRet [33]、NOVO [40]、ASI [44] 和 RIPOR [46]。对于稠密检索与生成式检索基线，我们将图书文本切分为多个片段，然后组成多个"图书片段-查询"数据对用于训练。

此外，对于生成式检索基线，我们还为索引任务构建多个"图书片段-对应标识符"数据对。所有生成式检索基线均采用编码器-解码器架构并以 MLE 优化。关于基线的更多细节请参见附录 A。

### 3.1 实现细节

**骨干模型。** 对于包含中英文图书的百度图书检索数据集，我们采用 Mengzi-T5-base [49]——一个在中英文语料上预训练的语言模型——作为骨干。对于全英文的 WhatsThatBook 数据集，我们使用生成式检索研究中广泛采用的 T5-base 模型 [25] 作为骨干 [36, 39, 46]。两个模型的隐层大小均为 768，前馈层大小为 3072，自注意力头数为 12，Transformer 层数为 12。Decoder-only 架构（如 GPT 系列模型 [24]）留待未来研究探索。

**超参数。** 关于章级和节级语义结构化编号，遵循 [36]，我们使用一个 8 层的小型 BERT 模型编码文本，并将聚类数量设为 10，每层最大阈值为 100。对于串行解码，我们将 $\beta$ 设为 1，$\gamma$ 设为 0.5。

**训练与推理。** GBS 使用 PyTorch 1.9.0 和 HuggingFace transformers 4.16.2 实现；我们重新实现了 DSI，其他基线使用开源代码。用于抽取图书关键词和摘要的模型为 TextRank，通过 summa API [2] 实现。训练所用（全文, book-id）数据对中的全文由图书的前 100 章构成。在推理时，我们使用带 20 个束的约束束搜索来解码标识符。我们采用 Adam 优化器，并在前 10% 的步数上进行线性预热。学习率设为 5e-5，标签平滑为 0.1，权重衰减为 0.01，最大训练步数为 5M，批大小为 128。我们将输入长度设为 128K，超出该限制的图书部分将被截断。我们的模型在八块 NVIDIA Tesla A100 80GB GPU 上训练。（译注：原文此段语序因排版原因有错位，以上按正确语义顺序翻译。）

## 4 实验结果

本节呈现实验发现。

表 2：在 BBS 和 WhatsThatBook 上的检索性能。最佳结果以粗体显示。∗ 表示相对表现最佳的生成式检索基线 RIPOR 有统计学显著提升（$p\leq 0.05$）。

| 方法 | BBS 10K Hits@10 | BBS 10K MRR@20 | BBS 20K Hits@10 | BBS 20K MRR@20 | BBS 40K Hits@10 | BBS 40K MRR@20 | WhatsThatBook Hits@10 | WhatsThatBook MRR@20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 41.8 | 30.5 | 40.6 | 30.1 | 40.1 | 29.8 | 45.3 | 41.6 |
| DocT5query | 46.6 | 39.3 | 42.5 | 35.4 | 37.5 | 30.7 | 48.2 | 42.8 |
| RepBERT | 53.1 | 46.4 | 48.3 | 41.9 | 45.6 | 37.3 | 56.8 | 48.1 |
| DPR | 51.3 | 43.6 | 46.4 | 40.3 | 42.1 | 35.9 | 54.2 | 45.7 |
| DSI | 20.7 | 13.4 | 18.5 | 11.6 | 13.5 | 7.2 | 22.6 | 15.4 |
| GENRE | 26.5 | 22.1 | 24.5 | 19.3 | 19.3 | 15.6 | 29.1 | 24.7 |
| SEAL | 27.8 | 23.6 | 25.7 | 20.6 | 20.5 | 16.2 | 30.5 | 24.8 |
| DSI-QG | 40.3 | 35.7 | 36.8 | 28.9 | 32.1 | 23.7 | 44.6 | 25.9 |
| NCI | 42.8 | 36.8 | 37.2 | 29.4 | 32.8 | 24.2 | 45.2 | 39.4 |
| CorpusBrain | 48.9 | 43.1 | 42.4 | 37.5 | 36.3 | 31.5 | 52.3 | 45.7 |
| Ultron | 48.6 | 44.5 | 41.3 | 36.3 | 35.7 | 30.1 | 51.9 | 45.2 |
| GenRet | 51.3 | 46.6 | 47.2 | 42.1 | 41.4 | 37.8 | 55.8 | 49.3 |
| NOVO | 52.7 | 47.3 | 47.6 | 42.8 | 41.8 | 38.4 | 56.5 | 49.7 |
| ASI | 58.4 | 53.6 | 53.5 | 44.6 | 46.6 | 40.1 | 56.2 | 49.5 |
| RIPOR | 62.5 | 52.8 | 56.8 | 46.2 | 52.9 | 42.7 | 66.7 | 55.4 |
| GBS^S（本文） | 66.4 | 54.1 | 61.3 | 49.5 | 56.3 | 46.5 | 70.4 | 58.1 |
| GBS^P（本文） | **66.7*** | **54.4*** | **61.6*** | **49.8*** | **56.7*** | **46.9*** | **70.7*** | **58.6*** |

（表 2 中基线按原文分组：BM25、DocT5query 为稀疏检索；RepBERT、DPR 为稠密检索；DSI 至 RIPOR 为生成式检索。）

### 4.1 主结果

表 2 展示了所提 GBS 与基线在 BBS 和 WhatsThatBook 数据集上的对比。

**稀疏检索与稠密检索基线的表现。** (i) BM25 在三个规模的 BBS 数据集上表现稳定，但随着图书数量的增加，性能略有下降。这种下降可能是因为将一本书划分为多个片段进行索引会增加主题相似片段的数量，使检索更加困难。(ii) DocT5query 总体上优于 BM25。这可能是因为生成的伪查询基于图书的前部章节，可能包含图书更多的关键信息。(iii) 两个稠密检索基线 RepBERT 和 DPR 优于稀疏检索基线。这可能是因为稠密嵌入能捕捉更多语义信息。

**生成式检索基线的表现。** (i) DSI 的表现差于稀疏检索基线，可能是因为图书信息过于丰富，仅使用语义结构化编号作为标识符会丢失太多信息。尽管 GENRE 相比 DSI 有所改进，但表现仍然较差，可能是因为使用书名作为标识符虽然比语义结构化编号包含更多信息，但仍只代表单一标识符，导致信息损失。(ii) DSI-QG 和 NCI 相较上述生成式检索基线有显著提升。这些提升可能得益于它们在训练中使用了额外生成的伪查询。(iii) CorpusBrain 和 Ultron 由于模型预训练而取得进一步提升。(iv) GenRet、NOVO 和 ASI 的表现与稠密检索基线相当，其中 ASI 甚至取得了更好的结果。这可能是因为这三个基线专门为检索学习合适的标识符。(v) RIPOR 优于其他基线，可能是因为其多阶段学习和负采样策略增强了模型对图书语料及相关性的理解。

**GBS 的表现。** (i) 两个变体 GBS^S 和 GBS^P 均取得优于基线的结果。具体而言，GBS^P 在 BBS 40K 数据集的 MRR@20 指标上超过最佳基线 RIPOR 9.8%，在 WhatsThatBook 数据集的 Hits@10 指标上超过 6%。这表明我们的方法对图书搜索的有效性。(ii) GBS^P 的表现略优于 GBS^S。这可能是由于生成章级标识符方式的差异。GBS^S 的串行解码保持了章级标识符相对于书级标识符列表的相对顺序，而并行解码则不然。当书级标识符列表不理想时，并行解码可以对此有所补偿。

### 4.2 消融研究

为验证 GBS 中各组件的有效性，我们在 BBS 40K 和 WhatsThatBook 数据集上进行消融研究。我们以 GBS^P 为重点，其检索性能见表 3；基于 GBS^S 的消融结果见表 4，趋势类似。我们的发现如下：

(i) 不使用增强多样性的查询增强时（即第 2 行），模型不学习检索任务，其在两个数据集上的检索性能相比 GBS^P（即第 1 行）均显著下降。这凸显了所生成伪查询对模型学习相关性的重要性。关于伪查询数量的分析请参见 4.4 节。

(ii) 不使用促进覆盖的图书标识符增强时，在学习每本书的内容时，将书划分为多个片段，学习"片段-书级标识符"数据对。该变体（即第 3 行）在两个数据集上相比 GBS^P 均出现明显性能下降。这表明仅依靠简单的图书片段与标识符对进行索引，不足以让模型充分学习图书信息，从而验证了促进覆盖的图书标识符增强的必要性与有效性。

(iii) 省略大纲导向的双层位置编码、仅使用骨干模型原有的相对位置编码时，该变体（即第 4 行）在两个数据集上相比 GBS^P 出现一定性能下降。这表明单层位置编码不足以表示图书信息，从而证实了我们的大纲导向双层位置编码对于书籍编码的必要性。

(iv) 移除大纲导向的保持式注意力、在编码器中使用 Transformer 默认的标准 MHA 时，该变体（即第 5 行）在两个数据集上相比 GBS^P 出现轻微性能下降。这表明我们的大纲导向保持式注意力确实有益于捕捉更长的输入信息。

表 4 展示了基于 GBS^S 的消融结果，其趋势与 GBS^P 的结果相同。

表 3：GBS^P 在 BBS 40K 和 WhatsThatBook 上的消融研究。

| 序号 | 方法 | BBS 40K Hits@10 | WhatsThatBook Hits@10 |
| --- | --- | --- | --- |
| 1 | GBS^P | 56.7 | 70.7 |
| 2 | 不用查询增强 | 50.6 | 64.9 |
| 3 | 不用标识符增强 | 45.3 | 60.8 |
| 4 | 不用双层位置编码 | 52.8 | 65.2 |
| 5 | 不用保持式注意力 | 53.5 | 67.3 |

表 4：GBS^S 在 BBS 40K 和 WhatsThatBook 上的消融研究。

| 序号 | 方法 | BBS 40K Hits@10 | WhatsThatBook Hits@10 |
| --- | --- | --- | --- |
| 1 | GBS^S | 56.3 | 70.4 |
| 2 | 不用查询增强 | 50.2 | 64.5 |
| 3 | 不用标识符增强 | 45.1 | 60.6 |
| 4 | 不用双层位置编码 | 52.4 | 64.8 |
| 5 | 不用保持式注意力 | 53.3 | 67.1 |

### 4.3 输入长度分析

为考察输入文本长度对图书搜索有效性的影响，我们变化输入长度。我们在 BBS 40K 数据集上设置 2K、4K、16K、32K、64K、128K 和 256K 的输入长度，GBS^P 与 RIPOR 的 Hits@10 结果见图 3。我们发现：

(i) 当输入长度在 4K 到 32K 之间时，RIPOR 的表现优于我们的方法。这可能是因为 RIPOR 的多阶段优化和负采样策略使其能够从图书的前部学习重要信息。此外，一旦输入长度超过 4K，RIPOR 的性能便开始下降，随后趋于稳定，表明 RIPOR 处理图书这类超长文本的能力有限。

(ii) 当输入长度在 64K 到 128K 之间时，我们的方法优于 RIPOR，且性能提升随输入长度增长而增大。这表明更多的输入内容有助于模型学习更全面的图书信息。我们还观察到在 2K 和 4K 输入长度下检索性能较弱，这进一步证实了图书信息的复杂性以及专门设计学习方法的必要性。

(iii) 当输入长度超过 128K 后，GBS^P 的检索性能略有下降。这可能是因为部分图书尾部内容的语义已包含在前部内容中，导致冗余而无额外增益。

图 3：GBS^P 与 RIPOR 在 BBS 40K 数据集上不同输入长度下的 Hits@10 性能。

### 4.4 伪查询数量的影响

我们提出的增强多样性的查询增强策略生成 $X$ 个两类的伪查询，其数量是影响检索性能的重要因素。因此，我们分析变化 $X$ 的影响。具体地，在 BBS 40K 数据集上，我们将 $X$ 设为 1 到 6 的整数，GBS^P 的 Hits@10 性能见图 4。我们观察到：

(i) 当 $X$ 不超过 5 时，增大 $X$ 带来较大的检索性能提升。这表明图书的伪查询有助于模型更好地学习相关性，增强查询、图书信息与标识符之间的联系。

(ii) 当 $X$ 设为 6 时，检索性能几乎无额外增益。这可能源于模型容量的限制，或 $X=5$ 已为模型有效学习图书细节提供了充分的信息。

图 4：GBS^P 在 BBS 40K 数据集上使用不同数量 $X$ 的多样性增强伪查询时的 Hits@10 性能。

### 4.5 案例研究

为更细致地分析所提 GBS 的表现，我们进行案例研究。具体地，我们从 BBS 40K 数据集的测试集中采样一个单章可答查询和一个多章可答查询，分析 GBS^P 与最佳基线 RIPOR 生成的标识符列表。如附录 B 的表 5 所示：

(i) 对于单章可答查询，我们的 GBS^P 成功地将书级和章级的相关标识符均排在首位，而 RIPOR 仅将其排在第二位。这表明我们的方法在处理此类查询上更为有效。

(ii) 对于多章可答查询，GBS^P 同样表现良好，而 RIPOR 未能在生成的前 50 个标识符中预测出正确的标识符。这凸显了图书搜索比一般网页搜索更困难，并进一步验证了我们的方法更擅长处理图书检索任务。

## 5 相关工作

**图书搜索。** 图书搜索涉及根据用户查询定位并检索图书。与处理多样文档的通用网页搜索不同，图书搜索必须处理图书内部的复杂结构，包括元数据、大纲和正文，以准确匹配用户需求 [16]。传统方法通常依赖词项匹配 [12, 42] 和索引来连接查询与图书元数据。更先进的技术利用自然语言处理（NLP）来理解和解释查询与图书内容，从而提升相关性 [4, 26]。

**生成式检索。** 生成式检索是一种新的搜索范式，它将整个语料库整合进一个统一的模型，使其能直接从查询生成相关的 docID [19, 36]。为此，它涉及两个核心操作 [36]：索引，学习文档与其 docID 之间的关系；检索，将查询映射到相关 docID。凭借在各类检索任务上的强劲表现，生成式检索受到越来越多的关注 [36, 46, 9, 34, 35, 47]。在本工作中，我们尝试将生成式检索应用于图书搜索，鉴于图书的独特特性，这是一项具有挑战性且尚未被探索的任务。

**长文本建模。** 长文本建模 [10] 对处理学术文章和报告等长篇文档至关重要，这类文档因其长度和复杂性而带来挑战。传统 RNN [45]（包括 LSTM [8] 和 GRU [45]）难以捕捉长程依赖，而基于 Transformer 的模型 [38]，尤其是预训练语言模型（PLM），虽已展现潜力，但面临固定输入长度和高计算成本的问题。让这些模型适配长文本，需要解决预处理以适配上下文长度，以及设计高效架构以有效管理长期依赖和层级结构的问题。一些研究通过增强位置编码来捕捉内在关系，从而改进长文本建模 [13, 28, 51]。另一些工作专注于增强骨干模型，例如在 Transformer 中引入额外的记忆模块以保留关键的长程信息 [20]。

## 6 结论

我们提出并评估了 GBS，一个专为图书搜索设计的生成式检索框架。我们的方法通过结合数据增强策略与大纲导向的编码技术，应对了图书的独特挑战。在工业界百度数据集和公开数据集上的实验表明，GBS 显著优于现有的最先进方法。这证实了我们的方法在提升图书搜索方面的有效性。

不过，仍存在一些未来可以改进的局限：(i) 由于图书的长度，我们构建了多个数据对来有效学习图书信息，这导致较高的训练成本。未来我们将探索平衡性能与学习成本的方法。(ii) 模型容量也是影响性能的一个因素。未来我们将探索更大容量骨干模型的影响。

## 致谢

本工作受国家自然科学基金（NSFC，资助号 62472408）、中国科学院战略性先导科技专项（资助号 XDB0680102）、国家重点研发计划（资助号 2023YFA1011602）、联想-中科院联合实验室青年科学家项目，以及资助号 JCKY2022130C039 项目的资助。本研究还（部分）受混合智能中心资助——这是荷兰教育、文化和科学部通过荷兰科学研究组织资助的 10 年期项目，https://hybrid-intelligence-centre.nl，项目编号 024.004.022；NWA ORC 2020/21 研究计划下编号 NWA.1389.20.183 的 LESSEN 项目，部分由荷兰研究理事会（NWO）资助；编号 KICH3.LTP.20.006 的 ROBUST 项目，部分由荷兰研究理事会（NWO）、DPG Media、RTL 以及荷兰经济事务与气候政策部（EZK）在 LTP KIC 2020-2023 计划下资助；以及获得欧盟"地平线欧洲"研究与创新计划资助协议号 101070212 资助的 FINDHR（人类推荐中的公平性与交叉非歧视）项目。所有内容仅代表作者观点，未必获得其各自雇主和/或资助方的认同或背书。

## 参考文献（原样保留，不翻译）

[1] Anserini 2020. Anserini. https://github.com/castorini/anserini.

[2] Federico Barrios, Federico López, Luis Argerich, and Rosa Wachenchauzer. 2016. Variations of the Similarity Function of TextRank for Automated Summarization. CoRR abs/1602.03606 (2016). arXiv:1602.03606 http://arxiv.org/abs/1602.03606

[3] Michele Bevilacqua, Giuseppe Ottaviano, Patrick Lewis, Wen-tau Yih, Sebastian Riedel, and Fabio Petroni. 2022. Autoregressive Search Engines: Generating Substrings as Document Identifiers. In Advances in Neural Information Processing Systems. 31668–31683.

[4] Avi Bleiweiss. 2017. A Hierarchical Book Representation of Word Embeddings for Effective Semantic Clustering and Search. In International Conference on Agents and Artificial Intelligence, Vol. 2. SCITEPRESS, 154–163.

[5] Shubham Chatterjee and Laura Dietz. 2021. Entity Retrieval Using Fine-grained Entity Aspects. In proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval. 1662–1666.

[6] Jiangui Chen, Ruqing Zhang, Jiafeng Guo, Yiqun Liu, Yixing Fan, and Xueqi Cheng. 2022. CorpusBrain: Pre-train a Generative Retrieval Model for Knowledge-Intensive Language Tasks. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 191–200.

[7] Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter. 2015. Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs). arXiv preprint arXiv:1511.07289 (2015).

[8] Arman Cohan, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, and Nazli Goharian. 2018. A Discourse-aware Attention Model for Abstractive Summarization of Long Documents. arXiv preprint arXiv:1804.05685 (2018).

[9] Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2021. Autoregressive Entity Retrieval. In International Conference on Learning Representations.

[10] Zican Dong, Tianyi Tang, Lunyi Li, and Wayne Xin Zhao. 2023. A Survey on Long Text Modeling with Transformers. arXiv preprint arXiv:2302.14502 (2023).

[11] Skhumbuzo Dube and Ritesh Ajoodha. 2021. Improving Library Book Retrieval By Using Topic Modeling. In Interdisciplinary Research in Technology and Management. CRC Press, 585–590.

[12] M Ali Fauzi, Agus Zainal Arifin, and Anny Yuniarti. 2017. Arabic Book Retrieval Using Class and Book Index Based Term Weighting. International Journal of Electrical and Computer Engineering 7, 6 (2017), 3705.

[13] Zhenyu He, Guhao Feng, Shengjie Luo, Kai Yang, Di He, Jingjing Xu, Zhi Zhang, Hongxia Yang, and Liwei Wang. 2024. Two Stones Hit One Bird: Bilevel Positional Encoding for Better Length Extrapolation. arXiv preprint arXiv:2401.16421 (2024).

[14] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, and Wu. 2020. Dense Passage Retrieval for Open-domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing. 6769–6781.

[15] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. In International Conference on Machine Learning. PMLR, 5156–5165.

[16] Shah Khusro, Irfan Ullah, Azhar Rauf, and Saeed Mahfooz. 2014. Issues and Challenges in Book Information Retrieval. International Information Institute (Tokyo). Information 17, 6 (2014), 2055.

[17] Yongqi Li, Nan Yang, Liang Wang, Furu Wei, and Wenjie Li. 2023. Multiview Identifiers Enhanced Generative Retrieval. In 61st Annual Meeting of the Association for Computational Linguistics. 6636–6648.

[18] Kevin Lin, Kyle Lo, Joseph E Gonzalez, and Dan Klein. 2023. Decomposing Complex Queries for Tip-of-the-tongue Retrieval. arXiv preprint arXiv:2305.15053 (2023).

[19] Donald Metzler, Yi Tay, Dara Bahri, and Marc Najork. 2021. Rethinking Search: Making Domain Experts Out of Dilettantes. SIGIR Forum 55, 1 (2021), 1–27.

[20] Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth Gopal. 2024. Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention. arXiv preprint arXiv:2404.07143 (2024).

[21] Tsendsuren Munkhdalai, Alessandro Sordoni, Tong Wang, and Adam Trischler. 2019. Metalearned Neural Memory. Advances in Neural Information Processing Systems 32 (2019).

[22] Ping Nie, Yuyu Zhang, Xiubo Geng, Arun Ramamurthy, Le Song, and Daxin Jiang. 2020. DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 1829–1832.

[23] Rodrigo Nogueira and Jimmy Lin. 2019. From doc2query to docTTTTTquery. An MS MARCO Passage Retrieval Task Publication. University of Waterloo.

[24] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training Language Models to Follow Instructions with Human Feedback. In Advances in Neural Information Processing Systems, Vol. 35. 27730–27744.

[25] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the Limits of Transfer Learning with a Unified Text-to-text Transformer. The Journal of Machine Learning Research 21, 1 (2020), 5485–5551.

[26] Julian Risch, Samuele Garda, and Ralf Krestel. 2018. Book Recommendation Beyond the Usual Suspects: Embedding Book Plots together with Place and Time Information. In Maturity and Innovation in Digital Libraries: 20th International Conference on Asia-Pacific Digital Libraries, ICADL 2018, Hamilton, New Zealand, November 19-22, 2018, Proceedings 20. Springer, 227–239.

[27] Stephen E. Robertson, Steve Walker, Susan Jones, Micheline M. Hancock-Beaulieu, and Mike Gatford. 1995. Okapi at TREC-3. In TREC. 109–126.

[28] Anian Ruoss, Grégoire Delétang, Tim Genewein, Jordi Grau-Moya, Róbert Csordás, Mehdi Bennani, Shane Legg, and Joel Veness. 2023. Randomized Positional Encodings Boost Length Generalization of Transformers. arXiv preprint arXiv:2305.16843 (2023).

[29] Imanol Schlag, Tsendsuren Munkhdalai, and Jürgen Schmidhuber. 2020. Learning Associative Inference using Fast Weight Memory. arXiv preprint arXiv:2011.07831 (2020).

[30] Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. 2018. Self-attention with Relative Position Representations. arXiv preprint arXiv:1803.02155 (2018).

[31] Zhuoran Shen, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, and Hongsheng Li. 2021. Efficient Attention: Attention with Linear Complexities. In Proceedings of the IEEE/CVF winter conference on applications of computer vision. 3531–3539.

[32] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing 568 (2024), 127063.

[33] Weiwei Sun, Lingyong Yan, Zheng Chen, Shuaiqiang Wang, Haichao Zhu, Pengjie Ren, Zhumin Chen, Dawei Yin, Maarten de Rijke, and Zhaochun Ren. 2023. Learning to Tokenize for Generative Retrieval. In Advances in Neural Information Processing Systems, Vol. 36.

[34] Yubao Tang, Ruqing Zhang, Jiafeng Guo, Jiangui Chen, Zuowei Zhu, Shuaiqiang Wang, Dawei Yin, and Xueqi Cheng. 2023. Semantic-Enhanced Differentiable Search Index Inspired by Learning Strategies. In 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 4904–4913.

[35] Yubao Tang, Ruqing Zhang, Jiafeng Guo, and Maarten de Rijke. 2023. Recent Advances in Generative Information Retrieval. In SIGIR-AP 2023: 1st International ACM SIGIR Conference on Information Retrieval in the Asia Pacific. ACM, 294–297.

[36] Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, and Donald Metzler. 2022. Transformer Memory as a Differentiable Search Index. In Advances in Neural Information Processing Systems, Vol. 35. 21831–21843.

[37] Irfan Ullah and Shah Khusro. 2020. Social Book Search: The Impact of the Social Web on Book Retrieval and Recommendation. Multimedia Tools and Applications 79, 11 (2020), 8011–8060.

[38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is All You Need. Advances in neural information processing systems 30 (2017).

[39] Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Hao Sun, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, Xing Xie, Hao Sun, Weiwei Deng, Qi Zhang, and Mao Yang. 2022. A Neural Corpus Indexer for Document Retrieval. In Advances in Neural Information Processing Systems, Vol. 35. 25600–25614.

[40] Zihan Wang, Yujia Zhou, Yiteng Tu, and Zhicheng Dou. 2023. NOVO: Learnable and Interpretable Document Identifiers for Model-Based IR. In Proceedings of the 32nd ACM Conference on Information and Knowledge Management.

[41] Hengzhi Wu, Gabriella Kazai, and Michael Taylor. 2008. Book Search Experiments: Investigating IR Methods for the Indexing and Retrieval of Books. In Advances in Information Retrieval: 30th European Conference on IR Research, ECIR 2008, Glasgow, UK, March 30-April 3, 2008. Proceedings 30. Springer, 234–245.

[42] Haihua Xie, Zhiyou Chen, Yabo Li, Shaoling Jing, Xiaoqing Lyu, and Zhi Tang. 2020. Sembrs: A Semantic Analysis Based Book Retrieval Approach. In 2020 IEEE conference on multimedia information processing and retrieval (MIPR). IEEE, 101–104.

[43] Caiming Xiong, Victor Zhong, and Richard Socher. 2017. DCN+: Mixed Objective And Deep Residual Coattention for Question Answering. In International Conference on Learning Representations.

[44] Tianchi Yang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, and Qi Zhang. 2023. Auto Search Indexer for End-to-End Document Retrieval. In Findings of the Association for Computational Linguistics: EMNLP 2023.

[45] Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical Attention Networks for Document Classification. In Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies. 1480–1489.

[46] Hansi Zeng, Chen Luo, Bowen Jin, Sheikh Muhammad Sarwar, Tianxin Wei, and Hamed Zamani. 2024. Scalable and Effective Generative Information Retrieval. In The 2024 ACM Web Conference.

[47] Hansi Zeng, Chen Luo, and Hamed Zamani. 2024. Planning Ahead in Generative Retrieval: Guiding Autoregressive Generation through Simultaneous Decoding. In Proceedings of the 2024 ACM SIGIR International Conference on Theory of Information Retrieval.

[48] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2020. RepBERT: Contextualized Text Embeddings for First-stage Retrieval. arXiv preprint arXiv:2006.15498 (2020).

[49] Zhuosheng Zhang, Hanqing Zhang, Keming Chen, Yuhang Guo, Jingyun Hua, Yulong Wang, and Ming Zhou. 2021. Mengzi: Towards Lightweight yet Ingenious Pre-trained Models for Chinese. arXiv:2110.06696 [cs.CL]

[50] Yujia Zhou, Jing Yao, Zhicheng Dou, Ledell Wu, Peitian Zhang, and Ji-Rong Wen. 2022. Ultron: An Ultimate Retriever on Corpus with a Model-based Indexer. arXiv preprint arXiv:2208.09257 (2022).

[51] Dawei Zhu, Nan Yang, Liang Wang, Yifan Song, Wenhao Wu, Furu Wei, and Sujian Li. 2023. Pose: Efficient Context Window Extension of LLMs via Positional Skip-wise Training. arXiv preprint arXiv:2309.10400 (2023).

[52] Shengyao Zhuang, Houxing Ren, Linjun Shou, Jian Pei, Ming Gong, Guido Zuccon, and Daxin Jiang. 2023. Bridging the Gap between Indexing and Retrieval for Differentiable Search Index with Query Generation. In Gen-IR@SIGIR 2023: The First Workshop on Generative Information Retrieval.

## 附录

### 附录 A 基线细节

基线方法描述如下：

**稀疏检索基线：**
(i) BM25 [27] 是常用的有效词项匹配方法。我们使用 Anserini 工具包 [1] 实现它。我们将图书切分为多个片段进行索引。
(ii) DocT5Query [23] 用一个微调过的 T5 [25] 以原始文档为条件预测伪查询，从而扩展文档，然后执行 BM25 检索。这里我们取前 10 章作为生成伪查询的输入，因为该基线难以编码图书全文。

**稠密检索基线：**
(i) RepBERT [48] 是采用暴力检索的双编码器模型；
(ii) DPR [14] 是经典的基于 BERT 的双编码器模型，对输入文本使用稠密嵌入。

**生成式检索基线：**
(i) DSI [36] 是第一个生成式检索工作，通过 K-means 聚类算法使用语义结构化编号作为 docID。
(ii) GENRE [9] 使用书名作为 docID，学习"文档-docID"数据对。
(iii) SEAL [3] 使用 n-gram 作为标识符，基于 FM-index 生成标识符，骨干模型为 BART-large。
(iv) DSI-QG [52] 使用 docT5query [23] 以图书内容为条件生成伪查询，并将其与标识符配对用于训练。它使用唯一的整数字符串作为标识符。
(v) NCI [39] 采用语义结构化编号作为标识符，使用"伪查询-标识符"对训练模型，并设计了前缀感知解码器。
(vi) CorpusBrain [6] 在预训练期间采用维基百科的唯一书名作为标识符。
(vii) Ultron [50] 采用乘积量化码作为标识符，先用"图书片段-docID"对进行预训练，再在下游任务上用标注查询和生成的伪查询进行有监督微调。
(viii) GenRet [33] 使用自编码器为图书生成标识符，将图书内容压缩进标识符并将 docID 重构回图书内容，与检索任务联合学习。
(ix) NOVO [40] 从图书中选择重要词语作为标识符，模型通过带标注信息的有监督学习进行训练。
(x) ASI [44] 引入额外的线性层辅助文档生成 docID，并引入负样本增强。
(xi) RIPOR [46] 使用多阶段优化策略和负例挖掘技术训练生成式检索模型。

### 附录 B 案例研究

表 5 展示了在给定两类查询时，GBS^P 与 RIPOR 生成的前 3 个标识符。

表 5：给定查询，GBS^P 和 RIPOR 返回前 3 个束的结果。正确结果以斜体显示（本文中以「*」标出）。

**单章可答查询：** Who is the author of the book "The Heart of a Boy"?（译：《一个少年的心》（The Heart of a Boy）这本书的作者是谁？）

| 排名 | GBS^P 书级标识符 | GBS^P 章级标识符 | RIPOR 图书标识符 |
| --- | --- | --- | --- |
| 1 | *The Heart of a Boy#Edmondo De Amicis#Laird & Lee* | *The Heart of a Boy#Edmondo De Amicis#Laird & Lee#October#068834* | 17-3-8-11-3-24-6-3-12-76-37-37-43-87-33-68-44-174-38-96-221-56-43-78-43-83-7-8-238-1-44-123 |
| 2 | Pig Heart Boy#Malorie Blackman#Penguin Random House Children's UK | The Heart of a Boy#Edmondo De Amicis#Laird & Lee#March#068815 | 17-3-8-33-75-32-123-65-168-38-63-183-211-48-95-63-58-168-43-67-83-65-128-46-37-98-68-43-16-43-76-32 |
| 3 | The Boy with a Broken Heart#Durjoy Datta#Penguin Metro Reads | The Heart of a Boy#Edmondo De Amicis#Laird & Lee#November#068753 | 17-3-8-11-67-127-39-105-18-35-15-207-48-53-8-178-157-47-36-85-43-17-43-87-9-4-178-164-105-36-41-38 |

**多章可答查询：** Introducing Enrico（译：介绍恩里科）

| 排名 | GBS^P 书级标识符 | GBS^P 章级标识符 | RIPOR 图书标识符 |
| --- | --- | --- | --- |
| 1 | *The Heart of a Boy#Edmondo De Amicis#Laird & Lee* | *The Heart of a Boy#Edmondo De Amicis#Laird & Lee#October#068834* | 17-3-8-11-67-127-39-105-18-35-15-207-48-53-8-178-157-47-36-85-43-17-43-87-9-4-178-164-105-36-41-38 |
| 2 | An Introduction to the Basics of Reliability and Risk Analysis#Enrico Zio#World Scientific Pub Co Inc | The Heart of a Boy#Edmondo De Amicis#Laird & Lee#November#068883 | 17-3-8-11-3-24-6-3-12-76-37-37-43-87-33-68-44-174-38-96-221-56-43-78-43-83-7-8-238-1-44-123 |
| 3 | Enrico Baj: The Artist's Home#Michael Reynolds#Skira Rizzoli | The Heart of a Boy#Edmondo De Amicis#Laird & Lee#December#068659 | 17-3-8-11-53-62-58-107-38-95-157-23-52-21-230-68-54-89-167-208-32-57-14-58-3-9-54-16-37-48-61-97 |

（全篇完。参考文献共 52 条，原样保留英文。）
