# 小样本下材料领域信息抽取Pipeline设计与实现

1.	摘要 
-	简要介绍问题背景、前人的工作，以及现有工作的创新
2.	引言 
-	介绍问题背景、难点、前人工作
-	介绍现有工作的解决方案、创新点
-	预期达到的成果
3.	研究方法 
-	数据来源
-	论文PDF数据解析，主要为-CR（待设计）
-	数据标注规则、Prompt设计
-	模型选择与介绍（暂定T5，看实际效果）
-	训练trick细节
4.	实验数据与结果分析 
-	具体数据、准召、F1值
-	对比前人算法效果
5.	总结 
- 总结实验结果
- 项目的应用场景（构建数据库）
- 展望与改进方向

# model

- bert-base-uncased: https://zenodo.org/record/7268337#.Y2CgbexByhY

- T5: https://zenodo.org/record/7268223#.Y2CgsuxByhY

  
# data

- papers:
- chemical:
  

# 摘要

​	人工智能的发展离不开设计精妙的算法，更离不开大量可靠的数据。在有机合成、药物合成、催化剂、材料等领域想要应用人工智能技术来辅助分子逆合成路线设计，辅助预测反应产物、产率与反应条件，就往往需要大量的数据（reference ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction / Automated pipeline for superalloy data by text mining）。这些数据基本上都需要从大量文献与专利中进行高质量的收集。而文献中的数据大多为非结构化的纯文本以及半结构化的表格隐藏在文献中，并且格式和论述方法风格不尽相同。因此想要大规模批量自动化提取这些数据，就需要通过自然语言处理的相关算法来实现。

​	一般来说，化学反应数据收集常用的策略是识别文本中化学物质、反应温度、时间及其他条件与信息，即将文本化学信息进行结构化处理。这其实与自然语言处理中的命名体识别（NER）任务非常相似。因此，除了一些基于人工设定化学规则的方法外，许多基于深度学习的NER算法被应用于这个问题，并表现出一定的潜力，如基于传统算法的patent-reactionextraction项目和最新大规模预训练模型的ChEMU以及ChemRxnExtractor项目（ChemDataExtractor: A Toolkit for Automated Extraction of Chemical Information from the Scientific Literature / ChEMU: Named Entity Recognition and Event Extraction of Chemical Reactions from Patents）。但大多数论文数据提取后仅仅做了部分后处理如消歧、拼接、聚类后就作为输出或写入数据库，并没有对数据做语义解析 & 语法解析等结构化处理，后续研究我们可以将得到的数据再进一步做结构化处理，如结合知识图谱让实体之间产生关联，进一步提升数据库的可用性和可扩展性（From database to knowledge graph — using data in chemistry）。

​	总体而言，目前大多数学术界开发的有机反应收集工具，仍存在不少提升空间，抑或是不足以完成整个反应数据流的自动收集工作。一方面借助nlp领域飞速发展的算法提高语言模型表现，同时也可以结合图像处理相关的OCR技术识别PDF中的图表与文本，继续进行信息抽取，实现自动化流水线，进而将大量的论文数据结构化为大规模数据集。

# 引言

## 问题描述

### 背景介绍

​	近年来，深度神经网络在各种分子、材料属性预测应用领域十分火热（Deep learning for computational chemistry），然而每个新的数据都需要昂贵且耗时的实验，并且这些数据大多存在于纸面论文中，并不能方便的为计算科学所用，因此数据稀缺性仍然是机器学习的根本挑战。

​	由此分析可知，从材料领域科学文献中准确抽取相关实体以及实体之间关系的任务是对该领域知识进行深层次分析的基础，该任务对材料属性预测、催化方案生成以及新材料发现等方面具有重大意义。然而，材料领域中相关实体名称组成复杂、结构嵌套，且标注困难、专业要求高，因此缺乏大规模人工标注语料库，为抽取统一、完整、准确的材料信息带来了困难。使用大量未被标注的纯文本数据，从中抽取出目标信息数据的有效方法仍然是一个悬而未决的重要挑战。

​	在过去的数年间，基于Transfomer的大规模预训练模型在自然语言处理领域大放异彩，多家中外行业巨头如谷歌、微软、脸书、百度、OPENAI、华为等都研发出了适用于他们应用场景的预训练模型（BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding / OPT: Open Pre-trained Transformer Language Models / Language Models are Few-Shot Learners / Exploring the Limits of Transfer Learning with a Uniﬁed Text-to-Text Transformer / ERNIE: Enhanced Representation through Knowledge Integration）。不同于传统的RNN循环神经网络训练相关信息抽取任务时需要大量标注数据，此类模型训练的数据只需要非标注数据进行预训练，之后可用少量标注数据进行微调，便可达到甚至远超越传统RNN模型的性能。

### 难点

​	相较于日常应用场景，现阶段并没有较为好用的材料领域开源预训练模型，因此需要重新筹集适用于材料领域的数据集来对模型进行预训练。该项任务不仅需要数以百万计的论文语料数据，还需要耗费大量的GPU计算资源和时间。其次，由于专业性过高导致标注困难，缺乏大规模人工标注语料库，使得模型的训练和评估变得十分困难。最后，由于其交叉领域的特性，需要团队同时具备化学以及材料和计算机方面的同学以及老师通力合作才能够较好的完成任务，项目管理难度陡增。

## 解决方案

### 数据与模型

​	该项目从数据和模型两个方面切入，我们需要训练材料相关领域的大规模预训练模型和构建少量高质量的标注数据集。

​	在模型方面，大规模论文语料收集自github开源项目[S2ORC](https://github.com/allenai/s2orc)。其包含了八千一百万以上篇论文数据，其中七千三百万篇拥有摘要，八百万篇拥有论文全文，涉及化学领域的数据也有数百万篇，是十分理想的数据集。

​	在数据方面，为解决数据集的标注规则不统一和标注密度过低的问题， 我们基于论文内容、结构的分析与整理，提出了更适合小样本下材料领域信息抽取的标注方案，并由相关专业的同学人工标注约 500 份的高质量数据。	



### 创新点

1)	基于材料相关领域论文的分析与整理，提出了统一、逻辑合理且包含上下文推理的标注规则。设立了额外的大标签进行多任务目标实体预测，以提高模型对主任务的预测效果。增加实体之间关系的标注，如催化反应的催化剂、反应物、产物、催化条件、反应效率之间的关系。。
2)	使用了当下由谷歌公司开发的最新的T5预训练模型（Text-To-Text Transfer Transformer），采用迁移学习与Prompt提示学习的方法对模型进行训练，对小样本数据集更为友好。
3)	针对专业领域的标注数据集较小、相对复杂的标注规则以及大量的非实体标签O，我们对训练数据进行了数据冗余和数据增强的处理，广泛采用字典方式进行自动标注，辅助和加快繁琐的人工标注过程，并且设置冗余标签，减少非实体的比例，减小了模型的过拟合程度，增加了模型的鲁棒性和对于极少出现的标签识别的准确率。同时使用对比学习算法，减少了模型的表示空间的坍缩问题（ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer）。

## 预期达到的成果

​	该项目将提供从论文pdf到结构化数据的pipeline流水线处理工具，并且其数据抽取效果评估F1值高于目前最新的化学领域抽取工具ChemRxnExtractor。

​	最后，该项目将基于大规模的论文数据构建材料领域的数据库以方便计算化学的应用。



