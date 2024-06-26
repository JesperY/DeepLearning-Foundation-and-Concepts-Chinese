# 序言

深度学习使用经过大型数据集训练的多层神经网络来解决复杂的信息处理任务，并且已经成为了机器学习领域最成功的范式。在过去十年里，深度学习已经在包括计算机视觉、语音识别和自然语言处理在内的多个领域引发了革命，并被用于越来越多的应用场景，这些应用场景横跨医疗保障、制造业、商业、金融业、科学研究等多个方面。最近，被称为大语言模型的大型神经网络，包含了大约一万亿个可学习参数，已经显现出通用人工智能的初步迹象，正在引领着技术史上的最大颠覆之一。

## 本书的目的

这种影响的扩大伴随着机器学习领域出版物在数量和广度上的爆炸式发展，并且创新的步伐仍在持续加速。对于这个领域的新人来说，理解关键思想的挑战就令人望而生畏，更不用说追赶研究前沿。在这种背景下，_Deep Learning: Foundations and Concepts_ 致力于带给新人和有经验的从业者深刻的理解，包括支撑深度学习的基本思想以及现代深度学习架构与技术的关键概念。这些资料会为读者提供强大的基础以保证未来的专业化。由于这个领域的变化日新月异，我们尽量避免避免关于最新研究的综合性概述。相反，本书最大的价值来源于对关键思想的提炼，虽然这个领域很可能继续快速发展，但是这些基础和概念大概率经得住时间的考验。例如，大语言模型在本书编写的时候仍在迅速发展，但是其最基础的 transformer 架构和 attention 机制在过去的五年里并没有太大的变化，同时许多机器学习的核心原理已经被人们熟知了数十年。

## 负责任地使用技术

深度学习是一个具有广泛应用地强大技术，在为世界创造巨大价值和应对社会中最紧迫地挑战等方面具有很大的潜力。然而，这同样意味着深度学习也可能被蓄意滥用或者造成意料之外的损害。我们已经决定不去讨论深度学习技术的道德或者社会问题，因为这些问题非常重要和复杂，它们需要更加详尽彻底的讨论。然而这些人文伦理的讨论应该建立在底层技术和原理的坚实基础之上，所以我们希望这本书可以对这些重要的课题提供一些充满价值的贡献。我们强烈鼓励读者关注他们的工作并在学习如何使用这项技术的同时，了解如何负责的使用深度学习和人工智能。

## 本书的结构

这本书被组织成很多短小的章节，每一章探索了一个特定的主题。采用一种线性结构，每一章只依赖于前几章所涉及的内容。这很适合作为一个两学期的本科生或研究生的机器学习课程，但是也同样适用研究人员和自学者。

仅通过使用某些数学知识就可以建立一个关于机器学习的清晰理解。尤其是涉及到机器学习核心的三个数学领域：概率论、线性代数和多元微积分。本书对于需要的概率论的知识提供了齐全的介绍，并且在附件中总结了线性代数中一些有用的结论。虽然我们在附件中提供对于变分法和拉格朗日乘子的基本介绍，但是我们假定读者对多元微积分的概念基本熟悉。本书的目的是传达一种对思想的清晰理解，重点关注于具有实际使用价值的技术而不是抽象的理论。在可能的情况下我们会使用文字描述、图表或者数学公式等多种视角来表达一些过于复杂的概念。另外很多在本书中的讨论的关键算法被总结在单独的框体中。这并不足以解决计算效率的问题，但是尽可能充分地提供了数学解释。我们希望各种背景的读者都可以获取到本书的内容。

从概念上讲，本书也许会被很自然的看作 _Neural Networks for Pattern Recognition_（Bishop,1995b）的续作，其首次从统计的视角对待神经网络。也可能被认为是 _Pattern Recognition and Machine Learning_（Bishop,2006）的姐妹篇，虽然它早于深度学习革命，但其中广泛涉及了机器学习的很多概念。为了保证这本新书是自包含的，很多来袭 Bishop（2006）的合适内容被保留并且重新组织以着重于那些深度学习所需要的基础思想。这意味着那些在保留至今的机器学习中的有趣主题在本书中被省略。例如 Bishop（2006）中讨论过贝叶斯方法，但是本书几乎完全没有贝叶斯。

一个相关的网站提供了一些支撑性内容，包括免费的电子版书籍、练习的解决方案以及可下载 PDF 和 JEPG 格式的图片。
https://www.bishopbook.com
引用本书的 BibTex 条目：
```latex
@book{Bishop:DeepLearning24,
author = {Christopher M. Bishop and Hugh Bishop},
title = {Deep Learning: Foundations and Concepts},
year = {2024},
publisher = {Springer}
}
```
反馈请发送到 feedback@bishopbook.com

## 参考

本着专注于核心思想的精神，我们并没有试图提供一个全面的文献综述，考虑到该领域的规模和变化速度，这本身就是不可能的。然而，我们确实提供了一些关键研究论文以及综述文章和其他进一步阅读的资源的参考。在许多情况下，这些还提供了我们在文本中未详细讨论的重要实现细节，以免分散读者对正在讨论的核心概念的注意力。
关于机器学习和特别是深度学习的主题，已经有许多书籍出版。与本书水平和风格最接近的包括 Bishop (2006)、Goodfellow、Bengio 和 Courville (2016)、Murphy (2022)、Murphy (2023) 和 Prince (2023)。
在过去的十年中，机器学习学术研究的性质发生了显著变化，许多论文在提交给同行评审的会议和期刊之前，甚至代替提交，被在线发布在档案网站上。其中最受欢迎的网站是 arXiv，发音为‘archive’，网址为 https://arXiv.org
该网站允许论文更新，常常导致与不同日历年份相关联的多个版本，这可能会导致关于应该引用哪个版本以及哪一年的某些模糊性。它还提供每篇论文的 PDF 免费访问。因此，我们采取了一种简单的引用方法，按照首次上传的年份来引用论文，尽管我们推荐阅读最新版本。
arXiv 上的论文使用一个标记法 arXiv:YYMM.XXXXX 进行索引，其中 YY 和 MM 分别表示首次上传的年份和月份。后续版本通过在形式为 arXiv:YYMM.XXXXXvN 的版本号 N 进行标记。

## 练习

每章的结尾都有一组旨在巩固文中解释的关键思想或以重要方式发展和推广这些思想的练习题。这些练习是文本的重要组成部分，每个练习都根据难度进行了评级，从（⭐）表示一个简单的练习，需要几分钟就能完成，到（⭐⭐⭐），表示一个明显更复杂的练习。强烈鼓励读者尝试这些练习，因为积极参与材料的学习大大提高了学习效果。所有练习的解答都可以从书籍网站下载PDF文件获得。

## 数学标记

暂略，如有需要再做翻译。

## 致谢

略 