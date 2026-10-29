---
Title: Fast On-device LLM Inference with NPUs - 中英双语版
Description: arXiv:2407.05858 英文原文与中文机器辅助译文，含原图。
Sort: 1
---

来源：[arXiv:2407.05858](https://arxiv.org/abs/2407.05858)。中文为机器辅助翻译，仅供学习；术语与结论以英文原文为准。

# Fast On-device LLM Inference with NPUs

> **中文标题：**使用 NPU 进行快速设备上 LLM 推理 感谢：\* 通讯作者。

> 双语学习版 ：英文内容、公式、图表和参考文献来自 arXiv 原文；中文为机器辅助翻译，仅供学习，术语和结论请以英文原文为准。 Source: arXiv:2407.05858

Conference: Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1; March 30–April 3, 2025; Rotterdam, Netherlands.Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1 (ASPLOS ’25), March 30–April 3, 2025, Rotterdam, NetherlandsISBN: 979-8-4007-0698-1/25/03DOI: [10.1145/3669940.3707239](https://doi.org/10.1145/3669940.3707239)CCS: Human-centered computing Ubiquitous and mobile computingCCS: Computer systems organization Heterogeneous (hybrid) systemsCCS: Computing methodologies Natural language processing

Daliang Xu
Affiliation: Key Lab of HCST (PKU), MOE; SCS, Peking University, China
email: [xudaliang@pku.edu.cn](mailto:xudaliang@pku.edu.cn)
, 
Hao Zhang
Affiliation: Beijing University of Posts and Telecommunications, China
email: [hao\_zhang@bupt.edu.cn](mailto:hao_zhang@bupt.edu.cn)
, 
Liming Yang
Affiliation: Key Lab of HCST (PKU), MOE; SCS, Peking University, China
email: [yangliming2018@pku.edu.cn](mailto:yangliming2018@pku.edu.cn)
, 
Ruiqi Liu
Affiliation: Key Lab of HCST (PKU), MOE; SCS, Peking University, China
email: [rich@stu.pku.edu.cn](mailto:rich@stu.pku.edu.cn)
, 
Gang Huang
Affiliation: Key Lab of HCST (PKU), MOE; SCS, Peking University, China
, National Key Laboratory of Data Space Technology and System, China
email: [hg@pku.edu.cn](mailto:hg@pku.edu.cn)
, 
Mengwei Xu∗
Affiliation: Beijing University of Posts and Telecommunications, China
email: [mwx@bupt.edu.cn](mailto:mwx@bupt.edu.cn)
 and 
Xuanzhe Liu∗
Affiliation: Key Lab of HCST (PKU), MOE; SCS, Peking University, China
email: [liuxuanzhe@pku.edu.cn](mailto:liuxuanzhe@pku.edu.cn)

© acmlicensed

###### Abstract.

> **中文标题：**抽象的。

On-device inference for Large Language Models (LLMs), driven by increasing privacy concerns and advancements of mobile-sized models, has gained significant interest.
However, even mobile-sized LLMs (e.g., Gemma-2B) encounter unacceptably high inference latency, often bottlenecked by the prefill stage in tasks like screen UI understanding.

> **中文译文：**在日益增长的隐私问题和移动规模模型的进步的推动下，大型语言模型 (LLM) 的设备端推理引起了人们的极大兴趣。然而，即使是移动大小的 LLM（例如 Gemma-2B）也会遇到不可接受的高推理延迟，通常会受到屏幕 UI 理解等任务中预填充阶段的瓶颈。

We present llm.npu, the first LLM inference system utilizing on-device Neural Processing Unit (NPU) offloading to reduce prefill latency.
llm.npu enhances NPU offloading efficiency by re-constructing the prompt and model in three levels:
(1) At prompt level, it divides variable-length prompts into multiple fixed-sized chunks while maintaining data dependencies;
(2) At tensor level, it identifies and extracts significant outliers to run on the CPU/GPU in parallel with minimal overhead;
(3) At block level, it schedules Transformer blocks in an out-of-order manner to the CPU/GPU and NPU based on their hardware affinity and sensitivity to accuracy.
Compared to competitive baselines, llm.npu achieves 22.4$\times$ faster prefill speed and 30.7$\times$ energy savings on average, and up to 32.8$\times$ speedup in an end-to-end real-world application.
For the first time, llm.npu achieves more than 1,000 tokens/sec prefilling for a billion-sized model.

> **中文译文：**我们推出了 llm.npu，这是第一个利用设备上神经处理单元 (NPU) 卸载来减少预填充延迟的 LLM 推理系统。 llm.npu通过在三个层面上重构提示和模型来提高NPU卸载效率：（1）在提示层面，它将可变长度的提示划分为多个固定大小的块，同时保持数据依赖性； (2) 在张量级别，它识别并提取显着的异常值，以最小的开销在 CPU/GPU 上并行运行； (3) 在块级别，它根据硬件亲和性和对精度的敏感度，以乱序方式将 Transformer 块调度到 CPU/GPU 和 NPU。与竞争基准相比，llm.npu 的预填充速度平均提高了 22.4 倍，节能 30.7 倍，在端到端实际应用中加速高达 32.8 倍。 llm.npu 首次实现了十亿规模模型每秒超过 1,000 个代币的预填充。

###### Keywords:

> **中文标题：**关键词：

Mobile computing, large language model, NPU

## 1. Introduction

> **中文标题：**一、简介

With rising privacy concerns ([5](#bib.bib5)), there is growing interest in running Large Language Models (LLMs) locally on mobile devices, known as on-device LLMs ([102](#bib.bib102), [100](#bib.bib100), [99](#bib.bib99)), e.g, Apple Intelligence ([2](#bib.bib2)) and Android AI Core ([1](#bib.bib1)).
Meanwhile, the advancement of mobile-sized language models (1B–10B parameters), such as Qwen2-1.5B and Phi3-3.7B, has demonstrated their ability to perform comparably to significantly larger models like GPT-3, despite having a reduced parameter count ([26](#bib.bib26), [27](#bib.bib27), [24](#bib.bib24), [60](#bib.bib60)). This progress makes the deployment of on-device language models feasible.
Without giving away private data, on-device LLM inference catalyzes novel mobile applications ([90](#bib.bib90)), such as UI task automation ([55](#bib.bib55)) (e.g., translating users’ language commands into UI operations such as “forward the unread emails to Alice”)
and automated message reply ([20](#bib.bib20)).

> **中文译文：**随着隐私问题的日益严重 (5)，人们对在移动设备上本地运行大型语言模型 (LLM) 越来越感兴趣，称为设备上 LLM (102、100、99)，例如 Apple Intelligence (2) 和 Android AI Core (1)。与此同时，移动大小的语言模型（1B-10B 参数）（例如 Qwen2-1.5B 和 Phi3-3.7B）的进步已经证明了它们的性能与 GPT-3 等更大的模型相当，尽管参数数量减少了（26、27、24、60）。这一进展使得设备上语言模型的部署变得可行。在不泄露私人数据的情况下，设备上的 LLM 推理可以促进新颖的移动应用程序 (90)，例如 UI 任务自动化 (55)（例如，将用户的语言命令翻译为 UI 操作，例如“将未读电子邮件转发给 Alice”）和自动消息回复 (20)。

However, the high inference latency remains as a significant obstacle to practical on-device LLMs.
To accomplish a UI task, LLM needs to ingest the screen view hierarchy (typically 600-800 tokens ([82](#bib.bib82), [22](#bib.bib22))) to generate corresponding UI operations step by step ([81](#bib.bib81)).
As will be shown in $\lx@sectionsign$[2.1](#S2.SS1 "2.1. On-device LLM Inference Analysis ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs"), each such step takes 8.1 seconds for Qwen1.5-1.8B ([27](#bib.bib27)), and thereby more than 40 seconds to finish a 5-step UI task.
Similarly, the Gemma-2B model requires 26.7 seconds to automatically reply to an email by mimicking the user’s tone based on historical email data (with 1500 tokens).
Diving into those tasks, we find out that the prompt processing (prefill stage) often dominates the end-to-end inference latency, e.g., 94.4%–98.8% for UI automation tasks.
This is because on-device LLM tasks often involve long-context understanding for handle personalized tasks.
Unfortunately, existing research efforts primarily focus on accelerating the text generation speed (decoding stage), such as activation sparsity ([94](#bib.bib94), [74](#bib.bib74))
and speculative decoding ([50](#bib.bib50), [61](#bib.bib61), [95](#bib.bib95)).
Therefore, this work mainly targets improving the prefill speed of on-device LLMs.

> **中文译文：**然而，高推理延迟仍然是实际设备上 LLM 的重大障碍。为了完成UI任务，LLM需要摄取屏幕视图层次结构（通常600-800个令牌（82, 22））以逐步生成相应的UI操作（81）。正如 \lx@sectionsign 2.1 中所示，对于 Qwen1.5-1.8B (27)，每个此类步骤需要 8.1 秒，因此完成 5 步 UI 任务需要 40 秒以上。同样，Gemma-2B 模型根据历史电子邮件数据（包含 1500 个令牌）模仿用户的语气，需要 26.7 秒来自动回复电子邮件。深入研究这些任务，我们发现提示处理（预填充阶段）通常主导端到端推理延迟，例如 UI 自动化任务的 94.4%–98.8%。这是因为设备上的 LLM 任务通常涉及长上下文理解以处理个性化任务。不幸的是，现有的研究工作主要集中在加快文本生成速度（解码阶段），例如激活稀疏性（94, 74）和推测解码（50, 61, 95）。因此，这项工作的主要目标是提高设备上LLM的预填充速度。

LLM prefilling is compute-bounded ([106](#bib.bib106), [69](#bib.bib69), [83](#bib.bib83));
yet, mobile CPU and GPU have limited parallel computing capacity ([97](#bib.bib97), [40](#bib.bib40)).
Instead, we are motivated by a key opportunity that Neural Processing Units (NPUs) are ubiquitously available in modern mobile devices, e.g., Qualcomm Hexagon NPU and Google Edge TPU.
These mobile NPUs are efficient at integer vector operations, delivering computing capability up to 73 TOPS ([17](#bib.bib17)).
On CNNs, their improvements over mobile CPU/GPUs are demonstrated to be up to 18$\times$/4$\times$, respectively ([86](#bib.bib86)).
Mobile NPUs are also more energy-efficient, and have less workloads contention as compared to CPU/GPU.

> **中文译文：**LLM 预填充受计算限制 (106, 69, 83)；然而，移动CPU和GPU的并行计算能力有限(97, 40)。相反，我们受到现代移动设备中无处不在的神经处理单元 (NPU) 的关键机遇的激励，例如 Qualcomm Hexagon NPU 和 Google Edge TPU。这些移动 NPU 在整数向量运算方面非常高效，计算能力高达 73 TOPS (17)。在 CNN 上，它们相对于移动 CPU/GPU 的改进被证明分别高达 18 倍 /4 倍 (86)。与 CPU/GPU 相比，移动 NPU 也更节能，并且工作负载争用更少。

Surprisingly, with such promised advantages, there exists no systems supporting LLM inference on COTS mobile NPUs.
Indeed, our preliminary efforts show that directly employing mobile NPUs for LLM inference does not offer performance benefits due to the following challenges.

> **中文译文：**令人惊讶的是，尽管有这些承诺的优势，但目前还没有支持 COTS 移动 NPU 上的 LLM 推理的系统。事实上，我们的初步工作表明，由于以下挑战，直接使用移动​​ NPU 进行 LLM 推理并不能提供性能优势。

$\bullet$ Costly preparation for variable-length prompts.
Mobile NPUs typically support only inference on static shapes, while LLM prompt length is dynamic (with a max context length).
Re-preparing and optimizing the LLM execution graph on NPUs for each different-sized prompt is costly on mobile devices (e.g., 11 seconds for the Gemma-2B model).
On the other hand, simply padding the LLM request to the same max context length wastes precious computing resources.

> **中文译文：**\bullet 可变长度提示的昂贵准备。移动 NPU 通常仅支持静态形状的推理，而 LLM 提示长度是动态的（具有最大上下文长度）。在移动设备上，为每个不同大小的提示在 NPU 上重新准备和优化 LLM 执行图的成本很高（例如，Gemma-2B 模型需要 11 秒）。另一方面，简单地将 LLM 请求填充到相同的最大上下文长度会浪费宝贵的计算资源。

$\bullet$ Mismatch between LLM quantization algorithms and mobile NPU design.
Due to the presence of outlier activations ([84](#bib.bib84), [33](#bib.bib33)), state-of-the-art LLM quantization methods often use per-group quantization to maintain high accuracy.
It partitions original activations and weights tensors into multiple groups with independent quantization scales to avoid the impacts of outliers on others.
However, our investigation shows that mobile NPUs cannot perform per-group MatMul directly (Table [2](#S2.T2 "Table 2 ‣ 2.2. Opportunity: Mobile NPUs ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs")). Instead, they must split the MatMul into multiple group-sized sub-tensor MatMuls and then reduce the sub-tensor intermediate results using a float sum operation. This process hampers NPU efficiency and incurs up to 10.7$\times$ performance overhead.

> **中文译文：**\bullet LLM 量化算法与移动 NPU 设计之间的不匹配。由于异常激活的存在 (84, 33)，最先进的 LLM 量化方法通常使用每组量化来保持高精度。它将原始激活和权重张量划分为多个具有独立量化尺度的组，以避免异常值对其他组的影响。然而，我们的调查表明，移动 NPU 无法直接执行每组 MatMul（表 2）。相反，他们必须将 MatMul 拆分为多个组大小的子张量 MatMul，然后使用浮点求和运算来减少子张量中间结果。此过程会降低 NPU 效率并导致高达 10.7 倍的性能开销。

$\bullet$ Floating point (FP) operations cannot be eliminated.
Mobile NPUs generally provide significant integer-based MatMul acceleration but are weak at FP operations.
However, LLMs can be hardly quantized into integer-only execution with minimal accuracy loss.
Existing quantized LLMs still rely on float operators like LayerNorm and Attention.
Scheduling those FP operators out of NPU easily increases the inference critical path.

> **中文译文：**\bullet 浮点（FP）运算无法消除。移动 NPU 通常提供显着的基于整数的 MatMul 加速，但在 FP 运算方面较弱。然而，LLM 很难以最小的精度损失量化为纯整数执行。现有的量化 LLM 仍然依赖于像 LayerNorm 和 Attention 这样的浮点运算符。将这些 FP 算子调度到 NPU 之外很容易增加推理关键路径。

This work presents llm.npu, the first LLM inference system with efficient on-device NPU offloading.
The primary design goal of llm.npu is to reduce the prefill latency and energy consumption.
It targets the mainstream decoder-only transformer architecture of LLMs (e.g., LlaMA, GPT, etc).
The key idea is to maximize prefill execution on mobile NPUs to accelerate integer computation while keeping essential float operations on the CPU/GPU to maintain accuracy.
To overcome the aforementioned challenges and enhance NPU offloading efficiency, llm.npu re-constructs the prompt and model at three levels:
(1) At prompt level: llm.npu divides variable-length prompts into multiple fixed-sized chunks while maintaining data dependencies;
(2) At tensor level: llm.npu identifies and extracts significant outliers to run on the CPU/GPU;
(3) At block level: llm.npu schedules Transformer blocks to the CPU/GPU and NPU based on their hardware affinity and sensitivity to accuracy.
The corresponding novel techniques are detailed as follows:

> **中文译文：**这项工作提出了 llm.npu ，这是第一个具有高效设备上 NPU 卸载功能的 LLM 推理系统。 llm.npu 的主要设计目标是减少预填充延迟和能耗。它针对的是 LLM 的主流仅解码器 Transformer 架构（例如 LlaMA、GPT 等）。关键思想是最大限度地提高移动 NPU 上的预填充执行速度，以加速整数计算，同时在 CPU/GPU 上保留必要的浮点运算以保持准确性。为了克服上述挑战并提高NPU卸载效率，llm.npu在三个层面上重新构建了提示和模型：（1）在提示级别：llm.npu将可变长度提示划分为多个固定大小的块，同时保持数据依赖性； (2) 在张量级别：llm.npu 识别并提取显着异常值以在 CPU/GPU 上运行； (3) 在块级别：llm.npu 根据硬件亲和性和对精度的敏感度将 Transformer 块调度到 CPU/GPU 和 NPU。相应的新技术详述如下：

$\bullet$ Chunk-sharing graphs
($\lx@sectionsign$[3.2](#S3.SS2 "3.2. Chunk-sharing graph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs"))
llm.npu splits variable-length prompts into fixed-sized “chunks” with pre-built subgraphs, reducing graph preparation time.
Each variable-length prompt is then executed on those graphs with chunk-level causal dependency.
This approach leverages the insight that the token generation depends only on preceding tokens in the decoder-only LLMs.
However, loading multiple pre-built chunk graphs simultaneously incurs significant memory overhead, e.g., 2–4$\times$ more than the LLM weights.
To address this, llm.npu further identifies operators that are irrelevant to prompt size (e.g., FFN) and shares them across each chunk execution, reducing the memory overhead by up to 4$\times$.

> **中文译文：**\bullet 块共享图 ( \lx@sectionsign 3.2 ) llm.npu 将可变长度提示拆分为带有预构建子图的固定大小的“块”，从而减少图形准备时间。然后，在具有块级因果依赖性的那些图上执行每个可变长度提示。这种方法利用了令牌生成仅依赖于仅解码器 LLM 中的先前令牌的见解。然而，同时加载多个预构建的块图会产生显着的内存开销，例如，比 LLM 权重多 2-4 倍。为了解决这个问题，llm.npu 进一步识别与提示大小无关的运算符（例如 FFN），并在每个块执行中共享它们，从而将内存开销减少多达 4 倍。

$\bullet$ Shadow outlier execution
($\lx@sectionsign$[3.3](#S3.SS3 "3.3. Shadow outlier execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs"))
addresses the activation outlier problem without compromising NPU efficiency.
Without changing per-tensor MatMul on NPU, llm.npu extracts the activation outlier channels into a compact tensor and executes it on the CPU/GPU in parallel, since outliers are very sparse (0.1%–0.3% of total channels).
This design further raises issues of increased memory footprint due to duplicating MatMul weights in CPU memory and synchronization overhead between CPU/GPU and NPU.
Therefore, based on the observation that outliers are more likely to appear at a small set of channel positions, llm.npu optimizes memory usage by only keeping those “hot channels” weights in memory, and retrieves others from disk on demand.
llm.npu also prunes unimportant outliers at layer-level by measuring outliers’ importance to reduce the synchronization overhead.

> **中文译文：**\bullet Shadow 离群值执行 ( \lx@sectionsign 3.3 ) 在不影响 NPU 效率的情况下解决了激活离群值问题。在不更改 NPU 上每个张量 MatMul 的情况下，llm.npu 将激活离群值通道提取到紧凑张量中，并在 CPU/GPU 上并行执行，因为离群值非常稀疏（占总通道的 0.1%–0.3%）。由于 CPU 内存中重复的 MatMul 权重以及 CPU/GPU 和 NPU 之间的同步开销，该设计进一步引发了内存占用增加的问题。因此，基于异常值更有可能出现在一小部分通道位置的观察结果，llm.npu 通过仅将那些“热通道”权重保留在内存中来优化内存使用，并根据需要从磁盘检索其他权重。 llm.npu 还通过测量异常值的重要性来在层级别修剪不重要的异常值，以减少同步开销。

$\bullet$ Out-of-order subgraph execution
($\lx@sectionsign$[3.4](#S3.SS4 "3.4. Out-of-order subgraph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs"))
llm.npu is guided by a key insight that multiple subgraphs can be scheduled in an out-of-order manner, without strictly following the chunk sequence in the original prompt.
This significantly enlarges the scheduling space of llm.npu to minimize the execution bubbles brought by CPU/GPU float operation.
Given that finding the optimal out-of-order execution order is an NP-hard problem, llm.npu employs a microsecond-level online scheduling algorithm. This algorithm is based on the observation that the workload of the NPU is heavier and constitutes the critical path. Therefore, when selecting which subgraph to execute, llm.npu prioritizes those subgraphs having a more significant impact on reducing NPU stalls, rather than focusing solely on the execution latency of the subgraph.
Notably, llm.npu’s scheduling algorithm does not maximize parallel processing capability. Instead, llm.npu aims to maximize the utilization of NPUs while minimizing the impact of CPU/GPU workloads.

> **中文译文：**\bullet 无序子图执行 ( \lx@sectionsign 3.4 ) llm.npu 以一个关键见解为指导，即可以以无序方式安排多个子图，而无需严格遵循原始提示中的块顺序。这显着扩大了llm.npu的调度空间，以最大限度地减少CPU/GPU浮点运算带来的执行气泡。鉴于寻找最优乱序执行顺序是一个NP难题，llm.npu采用了微秒级在线调度算法。该算法基于观察到NPU的工作负载较重并构成关键路径。因此，在选择执行哪个子图时，llm.npu会优先考虑那些对减少NPU停顿影响更显着的子图，而不是仅仅关注子图的执行延迟。值得注意的是，llm.npu的调度算法并没有最大化并行处理能力。相反，llm.npu 的目标是最大限度地提高 NPU 的利用率，同时最大限度地减少 CPU/GPU 工作负载的影响。

Implementation and evaluations.
We have implemented llm.npu on top of MLLM ([23](#bib.bib23)) and QNN ([25](#bib.bib25)),
with 10K lines of C/C++ and assembly code.
We evaluated llm.npu with five mobile-sized LLMs (Qwen1.5-1.8B ([27](#bib.bib27)), Gemma-2B ([9](#bib.bib9)), phi2-2.7B ([16](#bib.bib16)), LlaMA-2-7B ([11](#bib.bib11)), and Mistrial-7B ([14](#bib.bib14))), four LLM benchmarks,
and two mobile devices (Xiaomi 14 and Redmi K60 Pro).
We compared llm.npu with five competitive baselines, including three industrial open-source engines (llama.cpp ([59](#bib.bib59)), TFLite ([57](#bib.bib57)), MNN ([49](#bib.bib49))) and two state-of-the-art research prototypes (MLC-LLM ([75](#bib.bib75)) and PowerInfer-v2 ([94](#bib.bib94))).
The experiments show that llm.npu significantly and consistently outperforms all baselines in terms of prefill latency and energy consumption while preserving inference accuracy (<1% loss compared to FP16).
It is 7.3$\times$–18.4$\times$ faster than baselines on CPU, and 1.3$\times$–43.6$\times$ on GPU with a prompt length of 1024.
It also achieves a 1.9$\times$–59.5$\times$ energy reduction.
To our best knowledge, llm.npu is the first system that achieves $>$1000 tokens/sec of prefill speed on COTS mobile devices for billion-sized LLMs.
In end-to-end real-world applications, llm.npu reduces the inference latency (prefill+decode) by 1.4$\times$–32.8$\times$ compared to the baselines.

> **中文译文：**实施和评估。我们在 MLLM (23) 和 QNN (25) 之上实现了 llm.npu，其中包含 10K 行 C/C++ 和汇编代码。我们使用五个移动大小的 LLM（Qwen1.5-1.8B (27)、Gemma-2B (9)、phi2-2.7B (16)、LlaMA-2-7B (11) 和 MisTrial-7B (14)）、四个 LLM 基准测试和两个移动设备（小米 14 和 Redmi K60）评估了 llm.npu亲）。我们将 llm.npu 与五个竞争基准进行了比较，包括三个工业开源引擎（llama.cpp (59)、TFLite (57)、MNN (49)）和两个最先进的研究原型（MLC-LLM (75) 和 PowerInfer-v2 (94)）。实验表明，llm.npu 在预填充延迟和能耗方面始终显着优于所有基线，同时保持推理精度（与 FP16 相比损失 <1%）。它比 CPU 上的基准快 7.3 倍 –18.4 倍，在 GPU 上比基准快 1.3 倍 –43.6 倍，提示长度为 1024。它还实现了 1.9 倍 –59.5 倍的能耗降低。据我们所知，llm.npu 是第一个在 COTS 移动设备上为数十亿规模的 LLM 实现超过 1000 个令牌/秒预填充速度的系统。在端到端的实际应用中，llm.npu 与基线相比，将推理延迟（预填充+解码）减少了 1.4 倍 – 32.8 倍。

Contributions are summarized as follows:

> **中文译文：**贡献总结如下：

- •

  We thoroughly investigate the challenges and opportunities of accelerating LLM prefilling with mobile NPUs.

  > **中文译文：**我们深入研究了使用移动 NPU 加速 LLM 预填充的挑战和机遇。
- •

  We present the first LLM inference engine with efficient mobile NPU offloading, featuring three novel techniques: chunk-sharing graph, shadow outlier execution, and out-of-order subgraph execution.

  > **中文译文：**我们提出了第一个具有高效移动 NPU 卸载功能的 LLM 推理引擎，具有三种新颖的技术：块共享图、影子异常值执行和乱序子图执行。
- •

  We perform comprehensive experiments on llm.npu that demonstrate its superior performance over competitive baselines. The code of llm.npu is available at <https://github.com/UbiquitousLearning/mllm>.

  > **中文译文：**我们在 llm.npu 上进行了全面的实验，证明了其优于竞争基线的性能。 llm.npu 的代码可在 https://github.com/UbiquitousLearning/mllm 获取。

## 2. Background

> **中文标题：**2. 背景

### 2.1. On-device LLM Inference Analysis

> **中文标题：**2.1.设备上 LLM 推理分析

On-device LLMs are increasingly used in cutting-edge scenarios such as Apple Intelligence ([2](#bib.bib2)), UI automation ([81](#bib.bib81)), and automated email reply ([20](#bib.bib20)), due to the enhanced privacy protections. To empower these applications, many lightweight LLMs have been developed, as summarized in Table [1](#S2.T1 "Table 1 ‣ 2.1. On-device LLM Inference Analysis ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs").
For instance, the Qwen1.5-1.8B model on the llama.cpp exhibits delays of 8.1 seconds for one-step UI automation, 21.7 seconds for automated email replies, and 8.4 seconds for chat summary on average using mobile CPUs, as evaluated on the DroidTask ([82](#bib.bib82), [22](#bib.bib22)), LongBench datasets ([28](#bib.bib28)) and Persona-Chat ([48](#bib.bib48)), which is impractical for real-world deployment.
However, their inference latency remains a significant challenge.

> **中文译文：**由于增强的隐私保护，设备上的 LLM 越来越多地应用于 Apple Intelligence (2)、UI 自动化 (81) 和自动电子邮件回复 (20) 等前沿场景。为了支持这些应用程序，已经开发了许多轻量级法学硕士，如表 1 所示。例如，根据 DroidTask ( 82 , 22 ) 、 LongBench 数据集 ( 28 ) 和 Persona-Chat ( 48 ) 的评估，llama.cpp 上的 Qwen1.5-1.8B 模型在使用移动 CPU 时，单步 UI 自动化延迟为 8.1 秒，自动电子邮件回复延迟为 21.7 秒，聊天摘要延迟为 8.4 秒。对于现实世界的部署来说不切实际。然而，它们的推理延迟仍然是一个重大挑战。

To explore this issue in detail, we break down the inference procedure discussed above.
The results are summarized in Figure [1](#S2.F1 "Figure 1 ‣ 2.1. On-device LLM Inference Analysis ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs").
Surprisingly, we find that compared to the decoding phase that has been extensively studied, the prefill stage is often the bottleneck in typical mobile applications. For example, the prefill stage constitutes 88.3%–98.8% of the total latency on a mobile CPU for UI task automation, chat summaries, and context-aware generation tasks. Even when using mobile GPUs, the prefill phase remains predominant, accounting for 54.2% to 91.7% of the total latency.
As the prompt length increases, the prefill stage’s proportion of the total inference time also rises.

> **中文译文：**为了详细探讨这个问题，我们分解了上面讨论的推理过程。结果总结于图 1 中。令人惊讶的是，我们发现与已广泛研究的解码阶段相比，预填充阶段通常是典型移动应用中的瓶颈。例如，预填充阶段占 UI 任务自动化、聊天摘要和上下文感知生成任务的移动 CPU 总延迟的 88.3%–98.8%。即使使用移动 GPU，预填充阶段仍然占主导地位，占总延迟的 54.2% 到 91.7%。随着提示长度的增加，预填充阶段占总推理时间的比例也会增加。

Several factors contribute to this situation:
(1) Mobile CPUs/GPUs lack the parallelism capabilities of cloud GPUs ([97](#bib.bib97), [40](#bib.bib40)), being primarily designed for handling application logic or rendering tasks.
(2) Mobile LLM tasks often require long prompts for personalized, context-aware generation. For instance, automated email reply often requires extensive user data in prompt, such as historical emails, schedules, and location information, amounting to 1,168–1,835 tokens in the LongBench dataset ([28](#bib.bib28)). Similarly, on-device LLMs engaged in UI automation must process a substantial volume of UI annotation tokens (XML or HTML) and user commands, ranging from 505–827 tokens in the DroidTask dataset ([82](#bib.bib82), [22](#bib.bib22)). Yet, the output lengths for these applications are typically short, averaging 7.9 tokens for the LongBench dataset and 3.5 tokens for the DroidTask dataset. In the context of chat summaries, such as the Persona-Chat dataset ([48](#bib.bib48)), the prompt and output lengths are relatively balanced, with input sizes between 488–584 tokens and an average output of 44 tokens.
(3) Mobile LLMs now support long context windows.
For instance, recent models like Qwen2-1.5B can accommodate context windows of up to 32K tokens, as illustrated in Table [1](#S2.T1 "Table 1 ‣ 2.1. On-device LLM Inference Analysis ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs").

> **中文译文：**有几个因素导致了这种情况：（1）移动CPU/GPU缺乏云GPU（97, 40）的并行能力，主要是为处理应用程序逻辑或渲染任务而设计的。 (2) 移动法学硕士任务通常需要很长的提示才能进行个性化、上下文感知的生成。例如，自动电子邮件回复通常需要大量的提示用户数据，例如历史电子邮件、日程安排和位置信息，在 LongBench 数据集中总计有 1,168–1,835 个令牌 (28)。同样，从事 UI 自动化的设备上法学硕士必须处理大量 UI 注释标记（XML 或 HTML）和用户命令，范围从 DroidTask 数据集中的 505-827 个标记（82, 22）。然而，这些应用程序的输出长度通常很短，LongBench 数据集平均为 7.9 个标记，DroidTask 数据集平均为 3.5 个标记。在聊天摘要的上下文中，例如 Persona-Chat 数据集（48），提示和输出长度相对平衡，输入大小在 488-584 个令牌之间，平均输出为 44 个令牌。 (3) 移动法学硕士现在支持长上下文窗口。例如，Qwen2-1.5B 等最新模型可以容纳最多 32K 个标记的上下文窗口，如表 1 所示。

Table 1. The max context length for mobile-sized LLMs.

> **中文图注：**表 1.移动大小的 LLM 的最大上下文长度。

| Model | Max Context | Year | Model | Max Context | Year |
| --- | --- | --- | --- | --- | --- |
| Opt-1.3B | 2K | 2022.5 | TinyLLaMA-1.1B | 2K | 2023.9 |
| StableLLM-3B | 4K | 2023.10 | phi-2-2.7B | 2K | 2023.12 |
| Gemma-2B | 8K | 2024.2 | Qwen1.5-1.8B | 32K | 2024.2 |
| Phi3-mini-3.8B | 128K | 2024.5 | Qwen2-1.5B | 32K | 2024.6 |


(a) CPU results

> **中文图注：**(a)CPU结果

(b) GPU results

> **中文图注：**(b) GPU 结果

Figure 1. Breakdown of end-to-end inference latency for UI automation, context-aware QA, and chat summaries. CPU evaluation uses llama.cpp as the on-device inference engine, while GPU evaluation uses TFLite as the simulation engine.

> **中文图注：**图 1. UI 自动化、上下文感知 QA 和聊天摘要的端到端推理延迟细分。 CPU 评估使用 llama.cpp 作为设备端推理引擎，GPU 评估使用 TFLite 作为模拟引擎。

### 2.2. Opportunity: Mobile NPUs

> **中文标题：**2.2.机会：移动 NPU

To optimize prefill latency, llm.npu leverages a key opportunity: modern mobile SoCs ubiquitously include mobile neural processing units (NPUs) that are well-suited for integer operations, such as INT8-based matrix multiplication. Table [2](#S2.T2 "Table 2 ‣ 2.2. Opportunity: Mobile NPUs ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs") summarizes the specifications of well-known mobile NPUs provided by mainstream vendors. For example, Qualcomm’s mobile SoCs feature Hexagon NPUs, achieving up to 73 trillion INT8 operations per second. According to AI-Benchmark ([18](#bib.bib18)), the Hexagon NPU in the Xiaomi 14 can infer the MobileNet-V2 model in just 0.6 ms, 23$\times$ faster than a mobile CPU and 3.2$\times$ faster than a mobile GPU.

> **中文译文：**为了优化预填充延迟，llm.npu 利用了一个关键机会：现代移动 SoC 普遍包含非常适合整数运算（例如基于 INT8 的矩阵乘法）的移动神经处理单元 (NPU)。表2总结了主流厂商提供的知名移动NPU的规格。例如，高通的移动 SoC 采用 Hexagon NPU，每秒可实现高达 73 万亿次 INT8 运算。根据 AI-Benchmark（18）的数据，小米 14 中的 Hexagon NPU 只需 0.6 毫秒即可推理出 MobileNet-V2 模型，比移动 CPU 快 23 倍，比移动 GPU 快 3.2 倍。

Table 2. Specifications of well-known mobile NPUs provided by mainstream vendors.

> **中文图注：**表 2.主流厂商提供的知名移动NPU规格。

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Vendor | Latest NPU | SDK | Open | Group | INT8 Perf. |
| Qualcomm | Hexagon NPU ([17](#bib.bib17)) | QNN ([25](#bib.bib25)) | $\times$ | $\times$ | 73 TOPS |
| Google | Edge TPU ([19](#bib.bib19)) | Edge TPU API ([8](#bib.bib8)) | $\times$ | $\times$ | 4 TOPS |
| MediaTek | MediaTek APU 790 ([12](#bib.bib12)) | NeuroPilot ([15](#bib.bib15)) | $\times$ | N/A | 60 TOPS |
| Huawei | Ascend NPU ([7](#bib.bib7)) | HiAI ([10](#bib.bib10)) | $\times$ | $\times$ | 16 TOPS |
| |  | | --- | | "Open": Open-source?; "Group": Support per-group quantization MatMul? "N/A": No available | | documents for public; "INT8 Perf.": Int8 performance. | | | | | | |

Mobile NPU architecture. Mobile NPUs deliver significant performance benefits by single instruction multiple data (SIMD) architecture.
For instance, Hexagon NPUs support 1024-bit INT8 vector arithmetic, allowing multiple SIMD instructions to execute in parallel. However, their floating-point computation capabilities are relatively weak compared to mobile GPUs. With clock frequencies between 500 and 750 MHz, mobile NPUs are more energy-efficient than mobile CPUs and GPUs. Additionally, unlike cloud GPUs that have separate physical memory, mobile NPUs are integrated within mobile SoCs, sharing the same physical memory with mobile CPUs, eliminating the need for memory copying during NPU execution.

> **中文译文：**移动NPU架构。移动 NPU 通过单指令多数据 (SIMD) 架构提供显着的性能优势。例如，Hexagon NPU 支持 1024 位 INT8 向量运算，允许多个 SIMD 指令并行执行。然而，与移动 GPU 相比，它们的浮点计算能力相对较弱。移动 NPU 的时钟频率在 500 至 750 MHz 之间，比移动 CPU 和 GPU 更节能。此外，与拥有独立物理内存的云GPU不同，移动NPU集成在移动SoC中，与移动CPU共享相同的物理内存，从而无需在NPU执行期间进行内存复制。

Table 3. Execution latency (ms) for various MatMul sizes on Redmi K70 Pro.

> **中文图注：**表 3. Redmi K70 Pro 上各种 MatMul 大小的执行延迟（毫秒）。

| Matrix A | Matrix B | NPU INT8 | CPU INT8 | GPU FP16 | NPU FP16 |
| --- | --- | --- | --- | --- | --- |
| 64\*2048 | 2048\*2048 | 0.9 | 4.2 (4.6$\times$) | 1.7 (1.9$\times$) | 252 (193$\times$) |
| 64\*2048 | 2048\*8192 | 1.5 | 6.8 (4.5$\times$) | 4.8 (3.2$\times$) | 986 (657$\times$) |
| 64\*2048 | 2048\*11008 | 2.0 | 11.6 (5.8$\times$) | 6.9 (3.5$\times$) | 1207 (603$\times$) |
| 32\*4096 | 4096\*4096 | 1.7 | 7.5 (4.4$\times$) | 3.1 (1.8$\times$) | 1054 (620$\times$) |
| 32\*4096 | 4096\*8192 | 2.9 | 13.1 (4.5$\times$) | 7.7 (2.6$\times$) | 2009 (692$\times$) |
| 32\*4096 | 4096\*11008 | 4.1 | 19.6 (4.8$\times$) | 10.4 (2.5$\times$) | 3112 (759$\times$) |

Micro benchmark comparison.
To evaluate the performance of INT8 MatMul on mobile NPUs, we conducted preliminary experiments on the Redmi K70 Pro using MatMul sizes commonly used in mobile LLMs, as summarized in Table [3](#S2.T3 "Table 3 ‣ 2.2. Opportunity: Mobile NPUs ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs"). INT8 MatMul on mobile NPUs achieved a 4.5–5.8$\times$ speedup compared to CPU INT8 and a 1.8–3.5$\times$ improvement over GPU FP16. The performance gains increase with larger computational workloads. However, performing FP16 MatMul on the mobile NPU resulted in performance up to 159$\times$ slower than CPU INT8. These results align with the INT8 SIMD architecture of mobile NPUs, confirming that mobile NPUs are best suited for accelerating INT8 matrix multiplication.

> **中文译文：**微观基准比较。为了评估 INT8 MatMul 在移动 NPU 上的性能，我们使用移动 LLM 中常用的 MatMul 尺寸在 Redmi K70 Pro 上进行了初步实验，如表 3 所示。与 CPU INT8 相比，移动 NPU 上的 INT8 MatMul 实现了 4.5–5.8 倍的加速，比 GPU FP16 实现了 1.8–3.5 倍的加速。计算工作量越大，性能提升就越明显。然而，在移动 NPU 上执行 FP16 MatMul 导致性能比 CPU INT8 慢 159 倍。这些结果与移动 NPU 的 INT8 SIMD 架构一致，证实移动 NPU 最适合加速 INT8 矩阵乘法。

![Refer to caption](%image_url%/fast-on-device-llm-inference-with-npus/figure_02.png)


Figure 2. The workflow of executing DNNs on mobile NPUs, with latencies for each procedure on QNN ([25](#bib.bib25)).

> **中文图注：**图2.在移动 NPU 上执行 DNN 的工作流程，以及 QNN 上每个过程的延迟 (25)。

DNN execution workflow on mobile NPUs.
Executing DNNs on mobile NPUs involves configuring the NPU environment, creating the compute graph, optimizing the graph, executing the graph, and freeing the graph, as shown in Figure [2](#S2.F2 "Figure 2 ‣ 2.2. Opportunity: Mobile NPUs ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs").
Typically, creating and optimizing the compute graph are most time-consuming, where the former includes translating models into the NPU-required intermediate representation, and memory allocation, that takes 300–500ms, while the latter includes optimization for adjusting memory layout, execution order, and operator fusion, taking many seconds.
In addition, the closed-source nature of NPU SDKs limits further adaptation for LLMs.

> **中文译文：**移动 NPU 上的 DNN 执行工作流程。在移动NPU上执行DNN涉及配置NPU环境、创建计算图、优化图、执行图和释放图，如图2所示。通常，创建和优化计算图是最耗时的，前者包括将模型转换为 NPU 所需的中间表示和内存分配，需要 300-500 毫秒，而后者包括调整内存布局、执行顺序和算子融合的优化，需要很多秒。此外，NPU SDK 的闭源性质限制了 LLM 的进一步适配。

### 2.3. Gaps between LLMs and Mobile NPUs

> **中文标题：**2.3. LLM 和移动 NPU 之间的差距

Given its inherent advantages, we are surprised to find that none of existing DNN engines support LLM acceleration on mobile NPUs.
We then dig into the underlying reasons and find a huge gap between existing mobile NPUs design and LLM inference pipeline.

> **中文译文：**鉴于其固有的优势，我们惊讶地发现现有的 DNN 引擎都不支持移动 NPU 上的 LLM 加速。然后，我们深入研究根本原因，发现现有移动 NPU 设计与 LLM 推理流程之间存在巨大差距。

$\bullet$ LLM prefill phase relies on variable-length prompts, leading to excessive time spent on building and compiling the NPU graph.
As illustrated in Figure [2](#S2.F2 "Figure 2 ‣ 2.2. Opportunity: Mobile NPUs ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs"), before the compute graph can be executed on the mobile NPU, it must be built and optimized, a process taking tens of seconds. For instance, building the graph for the Gemma 2B model using QNN framework takes 360 ms, and graph optimization requires 11.54 seconds. Unlike CNN models, which are built and optimized once and can be executed multiple times with the same input shape, the LLM prefill phase must handle variable-length prompts, necessitating rebuilding and re-optimization of compute graphs for each inference. Consequently, using mobile NPUs in this scenario offers no performance benefit and is often slower than using a CPU.

> **中文译文：**\bullet LLM 预填充阶段依赖于可变长度的提示，导致在构建和编译 NPU 图上花费过多的时间。如图 2 所示，在移动 NPU 上执行计算图之前，必须对其进行构建和优化，该过程需要数十秒。例如，使用 QNN 框架为 Gemma 2B 模型构建图需要 360 毫秒，图优化需要 11.54 秒。与 CNN 模型不同，CNN 模型构建和优化一次，并且可以使用相同的输入形状多次执行，LLM 预填充阶段必须处理可变长度的提示，从而需要为每个推理重建和重新优化计算图。因此，在这种情况下使用移动 NPU 不会带来任何性能优势，而且速度通常比使用 CPU 慢。

Figure 3. Per-tensor quantization MatMul and per-group quantization MatMul. $seq$, $hds$, $group$ represent sequence length, hidden size, and group number, respectively.

> **中文图注：**图3.每张量量化 MatMul 和每组量化 MatMul。 seq 、 hds 、 group 分别表示序列长度、隐藏大小和组数。


Figure 4. The prefill latency and accuracy on HelloSwag datasets among different quantization algorithms atop Xiaomi 14 using Qualcomm QNN framework.

> **中文图注：**图4.使用 Qualcomm QNN 框架的小米 14 上不同量化算法在 HelloSwag 数据集上的预填充延迟和准确性。

$\bullet$ The existence of activation outliers makes LLM difficult to quantize at whole-tensor level, yet a more fine-grained group-level quantization hampers NPU efficiency.
Our preliminary experiments, shown in Figure [4](#S2.F4 "Figure 4 ‣ 2.3. Gaps between LLMs and Mobile NPUs ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs"), indicate that two popular quantization algorithms (K-Quant ([59](#bib.bib59)) and AWQ ([56](#bib.bib56))) incur significant inference overhead by 8.1$\times$–10.7$\times$, as compared to per-tensor quantization (Figure [3](#S2.F3 "Figure 3 ‣ 2.3. Gaps between LLMs and Mobile NPUs ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs")(a)).
This is because algorithms like K-Quant and AWQ use fine-grained per-group quantization (Figure [3](#S2.F3 "Figure 3 ‣ 2.3. Gaps between LLMs and Mobile NPUs ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs")(b)) to maintain high accuracy. These algorithms divide activations and weights into multiple groups, each with an independent quantization scale. On NPUs, this approach requires dividing the MatMul operation into several sub-tensor MatMuls, which fails to fully leverage the capabilities of mobile NPUs. Additionally, it necessitates aggregating intermediate results with floating-point additions, resulting in extra floating-point computations. The exception is SmoothQuant ([84](#bib.bib84)), which uses per-tensor quantization but suffers from significant accuracy loss, such as a 3.9% and 8.4% drop on the HelloSwag dataset for the LlaMA-2-7B and Qwen1.5-1.8B model, respectively.

> **中文译文：**\bullet 激活异常值的存在使得 LLM 难以在整个张量级别上进行量化，而更细粒度的组级别量化会阻碍 NPU 效率。我们的初步实验如图 4 所示，表明两种流行的量化算法（K-Quant ( 59 ) 和 AWQ ( 56 ) ）与每张量量化相比（图 3 (a)）会产生 8.1 × –10.7 × 的显着推理开销。这是因为 K-Quant 和 AWQ 等算法使用细粒度的每组量化（图 3 (b)）来保持高精度。这些算法将激活和权重分为多个组，每个组都有独立的量化尺度。在NPU上，这种方法需要将MatMul操作划分为多个子张量MatMul，这无法充分利用移动NPU的能力。此外，它需要通过浮点加法来聚合中间结果，从而导致额外的浮点计算。 SmoothQuant (84) 是个例外，它使用每张量量化，但精度损失显着，例如 LlaMA-2-7B 和 Qwen1.5-1.8B 模型的 HelloSwag 数据集分别下降了 3.9% 和 8.4%。

![Refer to caption](%image_url%/fast-on-device-llm-inference-with-npus/figure_03.png)


Figure 5. The workflow of quantized on-device LLM inference. Operations shown in blue are computed using INT4/INT8 formats, while those in orange are computed using float data formats.

> **中文图注：**图5.量化设备上 LLM 推理的工作流程。蓝色显示的运算使用 INT4/INT8 格式计算，而橙色显示的运算使用浮点数据格式计算。


Table 4. The operator data formats in the state-of-the-art quantization inference approaches. Each module in the first row is illustrated in Figure [5](#S2.F5 "Figure 5 ‣ 2.3. Gaps between LLMs and Mobile NPUs ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs").

> **中文图注：**表 4.最先进的量化推理方法中的运算符数据格式。第一行中的每个模块如图 5 所示。

| Quantization | Type | Acc. | Cal QKV | Atten. | Cal O | Norm. | FFN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K-Quant ([59](#bib.bib59)) | Per-Group | Low | INT8 | FP16 | INT8 | FP16 | INT8 |
| GPTQ ([35](#bib.bib35)) | Per-Group | High | FP16 | FP16 | FP16 | FP16 | FP16 |
| AWQ ([56](#bib.bib56)) | Per-Group | High | FP16 | FP16 | FP16 | FP16 | FP16 |
| SmoothQuant ([84](#bib.bib84)) | Per-tensor | Low | INT8 | FP16 | INT8 | FP16 | INT8 |
| "Atten.":Attention; "Norm.": Normalization. | | | | | | | |

$\bullet$ On-device LLM inference relies on floating-point operations, conflicting with the NPU’s design for INT8 acceleration.
Figure [5](#S2.F5 "Figure 5 ‣ 2.3. Gaps between LLMs and Mobile NPUs ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs") illustrates a typical workflow for quantized inference.
To ensure inference accuracy, only linear layers (highlighted in blue) perform matrix multiplication in the INT8/INT4 data format. For these layers, the activation $x$ is quantized to INT8/INT4 before performing the dot product with weights. Other operations, such as Attention and LayerNorm (highlighted in orange), are computed in floating-point format.
Table [4](#S2.T4 "Table 4 ‣ 2.3. Gaps between LLMs and Mobile NPUs ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs") further summarize the operator data formats in state-of-the-art quantization inference algorithms. All of them depend on float Attention and normalization operations. Given that float operations dramatically degrade performance on mobile NPUs, these operations cannot be efficiently executed on the NPU with significant overhead.

> **中文译文：**\bullet 设备上 LLM 推理依赖于浮点运算，与 NPU 的 INT8 加速设计相冲突。图 5 展示了量化推理的典型工作流程。为了确保推理精度，只有线性层（以蓝色突出显示）以 INT8/INT4 数据格式执行矩阵乘法。对于这些层，在执行与权重的点积之前，激活 x 被量化为 INT8/INT4。其他操作，例如 Attention 和 LayerNorm（以橙色突出显示），以浮点格式计算。表 4 进一步总结了最先进的量化推理算法中的运算符数据格式。所有这些都依赖于浮动注意力和标准化操作。鉴于浮点运算会极大地降低移动 NPU 的性能，因此这些运算无法在 NPU 上高效执行，且开销较大。

## 3. llm.npu Design

> **中文标题：**3.llm.npu设计

![Refer to caption](%image_url%/fast-on-device-llm-inference-with-npus/figure_04.png)


Figure 6. The workflow of llm.npu.

> **中文图注：**图6. llm.npu 的工作流程。

### 3.1. Overview of llm.npu

> **中文标题：**3.1. llm.npu 概述

Design goal.
llm.npu aims to reduce prefill latency and energy consumption for mobile-sized LLMs through on-device NPU offloading. It supports various mobile-sized LLMs on devices and can be integrated as part of LLM-as-a-System-Service in mobile OS or mobile application services ([99](#bib.bib99), [102](#bib.bib102)).

> **中文译文：**设计目标。 llm.npu 旨在通过设备上的 NPU 卸载来减少移动规模 LLM 的预填充延迟和能耗。它支持设备上的各种移动大小的LLM，并且可以作为LLM即系统服务的一部分集成到移动操作系统或移动应用程序服务中（99、102）。

Workflow.
Figure [6](#S3.F6 "Figure 6 ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs") illustrates the workflow of llm.npu.
The key idea of llm.npu is to maximize its execution on mobile NPU for integer operation acceleration; while keep necessary floating point operations on CPU/GPU to not compromise accuracy.
To enable more efficient NPU offloading, llm.npu re-constructs the prompt and model in following ways:
(1) At prompt level: variable-length prompt is reduced into multiple fixed-sized chunks with data dependency preserved;
(2) At block level: Transformer block is scheduled into CPU/GPU and NPU based on their hardware infinity and accuracy sensitivity;
(3) At tensor level: important outliers are identified and extracted to run on CPU/GPU.

> **中文译文：**工作流程。图 6 说明了 llm.npu 的工作流程。 llm.npu的核心思想是最大化其在移动NPU上的执行，以实现整数运算加速；同时在 CPU/GPU 上保留必要的浮点运算，以免影响精度。为了实现更高效的 NPU 卸载，llm.npu 通过以下方式重新构建提示和模型：（1）在提示级别：将可变长度提示缩减为多个固定大小的块，并保留数据依赖性； （2）块级：Transformer块根据硬件无限性和精度敏感度被调度到CPU/GPU和NPU中； (3) 在张量级别：识别并提取重要的异常值以在CPU/GPU上运行。

$\bullet$ Preparation stage.
llm.npu first uses an enhanced per-tensor quantization algorithm to quantize LLMs into W8A8 format.
The quantization algorithm differs from existing ones as it filters out most unimportant activation outliers and extracts the rest of them into independent, lightweight operators that are complementary to the original one.
llm.npu also generates fixed-length chunk-sharing graphs ($\lx@sectionsign$[3.2](#S3.SS2 "3.2. Chunk-sharing graph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs")) to efficiently handle variable-length prompts.

> **中文译文：**\项目符号准备阶段。 llm.npu 首先使用增强的每张量量化算法将 LLM 量化为 W8A8 格式。量化算法与现有算法不同，因为它过滤掉最不重要的激活异常值，并将其余的提取到独立的、轻量级的算子中，与原始算子互补。 llm.npu 还生成固定长度的块共享图 ( \lx@sectionsign 3.2 ) 以有效处理可变长度的提示。

$\bullet$ Execution stage.
When receiving a prompt, llm.npu divides it into fixed-sized chunks and processes them causally.
These chunk graphs will be split into subgraphs scheduled onto the CPU/GPU and NPU according to their data formats for efficient execution.
To preserve accuracy, certain INT8-based linear layers undergo sparse float outlier shadow execution on the CPU/GPU in parallel to compensate for quantization errors from outliers ($\lx@sectionsign$[3.3](#S3.SS3 "3.3. Shadow outlier execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs")).
To enhance execution efficiency, llm.npu judiciously schedules the chunks in out-of-order manner ($\lx@sectionsign$[3.4](#S3.SS4 "3.4. Out-of-order subgraph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs")).

> **中文译文：**\bullet 执行阶段。当收到提示时，llm.npu 将其分为固定大小的块并按因果关系进行处理。这些块图将根据数据格式分成调度到 CPU/GPU 和 NPU 上的子图，以实现高效执行。为了保持准确性，某些基于 INT8 的线性层在 CPU/GPU 上并行进行稀疏浮点离群值阴影执行，以补偿离群值的量化误差 ( \lx@sectionsign 3.3 )。为了提高执行效率，llm.npu 明智地以无序方式调度块（ \lx@sectionsign 3.4 ）。

### 3.2. Chunk-sharing graph execution

> **中文标题：**3.2.块共享图执行

To tackle the dynamic prompt length challenge, an intuitive solution is to set a fixed length compute graph ahead and use padding ([77](#bib.bib77), [67](#bib.bib67), [3](#bib.bib3)).
However, this method lacks flexibility and excessive padding wastes compute resources.

> **中文译文：**为了解决动态提示长度的挑战，一个直观的解决方案是预先设置固定长度的计算图并使用填充 (77, 67, 3) 。然而，这种方法缺乏灵活性，并且过多的填充会浪费计算资源。

Figure 7. The illustration of prompt graph, chunk graph and chunk-sharing graph. The chunk length is 32.

> **中文图注：**图 7.提示图、块图和块共享图的图示。块长度为 32。

Chunk-wise prefill.
To enhance flexibility and minimize padding for variable-length prompts, we recognize that processing a long prompt in a LLM is equivalent to processing several split sub-prompts, or “chunks”, causally.
This is feasible because popular LLMs use a decoder-only architecture, where the result of the $i$-th token depends only on the preceding tokens.
To that end, llm.npu first pre-builds and pre-optimizes fixed-length chunk-based NPU compute graph at the preparation stage.
During inference, llm.npu splits long prompts into several chunks and processes them using these pre-built chunk graphs, as illustrated in Figure [7](#S3.F7 "Figure 7 ‣ 3.2. Chunk-sharing graph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs")(b).

> **中文译文：**分块预填充。为了增强灵活性并最大限度地减少可变长度提示的填充，我们认识到在 LLM 中处理长提示相当于按因果关系处理多个拆分子提示或“块”。这是可行的，因为流行的 LLM 使用仅解码器架构，其中第 i 个标记的结果仅取决于前面的标记。为此，llm.npu 首先在准备阶段预构建和预优化基于固定长度块的 NPU 计算图。在推理过程中，llm.npu 将长提示分割成几个块，并使用这些预先构建的块图对其进行处理，如图 7 (b) 所示。

However, solely using chunk graphs is not scalable, as llm.npu would need to store numerous distinct chunk graphs in memory, significantly increasing memory overhead. This is because different chunk graphs have attention operators of varying sizes.
For instance, considering a prompt length of 1024 and a chunk length of 32, the QKV dimension sizes of the attention operators for the first chunk are all $32\*hds$, while for the last chunk they are $32\*hds$, $1024\*hds$, and $1024\*hds$ respectively, as shown in Figure [7](#S3.F7 "Figure 7 ‣ 3.2. Chunk-sharing graph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs")(b).

> **中文译文：**然而，单独使用块图是不可扩展的，因为 llm.npu 需要在内存中存储大量不同的块图，从而显着增加内存开销。这是因为不同的块图具有不同大小的注意力运算符。例如，考虑到提示长度为 1024，块长度为 32，第一个块的注意力算子的 QKV 维度大小均为 32\*hds ，而最后一个块的 QKV 维度大小分别为 32\*hds 、 1024\*hds 和 1024\*hds ，如图 7 (b) 所示。

Chunk-sharing graph.
llm.npu introduces a chunk-sharing graph, shown in Figure [7](#S3.F7 "Figure 7 ‣ 3.2. Chunk-sharing graph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs")(c), based on the insight that LLM operators fall into two distinct categories: (1) static operators (in green), such as Linear and LayerNorm, which depend only on the chunk length and can be shared across different chunks; and (2) dynamic operators (in red), such as Attention, which depend on both chunk length and chunk sequence and cannot be shared among different chunks.
Consequently, llm.npu divides the LLM into several subgraphs based on the shareability of operators. The shared subgraphs are built and optimized once, whereas non-shared subgraphs are constructed individually for different chunks. During the prefill phase, activations from different chunks pass through the same static operator subgraphs while dynamically selecting the appropriate dimension-specific dynamic operators. This method significantly reduces memory overhead and enhances scalability, as most dynamic operators, like Attention, do not contain weights, requiring only activation buffers.

> **中文译文：**块共享图。 llm.npu 引入了一个块共享图，如图 7（c）所示，基于 LLM 运算符分为两个不同类别的见解：（1）静态运算符（绿色），例如 Linear 和 LayerNorm，它们仅依赖于块长度并且可以在不同块之间共享； （2）动态算子（红色），例如Attention，它依赖于块长度和块序列，并且不能在不同块之间共享。因此，llm.npu根据操作者的可共享性将LLM分为几个子图。共享子图构建并优化一次，而非共享子图针对不同块单独构建。在预填充阶段，来自不同块的激活通过相同的静态运算符子图，同时动态选择适当的特定于维度的动态运算符。这种方法显着减少了内存开销并增强了可扩展性，因为大多数动态运算符（例如注意力）不包含权重，只需要激活缓冲区。

Our experiments show that 120 out of 144 subgraphs can be shared in Qwen1.5-1.8B models, reducing memory consumption by up to 75% (7.2GB) for a prompt length as 1024 and a chunk length as 256.

> **中文译文：**我们的实验表明，144 个子图中的 120 个可以在 Qwen1.5-1.8B 模型中共享，对于提示长度为 1024 和块长度为 256 的情况，内存消耗最多可减少 75% (7.2GB)。

Figure 8. The per-token latency of QKV Linears and FFN under different chunk lengths.

> **中文图注：**图 8. QKV Linears 和 FFN 在不同块长度下的每个令牌延迟。

We also conducted extensive experiments on selecting a proper chunk length.
The results of two popular LLMs (Qwen1.5-1.8B and Gemma-2B) on Xiaomi 14 device is illustrated in Figure [8](#S3.F8 "Figure 8 ‣ 3.2. Chunk-sharing graph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs").
Based on the observations, llm.npu empirically chooses a chunk length of 256 for Xiaomi 14 device, which effectively utilizes the capabilities of mobile NPUs while reducing intra-chunk padding.
In practice, such profiling needs to be performed across different NPUs.

> **中文译文：**我们还进行了广泛的实验来选择合适的块长度。两种流行的LLM（Qwen1.5-1.8B和Gemma-2B）在小米14设备上的结果如图8所示。根据观察，llm.npu 经验性地为小米 14 设备选择了 256 的 chunk 长度，有效利用了移动 NPU 的能力，同时减少了 chunk 内的 padding。实际上，这种分析需要在不同的 NPU 上执行。

![Refer to caption](%image_url%/fast-on-device-llm-inference-with-npus/figure_05.png)


Figure 9. The workflow of shadow outlier execution.

> **中文图注：**图 9.影子异常值执行的工作流程。

### 3.3. Shadow outlier execution

> **中文标题：**3.3.影子异常值执行

To enable NPU-friendly, per-tensor activation quantization without compromising LLM accuracy, llm.npu adopts a novel approach termed shadow outlier execution.
As shown in Figure [9](#S3.F9 "Figure 9 ‣ 3.2. Chunk-sharing graph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs"), llm.npu extracts the activation channels with outliers at runtime into a more compact tensor, executes it on CPU, and merges it back to the outcome of original operator on NPU.
This procedure can be formulated as follows:

> **中文译文：**为了在不影响 LLM 精度的情况下实现 NPU 友好的每张量激活量化，llm.npu 采用了一种称为阴影异常值执行的新颖方法。如图9所示，llm.npu将运行时存在异常值的激活通道提取为更紧凑的张量，在CPU上执行，并在NPU上将其合并回原始算子的结果。该程序可以表述如下：

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| (1) |  | $$ \displaystyle\frac{x}{s}\odot w= $$ | $$ \displaystyle\left\{\min\left[\max\left(\frac{x}{s},-127\right),128\right]+\lfloor\frac{x}{s}/128\rfloor\times 128\right\}\odot w $$ |  |
|  | $$ \displaystyle= $$ | $$ \displaystyle\min\left[\max\left(\frac{x}{s},-127\right),128\right]\odot w\hskip 9.24994pt\hskip 9.24994pt\textbf{on NPU} $$ |  |
|  | $$ \displaystyle+ $$ | $$ \displaystyle extract\left(\lfloor\frac{x}{s}/128\rfloor\times 128\right)\odot w\hskip 9.24994pt\hskip 9.24994pt\;\textbf{on CPU} $$ |  |

where $x$, $w$, $s$, $\odot$, and $extract$ represent the original float activation, INT8 weights, the quantization scale factor, the MatMul operation, and the function of extracting activation outliers into a more compact tensor, respectively.
Specifically, the MatMul $\frac{x}{s}\odot w$ can be equivalently divided into the sum of two parts according to the associative law:
(1) Mobile NPU for MatMul within the scale. llm.npu first quantizes and rounds $x$ to the range of -127 to 128 based on the scale factor $s$. It then obtains intermediate results by performing a standard W8A8 per-tensor MatMul with weights $w$.
(2) Mobile CPUs/GPUs for MatMul beyond the scale. llm.npu calculates the partial values exceeding $s$. Since these outliers are rare, llm.npu extracts these values from the tensor, compresses them into a dense tensor, and performs a MatMul with weights $w$.

> **中文译文：**其中 x 、 w 、 s 、 \odot 和 extract 分别表示原始浮点激活、INT8 权重、量化比例因子、MatMul 操作以及将激活异常值提取到更紧凑的张量中的函数。具体来说，MatMul \frac{x}{s}\odot w 可以根据结合律等价地分为两部分之和： (1) 尺度内 MatMul 的移动 NPU。 llm.npu 首先根据比例因子 s 将 x 量化并舍入到 -127 到 128 的范围。然后，它通过使用权重 w 执行标准 W8A8 每个张量 MatMul 来获得中间结果。 (2) 超出规模的用于 MatMul 的移动 CPU/GPU。 llm.npu 计算超过 s 的部分值。由于这些异常值很少见，因此 llm.npu 从张量中提取这些值，将它们压缩为密集张量，并使用权重 w 执行 MatMul 。

Since outliers are very sparse (around 5–15 channels, accounting for only 0.1%–0.3% of total channels, as shown in Figure [10](#S3.F10 "Figure 10 ‣ 3.3. Shadow outlier execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs")), the shadow execution on CPU is much faster than the execution of original tensor on NPU, and its execution time can be totally hidden by overlapping.
To further minimize the overhead of this extra process, llm.npu determines an outlier threshold (i.e., $s$ in Equation 1) by profiling a large corpora at offline, thereby can identify the outliers by simply comparing the activation numbers to this threshold.
The design of shadow outlier execution is compatible with any per-tensor quantization algorithms, and the current prototype of llm.npu is based on the simple max-min symmetry quantization ([47](#bib.bib47)).

> **中文译文：**由于异常值非常稀疏（大约 5-15 个通道，仅占总通道的 0.1%-0.3%，如图 10 所示），因此 CPU 上的影子执行速度比 NPU 上原始张量的执行速度快得多，并且其执行时间可以通过重叠完全隐藏。为了进一步最小化这个额外过程的开销，llm.npu 通过离线分析大型语料库来确定异常值阈值（即等式 1 中的 s），从而可以通过简单地将激活数与该阈值进行比较来识别异常值。阴影异常值执行的设计与任何每张量量化算法兼容，并且 llm.npu 的当前原型基于简单的最大最小对称量化（47）。

While the shadow outlier execution seems to have well balanced the NPU affinity and LLM accuracy, two more crucial issues need to be addressed to make it practical.
First, though mobile SoC uses a unified memory chip for heterogeneous processors, they use separated memory space.
To enable shadow execution of activation outliers, llm.npu has to keep another copy of each MatMul weights on CPU’s memory space.
This increases the memory footprint by almost 2 times.
Second, while the execution of outlier is fast even on CPU, the synchronization of the reduced sum between CPU and NPU still takes non-trivial overhead, e.g.,
29.7% end-to-end latency and 20.1% energy consumption on Qwen1.5-1.8B.

> **中文译文：**虽然影子异常值执行似乎很好地平衡了 NPU 亲和力和 LLM 准确性，但需要解决两个更关键的问题才能使其实用。首先，虽然移动SoC对异构处理器使用统一的存储芯片，但它们使用独立的存储空间。为了启用激活异常值的影子执行，llm.npu 必须在 CPU 内存空间上保留每个 MatMul 权重的另一个副本。这使内存占用增加了近 2 倍。其次，虽然异常值的执行速度即使在 CPU 上也很快，但 CPU 和 NPU 之间的降和同步仍然需要不小的开销，例如，Qwen1.5-1.8B 上的端到端延迟为 29.7%，能耗为 20.1%。

Figure 10. The average number and percentage of outlier channels per layer on Qwen1.5-1.8B model using the wikitext dataset under 2048 inference. Less than 0.3% channels have outliers during one inference.

> **中文图注：**图 10.使用 wikitext 数据集在 2048 推断下的 Qwen1.5-1.8B 模型上每层离群值通道的平均数量和百分比。在一次推理过程中，只有不到 0.3% 的通道存在异常值。


Figure 11. The outlier times per channel on Qwen1.5-1.8B model using the wikitext dataset under 2048 inference. Less than 3% channels contribute over 80% outliers.

> **中文图注：**图 11.使用 wikitext 数据集在 2048 推断下的 Qwen1.5-1.8B 模型上每个通道的异常值时间。不到 3% 的渠道贡献了超过 80% 的异常值。

Most outliers tend to appear in a small set of channel positions.
We observe that, while outliers appear in a wide range of channel positions during processing a long prompt (e.g., 78%), such appearance is highly skewed – few channels dominate the appearance of outliers.
For instance, as shown in Figure [11](#S3.F11 "Figure 11 ‣ 3.3. Shadow outlier execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs"), less than 3% of channels contribute to the majority of outliers (over 80%) across various inputs on Qwen1.5-1.8B and LlaMA-2-7B models.
Therefore, for shadow outlier execution, llm.npu only keeps the tensor weights that need to be used by those “hot channels” in CPU memory space, and retrieve the rest of them from disk if outliers on those positions are extracted (which is rare) at runtime.
Note that the weights retrieval can be also overlapped with the NPU execution of the original MatMul.
This approach reduces the memory overhead of shadow execution by 34.3% with negligible latency.

> **中文译文：**大多数异常值往往出现在一小部分通道位置中。我们观察到，虽然在处理长提示期间（例如 78%），异常值出现在各种通道位置，但这种外观是高度倾斜的——很少有通道主导异常值的出现。例如，如图 11 所示，在 Qwen1.5-1.8B 和 LlaMA-2-7B 模型的各种输入中，不到 3% 的通道对大多数异常值（超过 80%）有贡献。因此，对于影子异常值执行，llm.npu 仅将那些“热通道”需要使用的张量权重保留在 CPU 内存空间中，如果在运行时提取了这些位置上的异常值（这种情况很少见），则从磁盘检索其余部分。请注意，权重检索也可以与原始 MatMul 的 NPU 执行重叠。这种方法将影子执行的内存开销减少了 34.3%，并且延迟可以忽略不计。

Most outliers can be pruned without impacts on accuracy.
Surprisingly, we observe that the activation outliers on most MatMul operators are unimportant to the LLM accuracy and can be simply removed.
Here, the importance of an outlier is measured as the ratio between the largest outlier and the quantization scale ($s$ in Equation 1).
The larger ratio indicates a more dispersed activation distribution, resulting in more significant quantization errors.
llm.npu profiles these outlier importance using large corpora data at offline stage, (Figure [12](#S3.F12 "Figure 12 ‣ 3.3. Shadow outlier execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs")), and prunes most of unimportant layers’ outliers.
Typically, we observed layers near the inputs and outputs have a higher importance.
This is because layers close to inputs are easily influenced by the tokens disparity, exhibiting greater fluctuations, while layers approaching to outputs easily accumulate the errors from the shallow layers.
Based on the observation, llm.npu prunes the outliers of top 85% most unimportant layers through offline profiling, so that the CPU-NPU synchronization is eliminated.

> **中文译文：**大多数异常值都可以被修剪，而不影响准确性。令人惊讶的是，我们观察到大多数 MatMul 算子上的激活异常值对 LLM 精度并不重要，可以简单地删除。此处，异常值的重要性通过最大异常值与量化尺度（等式 1 中的 s）之间的比率来衡量。比率越大表示激活分布越分散，导致量化误差越显着。 llm.npu 在离线阶段使用大型语料库数据来描述这些异常值的重要性（图 12），并修剪大部分不重要层的异常值。通常，我们观察到输入和输出附近的层具有更高的重要性。这是因为靠近输入的层很容易受到令牌差异的影响，表现出更大的波动，而靠近输出的层很容易积累来自浅层的误差。根据观察，llm.npu通过离线分析修剪了前8​​5%最不重要层的异常值，从而消除了CPU-NPU同步。

Figure 12. Left: Outlier importance of different layers in Qwen1.5-1.8B model. Right: Relationship between accuracy and pruned layers on HellaSwag and Winograde datasets.

> **中文图注：**图 12.左：Qwen1.5-1.8B 模型中不同层的异常重要性。右图：HellaSwag 和 Winograde 数据集上的准确度与修剪层之间的关系。

### 3.4. Out-of-order subgraph execution

> **中文标题：**3.4.子图乱序执行

As elaborated in ($\lx@sectionsign$[2.3](#S2.SS3 "2.3. Gaps between LLMs and Mobile NPUs ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs")), LLM quantization algorithms cannot fully eliminate floating point operations, llm.npu thereby divides its execution flow into NPU and CPU/GPU collaboratively.
Typically, LayerNorm, Attention, as well as the shadow outlier computation are placed on the CPU/GPU; while the other linear layers are processed on the NPU.
However, we found simply overlapping their execution is inefficient, resulting in large execution bubbles (37% bubble rate in critical path), as illustrated in Figure [13](#S3.F13 "Figure 13 ‣ 3.4. Out-of-order subgraph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs")(a).

> **中文译文：**正如（ \lx@sectionsign 2.3 ）中所述，LLM 量化算法无法完全消除浮点运算，因此 llm.npu 将其执行流程分为 NPU 和 CPU/GPU 协同执行。通常，LayerNorm、Attention 以及阴影异常值计算都放在 CPU/GPU 上；而其他线性层则在 NPU 上处理。然而，我们发现简单地重叠执行效率很低，会导致较大的执行泡沫（关键路径中的泡沫率为 37%），如图 13 (a) 所示。

Out-of-order execution.
To reduce these execution bubbles, llm.npu is guided by a key insight that, after being partitioned at both chunk and subgraphs levels, the LLM subgraphs can be scheduled in an out-of-order manner.
More specifically, any input-ready subgraph can be executed without strictly following the chunk sequence.
For instance, the first subgraph of the third chunk (C3-Graph1) can be executed during the bubble period when C2-Graph1 finishes.

> **中文译文：**无序执行。为了减少这些执行泡沫，llm.npu 以一个关键见解为指导，即在块和子图级别进行分区后，可以以无序方式调度 LLM 子图。更具体地说，任何输入就绪的子图都可以在不严格遵循块序列的情况下执行。例如，当 C2-Graph1 完成时，可以在冒泡期间执行第三个块 (C3-Graph1) 的第一个子图。

Figure 13. Out-of-order subgraph execution.

> **中文图注：**图 13.无序子图执行。

To preserve correctness, llm.npu considers two types of dependency:
(1) Cross-chunk dependency. Operators like Attention rely on data from previous chunks. This means the $i$-th chunk $j$-th subgraph $G\_{i,j}$ depends on the $j-1$-th subgraph of the $0,1,\dots,i-1$ chunks:

> **中文译文：**为了保持正确性，llm.npu 考虑两种类型的依赖关系： (1) 跨块依赖关系。像 Attention 这样的算子依赖于之前块的数据。这意味着第 i 个块 j 个子图 G\_{i,j} 取决于 0,1,\dots,i-1 个块的第 j-1 个子图：

|  |  |  |  |
| --- | --- | --- | --- |
| (2) |  | $$ \displaystyle G\_{i,j}\leftarrow{G\_{0,j-1},G\_{1,j-1},\dots,G\_{i,j-1}} $$ |  |

(2) Intra-chunk dependency. Operators like LayerNorm, Linear, and Quantize rely only on previous subgraphs within the same chunk. Therefore, the $i$-th chunk’s $j$-th subgraph $G\_{i,j}$ depends on the $j-1$-th subgraph of the same chunk:

> **中文译文：**(2)块内依赖性。 LayerNorm、Linear 和 Quantize 等运算符仅依赖于同一块中的先前子图。因此，第 i 个块的第 j 个子图 G\_{i,j} 取决于同一块的第 j-1 个子图：

|  |  |  |  |
| --- | --- | --- | --- |
| (3) |  | $$ \displaystyle G\_{i,j}\leftarrow{G\_{i,j-1}} $$ |  |

As mobile processors are weak at parallelism and preemption ([97](#bib.bib97), [40](#bib.bib40), [86](#bib.bib86)), to ensure efficiency, a processor is capable of executing only one subgraph at any given time.:

> **中文译文：**由于移动处理器在并行性和抢占方面较弱（97、40、86），为了确保效率，处理器在任何给定时间只能执行一个子图：

|  |  |  |  |
| --- | --- | --- | --- |
| (4) |  | $$ \displaystyle\sum\_{i=0}^{N}\sum\_{j=0}^{M}P\_{i,j,t}=1,\forall t $$ |  |

where $P\_{i,j,t}=1$ indicates that subgraph $G\_{i,j}$ is running on processor $P$ at time $t$, and $N$ and $M$ represent the maximum number of chunks and subgraphs, respectively.
llm.npu aims to find an execution order minimizing the total execution time of all subgraphs under these constraints.
Unfortunately, this scheduling problem can be reduced to a classical NP-Hard Traveling Salesman Problem ([44](#bib.bib44)).
Moreover, because the number of chunks varies with user prompts, an optimal scheduling strategy cannot be generated offline.

> **中文译文：**其中 P\_{i,j,t}=1 表示子图 G\_{i,j} 在时间 t 正在处理器 P 上运行，N 和 M 分别表示块和子图的最大数量。 llm.npu 的目标是找到一个执行顺序，在这些约束下最小化所有子图的总执行时间。不幸的是，这个调度问题可以简化为经典的 NP 困难旅行商问题 (44)。而且，由于块的数量随用户提示而变化，无法离线生成最优调度策略。

Instead, llm.npu utilizes an online heuristic algorithm. The key idea is to focus not on the execution time of the subgraph $g$, but on how executing $g$ contributes to reducing NPU stalls, motivated by the observation that during the prefill phase, NPU execution time often dominates inference latency, being the critical path. For instance, with a prompt length of 256 using the Qwen1.5-1.8B model, NPU execution takes 315ms, about twice that of the CPU.

> **中文译文：**相反，llm.npu 使用在线启发式算法。关键思想是不要关注子图 g 的执行时间，而是关注执行 g 如何有助于减少 NPU 停顿，其动机是观察到在预填充阶段，NPU 执行时间通常主导推理延迟，是关键路径。例如，使用 Qwen1.5-1.8B 模型，提示长度为 256，NPU 执行需要 315ms，大约是 CPU 的两倍。

Specifically, we define a subgraph $g$’s contribution to reduce NPU stalls as follows: If subgraph $g$ is to be executed on the CPU/GPU, let $S$ be the set of new subgraphs that can be executed after $g$ is completed. $S$ will be executed on the NPU. A longer execution time of $S$ is beneficial for reducing NPU stalls. Thus, $g$’s contribution is defined as the total execution time of $S$. Conversely, if $g$ is executed on the NPU, a shorter execution time of $S$ is beneficial, with the negative value of $S$’s execution time as $g$’s contribution, formulated as:

> **中文译文：**具体来说，我们定义子图 g 对减少 NPU 停顿的贡献如下：如果子图 g 要在 CPU/GPU 上执行，则令 S 为 g 完成后可以执行的新子图的集合。 S将在NPU上执行。 S 较长的执行时间有利于减少 NPU 停顿。因此，g 的贡献被定义为 S 的总执行时间。相反，如果g在NPU上执行，则S的执行时间较短是有利的，S执行时间的负值作为g的贡献，公式为：

|  |  |  |
| --- | --- | --- |
|  | $$ \displaystyle C=\begin{array}[]{l}\left\{\begin{matrix}\sum T\_{i},\forall i\in S\text{ if $g$ is on the CPU/GPU}\\ -\sum T\_{i},\forall i\in S\text{ if $g$ is on the NPU}\\ \end{matrix}\right.\end{array} $$ |  |

where $T$ is the subgraph execution time.
llm.npu always chooses the subgraph with the largest $C$, meaning the subgraph $g$ with $S$ having the longest execution time on the NPU or the shortest execution time on the CPU/GPU.

> **中文译文：**其中 T 是子图执行时间。 llm.npu 总是选择 C ​​最大的子图，这意味着 S 的子图 g 在 NPU 上执行时间最长，或者在 CPU/GPU 上执行时间最短。

In a nut shell, llm.npu profiles all the subgraph execution time and their dependency offline at the preparation stage. During the prefill stage, it calculates all the pending subgraphs $C$ value and selects one with maximum $C$ to run, with microsecond-level performance overhead.

> **中文译文：**简而言之，llm.npu 在准备阶段离线分析所有子图的执行时间及其依赖关系。在预填充阶段，它会计算所有待处理子图的C值，并选择C最大的子图来运行，性能开销为微秒级。

![Refer to caption](%image_url%/fast-on-device-llm-inference-with-npus/figure_06.png)


Figure 14. Prefill speed under different prompt lengths on different devices (datasets: Longbench-2wiki-Multi-doc QA).

> **中文图注：**图 14.不同设备上不同提示长度下的预填充速度（数据集：Longbench-2wiki-Multi-doc QA）。


Figure 15. Energy consumption under different prompt lengths on Redmi K60 Pro (datasets: Longbench-2wiki-Multi-doc QA).

> **中文图注：**图 15. Redmi K60 Pro 不同提示时长下的能耗（数据集：Longbench-2wiki-Multi-doc QA）。

## 4. Implementation and Evaluation

> **中文标题：**4. 实施与评估

We have fully implemented llm.npu for Qualcomm Hexagon NPUs, comprising 10K lines of code in C/C++ and assembly language.
We choose Qualcomm SoCs as the target platform for its popularity on mobile devices and powerful NPU capacity.
Qualcomm Hexagon is also the only mobile NPU with an open instruction set architecture.
llm.npu is built on the MLLM ([23](#bib.bib23)), one state-of-the-art mobile LLM engines, and QNN framework ([25](#bib.bib25)), the Qualcomm Neural Processing SDK.
It supports standard LLM formats exported from Hugging Face ([21](#bib.bib21)).
To facilitate LLM execution, we implemented specific operators like KVCache, SiLU, RMSNorm, ROPE, and etc, in addition to what have been supported by QNN.
To reduce context switching overhead between CPUs/GPUs and NPUs, llm.npu leverages shared buffers to synchronize intermediate results from different processors.
For end-to-end inference, llm.npu is compatible with any decoding engine and utilizes the MLLM CPU backend for decoding stage as easy implementation, with a default chunk length of 256.
The default pruning rate for outlier layers is 85%.
llm.npu does not consider resource detection and contention in the current implementation.

> **中文译文：**我们已经为 Qualcomm Hexagon NPU 全面实现了 llm.npu，其中包含 10K 行 C/C++ 和汇编语言代码。我们选择高通 SoC 作为目标平台，因为它在移动设备上的受欢迎程度和强大的 NPU 能力。 Qualcomm Hexagon也是唯一一款具有开放指令集架构的移动NPU。 llm.npu 构建于 MLLM (23)（一种最先进的移动 LLM 引擎）和 QNN 框架 (25)（高通神经处理 SDK）之上。它支持从 Hugging Face (21) 导出的标准 LLM 格式。为了促进 LLM 执行，除了 QNN 支持的操作之外，我们还实现了 KVCache、SiLU、RMSNorm、ROPE 等特定操作。为了减少 CPU/GPU 和 NPU 之间的上下文切换开销，llm.npu 利用共享缓冲区来同步来自不同处理器的中间结果。对于端到端推理，llm.npu 与任何解码引擎兼容，并利用 MLLM CPU 后端进行解码阶段，易于实现，默认块长度为 256。异常层的默认剪枝率为 85%。 llm.npu 在当前实现中不考虑资源检测和争用。

The prototype further incorporates two optimizations.
(1)
Our extensive experiments show that mobile NPUs ofte favor tensor sizes (e.g., equal “height” and “width”) in CNN architectures.
For example, a linear layer with weights of 2048$\times$2048 produces the same results for inputs of 1024×1×2048 and 32$\times$32$\times$2048, but using 32$\times$32$\times$2048 reduces execution latency by 1.62$\times$.
Therefore, llm.npu profiles all possible equivalent shapes for linear layers during the preparation stage and selects the most efficient one.
(2)
Mobile NPUs typically access limited memory regions (e.g., 4GB for Hexagon NPU), which can be smaller than the size of LLM weights. To maximize prefill acceleration within this limited memory, llm.npu prioritizes executing computationally intensive tasks, such as FFN, on the NPU to enhance efficiency.

> **中文译文：**该原型还包含两项优化。 (1) 我们的大量实验表明，移动 NPU 通常更喜欢 CNN 架构中的张量大小（例如，相等的“高度”和“宽度”）。例如，权重为 2048 × 2048 的线性层对于 1024×1×2048 和 32 × 32 × 2048 的输入产生相同的结果，但使用 32 × 32 × 2048 可以将执行延迟减少 1.62 × 。因此，llm.npu 在准备阶段会分析线性层的所有可能的等效形状，并选择最有效的形状。 (2) 移动 NPU 通常访问有限的内存区域（例如，Hexagon NPU 为 4GB），该内存区域可能小于 LLM 权重的大小。为了在有限的内存内最大化预填充加速，llm.npu 优先在 NPU 上执行计算密集型任务（例如 FFN）以提高效率。

### 4.1. Experiment setups

> **中文标题：**4.1.实验设置

Hardware setup.
We test llm.npu on two smartphones with different Qualcomm SoCs: Redmi K70 Pro (Snapdragon 8gen3, 24GB memory) and Redmi K60 Pro (Snapdragon 8gen2, 16GB memory).
All devices run Android OS 13.

> **中文译文：**硬件设置。我们在两部配备不同 Qualcomm SoC 的智能手机上测试 llm.npu：Redmi K70 Pro（Snapdragon 8gen3，24GB 内存）和 Redmi K60 Pro（Snapdragon 8gen2，16GB 内存）。所有设备均运行 Android OS 13。

Models and datasets.
We conducted tests using a variety of typical mobile-sized LLMs: Qwen1.5-1.8B ([27](#bib.bib27)), Gemma-2B ([9](#bib.bib9)), Phi2-2.7B ([16](#bib.bib16)), LLaMA2-Chat-7B ([11](#bib.bib11)), and Mistral-7B ([14](#bib.bib14)). To evaluate llm.npu’s quantization accuracy, we employed widely recognized LLM benchmarks, including LAMBADA ([68](#bib.bib68)), HellaSwag ([103](#bib.bib103)), WinoGrande ([4](#bib.bib4)), OpenBookQA ([62](#bib.bib62)) and MMLU ([43](#bib.bib43)). For inference speed experiments, we selected retrieval-based datasets from Longbench, 2wikimqa and TriviaQA ([28](#bib.bib28)), for simulating context-aware generate tasks like automated email reply. Additionally, we assessed llm.npu in screen question-answering and mapping instruction to UI action scenarios using DroidTask datasets ([22](#bib.bib22)) to simulate agent-based UI automation tasks.

> **中文译文：**模型和数据集。我们使用各种典型的移动大小的 LLM 进行了测试：Qwen1.5-1.8B (27)、Gemma-2B (9)、Phi2-2.7B (16)、LLaMA2-Chat-7B (11) 和 Mistral-7B (14)。为了评估 llm.npu 的量化精度，我们采用了广泛认可的 LLM 基准，包括 LAMBADA (68)、HellaSwag (103)、WinoGrande (4)、OpenBookQA (62) 和 MMLU (43)。对于推理速度实验，我们选择了来自 Longbench、2wikimqa 和 TriviaQA (28) 的基于检索的数据集，用于模拟上下文感知生成任务，例如自动电子邮件回复。此外，我们还使用 DroidTask 数据集 (22) 评估了 llm.npu 的屏幕问答和将指令映射到 UI 操作场景，以模拟基于代理的 UI 自动化任务。

Baselines.
We mainly compare llm.npu with 5 baselines, including 3 widely used mobile LLM engines (TFLite ([57](#bib.bib57)), MNN ([49](#bib.bib49)), and llama.cpp ([59](#bib.bib59))).
Those engines support only mobile CPU and GPU;
2 advanced baselines are MLC-LLM ([75](#bib.bib75)), an LLM compiler for on-device GPUs, and PowerInfer-v2, which also utilizes mobile NPUs to accelerate prefilling ([94](#bib.bib94)).
Since PowerInfer-v2 is not open-sourced, we use the reported data from its paper.
To be noted, those baselines often support only a subset of 5 LLMs we evaluated.

> **中文译文：**基线。我们主要将 llm.npu 与 5 个基线进行比较，包括 3 个广泛使用的移动 LLM 引擎（TFLite (57)、MNN (49) 和 llama.cpp (59)）。这些引擎仅支持移动CPU和GPU； 2 个高级基线是 MLC-LLM (75)，一种用于设备上 GPU 的 LLM 编译器，以及 PowerInfer-v2，它也利用移动 NPU 来加速预填充 (94)。由于 PowerInfer-v2 不是开源的，我们使用其论文中报告的数据。需要注意的是，这些基线通常仅支持我们评估的 5 个法学硕士的子集。

Metrics.
We mainly measure LLM inference accuracy, prefill latency, prefill energy consumption, prefill memory consumption and end-to-end inference latency.
The energy consumption is obtained through /sys /class/power\_supply in Android OS
by profiling every 100ms.
All experiments are repeated three times and we report the average
numbers.

> **中文译文：**指标。我们主要测量LLM推理精度、预填充延迟、预填充能耗、预填充内存消耗和端到端推理延迟。 Android 操作系统中通过 /sys /class/power\_supply 每 100ms 进行一次分析来获取能耗。所有实验重复三次，我们报告平均数。

Table 5. End-to-end latency comparison using real mobile applications execution on Redmi K70 Pro. Each cell is organized in the following format: end-to-end latency (prefill latency, decode latency). Unit: seconds.

> **中文图注：**表 5.使用 Redmi K70 Pro 上的真实移动应用程序执行进行端到端延迟比较。每个单元按以下格式组织：端到端延迟（预填充延迟、解码延迟）。单位：秒。

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLM | Datasets | MLC | LCPP | MNN | PI | TFLite | Ours | Speedup |
| Qwen1.5-1.8B | Longbench: 2wiki-Multi-doc QA  prompt length: 1451–1672 tokens  output length: 2–4 tokens | |  | | --- | | 45.6 | | (45.43, 0.17) | | |  | | --- | | 26.7 | | (26.43, 0.24) | | |  | | --- | | 10.6 | | (10.01, 0.59) | | - | - | |  | | --- | | 1.7 | | (1.49, 0.24) | | 6.2-26.8$\times$ |
| Gemma-2B | |  | | --- | | 78.4 | | (78.03, 0.32) | | |  | | --- | | 34.6 | | (34.31, 0.25) | | - | - | |  | | --- | | 2.6 | | (2.40, 0.19) | | |  | | --- | | 1.9 | | (1.68, 0.25) | | 1.4-41.3$\times$ |
| Phi-2-2.7B | |  | | --- | | 87 | | (86.70, 0.28) | | |  | | --- | | 53.3 | | (53.00, 0.34) | | |  | | --- | | 13 | | (11.98, 1.05) | | - | |  | | --- | | 6.3 | | (6.06, 0.28) | | |  | | --- | | 3.1 | | (2.79, 0.34) | | 2.0-28.1$\times$ |
| LlaMA-2-7B | |  | | --- | | 184.7 | | (184.20, 0.45) | | |  | | --- | | 146 | | (145.25, 0.70) | | |  | | --- | | 22.4 | | (22.35, 0.05) | | |  | | --- | | 19.8 | | (19.04, 0.78) | | - | |  | | --- | | 5.3 | | (4.62, 0.70) | | 3.7-34.8$\times$ |
| Mistral-7b | |  | | --- | | 254.2 | | (253.45, 0.72) | | |  | | --- | | 200.2 | | (199.37, 0.87) | |  | |  | | --- | | 20 | | (19.22, 0.78) | | - | |  | | --- | | 5.5 | | (4.61, 0.87) | | 3.6-46.2$\times$ |
| Geo-mean (speedup) | | 34.7$\times$ | 21.8$\times$ | 4.8$\times$ | 3.7$\times$ | 1.7$\times$ | - |  |
| Qwen1.5-1.8B | Longbench: TriviaQA  prompt length: 1511–1787 tokens  output length: 5–11 tokens) | |  | | --- | | 46 | | (45.68) | | |  | | --- | | 27 | | (26.57, 0.47) | | |  | | --- | | 11.2 | | (10.07, 1.15) | | - | - | |  | | --- | | 2.0 | | (1.49, 0.47) | | 5.6-23.0$\times$ |
| Gemma-2B | |  | | --- | | 81.8 | | (81.25, 0.26) | | |  | | --- | | 36.2 | | (35.72, 0.43) | | - | - | |  | | --- | | 2.8 | | (2.50, 0.32) | | |  | | --- | | 2.2 | | (1.75, 0.43) | | 1.3-37.2$\times$ |
| Phi-2-2.7B | |  | | --- | | 91.4 | | (90.94, 0.50) | | |  | | --- | | 56.3 | | (55.60, 0.68) | | |  | | --- | | 14.7 | | (12.57, 2.11) | | - | |  | | --- | | 6.8 | | (6.36, 0.40) | | |  | | --- | | 3.6 | | (2.94, 0.68) | | 1.9-25.4$\times$ |
| LlaMA-2-7B | |  | | --- | | 197.3 | | (196.43, 0.83) | | |  | | --- | | 156.2 | | (154.90, 1.28) | | |  | | --- | | 23.8 | | (20.83, 3.00) | | |  | | --- | | 21.8 | | (20.31, 1.34) | | - | |  | | --- | | 6.2 | | (4.93, 1.28) | | 3.5-31.8$\times$ |
| Mistral-7b | |  | | --- | | 266.2 | | (264.93, 1.31) | | |  | | --- | | 210 | | (208.41, 1.57) | | - | |  | | --- | | 21.5 | | (20.10, 0.41) | | - | |  | | --- | | 6.4 | | (4.82, 1.57) | | 3.4-41.6$\times$ |
| Geo-mean (speedup) | | 31.0$\times$ | 19.6$\times$ | 4.4$\times$ | 3.4$\times$ | 1.6$\times$ | - |  |
| Qwen1.5-1.8B | DroidTask: clock  prompt length: 656–827 tokens  output length: 1–5 tokens) | |  | | --- | | 21 | | (20.84, 0.14) | | |  | | --- | | 10.4 | | (10.16, 0.25) | | |  | | --- | | 3.9 | | (3.25, 0.61) | | - | - | |  | | --- | | 1.4 | | (1.16, 0.25) | | 2.80-15$\times$ |
| Gemma-2B | |  | | --- | | 39.4 | | (39.11, 0.25) | | |  | | --- | | 16.5 | | (16.25, 0.21) | | - | - | |  | | --- | | 2.5 | | (2.30, 0.18) | | |  | | --- | | 1.2 | | (0.97, 0.23) | | 2.1-32.8$\times$ |
| Phi-2-2.7B | |  | | --- | | 46.6 | | (46.40, 0.24) | | |  | | --- | | 25 | | (24.62, 0.34) | | |  | | --- | | 7.4 | | (6.34, 1.06) | | - | |  | | --- | | 4.2 | | (3.98, 0.2) | | |  | | --- | | 3.1 | | (2.72, 0.34) | | 1.4-15.0$\times$ |
| LlaMA-2-7B | |  | | --- | | 87.7 | | (87.38, 0.33) | | |  | | --- | | 60.4 | | (59.83, 0.63) | | |  | | --- | | 10.6 | | (8.58, 2.11) | | |  | | --- | | 11.1 | | (10.74, 0.37) | | - | |  | | --- | | 4.8 | | (4.28, 0.53) | | 2.2-18.3$\times$ |
| Mistral-7b | |  | | --- | | 122.3 | | (121.82, 0.51) | | |  | | --- | | 68.6 | | (67.91, 0.65) | | - | |  | | --- | | 12 | | (11.64, 0.37) | | - | |  | | --- | | 4.9 | | (4.26, 0.65) | | 2.4-25.0$\times$ |
| Geo-mean (speedup) | | 20.2$\times$ | 10.8$\times$ | 2.4$\times$ | 2.4$\times$ | 1.7$\times$ | - |  |
| Qwen1.5-1.8B | DroidTask: clock  prompt length: 505–645 tokens  output length: 3–5 tokens | |  | | --- | | 16.2 | | (16.05, 0.14) | | |  | | --- | | 8.1 | | (7.83, 0.25) | | |  | | --- | | 3.1 | | (2.5, 0.61) | | - | - | |  | | --- | | 1.1 | | (0.90, 0.25) | | 2.8-14.7$\times$ |
| Gemma-2B | |  | | --- | | 29.4 | | (29.14, 0.52) | | |  | | --- | | 12.3 | | (12.11, 0.20) | | - | - | |  | | --- | | 1.9 | | (1.71, 0.18) | | |  | | --- | | 0.9 | | (0.72, 0.23) | | 2.1-32.7$\times$ |
| Phi-2-2.7B | |  | | --- | | 35.4 | | (35.15, 0.23) | | |  | | --- | | 19 | | (18.65, 0.34) | | |  | | --- | | 5.9 | | (4.80, 1.05) | | - | |  | | --- | | 3.2 | | (3.01, 1.2) | | |  | | --- | | 2.4 | | (2.07, 0.34) | | 1.3-14.8$\times$ |
| LlaMA-2-7B | |  | | --- | | 63.7 | | (63.38, 0.33) | | |  | | --- | | 43.9 | | (43.40, 0.52) | | |  | | --- | | 7.7 | | (5.06. 1.64) | | |  | | --- | | 8.2 | | (7.79, 0.37) | | - | |  | | --- | | 3.6 | | (3.10, 0.52) | | 2.1-17.7$\times$ |
| Mistral-7b | |  | | --- | | 90.1 | | (89.60, 0.51) | | |  | | --- | | 50.6 | | (49.95, 0.65) | | - | |  | | --- | | 8.9 | | (8.56, 0.37) | | - | |  | | --- | | 3.8 | | (3.13, 0.65) | | 2.3-23.7$\times$ |
| Geo-mean (speedup) | | 19.7$\times$ | 10.5$\times$ | 2.5$\times$ | 2.3$\times$ | 1.7$\times$ | - |  |
| Qwen1.5-1.8B | Persona-Chat  prompt length: 488–584 tokens  output length: 35–57 tokens | |  | | --- | | 18.74 | | (16.54, 2.20) | | |  | | --- | | 11.86 | | (8.07, 3.79) | | |  | | --- | | 11.87 | | (2.58, 9.29) | | - | - | |  | | --- | | 6.72 | | (2.92, 3.80) | | 1.8-2.8$\times$ |
| Gemma-2B | |  | | --- | | 30.9 | | (27.05, 3.82) | | |  | | --- | | 14.3 | | (11.24, 3.02) | | - | - | |  | | --- | | 4.3 | | (1.59, 2.67) | | |  | | --- | | 4.2 | | (0.67, 3.48) | | 1.02-7.4$\times$ |
| Phi-2-2.7B | |  | | --- | | 36.1 | | (32.51, 3.57) | | |  | | --- | | 22.5 | | (15 .24, 3.22) | | |  | | --- | | 20.6 | | (4.43, 16.19) | | - | |  | | --- | | 5.9 | | (2.79, 3.06) | | |  | | --- | | 5.1 | | (1.91, 3.22) | | 1.2-7.1$\times$ |
| LlaMA-2-7B | |  | | --- | | 64.2 | | (58.96, 5.20) | | |  | | --- | | 48.7 | | (40.37, 8.30) | | |  | | --- | | 45.5 | | (7.15, 38.23) | | |  | | --- | | 13.0 | | (7.25, 5.77) | | - | |  | | --- | | 11.2 | | (2.88, 8.30) | | 1.2-5.7$\times$ |
| Mistral-7b | |  | | --- | | 90.5 | | (82.54, 8.0) | | |  | | --- | | 56.3 | | (46.02, 10.28) | | - | |  | | --- | | 13.7 | | (7.89, 5.76) | | - | |  | | --- | | 13.2 | | (2.88, 10.28) | | 1.04-6.9$\times$ |
| Geo-mean (speedup) | | 10.5$\times$ | 3.5$\times$ | 3.1$\times$ | 1.1$\times$ | 1.1$\times$ | - |  |
| \*LCPP and PI in the first row represent llama.cpp and PowerInfer-V2, respectively. | | | | | | | | |

### 4.2. Prefill performance.

> **中文标题：**4.2.预填充性能。

We evaluate the llm.npu’s prefill performance (speed and energy consumption) at prompt lengths of 64, 256 and 1024 tokens on two devices, as shwon in Figure [14](#S3.F14 "Figure 14 ‣ 3.4. Out-of-order subgraph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs") and [15](#S3.F15 "Figure 15 ‣ 3.4. Out-of-order subgraph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs").
Despite outlier variations across datasets, the overall impact on prefill performance is minimal, so we report results from the LongBench dataset.
The results show that llm.npu consistently outperforms all baselines across both metrics, with benefits becoming more pronounced as prompt length increases.

> **中文译文：**我们在两台设备上以 64、256 和 1024 个令牌的提示长度评估 llm.npu 的预填充性能（速度和能耗），如图 14 和 15 所示。尽管数据集之间存在异常变化，但对预填充性能的总体影响很小，因此我们报告了 LongBench 数据集的结果。结果表明，llm.npu 在这两个指标上始终优于所有基线，并且随着提示长度的增加，优势变得更加明显。

Prefill speed.
For prompt length of 1024 tokens, llm.npu can reduce prefill latency by 18.17–38.4$\times$, 7.3$\times$, 32.5–43.6$\times$, and 1.27–2.34$\times$ on Redmi K70 Pro compared with llama.cpp-CPU, MNN-CPU, MLC-GPU, TFLite-GPU, respectively.
On the Redmi K60 Pro, these improvements are 21.3–41.3$\times$, 7.43$\times$, 37.2–69.3$\times$, and 1.3–2.6$\times$, respectively.
These speedups are due to llm.npu’s use of three innovative techniques that fully leverage mobile NPUs, including shadow outlier execution, high-efficiency per-tensor MatMul, and out-of-order subgraph execution.
Compared with PowerInfer-V2-NPU, a baseline also using NPU for prefilling, llm.npu can achieve 3.28–5.32$\times$ and 3.4–5.6$\times$ speedup on two devices, respectively, by employing NPU-friendly INT8 linear computation and fine-grained subgraph scheduling ($\lx@sectionsign$[3.4](#S3.SS4 "3.4. Out-of-order subgraph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs")).

> **中文译文：**预充速度。对于 1024 个 token 的提示长度，与 llama.cpp-CPU、MNN-CPU、MLC-GPU、TFLite-GPU 相比，llm.npu 在 Redmi K70 Pro 上可以分别将预填充延迟减少 18.17–38.4 ×、7.3 ×、32.5–43.6 ×和 1.27–2.34 ×。在Redmi K60 Pro上，这些改进分别为21.3-41.3×、7.43×、37.2-69.3×和1.3-2.6×。这些加速归功于 llm.npu 使用了三种充分利用移动 NPU 的创新技术，包括阴影离群值执行、高效每张量 MatMul 和乱序子图执行。与同样使用 NPU 进行预填充的基线 PowerInfer-V2-NPU 相比，llm.npu 通过采用 NPU 友好的 INT8 线性计算和细粒度子图调度（ \lx@sectionsign 3.4 ），可以在两个设备上分别实现 3.28–5.32 倍和 3.4–5.6 倍的加速。

For prompt length of 64 tokens, the prefill speed of llm.npu is 14.86–7.10$\times$, 1.69$\times$, 10.91–17.32$\times$, 1.48$\times$, and 1.81–2.51$\times$ faster than llama.cpp-CPU, MNN-CPU, MLC-GPU, TFLite-GPU, and PowerInfer-V2-NPU respectively, with speedups averaging 10.5$\times$, 4.31$\times$, 2.68$\times$, 1.02$\times$, and 1.96$\times$ lower than those for 1024-token prompts.
This is because a shorter prompt can lead to a padding problem and limit llm.npu’s out-of-order execution scheduling efficiency.

> **中文译文：**对于 64 个 token 的提示长度，llm.npu 的预填充速度比 llama.cpp-CPU、MNN-CPU、MLC-GPU、TFLite-GPU 和 llama.cpp-CPU、MNN-CPU、MLC-GPU、TFLite-GPU 快 14.86–7.10 \times、1.69 \times、10.91–17.32 \times、1.48 \times 和 1.81–2.51 \times PowerInfer-V2-NPU 分别比 1024 个令牌提示的加速速度平均低 10.5 × 、 4.31 × 、 2.68 × 、 1.02 × 和 1.96 × 。这是因为较短的提示可能会导致填充问题并限制 llm.npu 的乱序执行调度效率。

Prefill energy consumption.
Energy consumption was evaluated on the Redmi K60 Pro, the only rootable device.
PowerInfer-V2 was excluded due to the lack of energy consumption data and open-source code.
For 1024-token prompts, llm.npu reduces energy consumption by 35.63–59.52$\times$, 35.21– 59.25$\times$, and 1.85–4.32$\times$ compared to llama.cpp-CPU, MLC-GPU, and TFLite-GPU, respectively. For 64-token prompts, the savings are 10.38–14.12$\times$, 10.38–17.79$\times$, and 3.22–3.67$\times$, respectively. These savings are due to the high energy efficiency of mobile NPUs and llm.npu’s three novel techniques for maximizing NPU performance.
Typically, during the LLM prefill stage, all CPU cores are fully utilized, consuming the highest power; NPUs operate at 500-750 MHz, consuming the least power.

> **中文译文：**预填充能量消耗。能耗评估是在唯一可 Root 的设备 Redmi K60 Pro 上进行的。 PowerInfer-V2 由于缺乏能耗数据和开源代码而被排除在外。对于 1024 个令牌提示，与 llama.cpp-CPU、MLC-GPU 和 TFLite-GPU 相比，llm.npu 分别将能耗降低了 35.63–59.52 倍、35.21–59.25 倍和 1.85–4.32 倍。对于 64 个令牌提示，节省的成本分别为 10.38–14.12 \times 、 10.38–17.79 \times 和 3.22–3.67 \times 。这些节省归功于移动 NPU 的高能效以及 llm.npu 的三项新技术，可最大限度地提高 NPU 性能。通常，在LLM预填充阶段，所有CPU核心都被充分利用，消耗的功率最高； NPU 的运行频率为 500-750 MHz，功耗最低。

### 4.3. End-to-end performance

> **中文标题：**4.3.端到端性能

We evaluate the real-world performance of llm.npu against baseline systems using three workloads: UI automation on DroidTask datasets, context-aware automated email replies on LongBench datasets, chat summaries on Persona-Chat datasets.
Table [5](#S4.T5 "Table 5 ‣ 4.1. Experiment setups ‣ 4. Implementation and Evaluation ‣ Fast On-device LLM Inference with NPUs") presents the prompt and output lengths of the three datasets, along with their corresponding prefill, decode, and end-to-end inference latency results. End-to-end speedups in Table [5](#S4.T5 "Table 5 ‣ 4.1. Experiment setups ‣ 4. Implementation and Evaluation ‣ Fast On-device LLM Inference with NPUs") are the geometric mean of each sample.
Our key observation is that llm.npu always achieves the lowest inference latency across all three datasets.

> **中文译文：**我们使用三种工作负载根据基准系统评估 llm.npu 的实际性能：DroidTask 数据集上的 UI 自动化、LongBench 数据集上的上下文感知自动电子邮件回复、Persona-Chat 数据集上的聊天摘要。表 5 显示了三个数据集的提示和输出长度，以及相应的预填充、解码和端到端推理延迟结果。表 5 中的端到端加速是每个样本的几何平均值。我们的主要观察结果是 llm.npu 始终在所有三个数据集上实现最低的推理延迟。

For LongBench datasets, llm.npu shows significant speed improvements: 23.0–46.2$\times$ over llama.cpp-CPU, 16.5–36.4$\times$ over MLC-LLM-GPU, 4.08–4.19$\times$ over MNN-CPU, 3.51–3.73$\times$ over PowerInfer-V2-NPU, and 1.27–2.03$\times$ over TFLite-GPU.
This impressive performance is primarily due to llm.npu’s superior efficiency during the prefill stage. The speedup against TFLite-GPU is lower since llm.npu currently relies on a CPU backend for decoding with no optimization, while TFLite utilizes GPU. Notably, llm.npu is compatible with any decoding engine, which means
once TFLite is open-sourced, llm.npu can integrate it as the decoding backend, potentially enhancing performance further.

> **中文译文：**对于 LongBench 数据集，llm.npu 显示出显着的速度提升：比 llama.cpp-CPU 快 23.0–46.2 倍，比 MLC-LLM-GPU 快 16.5–36.4 倍，比 MNN-CPU 快 4.08–4.19 倍，比 PowerInfer-V2-NPU 快 3.51–3.73 倍，比 PowerInfer-V2-NPU 快 1.27–2.03 \ 超过 TFLite-GPU 的倍数。这种令人印象深刻的性能主要归功于 llm.npu 在预填充阶段的卓越效率。相对于 TFLite-GPU 的加速比较低，因为 llm.npu 目​​前依赖 CPU 后端进行解码，没有进行优化，而 TFLite 使用 GPU。值得注意的是，llm.npu 与任何解码引擎兼容，这意味着一旦 TFLite 开源，llm.npu 可以将其集成为解码后端，从而有可能进一步增强性能。

For the DroidTask datasets, llm.npu reduces end-to-end inference latency by 7.9–12.5$\times$ compared to llama.cpp-CPU, 15.0–32.8$\times$ compared to MLC-LLM-GPU, 2.38–2.45$\times$ compared to MNN-CPU, 2.27–2.44$\times$ compared to PowerInfer-V2-NPU, and 1.35–2.38$\times$ compared to TFLite-GPU. The performance gains are slightly smaller for DroidTask datasets due to shorter prompts in UI automation versus email writing.

> **中文译文：**对于 DroidTask 数据集，llm.npu 与 llama.cpp-CPU 相比，端到端推理延迟减少了 7.9–12.5 倍，与 MLC-LLM-GPU 相比减少了 15.0–32.8 倍，与 MNN-CPU 相比减少了 2.38–2.45 倍，与 PowerInfer-V2-NPU 相比减少了 2.27–2.44 倍，与 TFLite-GPU 相比，提高了 1.35–2.38 倍。 DroidTask 数据集的性能提升略小，因为 UI 自动化中的提示比电子邮件编写中的提示更短。

For the Persona-Chat datasets, llm.npu reduces end-to-end inference latency by an average of 3.5$\times$ compared to llama.cpp-CPU, 10.5$\times$ compared to MLC-LLM-GPU, 3.1$\times$ compared to MNN-CPU, 1.1 times compared to PowerInfer-V2-NPU, and 1.1$\times$ compared to TFLite-GPU. The primary reason for the reduced performance improvement relative to other datasets is the larger number of output tokens generated by this dataset. Since llm.npu currently supports only CPU-based decoding, this slower decoding backend results in increased end-to-end latency. This issue can be mitigated by utilizing a mobile GPU as the decoding backend (see Section [4.6](#S4.SS6 "4.6. GPU-NPU coordination analysis ‣ 4. Implementation and Evaluation ‣ Fast On-device LLM Inference with NPUs") for details).

> **中文译文：**对于 Persona-Chat 数据集，llm.npu 与 llama.cpp-CPU 相比，平均降低端到端推理延迟 3.5 倍，与 MLC-LLM-GPU 相比，平均降低 10.5 倍，与 MNN-CPU 相比，降低 3.1 倍，与 PowerInfer-V2-NPU 相比，降低 1.1 倍，与 TFLite-GPU 相比，降低 1.1 倍。相对于其他数据集，性能提升降低的主要原因是该数据集生成的输出标记数量较多。由于 llm.npu 目​​前仅支持基于 CPU 的解码，因此较慢的解码后端会导致端到端延迟增加。通过使用移动 GPU 作为解码后端可以缓解此问题（详细信息请参阅第 4.6 节）。

### 4.4. Inference accuracy

> **中文标题：**4.4.推理准确率

Overall accuracy.
We investigate the inference accuracy of llm.npu on 5 LLM benchmarks: LAMBADA ([68](#bib.bib68)), MMLU ([43](#bib.bib43)), WinoGrande ([4](#bib.bib4)), OpenBookQA ([62](#bib.bib62)) and HellaSwag ([103](#bib.bib103)).
For comparison, we evaluated 4 alternatives:
FP16 (non-quantization), K-Quant ([59](#bib.bib59)) (used in llama.cpp), SmoothQu-ant (state-of-the-art per-tensor method) ([84](#bib.bib84)), and LLM.Int8() (state-of-the-art float outlier handling method) ([33](#bib.bib33)).
llm.npu achieves negligible accuracy loss, and significantly outperforms the other quantization algorithms, as shown in Table [6](#S4.T6 "Table 6 ‣ 4.4. Inference accuracy ‣ 4. Implementation and Evaluation ‣ Fast On-device LLM Inference with NPUs").

> **中文译文：**总体准确度。我们研究了 llm.npu 在 5 个 LLM 基准上的推理准确性：LAMBADA (68)、MMLU (43)、WinoGrande (4)、OpenBookQA (62) 和 HellaSwag (103)。为了进行比较，我们评估了 4 个替代方案：FP16（非量化）、K-Quant (59)（在 llama.cpp 中使用）、SmoothQu-ant（最先进的每张量方法）(84) 和 LLM.Int8()（最先进的浮点异常值处理方法）(33)。 llm.npu 的精度损失可以忽略不计，并且明显优于其他量化算法，如表 6 所示。

Specifically, llm.npu is, on average, only 1% less accurate than FP16, and it shows an accuracy improvement of up to 32.9% over SmoothQuant and up to 70.9% over K-Quant.
This improvement over SmoothQuant, which uses static profiling to smooth outliers to normal values, is due to llm.npu’s dynamic handling of outlier positions with CPU float precision. llm.npu addresses outliers at the element level, providing higher precision than K-Quant that uses group-level quantization scales.
Furthermore, llm.npu achieves comparable accuracy (0.1% average loss) to LLM.Int8(), as both handle outliers with float precision. But
llm.npu better utilizes NPU-specific computational features, maintaining high accuracy and NPU efficiency.

> **中文译文：**具体来说，llm.npu 的平均准确度仅比 FP16 低 1%，但与 SmoothQuant 相比，准确度提高了 32.9%，比 K-Quant 提高了 70.9%。 SmoothQuant 使用静态分析将离群值平滑为正常值，这一改进归功于 llm.npu 以 CPU 浮点精度动态处理离群值位置。 llm.npu 在元素级别解决异常值，提供比使用组级别量化尺度的 K-Quant 更高的精度。此外，llm.npu 实现了与 LLM.Int8() 相当的精度（平均损失 0.1%），因为两者都以浮点精度处理异常值。但 llm.npu 更好地利用了 NPU 特有的计算特性，保持了高精度和 NPU 效率。

Table 6. LLM capability accuracy on llm.npu and baselines. "SQ": SmoothQuant; "Int8()": LLM.Int8(); "Degrad.": accuracy degradation compared to FP16.

> **中文图注：**表 6. llm.npu 和基线上的 LLM 能力准确性。 “SQ”：SmoothQuant； “Int8()”: LLM.Int8(); “Degrad.”：与 FP16 相比，精度下降。

| LAMBADA | FP16 | SQ | INT8() | K-Quant | Ours | Ours Degrad. |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen1.5-1.8B | 71.1% | 65.6% | 71.0% | 62.7% | 71.7% | +0.6% |
| Gemma2-2B | 59.6% | 45.8% | 59.2% | 56.9% | 59.4% | -0.2% |
| Phi-2-2.7B | 72.2% | 66.1% | 71.7% | 59.3% | 67.5% | -4.7% |
| LlaMA-2-7B | 87.5% | 71.9% | 88.0% | 15.6% | 86.3% | -1.2% |
| Mistral-7b | 84.8% | 51.2% | 85.3% | 23.9% | 84.1% | -0.7% |
| Avg. Degrad. | - | -14.9% | 0% | -31.3% | -1.2% |  |

| HellaSwag | FP16 | SQ | INT8() | K-Quant | Ours | Ours Degrad. |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen1.5-1.8B | 43.8% | 40.9% | 43.5% | 44.3% | 43.8% | 0% |
| Gemma2-2B | 46.5% | 43.8% | 46.1% | 45.4% | 47.3% | +0.8% |
| Phi-2-2.7B | 48.2% | 46.2% | 47.7% | 47.6% | 46.9% | -1.3% |
| LlaMA-2-7B | 52.8% | 44.4% | 53.1% | 50.5% | 53.5% | +0.7% |
| Mistral-7b | 57.4% | 44.9% | 57.9% | 57.0% | 57.0% | -0.4% |
| Avg. Degrad. | - | -5.7% | -0.1% | -0.8% | -0.0% |  |

| WinoGrande | FP16 | SQ | INT8() | K-Quant | ours | Ours Degrad. |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen1.5-1.8B | 58.3% | 51.0% | 58.2% | 59.0% | 59.3% | +1.0% |
| Gemma2-2B | 58.3% | 54.8% | 59.0% | 58.5% | 59.5% | +1.2% |
| Phi-2-2.7B | 72.2% | 68.9% | 72.4% | 72.5% | 70.2% | -2.0% |
| LlaMA-2-7B | 65.2% | 56.9% | 66.2% | 67.4% | 65.1% | -0.1% |
| Mistral-7b | 73.5% | 59.1% | 73.3% | 73.5% | 73.1% | -0.4% |
| Avg. Degrad. | - | -7.4% | +0.3% | +0.7% | -0.1% |  |

| OpenBookQA | FP16 | SQ | INT8() | K-Quant | ours | Ours Degrad. |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen1.5-1.8B | 28.8% | 23.0% | 28.5% | 28.0% | 26.6% | -2.2% |
| Gemma2-2B | 33.7% | 28.0% | 34.2% | 33.0% | 38.4% | +4.7% |
| Phi-2-2.7B | 41.0% | 35.9% | 40.2% | 39.5% | 37.7% | -3.3% |
| LlaMA-2-7B | 32.7% | 25.0% | 32.0% | 31.5% | 31.1% | -1.6% |
| Mistral-7b | 39.4% | 25.6% | 39.3% | 37.9% | 39.3% | -0.1% |
| Avg. Degrad. | - | -7.6% | -0.3% | -1.1% | -0.5% |  |

| MMLU | FP16 | SQ | INT8() | K-Quant | ours | Ours Degrad. |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen1.5-1.8B | 29.7% | 27.9% | 29.1% | 29.8% | 30.8% | +1.1% |
| Gemma2-2B | 35.7% | 32.1% | 35.1% | 35.1% | 36.4% | +0.7% |
| Phi-2-2.7B | 35.4% | 35.3% | 35.6% | 35.7% | 36.7% | +1.3% |
| LlaMA-2-7B | 37.8% | 29.2% | 38.1% | 34.4% | 36.9% | -0.9% |
| Mistral-7b | 42.1% | 30.9% | 41.4% | 42.3% | 41.0% | -1.1% |
| Avg. Degrad. | - | -5.1% | -0.3% | -0.7% | +0.2% |  |


(a) Qwen1.5-1.8B

> **中文图注：**(一)Qwen1.5-1.8B

(b) Gemma-2B

> **中文图注：**(b) Gemma-2B

Figure 16. Generation speed and accuracy across various outlier pruning rates. The text positioned above or below the plots represents the corresponding pruning rates.

> **中文图注：**图 16.各种异常值修剪率的生成速度和准确性。位于图上方或下方的文本表示相应的修剪率。

Trade-offs between accuracy and performance.
llm.npu balances generation accuracy and speed by adjusting outlier pruning rates, with higher pruning rates leading to faster generation but lower quality.
To comprehensively show how outlier pruning rates influence generate accuracy and speed, we conducted experiments using Qwen1.5-1.8B and Gemma-2B on the Redmi K70 Pro with the HellaSwag ([103](#bib.bib103)) and LAMBADA ([68](#bib.bib68)) datasets. The results are shown in Figure [16](#S4.F16 "Figure 16 ‣ 4.4. Inference accuracy ‣ 4. Implementation and Evaluation ‣ Fast On-device LLM Inference with NPUs").

> **中文译文：**准确性和性能之间的权衡。 llm.npu 通过调整异常剪枝率来平衡生成精度和速度，较高的剪枝率会导致生成速度更快，但质量较低。为了全面展示离群剪枝率如何影响生成精度和速度，我们在 Redmi K70 Pro 上使用 Qwen1.5-1.8B 和 Gemma-2B 以及 HellaSwag ( 103 ) 和 LAMBADA ( 68 ) 数据集进行了实验。结果如图16所示。

When no outliers are pruned (pruning rate = 0%), llm.npu achieves the highest accuracy, albeit with the slowest generation speed. Specifically, the Qwen1.5-1.8B model can attain an accuracy of 71.3% on the Lambada dataset, with a corresponding generation speed of 156 tokens per second. Conversely, the Gemma-2B model demonstrates an accuracy of 59.6% and a generation speed of 102 tokens per second.
Upon pruning 80% of outliers, the Qwen1.5-1.8B model’s accuracy decreases to 42.2%, while its generation speed increases to 544 tokens per second. Similarly, the Gemma-2B model’s accuracy declines to 57.4%, yet its generation speed rises to 285 tokens per second. When all outliers are pruned, generation accuracy reaches its lowest point, while generation speed is at its highest. Specifically, the Qwen1.5-1.8B model demonstrates an accuracy of merely 8.1% with a generation speed of 596 tokens/s; the Gemma-2B model reports an accuracy of 41.9% alongside a generation speed of 345 tokens/s.

> **中文译文：**当没有异常值被剪枝时（剪枝率 = 0%），llm.npu 实现了最高的准确率，尽管生成速度最慢。 Specifically, the Qwen1.5-1.8B model can attain an accuracy of 71.3% on the Lambada dataset, with a corresponding generation speed of 156 tokens per second.相反，Gemma-2B 模型的准确度为 59.6%，生成速度为每秒 102 个令牌。 Upon pruning 80% of outliers, the Qwen1.5-1.8B model’s accuracy decreases to 42.2%, while its generation speed increases to 544 tokens per second.同样，Gemma-2B 模型的准确率下降至 57.4%，但其生成速度却上升至每秒 285 个代币。当所有异常值都被修剪时，生成精度达到最低点，而生成速度达到最高点。具体来说，Qwen1.5-1.8B模型的准确率仅为8.1%，生成速度为596个token/s； Gemma-2B 模型的准确度为 41.9%，生成速度为 345 个令牌/秒。

### 4.5. Memory consumption

> **中文标题：**4.5.内存消耗

We compare llm.npu with INT8 weight baselines, as mobile NPUs only support INT8 weight computations. Memory consumption results on the Redmi K60 Pro, using a 512-token prompt, are presented in Figure [17](#S4.F17 "Figure 17 ‣ 4.5. Memory consumption ‣ 4. Implementation and Evaluation ‣ Fast On-device LLM Inference with NPUs"). llm.npu consumes up to 1.32 $\times$ more memory than llama.cpp and TFLite.
The overhead is due to the MLLM and QNN frameworks, which allocate independent activation buffers for each operator to enhance speed. The tiny additional memory overhead introduced by llm.npu is its $\lx@sectionsign$[3.3](#S3.SS3 "3.3. Shadow outlier execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs") shadow outlier execution technique (in black), which loads tiny float weights into memory, accounting for only 0.6%–1% of the total memory.

> **中文译文：**我们将 llm.npu 与 INT8 权重基线进行比较，因为移动 NPU 仅支持 INT8 权重计算。使用 512 令牌提示时，Redmi K60 Pro 上的内存消耗结果如图 17 所示。 llm.npu 消耗的内存比 llama.cpp 和 TFLite 多 1.32 倍。开销是由于 MLLM 和 QNN 框架造成的，它们为每个运算符分配独立的激活缓冲区以提高速度。 llm.npu 引入的微小额外内存开销是其 \lx@sectionsign 3.3 阴影离群值执行技术（黑色），它将微小的浮点权重加载到内存中，仅占总内存的 0.6%–1%。

Figure 17. Memory consumption of different baselines (prompt length=512).

> **中文图注：**图 17.不同基线的内存消耗（提示长度=512）。


(a) Prefill speed under different prompt length

> **中文图注：**(a) 不同提示长度下的预填充速度

(b) End-to-end latency on the Longbench dataset

> **中文图注：**(b) Longbench 数据集上的端到端延迟

Figure 18. Prefill speed and end-to-end latency comparison between CPU-NPU coordination and GPU-NPU coordination.

> **中文图注：**图 18. CPU-NPU协同和GPU-NPU协同的预填充速度和端到端延迟比较。

### 4.6. GPU-NPU coordination analysis

> **中文标题：**4.6。 GPU-NPU协调分析

llm.npu is compatible with mobile GPUs; however, we have currently implemented only the CPU-NPU version due to easy implementation. To illustrate the potential advantages of employing a mobile GPU, we simulate GPU-NPU co-scheduling, assuming TFLite serves as the GPU backend. The evaluation is conducted using the Gemma-2B model across varying prompt lengths and the Longbench dataset, specifically on the Redmi K70 Pro, as shown in Figure [18](#S4.F18 "Figure 18 ‣ 4.5. Memory consumption ‣ 4. Implementation and Evaluation ‣ Fast On-device LLM Inference with NPUs"). Our observations indicate that while GPU-NPU coordination does not enhance the prefilling speed, it effectively reduces the end-to-end latency.

> **中文译文：**llm.npu 与移动 GPU 兼容；不过，由于实现简单，我们目前只实现了 CPU-NPU 版本。为了说明使用移动 GPU 的潜在优势，我们模拟 GPU-NPU 协同调度，假设 TFLite 作为 GPU 后端。使用 Gemma-2B 模型在不同的提示长度和 Longbench 数据集上进行评估，特别是在 Redmi K70 Pro 上，如图 18 所示。我们的观察表明，虽然 GPU-NPU 协调并没有提高预填充速度，但它有效地降低了端到端延迟。

As illustrated in Figure [18](#S4.F18 "Figure 18 ‣ 4.5. Memory consumption ‣ 4. Implementation and Evaluation ‣ Fast On-device LLM Inference with NPUs")(a), the prefill speed achieved through GPU-NPU coordination is equivalent to that of CPU-NPU coordination. This phenomenon occurs because, within the design of llm.npu, the computational demands placed on mobile CPUs and GPUs are significantly lower than those on the NPU (details in Section [3.4](#S3.SS4 "3.4. Out-of-order subgraph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs")). Consequently, the computational latency of mobile CPUs and GPUs can be effectively hidden by the NPU computation, indicating that the choice of a mobile CPU or mobile GPU is not essential. On the other hand, GPU-NPU coordination can decrease end-to-end latency by 80–90 ms compared to CPU-NPU coordination. This improvement is attributed to the faster decoding speed of GPUs relative to CPUs.

> **中文译文：**如图18（a）所示，通过GPU-NPU协调实现的预填充速度与CPU-NPU协调的预填充速度相当。出现这种现象的原因是，在 llm.npu 的设计中，对移动 CPU 和 GPU 的计算需求明显低于对 NPU 的计算需求（详见 3.4 节）。因此，移动CPU和GPU的计算延迟可以被NPU计算有效隐藏，这表明移动CPU或移动GPU的选择并不是必需的。另一方面，与CPU-NPU协调相比，GPU-NPU协调可以将端到端延迟减少80-90毫秒。这一改进归功于 GPU 相对于 CPU 更快的解码速度。

Figure 19. Ablation study of llm.npu (prompt length=512).

> **中文图注：**图 19. llm.npu 的消融研究（提示长度=512）。

### 4.7. Ablation study.

> **中文标题：**4.7.消融研究。

We conduct a comprehensive breakdown analysis of the benefits brought by each of llm.npu ’s techniques using the Qwen1.5-1.8B, Gemma-2B, and LlaMA2-7B models, as shown in Figure [19](#S4.F19 "Figure 19 ‣ 4.6. GPU-NPU coordination analysis ‣ 4. Implementation and Evaluation ‣ Fast On-device LLM Inference with NPUs").
The leftmost bar represents the prefill speed on the CPU using llama.cpp. The second bar shows a naive implementation on mobile NPUs, followed by the incorporation of our three techniques, with the rightmost bar representing llm.npu.
The three techniques are represented by chunk ($\lx@sectionsign$[3.2](#S3.SS2 "3.2. Chunk-sharing graph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs")), outlier ($\lx@sectionsign$[3.3](#S3.SS3 "3.3. Shadow outlier execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs")), and OOE ($\lx@sectionsign$[3.4](#S3.SS4 "3.4. Out-of-order subgraph execution ‣ 3. llm.npu Design ‣ Fast On-device LLM Inference with NPUs")), respectively.
We observe that llm.npu ’s three techniques make a significant contribution to the overall improvement.

> **中文译文：**我们使用 Qwen1.5-1.8B、Gemma-2B 和 LlaMA2-7B 模型对 llm.npu 的各项技术带来的收益进行了全面的细分分析，如图 19 所示。最左边的条代表使用 llama.cpp 在 CPU 上的预填充速度。第二个条显示了移动 NPU 上的简单实现，然后是我们三种技术的结合，最右边的条代表 llm.npu 。这三种技术分别用 chunk ( \lx@sectionsign 3.2 ) 、 outlier ( \lx@sectionsign 3.3 ) 和 OOE ( \lx@sectionsign 3.4 ) 表示。我们观察到 llm.npu 的三种技术对整体改进做出了重大贡献。

Firstly, directly offloading LLM prefilling workloads to mobile NPUs results in 2.55–2.68$\times$ delays, due to the substantial gap between LLMs and mobile NPUs, as detailed in $\lx@sectionsign$[2.3](#S2.SS3 "2.3. Gaps between LLMs and Mobile NPUs ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs").
Additionally, chunk-sharing graph improves prefill speed by 1.46–5.09$\times$ by reducing graph building and optimization delays. The Gemma-2B model achieves the highest speedup, as it requires more time to build and optimize.
Furthermore, shadow outlier execution reduces prefill latency by 3.91–8.68$\times$, allowing per-tensor MatMul operations to fully utilize NPUs with minimal CPU/GPU overhead.
Lastly, the out-of-order subgraph execution reduces prefill latency by 18%–44% by reducing execution bubbles within the NPU.

> **中文译文：**首先，由于 LLM 和移动 NPU 之间存在巨大差距，直接将 LLM 预填充工作负载卸载到移动 NPU 会导致 2.55–2.68 倍的延迟，如 \lx@sectionsign 2.3 中详述。此外，块共享图通过减少图构建和优化延迟，将预填充速度提高了 1.46–5.09 倍。 Gemma-2B 模型实现了最高的加速，因为它需要更多的时间来构建和优化。此外，影子异常值执行将预填充延迟减少了 3.91–8.68 倍，允许每个张量 MatMul 操作以最小的 CPU/GPU 开销充分利用 NPU。最后，无序子图执行通过减少 NPU 内的执行气泡，将预填充延迟减少了 18%–44%。

## 5. Discussion and Future Work

> **中文标题：**5. 讨论和未来的工作

Advantages of NPUs over GPUs.
Typically, both mobile GPUs and NPUs serve as hardware accelerators for on-device DNN inference ([53](#bib.bib53), [86](#bib.bib86)).
However, as mobile GPUs are already extensively used for graphic rendering on devices, relying on them for LLM inference can cause significant resource contention.
In contrast, mobile NPUs, which are dedicated exclusively to deep learning tasks, have relatively lighter workloads, leaving ample time to run LLM inference.
Besides, as elaborated in Section [2.2](#S2.SS2 "2.2. Opportunity: Mobile NPUs ‣ 2. Background ‣ Fast On-device LLM Inference with NPUs"), mobile NPUs provide greater computational capacity, enhanced energy efficiency, and increased availability compared to mobile GPUs, making them more suitable for on-device LLM inference.

> **中文译文：**NPU 相对于 GPU 的优势。通常，移动 GPU 和 NPU 都充当设备上 DNN 推理的硬件加速器 (53, 86)。然而，由于移动 GPU 已经广泛用于设备上的图形渲染，依赖它们进行 LLM 推理可能会导致严重的资源争用。相比之下，专门用于深度学习任务的移动 NPU 的工作负载相对较轻，因此有充足的时间来运行 LLM 推理。此外，如第 2.2 节所述，与移动 GPU 相比，移动 NPU 提供了更大的计算能力、更高的能源效率和更高的可用性，使其更适合设备上的 LLM 推理。

llm.npu’s technique stability.
Despite the recent trend towards supporting FP16 in advanced NPUs ([13](#bib.bib13), [6](#bib.bib6), [17](#bib.bib17)), llm.npu remains essential for NPU-based prefilling acceleration for two reasons: (1) Integer operations are expected to maintain dominance in mobile NPU-side LLM inference due to their superior computation performance, memory efficiency, and energy savings, while CPU/GPUs serve as supplementary units for enhancing accuracy. Therefore, llm.npu’s three INT8-related techniques, which include co-scheduling across CPU/GPU and NPU to achieve high accuracy and efficiency, will continue to have a significant impact in this domain.
(2) llm.npu is capable of empowering legacy devices with on-device LLM capabilities, democratizing access to LLMs.

> **中文译文：**llm.npu 的技术稳定性。尽管最近有在高级 NPU 中支持 FP16 的趋势（13、6、17），但 llm.npu 对于基于 NPU 的预填充加速仍然至关重要，原因有两个：（1）整数运算因其卓越的计算性能、内存效率和节能而有望在移动 NPU 端 LLM 推理中保持主导地位，而 CPU/GPU 作为补充单元以提高精度。因此，llm.npu 的三种 INT8 相关技术，包括跨 CPU/GPU 和 NPU 的协同调度以实现高精度和高效率，将继续对该领域产生重大影响。 (2) llm.npu 能够为传统设备提供设备上的 LLM 功能，从而实现 LLM 访问的民主化。

Future hardware design implications.
llm.npu’s performance could be further enhanced with the following hardware optimizations:
(1) Dynamic shape-aware optimization: Hardware capable of dynamically reconfiguring to accommodate varying input and output shapes would markedly improve the acceleration of LLM inference.
(2) Increased data cache size: Mobile NPUs generally enhance inference performance by loading an entire layer’s weights into the data cache. Increasing the NPU cache size to align with LLM weight sizes is crucial for optimal performance.
(3) Mixed-precision operands in computing units: Mobile NPUs typically support only uniform precision operands, necessitating additional data format conversions to perform FP16 \* INT8 operations. This overhead can be eliminated by enabling support for mixed-precision operands in computing units.

> **中文译文：**未来硬件设计的影响。 llm.npu 的性能可以通过以下硬件优化进一步增强：（1）动态形状感知优化：能够动态​​重新配置以适应不同输入和输出形状的硬件将显着提高 LLM 推理的加速。 (2) 增加数据缓存大小：移动 NPU 通常通过将整个层的权重加载到数据缓存中来增强推理性能。增加 NPU 缓存大小以与 LLM 权重大小保持一致对于获得最佳性能至关重要。 (3)计算单元中的混合精度操作数：移动NPU通常仅支持统一精度操作数，需要额外的数据格式转换来执行FP16 \* INT8运算。通过在计算单元中启用对混合精度操作数的支持可以消除这种开销。

llm.npu with mobile GPUs.
Our current implementation relies solely on mobile CPUs for floating-point operators and the decoding phase due to ease of implementation. However, the design of llm.npu is compatible with both mobile CPUs and GPUs.
We plan to incorporate GPU-NPU coordination in future developments. The primary engineering efforts in integrating mobile GPUs into llm.npu involve configuring the GPU backend and implementing the necessary operators.
With mobile GPUs, llm.npu can significantly reduce end-to-end inference latency, as elaborated in Section [4.6](#S4.SS6 "4.6. GPU-NPU coordination analysis ‣ 4. Implementation and Evaluation ‣ Fast On-device LLM Inference with NPUs").

> **中文译文：**带有移动 GPU 的 llm.npu。由于易于实现，我们当前的实现仅依赖于移动 CPU 进行浮点运算符和解码阶段。不过，llm.npu 的设计同时兼容移动 CPU 和 GPU。我们计划在未来的开发中纳入 GPU-NPU 协调。将移动 GPU 集成到 llm.npu 的主要工程工作包括配置 GPU 后端和实现必要的运算符。借助移动 GPU，llm.npu 可以显着减少端到端推理延迟，如第 4.6 节中所述。

## 6. Related work

> **中文标题：**六、相关工作

On-device LLM optimization.
To reduce the substantial memory consumption of on-device LLM inference, various compression techniques have been proposed, including quantization and knowledge distillation ([96](#bib.bib96), [35](#bib.bib35), [38](#bib.bib38), [65](#bib.bib65), [80](#bib.bib80), [46](#bib.bib46), [101](#bib.bib101), [70](#bib.bib70), [88](#bib.bib88), [91](#bib.bib91), [98](#bib.bib98), [30](#bib.bib30), [92](#bib.bib92), [58](#bib.bib58)). To minimize on-device LLM computation, researchers have introduced token pruning ([79](#bib.bib79), [31](#bib.bib31), [51](#bib.bib51), [72](#bib.bib72), [29](#bib.bib29)), which prunes unnecessary tokens during the inference process.
Speculative decoding, a method that accelerates token generation by offloading tasks to a smaller LLM, has been widely adopted in open-source frameworks ([59](#bib.bib59), [3](#bib.bib3)) and extensively researched ([50](#bib.bib50), [61](#bib.bib61), [95](#bib.bib95), [42](#bib.bib42), [36](#bib.bib36), [32](#bib.bib32), [87](#bib.bib87)). As a system optimization, llm.npu is orthogonal and compatible with these algorithm-level optimizations.

> **中文译文：**设备上 LLM 优化。为了减少设备上LLM推理的大量内存消耗，人们提出了各种压缩技术，包括量化和知识蒸馏（96,35,38,65,80,46,101,70,88,91,98,30,92,58）。为了最大限度地减少设备上的 LLM 计算，研究人员引入了令牌修剪 (79, 31, 51, 72, 29)，它在推理过程中修剪不必要的令牌。推测解码是一种通过将任务卸载到较小的 LLM 来加速令牌生成的方法，已在开源框架中广泛采用 (59, 3) 并进行了广泛研究 (50, 61, 95, 42, 36, 32, 87)。作为系统优化，llm.npu 是正交的并且兼容这些算法级别的优化。

On-chip offloading for ML.
This has been thoroughly studied to enable faster DNN inference by leveraging heterogeneous mobile processors like GPUs and NPUs ([52](#bib.bib52), [104](#bib.bib104), [39](#bib.bib39), [53](#bib.bib53), [41](#bib.bib41), [105](#bib.bib105), [54](#bib.bib54), [37](#bib.bib37), [86](#bib.bib86), [94](#bib.bib94), [66](#bib.bib66), [63](#bib.bib63), [45](#bib.bib45), [89](#bib.bib89), [85](#bib.bib85), [93](#bib.bib93)). MobiSR ([54](#bib.bib54)) utilizes mobile NPUs to speed up super-resolution computation.
However, these methods do not address LLM-specific features and are unsuitable for on-device LLM scenarios.
The most relevant work is PowerInfer-V2 ([94](#bib.bib94)), which also utilizes mobile NPUs for the prefilling.
But, it does not comprehensively analyze the challenges of on-device NPU offloading for LLM prefilling or incorporate techniques to fully harness NPU capability.
Our experiments demonstrate that our approach is up to 5.32$\times$ more efficient in the prefill stage. llm.npu is inspired by these efforts and is the first LLM inference framework with efficient on-device NPU offloading.

> **中文译文：**ML 的片上卸载。我们对此进行了深入研究，以便通过利用 GPU 和 NPU（52、104、39、53、41、105、54、37、86、94、66、63、45、89、85、93）等异构移动处理器实现更快的 DNN 推理。 MobiSR（54）利用移动NPU来加速超分辨率计算。然而，这些方法并没有解决LLM特定的功能，并且不适合设备上的LLM场景。最相关的工作是 PowerInfer-V2 (94)，它也利用移动 NPU 进行预填充。但是，它没有全面分析 LLM 预填充的设备上 NPU 卸载的挑战，也没有整合充分利用 NPU 功能的技术。我们的实验表明，我们的方法在预填充阶段的效率提高了 5.32 倍。 llm.npu 受到这些努力的启发，是第一个具有高效设备上 NPU 卸载功能的 LLM 推理框架。

Mobile NPU execution optimization.
With the growing importance of mobile NPUs in smartphones, significant efforts have been made to optimize their execution efficiency ([73](#bib.bib73), [76](#bib.bib76), [64](#bib.bib64), [78](#bib.bib78), [71](#bib.bib71), [34](#bib.bib34)). Pitchfork ([73](#bib.bib73)) defines a portable fixed-point intermediate representation to optimize fixed-point execution efficiency. Isaria ([76](#bib.bib76)) proposes a framework for automatically generating vectorizing compilers for DSP architectures, creating efficient operator codes for mobile NPUs.
As a system framework, llm.npu is orthogonal and can leverage them to generate more efficient operator libraries as execution backend, further boosting performance.

> **中文译文：**移动NPU执行优化。随着移动 NPU 在智能手机中的重要性日益增加，人们在优化其执行效率方面做出了巨大努力 (73,76,64,78,71,34)。 Pitchfork（73）定义了可移植的定点中间表示以优化定点执行效率。 Isaria (76) 提出了一个框架，用于自动生成 DSP 架构的矢量化编译器，为移动 NPU 创建高效的运算符代码。作为一个系统框架，llm.npu 是正交的，可以利用它们生成更高效的算子库作为执行后端，进一步提升性能。

## 7. Conclusions

> **中文标题：**七、结论

This paper has proposed llm.npu, the first LLM inference system utilizing on-device NPU offloading to reduce prefill latency and energy consumption.
llm.npu has incorporated novel techniques: chunk-sharing graph, shadow outlier execution and out-of-order subgraph execution to enhance NPU offloading efficiency.
Extensive experiments have demonstrated llm.npu to show its superior performance benefits, e.g, up to 43.6$\times$ speedup and 59.5$\times$ energy savings.

> **中文译文：**本文提出了 llm.npu ，这是第一个利用设备上 NPU 卸载来减少预填充延迟和能耗的 LLM 推理系统。 llm.npu 融入了新技术：块共享图、影子离群值执行和乱序子图执行，以提高 NPU 卸载效率。大量实验证明 llm.npu 具有卓越的性能优势，例如高达 43.6 倍的加速和 59.5 倍的节能。

## 8. Acknowledgement

> **中文标题：**8. 致谢

This work was supported by the National Natural Science Foundation of China under grant number 62325201 and sponsored by the Huawei University Joint Research Program.

> **中文译文：**该工作得到了国家自然科学基金委（编号：62325201）和华为大学联合研究项目的资助。

## Appendix A Artifact Appendix

> **中文标题：**附录 A 工件附录

### A.1. Abstract

> **中文标题：**A.1.抽象的

The artificial evaluations primarily present the prefill performance results detailed in Section 4.2 and the inference accuracy evaluation results discussed in Section 4.4. These results can be found in the performance\_results and accuracy\_results directories in our GitHub repository11
1
<https://doi.org/10.5281/zenodo.14392760>.
To conduct these evaluations, please refer to the README.md file in the repository.
Each evaluation directory in the repository contains detailed instructions for installing and executing evaluations for llm.npu and all baseline methods. This includes guidance on environment setup, LLM and dataset downloads, source code compilation and execution, and clarity on results.
We recommend performing the accuracy evaluation first, followed by the prefill performance test. For optimal results, use a Redmi K70 Pro 24G for on-device LLM inference and an NVIDIA A100 40G GPU server for testing, as these setups have been validated by the authors. Most experiments utilize Qwen1.5-1.8B as a representative example to demonstrate the functionality and evaluation results of llm.npu. Users have the flexibility to tailor their own LLM models by adjusting the code, if needed. Once this artifact evaluation is accepted, we will archive the evaluation results.

> **中文译文：**人工评估主要呈现第 4.2 节中详述的预填充性能结果和第 4.4 节中讨论的推理准确性评估结果。这些结果可以在我们的 GitHub 存储库的 Performance\_results 和 Accuracy\_results 目录中找到 1 1 1 https://doi.org/10.5281/zenodo.14392760 。要进行这些评估，请参阅存储库中的 README.md 文件。存储库中的每个评估目录都包含有关安装和执行 llm.npu 和所有基线方法评估的详细说明。这包括环境设置、LLM 和数据集下载、源代码编译和执行以及结果清晰度的指导。我们建议首先执行准确性评估，然后进行预填充性能测试。为了获得最佳结果，请使用 Redmi K70 Pro 24G 进行设备上 LLM 推理，并使用 NVIDIA A100 40G GPU 服务器进行测试，因为这些设置已经过作者的验证。大多数实验都以 Qwen1.5-1.8B 作为代表性示例来演示 llm.npu 的功能和评估结果。如果需要，用户可以通过调整代码灵活地定制自己的 LLM 模型。一旦工件评估被接受，我们将存档评估结果。

### A.2. Artifact check-list (meta-information)

> **中文标题：**A.2.工件检查表（元信息）

- •

  Compilation: Support compilation from users.

  > **中文译文：**编译：支持用户编译。
- •

  Binary: Compiled binaries are available.

  > **中文译文：**二进制：已编译的二进制文件可用。
- •

  Model: Qwen1.5-1.8B-Chat is used as demonstration model. Gemma-2B, Phi-2-2.7B, LLaMA-2-7B, and Mistral-7B are also supported.

  > **中文译文：**示范模型采用型号：Qwen1.5-1.8B-Chat。还支持 Gemma-2B、Phi-2-2.7B、LLaMA-2-7B 和 Mistral-7B。
- •

  Dataset: HellaSwag ([103](#bib.bib103)) and LAMBADA ([68](#bib.bib68)) are used as demonstration datasets. OpenBookQA ([62](#bib.bib62)), WinoGrande ([4](#bib.bib4)), and MMLU ([43](#bib.bib43)) are also supported.

  > **中文译文：**数据集：HellaSwag（103）和LAMBADA（68）用作演示数据集。还支持 OpenBookQA (62)、WinoGrande (4) 和 MMLU (43)。
- •

  Server Hardware: NVIDIA-A100 40G

  > **中文译文：**服务器硬件：NVIDIA-A100 40G
- •

  Device Hardware: Redmi K70 Pro 24G

  > **中文译文：**设备硬件：Redmi K70 Pro 24G
- •

  Execution: C++, Python and bash scripts

  > **中文译文：**执行：C++、Python 和 bash 脚本
- •

  Metrics: inference accuracy and prefill speed

  > **中文译文：**指标：推理准确性和预填充速度
- •

  Output: log in terminal

  > **中文译文：**输出：登录终端
- •

  Experiments: Section 4.2 and Section 4.4

  > **中文译文：**实验：第 4.2 节和第 4.4 节
- •

  How much disk space required (approximately)?: 200GB

  > **中文译文：**需要多少磁盘空间（大约）：200GB
- •

  How much time is needed to prepare workflow (approximately)?: 5 hours

  > **中文译文：**准备工作流程需要多长时间（大约）：5小时
- •

  How much time is needed to complete experiments (approximately)?: 2 hours

  > **中文译文：**完成实验需要多长时间（大约）：2小时
- •

  Publicly available?: Yes

  > **中文译文：**是否公开？： 是
- •

  Code licenses (if publicly available)?: MIT

  > **中文译文：**代码许可证（如果公开可用）？：麻省理工学院
- •

  Data licenses (if publicly available)?: MIT

  > **中文译文：**数据许可证（如果公开可用）？：麻省理工学院
- •

  Workflow automation framework used?: mllm ([23](#bib.bib23))

  > **中文译文：**使用的工作流自动化框架？：mllm ( 23 )
- •

  Archived (provide DOI)?: <https://doi.org/10.5281/zenodo.14392760>.

  > **中文译文：**已存档（提供 DOI）？：https://doi.org/10.5281/zenodo.14392760 。

### A.3. Description

> **中文标题：**A.3.描述

#### A.3.1. How to access

> **中文标题：**A.3.1。如何访问

Our artificial evaluation is publicly accessible at

> **中文译文：**我们的人工评估可公开访问：

<https://doi.org/10.5281/zenodo.14392760>.

> **中文译文：**https://doi.org/10.5281/zenodo.14392760。

#### A.3.2. Hardware dependencies

> **中文标题：**A.3.2。硬件依赖性

For the accuracy results discussed in Section 4.4, reviewers will require access to a GPU server equipped with an NVIDIA A100 40GB GPU, x86 CPU cores, and preferably over 200GB of CPU memory. The server should operate on Ubuntu 20.04 OS.

> **中文译文：**对于第 4.4 节中讨论的准确性结果，审核者将需要访问配备 NVIDIA A100 40GB GPU、x86 CPU 内核以及最好超过 200GB CPU 内存的 GPU 服务器。服务器应在 Ubuntu 20.04 操作系统上运行。

To assess the performance results in Section 4.2, reviewers will need a Redmi K70 Pro with 24GB of memory.

> **中文译文：**为了评估第 4.2 节中的性能结果，评测人员需要配备 24GB 内存的 Redmi K70 Pro。

#### A.3.3. Software dependencies

> **中文标题：**A.3.3。软件依赖性

For the accuracy results outlined in Section 4.4, the server should have CUDA 12.5 installed and a Conda virtual environment with Python version 3.8 or higher.

> **中文译文：**为了获得第 4.4 节中概述的准确性结果，服务器应安装 CUDA 12.5 和 Python 版本 3.8 或更高版本的 Conda 虚拟环境。

#### A.3.4. Datasets

> **中文标题：**A.3.4。数据集

The datasets used for accuracy results include HellaSwag ([103](#bib.bib103)), LAMBADA, OpenBookQA ([62](#bib.bib62)), WinoGrande ([4](#bib.bib4)), and MMLU ([43](#bib.bib43)). All datasets can be automatically downloaded using our code from Hugging Face ([21](#bib.bib21)).

> **中文译文：**用于准确性结果的数据集包括 HellaSwag (103)、LAMBADA、OpenBookQA (62)、WinoGrande (4) 和 MMLU (43)。所有数据集都可以使用我们的 Hugging Face(21) 代码自动下载。

#### A.3.5. Models

> **中文标题：**A.3.5。型号

We demonstrate llm.npu’s functionality on devices using Qwen1.5-1.8B. Users may also perform additional evaluations by integrating customized models like Gemma-2B, Phi-2-2.7B, LLaMA-2-7B, and Mistral-7B.

> **中文译文：**我们在使用 Qwen1.5-1.8B 的设备上演示 llm.npu 的功能。用户还可以通过集成 Gemma-2B、Phi-2-2.7B、LLaMA-2-7B 和 Mistral-7B 等定制模型来执行其他评估。

### A.4. Installation

> **中文标题：**A.4.安装

To obtain the accuracy results discussed in Section 4.4, users should begin by using Conda to install the Python virtual environment specified in accuracy\_results/environment.yml. This environment includes all the necessary dependencies to reproduce the accuracy results of llm.npu.
When evaluating the baseline methods, particularly SmoothQuant ([84](#bib.bib84)), users need to install the SmoothQuant environment as detailed in accuracy\_results/baselines/README.md.

> **中文译文：**要获得第 4.4 节中讨论的准确度结果，用户应首先使用 Conda 安装 precision\_results/environment.yml 中指定的 Python 虚拟环境。该环境包含重现 llm.npu 准确性结果所需的所有依赖项。在评估基线方法时，特别是 SmoothQuant (84)，用户需要安装 SmoothQuant 环境，详细信息请参阅 precision\_results/baselines/README.md。

For the performance results outlined in Section 4.2, users are advised to refer to and follow the installation instructions provided in each respective README.md file within the performance\_results directory.

> **中文译文：**对于第 4.2 节中概述的性能结果，建议用户参考并遵循 Performance\_results 目录中每个相应 README.md 文件中提供的安装说明。

### A.5. Experiment workflow

> **中文标题：**A.5.实验流程

Initially, users should perform the accuracy results evaluation (accuracy\_results) to generate the quantized model necessary for on-device inference with llm.npu (accuracy\_results/ llm.npu/examples/act\_scales\_try/llmnpu\_get\_int8\_weights \_finalmodel \_int8bias\_ns.py). Subsequently, using the generated model, users should proceed with the performance results evaluation (performance\_results).

> **中文译文：**最初，用户应执行准确性结果评估 (accuracy\_results)，以使用 llm.npu 生成设备上推理所需的量化模型 (accuracy\_results/llm.npu/examples/act\_scales\_try/llmnpu\_get\_int8\_weights \_finalmodel \_int8bias\_ns.py)。随后，使用生成的模型，用户应该继续进行性能结果评估（performance\_results）。

### A.6. Evaluation and expected results

> **中文标题：**A.6.评价及预期结果

We offer two primary types of evaluation results: accuracy evaluation results for Section 4.4 and prefill performance evaluation results for Section 4.2. These results contribute to the data reported in Table 6 and illustrated in Figure 14.

> **中文译文：**我们提供两种主要类型的评估结果：第 4.4 节的准确性评估结果和第 4.2 节的预填充性能评估结果。这些结果构成了表 6 中报告的数据和图 14 中所示的数据。

### A.7. Methodology

> **中文标题：**A.7.方法论

Availability, Functional, and Reproducibility

> **中文译文：**可用性、功能性和可重复性

## References

- [1]

  AI Core.
  <https://developer.android.com/ai/aicore>.
  Accessed: [2023.7].
- [2]

  Apple Intelligence.
  <https://www.apple.com/apple-intelligence/>.
  Accessed: [2023.7].
- [3]

  Gboard - the Google Keyboard - Apps on Google Play —
  play.google.com.
  <https://github.com/NVIDIA/FasterTransformer>.
  [Accessed 22-Oct-2023].
- [4]

  Winogrande: An adversarial winograd schema challenge at scale.
  2019.
- [5]

  General data protection regulation.
  <https://gdpr-info.eu/>, 2021.
- [6]

  AMD Strix Point (Ryzen 300).
  <https://www.sellcell.com/blog/how-often-do-people-upgrade-their-phone-2023-statistics>,
  2023.
- [7]

  Ascend NPU.
  <https://www.hisilicon.com/en/products/Kirin/Kirin-flagship-chips/Kirin-9000>,
  2023.
- [8]

  Edge TPU API.
  <https://coral.ai/docs/edgetpu/inference/#general-purpose-operating-systems>,
  2023.
- [9]

  Gemma-2B.
  <https://huggingface.co/google/gemma-2b>, 2023.
- [10]

  HiAI Engine.
  <https://developer.huawei.com/consumer/cn/doc/hiai-References/overview-0000001053824513>,
  2023.
- [11]

  Llama-2-7b.
  <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>, 2023.
- [12]

  MediaTek APU 790.
  <https://corp.mediatek.com/news-events/press-releases/mediateks-new-all-big-core-design-for-flagship-dimensity-9300-chipset-maximizes-smartphone-performance-and-efficiency>,
  2023.
- [13]

  Meteor Lake.
  <https://www.sellcell.com/blog/how-often-do-people-upgrade-their-phone-2023-statistics>,
  2023.
- [14]

  Mistral-7B.
  <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>,
  2023.
- [15]

  Neuro Pilot.
  <https://neuropilot.mediatek.com/>, 2023.
- [16]

  Phi-2.
  <https://https://huggingface.co/microsoft/phi-2>, 2023.
- [17]

  Snapdragon 8gen3 soc.
  <https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-3-mobile-platform>,
  2023.
- [18]

  Ai benchmark.
  <https://ai-benchmark.com/ranking_detailed.html>, 2024.
- [19]

  Edgetpu.
  <https://cloud.google.com/edge-tpu>, 2024.
- [20]

  Gpt-based email writer.
  <https://hix.ai/ai-email-writer-email-generator>, 2024.
- [21]

  Hugging face.
  <https://huggingface.co/>, 2024.
- [22]

  Llamatouch.
  <https://github.com/LlamaTouch/LlamaTouch>, 2024.
- [23]

  Mllm.
  <https://github.com/UbiquitousLearning/mllm>, 2024.
- [24]

  Mmlu leader board.
  <https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu>,
  2024.
- [25]

  Qnn.
  <https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk>,
  2024.
- [26]

  Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah,
  Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl,
  et al.
  Phi-3 technical report: A highly capable language model locally on
  your phone.
  arXiv preprint arXiv:2404.14219, 2024.
- [27]

  Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan,
  Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji
  Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men,
  Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang,
  Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang,
  Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan,
  Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou,
  Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu.
  Qwen technical report.
  arXiv preprint arXiv:2309.16609, 2023.
- [28]

  Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang,
  Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al.
  Longbench: A bilingual, multitask benchmark for long context
  understanding.
  arXiv preprint arXiv:2308.14508, 2023.
- [29]

  Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph
  Feichtenhofer, and Judy Hoffman.
  Token merging: Your vit but faster.
  arXiv preprint arXiv:2210.09461, 2022.
- [30]

  Dongqi Cai, Yaozong Wu, Shangguang Wang, Felix Xiaozhu Lin, and Mengwei Xu.
  Efficient federated learning for modern nlp.
  In Proceedings of the 29th Annual International Conference on
  Mobile Computing and Networking, pages 1–16, 2023.
- [31]

  Han Cai, Ji Lin, Yujun Lin, Zhijian Liu, Haotian Tang, Hanrui Wang, Ligeng Zhu,
  and Song Han.
  Enable deep learning on mobile devices: Methods, systems, and
  applications.
  ACM Transactions on Design Automation of Electronic Systems
  (TODAES), 27(3):1–50, 2022.
- [32]

  Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen,
  and Tri Dao.
  Medusa: Simple llm inference acceleration framework with multiple
  decoding heads.
  arXiv preprint arXiv:2401.10774, 2024.
- [33]

  Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer.
  Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale.
  Advances in Neural Information Processing Systems,
  35:30318–30332, 2022.
- [34]

  Franz Franchetti and Markus Püschel.
  Generating simd vectorized permutations.
  In International Conference on Compiler Construction, pages
  116–131. Springer, 2008.
- [35]

  Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh.
  Gptq: Accurate post-training quantization for generative pre-trained
  transformers.
  arXiv preprint arXiv:2210.17323, 2022.
- [36]

  Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang.
  Break the sequential dependency of llm inference using lookahead
  decoding.
  arXiv preprint arXiv:2402.02057, 2024.
- [37]

  Petko Georgiev, Nicholas D Lane, Kiran K Rachuri, and Cecilia Mascolo.
  Dsp. ear: Leveraging co-processor support for continuous audio
  sensing on smartphones.
  In Proceedings of the 12th ACM Conference on Embedded Network
  Sensor Systems, pages 295–309, 2014.
- [38]

  Hui Guan, Shaoshan Liu, Xiaolong Ma, Wei Niu, Bin Ren, Xipeng Shen, Yanzhi
  Wang, and Pu Zhao.
  Cocopie: Enabling real-time ai on off-the-shelf mobile devices via
  compression-compilation co-design.
  Communications of the ACM, 64(6):62–68, 2021.
- [39]

  Donghee Ha, Mooseop Kim, KyeongDeok Moon, and Chi Yoon Jeong.
  Accelerating on-device learning with layer-wise processor selection
  method on unified memory.
  IEEE Sensors, 21(7):2364, 2021.
- [40]

  Lixiang Han, Zimu Zhou, and Zhenjiang Li.
  Pantheon: Preemptible multi-dnn inference on mobile edge gpus.
  In Proceedings of the 22nd Annual International Conference on
  Mobile Systems, Applications and Services, pages 465–478, 2024.
- [41]

  Myeonggyun Han, Jihoon Hyun, Seongbeom Park, Jinsu Park, and Woongki Baek.
  Mosaic: Heterogeneity-, communication-, and constraint-aware model
  slicing and execution for accurate and efficient inference.
  In 2019 28th International Conference on Parallel Architectures
  and Compilation Techniques, pages 165–177. IEEE, 2019.
- [42]

  Zhenyu He, Zexuan Zhong, Tianle Cai, Jason D Lee, and Di He.
  Rest: Retrieval-based speculative decoding.
  arXiv preprint arXiv:2311.08252, 2023.
- [43]

  Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn
  Song, and Jacob Steinhardt.
  Measuring massive multitask language understanding.
  Proceedings of the International Conference on Learning
  Representations (ICLR), 2021.
- [44]

  Karla L Hoffman, Manfred Padberg, Giovanni Rinaldi, et al.
  Traveling salesman problem.
  Encyclopedia of operations research and management science,
  1:1573–1578, 2013.
- [45]

  Kuan-Chieh Hsu and Hung-Wei Tseng.
  Simultaneous and heterogenous multithreading.
  In Proceedings of the 56th Annual IEEE/ACM International
  Symposium on Microarchitecture, pages 137–152, 2023.
- [46]

  Loc N Huynh, Youngki Lee, and Rajesh Krishna Balan.
  Deepmon: Mobile gpu-based deep learning framework for continuous
  vision applications.
  In Proceedings of the 15th Annual International Conference on
  Mobile Systems, Applications, and Services, pages 82–95, 2017.
- [47]

  Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew
  Howard, Hartwig Adam, and Dmitry Kalenichenko.
  Quantization and training of neural networks for efficient
  integer-arithmetic-only inference.
  In Proceedings of the IEEE conference on computer vision and
  pattern recognition, pages 2704–2713, 2018.
- [48]

  Pegah Jandaghi, XiangHai Sheng, Xinyi Bai, Jay Pujara, and Hakim Sidahmed.
  Faithful persona-based conversational dataset generation with large
  language models, 2023.
- [49]

  Xiaotang Jiang, Huan Wang, Yiliu Chen, Ziqi Wu, Lichuan Wang, Bin Zou, Yafeng
  Yang, Zongyang Cui, Yu Cai, Tianhang Yu, et al.
  Mnn: A universal and efficient inference engine.
  arXiv preprint arXiv:2002.12418, 2020.
- [50]

  Sehoon Kim, Karttikeya Mangalam, Jitendra Malik, Michael W Mahoney, Amir
  Gholami, and Kurt Keutzer.
  Big little transformer decoder.
  arXiv preprint arXiv:2302.07863, 2023.
- [51]

  Sehoon Kim, Sheng Shen, David Thorsley, Amir Gholami, Woosuk Kwon, Joseph
  Hassoun, and Kurt Keutzer.
  Learned token pruning for transformers.
  In Proceedings of the 28th ACM SIGKDD Conference on Knowledge
  Discovery and Data Mining, pages 784–794, 2022.
- [52]

  Youngsok Kim, Joonsung Kim, Dongju Chae, Daehyun Kim, and Jangwoo Kim.
  $\mu$layer: Low latency on-device inference using cooperative
  single-layer acceleration and processor-friendly quantization.
  In Proceedings of the Fourteenth EuroSys Conference 2019, pages
  1–15, 2019.
- [53]

  Nicholas D Lane, Sourav Bhattacharya, Petko Georgiev, Claudio Forlivesi, Lei
  Jiao, Lorena Qendro, and Fahim Kawsar.
  Deepx: A software accelerator for low-power deep learning inference
  on mobile devices.
  In 2016 15th ACM/IEEE International Conference on Information
  Processing in Sensor Networks, pages 1–12. IEEE, 2016.
- [54]

  Royson Lee, Stylianos I Venieris, Lukasz Dudziak, Sourav Bhattacharya, and
  Nicholas D Lane.
  Mobisr: Efficient on-device super-resolution through heterogeneous
  mobile processors.
  In The 25th Annual International Conference on Mobile Computing
  and Networking, pages 1–16, 2019.
- [55]

  Yuanchun Li, Hao Wen, Weijun Wang, Xiangyu Li, Yizhen Yuan, Guohong Liu,
  Jiacheng Liu, Wenxing Xu, Xiang Wang, Yi Sun, et al.
  Personal llm agents: Insights and survey about the capability,
  efficiency and security.
  arXiv preprint arXiv:2401.05459, 2024.
- [56]

  Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han.
  Awq: Activation-aware weight quantization for llm compression and
  acceleration.
  arXiv preprint arXiv:2306.00978, 2023.
- [57]

  TensorFlow Lite.
  Deploy machine learning models on mobile and iot devices, 2019.
- [58]

  Yi Liu, Yun Ma, Xusheng Xiao, Tao Xie, and Xuanzhe Liu.
  Legodroid: flexible android app decomposition and instant
  installation.
  Sci. China Inf. Sci., 66(4), 2023.
- [59]

  llama.cpp.
  Port of Facebook’s LLaMA model in C/C++ Resources.
  <https://github.com/ggerganov/llama.cpp>, Year of publication.
  Accessed: [2023.7].
- [60]

  Zhenyan Lu, Xiang Li, Dongqi Cai, Rongjie Yi, Fangming Liu, Xiwen Zhang,
  Nicholas D Lane, and Mengwei Xu.
  Small language models: Survey, measurements, and insights.
  arXiv preprint arXiv:2409.15790, 2024.
- [61]

  Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Rae
  Ying Yee Wong, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao
  Jia.
  Specinfer: Accelerating generative llm serving with speculative
  inference and token tree verification.
  arXiv preprint arXiv:2305.09781, 2023.
- [62]

  Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal.
  Can a suit of armor conduct electricity? a new dataset for open book
  question answering.
  In EMNLP, 2018.
- [63]

  Wei Niu, Gagan Agrawal, and Bin Ren.
  Sod2: Statically optimizing dynamic deep neural network.
  arXiv preprint arXiv:2403.00176, 2024.
- [64]

  Wei Niu, Jiexiong Guan, Xipeng Shen, Yanzhi Wang, Gagan Agrawal, and Bin Ren.
  Gcd 2: A globally optimizing compiler for mapping dnns to mobile
  dsps.
  In 2022 55th IEEE/ACM International Symposium on
  Microarchitecture (MICRO), pages 512–529. IEEE, 2022.
- [65]

  Wei Niu, Xiaolong Ma, Sheng Lin, Shihao Wang, Xuehai Qian, Xue Lin, Yanzhi
  Wang, and Bin Ren.
  Patdnn: Achieving real-time dnn execution on mobile devices with
  pattern-based weight pruning.
  In Proceedings of the Twenty-Fifth International Conference on
  Architectural Support for Programming Languages and Operating Systems, pages
  907–922, 2020.
- [66]

  Wei Niu, Md Musfiqur Rahman Sanim, Zhihao Shu, Jiexiong Guan, Xipeng Shen, Miao
  Yin, Gagan Agrawal, and Bin Ren.
  Smartmem: Layout transformation elimination and adaptation for
  efficient dnn execution on mobile.
  In Proceedings of the 29th ACM International Conference on
  Architectural Support for Programming Languages and Operating Systems, Volume
  3, pages 916–931, 2024.
- [67]

  Christopher Olston, Noah Fiedel, Kiril Gorovoy, Jeremiah Harmsen, Li Lao,
  Fangwei Li, Vinu Rajashekhar, Sukriti Ramesh, and Jordan Soyke.
  Tensorflow-serving: Flexible, high-performance ml serving.
  arXiv preprint arXiv:1712.06139, 2017.
- [68]

  Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham,
  Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel
  Fernandez.
  The LAMBADA dataset: Word prediction requiring a broad discourse
  context.
  In Proceedings of the 54th Annual Meeting of the Association for
  Computational Linguistics (Volume 1: Long Papers), pages 1525–1534, Berlin,
  Germany, August 2016. Association for Computational Linguistics.
- [69]

  Pratyush Patel, Esha Choukse, Chaojie Zhang, Íñigo Goiri, Aashaka Shah,
  Saeed Maleki, and Ricardo Bianchini.
  Splitwise: Efficient generative llm inference using phase splitting.
  arXiv preprint arXiv:2311.18677, 2023.
- [70]

  Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury,
  Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean.
  Efficiently scaling transformer inference.
  Proceedings of Machine Learning and Systems, 5, 2023.
- [71]

  Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo
  Durand, and Saman Amarasinghe.
  Halide: a language and compiler for optimizing parallelism, locality,
  and recomputation in image processing pipelines.
  Acm Sigplan Notices, 48(6):519–530, 2013.
- [72]

  Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh.
  Dynamicvit: Efficient vision transformers with dynamic token
  sparsification.
  Advances in neural information processing systems,
  34:13937–13949, 2021.
- [73]

  Alexander J Root, Maaz Bin Safeer Ahmad, Dillon Sharlet, Andrew Adams, Shoaib
  Kamil, and Jonathan Ragan-Kelley.
  Fast instruction selection for fast digital signal processing.
  In Proceedings of the 28th ACM International Conference on
  Architectural Support for Programming Languages and Operating Systems, Volume
  4, pages 125–137, 2023.
- [74]

  Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen.
  Powerinfer: Fast large language model serving with a consumer-grade
  gpu.
  arXiv preprint arXiv:2312.12456, 2023.
- [75]

  MLC team.
  MLC-LLM.
  <https://github.com/mlc-ai/mlc-llm>, 2023.
- [76]

  Samuel Thomas and James Bornholt.
  Automatic generation of vectorizing compilers for customizable
  digital signal processors.
  2024.
- [77]

  Philippe Tillet, Hsiang-Tsung Kung, and David Cox.
  Triton: an intermediate language and compiler for tiled neural
  network computations.
  In Proceedings of the 3rd ACM SIGPLAN International Workshop on
  Machine Learning and Programming Languages, pages 10–19, 2019.
- [78]

  Sander Vocke, Henk Corporaal, Roel Jordans, Rosilde Corvino, and Rick Nas.
  Extending halide to improve software development for imaging dsps.
  ACM Transactions on Architecture and Code Optimization (TACO),
  14(3):1–25, 2017.
- [79]

  Hanrui Wang, Zhekai Zhang, and Song Han.
  Spatten: Efficient sparse attention architecture with cascade token
  and head pruning.
  In 2021 IEEE International Symposium on High-Performance
  Computer Architecture (HPCA), pages 97–110. IEEE, 2021.
- [80]

  Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou.
  Minilm: Deep self-attention distillation for task-agnostic
  compression of pre-trained transformers.
  Advances in Neural Information Processing Systems,
  33:5776–5788, 2020.
- [81]

  Hao Wen, Yuanchun Li, Guohong Liu, Shanhui Zhao, Tao Yu, Toby Jia-Jun Li, Shiqi
  Jiang, Yunhao Liu, Yaqin Zhang, and Yunxin Liu.
  Empowering llm to use smartphone for intelligent task automation.
  arXiv preprint arXiv:2308.15272, 2023.
- [82]

  Hao Wen, Yuanchun Li, Guohong Liu, Shanhui Zhao, Tao Yu, Toby Jia-Jun Li, Shiqi
  Jiang, Yunhao Liu, Yaqin Zhang, and Yunxin Liu.
  Autodroid: Llm-powered task automation in android.
  In Proceedings of the 30th Annual International Conference on
  Mobile Computing and Networking, pages 543–557, 2024.
- [83]

  Bingyang Wu, Shengyu Liu, Yinmin Zhong, Peng Sun, Xuanzhe Liu, and Xin Jin.
  Loongserve: Efficiently serving long-context large language models
  with elastic sequence parallelism.
  arXiv preprint arXiv:2404.09526, 2024.
- [84]

  Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han.
  Smoothquant: Accurate and efficient post-training quantization for
  large language models.
  In International Conference on Machine Learning, pages
  38087–38099. PMLR, 2023.
- [85]

  Daliang Xu, Qing Li, Mengwei Xu, Kang Huang, Gang Huang, Shangguang Wang, Xin
  Jin, Yun Ma, and Xuanzhe Liu.
  Niagara: Scheduling dnn inference services on heterogeneous edge
  processors.
  In International Conference on Service-Oriented Computing,
  pages 67–85. Springer, 2023.
- [86]

  Daliang Xu, Mengwei Xu, Qipeng Wang, Shangguang Wang, Yun Ma, Kang Huang, Gang
  Huang, Xin Jin, and Xuanzhe Liu.
  Mandheling: Mixed-precision on-device dnn training with dsp
  offloading.
  In Proceedings of the 28th Annual International Conference on
  Mobile Computing And Networking, pages 214–227, 2022.
- [87]

  Daliang Xu, Wangsong Yin, Xin Jin, Ying Zhang, Shiyun Wei, Mengwei Xu, and
  Xuanzhe Liu.
  Llmcad: Fast and scalable on-device large language model inference.
  arXiv preprint arXiv:2309.04255, 2023.
- [88]

  Daliang Xu, Hao Zhang, Liming Yang, Ruiqi Liu, Mengwei Xu, and Xuanzhe Liu.
  Wip: Efficient llm prefilling with mobile npu.
  In Proceedings of the Workshop on Edge and Mobile Foundation
  Models, EdgeFM ’24, page 33–35, New York, NY, USA, 2024. Association for
  Computing Machinery.
- [89]

  Mengwei Xu, Dongqi Cai, Yaozong Wu, Xiang Li, and Shangguang Wang.
  $\{$FwdLLM$\}$: Efficient federated finetuning of large language
  models with perturbed inferences.
  In 2024 USENIX Annual Technical Conference (USENIX ATC 24),
  pages 579–596, 2024.
- [90]

  Mengwei Xu, Jiawei Liu, Yuanqiang Liu, Felix Xiaozhu Lin, Yunxin Liu, and
  Xuanzhe Liu.
  A first look at deep learning apps on smartphones.
  In The World Wide Web Conference, pages 2125–2136, 2019.
- [91]

  Mengwei Xu, Wangsong Yin, Dongqi Cai, Rongjie Yi, Daliang Xu, Qipeng Wang,
  Bingyang Wu, Yihao Zhao, Chen Yang, Shihe Wang, et al.
  A survey of resource-efficient llm and multimodal foundation models.
  arXiv preprint arXiv:2401.08092, 2024.
- [92]

  Mengwei Xu, Mengze Zhu, Yunxin Liu, Felix Xiaozhu Lin, and Xuanzhe Liu.
  Deepcache: Principled cache for mobile deep vision.
  In Proceedings of the 24th annual international conference on
  mobile computing and networking, pages 129–144, 2018.
- [93]

  Liu Xuanzhe, Yihao Zhao, Shufan Liu, Xiang Li, Xin Zhu, Yibo Liu, and Xin Jin.
  Muxflow: Efficient gpu sharing in production-level clusters with more
  than 10,000 gpus.
  SCIENCE CHINA Information Sciences, pages –, 2024.
- [94]

  Zhenliang Xue, Yixin Song, Zeyu Mi, Le Chen, Yubin Xia, and Haibo Chen.
  Powerinfer-2: Fast large language model inference on a smartphone.
  arXiv preprint arXiv:2406.06282, 2024.
- [95]

  Seongjun Yang, Gibbeum Lee, Jaewoong Cho, Dimitris Papailiopoulos, and Kangwook
  Lee.
  Predictive pipelined decoding: A compute-latency trade-off for exact
  llm decoding.
  arXiv preprint arXiv:2307.05908, 2023.
- [96]

  Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, and
  Yuxiong He.
  Zeroquant: Efficient and affordable post-training quantization for
  large-scale transformers.
  Advances in Neural Information Processing Systems,
  35:27168–27183, 2022.
- [97]

  Juheon Yi and Youngki Lee.
  Heimdall: mobile gpu coordination platform for augmented reality
  applications.
  In Proceedings of the 26th Annual International Conference on
  Mobile Computing and Networking, pages 1–14, 2020.
- [98]

  Rongjie Yi, Liwei Guo, Shiyun Wei, Ao Zhou, Shangguang Wang, and Mengwei Xu.
  Edgemoe: Fast on-device inference of moe-based large language models.
  arXiv preprint arXiv:2308.14352, 2023.
- [99]

  Wangsong Yin, Mengwei Xu, Yuanchun Li, and Xuanzhe Liu.
  Llm as a system service on mobile devices.
  arXiv preprint arXiv:2403.11805, 2024.
- [100]

  Wangsong Yin, Rongjie Yi, Daliang Xu, Gang Huang, Mengwei Xu, and Xuanzhe Liu.
  Elms: Elasticized large language models on mobile devices.
  arXiv preprint arXiv:2409.09071, 2024.
- [101]

  Zhao You, Shulin Feng, Dan Su, and Dong Yu.
  Speechmoe2: Mixture-of-experts model with improved routing.
  In ICASSP 2022-2022 IEEE International Conference on Acoustics,
  Speech and Signal Processing (ICASSP), pages 7217–7221. IEEE, 2022.
- [102]

  Jinliang Yuan, Chen Yang, Dongqi Cai, Shihe Wang, Xin Yuan, Zeling Zhang, Xiang
  Li, Dingge Zhang, Hanzi Mei, Xianqing Jia, et al.
  Mobile foundation model as firmware.
  In Proceedings of the 30th Annual International Conference on
  Mobile Computing and Networking, pages 279–295, 2024.
- [103]

  Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi.
  Hellaswag: Can a machine really finish your sentence?
  In Proceedings of the 57th Annual Meeting of the Association for
  Computational Linguistics, 2019.
- [104]

  Qunsong Zeng, Yuqing Du, Kaibin Huang, and Kin K Leung.
  Energy-efficient resource management for federated edge learning with
  cpu-gpu heterogeneous computing.
  IEEE Transactions on Wireless Communications,
  20(12):7947–7962, 2021.
- [105]

  Jinrui Zhang, Deyu Zhang, Xiaohui Xu, Fucheng Jia, Yunxin Liu, Xuanzhe Liu,
  Ju Ren, and Yaoxue Zhang.
  Mobipose: Real-time multi-person pose estimation on mobile devices.
  In Proceedings of the 18th Conference on Embedded Networked
  Sensor Systems, pages 136–149, 2020.
- [106]

  Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin
  Jin, and Hao Zhang.
  Distserve: Disaggregating prefill and decoding for goodput-optimized
  large language model serving.
  arXiv preprint arXiv:2401.09670, 2024.

