# HPCD
Hierarchical Physical-Chain Decoupling with Geo-Semantic MoLoRA for All-in-One Multi-modal Remote Sensing Image Restoration
# Overview
![alt text](https://github.com/zhentao-zou/HPCD/blob/main/Fig/Framework.png)
Figure 1: Overview of the proposed \textbf{HPCD} framework and its three-stage training strategy. (a) The framework fine-tunes a VLM to extract a **degradation-type token** for task-specific expert activation and a **Geo-context token** to provide semantic guidance for the diffusion process. (b) The **three-stage training recipe** progressively optimizes the network by exploiting physical homology in grouped degradations and addressing task heterogeneity through atomic expert refinement. In addition to task-specific restoration, we leverage the pre-trained weights from Stage I to evaluate the model's capacity for **mixed degradation removal and adversarial defense**.
# Motivation
![alt text](https://github.com/zhentao-zou/HPCD/blob/main/Fig/cmp.png)
Figure 2: Comparison of different training paradigms for all-in-one RS image restoration. **(a)** Agent-based approaches incur substantial inference overhead and lack robustness in mixed degradation scenarios due to the sequential invocation of multiple specialized networks. **(b)** Prompt-based and **(c)** direct LoRA training methods often suffer from a performance trade-off between task synergy and task compatibility. **(d)** Our proposed Hierarchical Physical-Chain Decoupling (\textbf{HPCD}) framework simultaneously exploits physical homology and task heterogeneity. By decomposing the restoration process into three progressive stages based on physical priors, **HPCD** enables the model to capture physics-grouped shared features in Stage II (e.g., understanding the atmospheric scattering model through joint cloud and fog training) while maintaining expert-level precision via task-specific refinement in Stage III.

# Introduction
All-in-one remote sensing (RS) multi-modal image restoration has emerged as a promising paradigm for addressing diverse degradations within a unified framework. However, existing methodologies encompassing Prompt-based, Agent-based, and LoRA-based approaches encounter a subtle architectural trade off. They often prioritize task heterogeneity at the expense of underlying physical homologies, or conversely, focus on cross-task shared priors while potentially overlooking unique task specific heterogeneities. Furthermore, these systems can involve considerable computational overhead when addressing mixed degradations due to the sequential invocation of multiple tools. To overcome these limitations, we propose a approach named hierarchical physical-chain decoupling (HPCD) aiming to simultaneously exploit physical homology and task heterogeneity, a novel framework decomposes restoration into three progressive stages according to physical priors: global geographic layout learning in Stage I, physics-driven group homology extraction across atmospheric, sensor, and transmission chains in Stage II, and task-specific atomic expert refinement in Stage III. Specifically, we implement the refinement stage via Mixture of Experts Low-Rank Adaptation (MoLoRA), which enables dynamic expert activation based on identified degradation types. We further introduce a Vision-Language Model (VLM) Guided Regularization paradigm to mitigate diffusion based hallucinations and develop Multi-LoRA Distribution Matching Distillation (Multi-LoDMD) for efficient one step inference. Finally, we establish UniRS-60K, a comprehensive benchmark for multi-modal and adversarial restoration. Experiments demonstrate that HPCD achieves state-of-the-art precision and task compatibility across three benchmarks. The code and UniRS-60K dataset are released at {\url{https://github.com/zhentao-zou/HPCD}}.
# Uni-RS-60K Dataset
The Dataset are avaliable at https://huggingface.co/datasets/zzt001/UniRS-60K
## CheckPoints
The Checkpoints are avaliable at  https://huggingface.co/zzt001/HPCD

# Vision Results
![alt text](https://github.com/zhentao-zou/HPCD/blob/main/Fig/final_comparison_3x3.png)

## Training
Researchers interested in extending our work to alternative datasets may utilize our provided training framework as a reference implementation.

## Reference
@article{zou2026 Hierarchical,
  title={Hierarchical Physical-Chain Decoupling with Geo-Semantic MoLoRA for All-in-One Multi-modal Remote Sensing Image Restoration},
  author={Zou, Zhentao},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026},
  publisher={IEEE}
}
