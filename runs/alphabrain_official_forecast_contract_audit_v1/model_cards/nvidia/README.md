---
base_model:
- nvidia/Cosmos-Predict2-2B-Video2World
---
# **Cosmos-Policy-LIBERO-Predict2-2B**

[**Cosmos Policy**](https://huggingface.co/collections/nvidia/cosmos-policy) | [**Code**](http://github.com/NVlabs/cosmos-policy) | [**White Paper**](https://arxiv.org/abs/2601.16163) | [**Website**](https://research.nvidia.com/labs/dir/cosmos-policy/)

# Model Overview

## Description:

Cosmos-Policy-LIBERO-Predict2-2B is a 2B-parameter robot manipulation policy model fine-tuned from the [NVIDIA Cosmos-Predict2-2B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World) video foundation model. This model achieves state-of-the-art performance on the LIBERO simulation benchmark with a 98.5% average success rate across four task suites.

Key features:

* **Single-stage fine-tuning**: Adapted from pretrained video model with no architectural modifications
* **Multimodal outputs**: Jointly predicts actions, future states, and values through unified video diffusion
* **High performance**: 98.5% average success rate on LIBERO (Spatial: 98.1%, Object: 100.0%, Goal: 98.2%, Long: 97.6%)

Use cases:

* Robotic manipulation and control in simulation environments
* Imitation learning and policy learning for table-top manipulation tasks
* Vision-based robot learning with multiple camera viewpoints
* Long-horizon task planning and execution
* Lifelong learning and transfer learning in robotics

This model is for research and development only.

**Model Developer**: NVIDIA

## Model Versions

Cosmos Policy models include the following:

- [Cosmos-Policy-LIBERO-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B): Given current state observations and a task description, generate action sequences, future state predictions, and value estimates for robot manipulation in simulated LIBERO environments.
- [Cosmos-Policy-RoboCasa-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-RoboCasa-Predict2-2B): Given current state observations and a task description, generate action sequences, future state predictions, and value estimates for robot manipulation in simulated RoboCasa environments.
- [Cosmos-Policy-ALOHA-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-ALOHA-Predict2-2B): Given current state observations and a task description, generate action sequences, future state predictions, and value estimates for robot manipulation in real-world ALOHA robot environments.
- [Cosmos-Policy-ALOHA-Planning-Model-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-ALOHA-Planning-Model-Predict2-2B): Given current state observations, a task description, and action sequences, generate future state predictions and value estimates for robot manipulation in real-world ALOHA robot environments. (This checkpoint is meant to be deployed alongside Cosmos-Policy-ALOHA-Predict2-2B, not independently.)

### License:

This model is released under the  [NVIDIA One-Way Noncommercial License (NSCLv1)](https://github.com/NVlabs/HMAR/blob/main/LICENSE). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).

Under the NVIDIA One-Way Noncommercial License (NSCLv1), NVIDIA confirms:

* Models are not for commercial use.
* NVIDIA does not claim ownership to any outputs generated using the Models or Derivative Models.

### Deployment Geography:

Global

### Use Case:

Physical AI: Robot manipulation and control, encompassing tabletop manipulation and imitation learning in simulation environments.

### Release Date:

GitHub [01/22/2026] via [https://github.com/nvlabs/cosmos-policy](https://github.com/nvlabs/cosmos-policy)

Hugging Face [01/22/2026] via [https://huggingface.co/collections/nvidia/cosmos-policy](https://huggingface.co/collections/nvidia/cosmos-policy)

## Model Architecture:

Architecture Type: A diffusion transformer with latent video diffusion, fine-tuned from Cosmos-Predict2-2B-Video2World.

Network Architecture: The model uses the same architecture as the base [Cosmos-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World) model (a diffusion transformer with latent video diffusion).

**Key adaptation**: Actions, proprioceptive states, and values are encoded as latent frames and injected directly into the video model's latent diffusion sequence, enabling the model to generate these modalities alongside predicted future images.

**Number of model parameters:**

2B (inherited from base model)

## Input

**Input Type(s)**: Text + Multi-view Images + Proprioceptive State

**Input Format(s)**:

* Text: String (natural language task description)
* Images: RGB images from multiple camera views
* Proprioception: Numerical array

**Input Parameters**:

* Text: One-dimensional (1D) - Task description (e.g., "put the black bowl on top of the cabinet")
* Images: Two-dimensional (2D) - Third-person camera (agentview): 224×224 RGB; Wrist-mounted camera (eye-in-hand): 224×224 RGB
* Proprioception: One-dimensional (1D) - 9-dimensional state (2 gripper joints + 3 end-effector position + 4 end-effector quaternion)

**Other Properties Related to Input**:

* Requires specific camera configuration (third-person + wrist views)
* Images resized to 224×224 pixels from original resolution
* Trained exclusively for Franka Emika Panda robot arm in LIBERO simulation environments

## Output

**Output Type(s)**: Action Sequence + Future State Predictions + Value Estimate

**Output Format**:

* Actions: Numerical array
* Future states: Images + Proprioception
* Value: Scalar

**Output Parameters**:

* Action chunk: 16-timestep sequence of 7-dimensional actions (6-DoF end-effector control + 1 gripper)
* Future robot proprioception: 9-dimensional state at timestep t+16
* Future state images: Third-person camera prediction (224×224 RGB) and wrist camera prediction (224×224 RGB) at timestep t+16
* Future state value: Expected cumulative reward from future state (scalar)

**Other Properties Related to Output**:

* Action chunk size: 16 timesteps
* Denoising steps: 5 (configurable without retraining)
* Noise level range: σ_min = 4.0, σ_max = 80.0
* Generation mode: Parallel (action, future state, and value generated simultaneously)

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration

**Runtime Engine(s):**

* [Transformers](https://github.com/huggingface/transformers)

**Supported Hardware Microarchitecture Compatibility:**

* NVIDIA Hopper (e.g., H100)

**Note**: We have only tested doing inference with BF16 precision.

**Operating System(s):**

* Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

# Usage

See [Cosmos Policy GitHub](http://github.com/NVlabs/cosmos-policy) for details.

## Training and Evaluation Sections:

### Training Datasets:

**Data Collection Method**:

* LIBERO-Cosmos-Policy: Hybrid: Human - Human-teleoperated demonstrations recorded in simulation environment

**Labeling Method**:

* LIBERO-Cosmos-Policy: Automated - Success/failure labels automatically determined by simulation environment evaluation; task descriptions from benchmark specification

##### Properties:

**Training Data**: [LIBERO-Cosmos-Policy](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy) dataset

- 4 task suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-Long
- 500 demonstrations per suite (50 demos × 10 tasks)
- Successful demonstrations used for policy training
- All demonstrations (including failures) used for world model and value function training

**Training Configuration**:

- **Base model**: NVIDIA Cosmos-Predict2-2B-Video2World (`model-480p-16fps.pt`)
- **Training steps**: 40,000 gradient steps
- **Batch size**: 1,920 (global)
- **GPUs**: 64 H100 GPUs
- **Training time**: ~48 hours
- **Optimization**: Full model fine-tuning (all weights updated)
- **Action chunk size**: 16 timesteps
- **Image resolution**: 224×224 pixels

**Training Objective**: The model is trained with a hybrid log-normal-uniform noise distribution (modified from the base model's log-normal distribution; see paper for details) to improve action prediction accuracy. Training batches are split 50/25/25 for policy, world model, and value function objectives, respectively.

### Evaluation Datasets:

Data Collection Method: Not Applicable

Labeling Method: Not Applicable

Properties:  Not Applicable - we use the LIBERO simulation environments for direct evaluations.

## Inference:

**Test Hardware:** H100, A100

See [Cosmos Policy GitHub](http://github.com/NVlabs/cosmos-policy) for details.

#### System Requirements and Performance

Inference with base Cosmos Policy only (i.e., no model-based planning):

* 1 GPU with 6.8 GB VRAM for LIBERO sim benchmark tasks
* 1 GPU with 8.9 GB VRAM for RoboCasa sim benchmark tasks
* 1 GPU with 6.0 GB VRAM for ALOHA robot tasks

#### Quality Benchmarks

### LIBERO Benchmark Results

| Task Suite        | Success Rate    |
| ----------------- | --------------- |
| LIBERO-Spatial    | 98.1%           |
| LIBERO-Object     | 100.0%          |
| LIBERO-Goal       | 98.2%           |
| LIBERO-Long       | 97.6%           |
| **Average** | **98.5%** |

Success rates are averaged over 500 trials per suite (10 tasks × 50 episodes) across 3 random seeds (6,000 trials total).

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Users are responsible for model inputs and outputs. Users are responsible for ensuring safe integration of this model, including implementing guardrails as well as other safety mechanisms, prior to deployment.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Related Resources

- **Base Model**: [Cosmos-Predict2-2B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)
- **Training Dataset**: [LIBERO-Cosmos-Policy](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy)
- **Paper**: [Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning](https://arxiv.org/abs/2601.16163)
- **Original LIBERO**: [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310)

## Citation

If you use this model, please cite the Cosmos Policy paper:

(Cosmos Policy BibTeX citation coming soon!)