# High-Precision In-Memory Flow Matching using Resistive Memory

This repository contains the official implementation of the paper **"High-precision in-memory flow matching using resistive memory"**.

We introduce a software-hardware co-design system that leverages **Flow Matching** and **Resistive Memory-based Computing-In-Memory (CIM)**. The system addresses the trade-off between computational efficiency and generation quality in generative AI. It features a training-free low-rank approximation method to compensate for resistive memory write noise, enabling high-fidelity generation tasks (2D distribution and spatiotemporal turbulence) with significant energy efficiency gains.

## 📁 Repository Structure

```text
memristor-flow-matching/
├── toy-demo/
│   └── toy.ipynb            # Task 1: 2D quarter-arc distribution demo
├── turbulence/              # Task 2: Spatiotemporal Turbulence Generation
│   ├── field_data/          # Dataset and generation results
│   ├── flow-matching/       # Latent Flow Matching model (Training & Inference)
│   ├── latent/              # Intermediate latent feature storage
│   └── nerf/                # Encoder and Conditional Neural Field (CNF) Decoder
└── requirements.txt         # Python dependencies
```

## 🛠️ Environment Setup

To install the necessary dependencies, please run:

```bash
pip install -r requirements.txt
```

**Dependencies:**

  * `tqdm==4.67.1`
  * `torch==2.9.1`
  * `numpy==2.1.3`
  * `einops==0.8.1`
  * `matplotlib==3.10.0`
  * `seaborn==0.13.2`
  * `pandas==2.2.3`
  * `scipy==1.15.3`
  * `tensorboard==2.20.0`

-----

## 🧪 Task 1: Toy Demo (2D Generation)

This is a lightweight demonstration of the Flow Matching framework. It trains a model to generate a **2D quarter-arc shaped distribution** from Gaussian noise.

  * **Code:** Located in `toy-demo/toy.ipynb`.
  * **Usage:** Open the Jupyter Notebook to walk through the training and sampling process for the 2D distribution task.

-----

## 🌊 Task 2: Spatiotemporal Turbulence Generation

This task generates spatiotemporal turbulence fields. The architecture consists of three main components:

1.  **Field Data:** Turbulence data storage.
2.  **NeRF (Encoder/Decoder):** Handles the encoding of field data and the **Conditional Neural Field (CNF)** decoding.
    * **Acknowledgement:** This module is developed based on the work: *Conditional neural field latent diffusion model for generating spatiotemporal turbulence* (Nature Communications 15, 10416, 2024) and its official repository [CoNFiLD](https://github.com/jx-wang-s-group/CoNFiLD). Special thanks to the original authors for their generous sharing. If you need to use this specific module, please refer to the original version of the code and literature.
3.  **Flow-Matching:** The Latent Flow Matching model that generates latent features.

### Inference

To generate turbulence fields using the pre-trained models, follow these three steps:

#### 1\. Generate Latent Features

Run the flow matching model to generate latent representations of the turbulence.

```bash
cd turbulence/flow-matching
python inference.py
```

  * **Outputs:**
      * `distribution_comparison.jpg`: Visual comparison of the generated latent feature distribution quality.
      * `../latent/gen_feature.pt`: The generated latent data saved for the next step.

#### 2\. Decode Physical Fields

Use the CNF decoder to transform the latent features into physical turbulence fields (Velocity and Pressure).

```bash
cd ../nerf
python inference.py case1.yml
```

  * **Outputs:**
      * `pressure_distribution.jpg`: Visualization of the generated pressure distribution.
      * `speed_distribution.jpg`: Visualization of the generated velocity magnitude distribution.
      * `../field_data/gen_data.npy`: The final generated raw data files.

#### 3\. Visualize Final Results

Plot the comparison between the software baseline, the uncompensated hardware result, and the compensated hardware result.

```bash
cd ../field_data
python plot_result.py
```

  * **Outputs:**
      * `results/field_comparison.png`: A comparison image showing the Baseline, Uncompensated Hardware (noisy), and Compensated Hardware (high-precision) results.

### Training

If you wish to retrain the models from scratch, follow this order:

#### 1\. Train the Autoencoder (NeRF/CNF)

First, train the external Encoder and Conditional Neural Field Decoder.

```bash
cd turbulence/nerf
python train.py case1.yml
```

  * **Details:**
      * Model checkpoints will be saved in `nerf/work/`.
      * Upon completion, it generates `../latent/train_latents.pt`, which is the training data required for the Flow Matching model.

#### 2\. Train the Latent Flow Matching Model

Train the flow matching model using the latents generated in the previous step.

```bash
cd ../flow-matching
python train.py
```

  * **Details:**
      * The trained model weights will be saved to `flow-matching/state_dict.pt`.

-----

## 📄 Citation

If you find this code or paper useful for your research, please cite:

```bibtex
@article{yang2025highprecision,
  title={High-precision in-memory flow matching using resistive memory},
  author={Yang, Jichang and Xu, Meng and Chen, Hegan and others},
  journal={arXiv preprint},
  year={2025}
}
```