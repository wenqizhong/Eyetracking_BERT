# Eye Movement Classification via Global Semantic Representation with Fixation-Guided BERT
This repo will contains the code for our paper *"Eye Movement Classification via Global Semantic Representation with Fixation-Guided BERT"*.
> Previous eye-tracking studies have typically focused on predicting common mechanisms and behaviors in visual attention among diverse subjects. However, leveraging eye movements as biometric traits for differential recognition, such as group recognition or personalized identity recognition, remains an important yet understudied issue. To address this, we propose an eye movement recognition framework for distinct eye movement classification tasks, which effectively integrates images and eye movement data to learn global semantic representation. We pre-train an image-based Bidirectional Encoder Representations from Transformers (BERT) on large-scale data for eye movement encoding. To enhance foreground learning and better capture the global context of scenes, we construct a fixation-based masking method for eye-tracking-guided encoding. Finally, we learn the mappings from eye movement representations to image-based results for comprehensive classification of each subject. In the experiments, we constructed a database for identifying children with autism spectrum disorders (ASD) comprising the data from 126 children, and a gender recognition database with the eye-tracking data from 30 university students. The proposed method outperforms state-of-the-art algorithms across five databases under four eye movement classification tasks. Notably, this method is the first to apply image-related high-level information to the challenging task of multi-class identity recognition, achieving significant improvements in both quantitative metrics and qualitative performance.

## &#x1F527; Usage
### Dependencies
'''
pip install -r requirements.txt
'''

### Dataset
This project utilizes two main eye-tracking datasets: the publicly available **Saliency4ASD** benchmark for comparative analysis and our proprietary **Collected ASD Eye-Tracking Dataset** for enhanced validation.

#### 1. Public Benchmark: Saliency4ASD

The Saliency4ASD dataset is a widely used public resource for eye-tracking research in Autism Spectrum Disorder (ASD). It serves as the primary benchmark for model evaluation.

* **Download:** Please obtain the dataset directly from the official source:
    [Saliency4ASD Download Link](https://saliency4asd.ls2n.fr/datasets/)

---

#### 2. Proprietary Dataset: Collected Eye-Tracking Data

This dataset was collected internally to further validate our model's performance and generalization on a specific cohort.

**Cohort and Collection Details:**

We recruited **126 children aged 2-8 years** from hospitals and medical institutions in Shaanxi and Fujian provinces for eye movement data collection. The subjects included **62 children with ASD** (Autism Spectrum Disorder) and **64 TD (Typically Developing) children**.

**Stimuli and Equipment:**

* **Stimuli:** Based on the eye movement behavior characteristics of children with ASD, we selected **437 stimuli** across four types: **facial images, social scenes, natural scenes, and psychophysical images**.
* **Eye Tracker:** We used a **SciEye 2.0 eye tracker**, recording data at a sampling rate of **$140 \text{ Hz}$**.
* **Display:** The screen resolution was **$1920 \times 1080 \text{ pixels}$**, with the central image size set to **$1040 \times 780 \text{ pixels}$**.
* **Procedure:** Participants were seated **$0.5 \text{ meters}$** from the monitor and instructed to **free-view** the images. Each image was displayed for **$2.5 \text{ seconds}$**, followed by a **$0.5 \text{-second}$** gray mask between consecutive images.

**Download and License:**

* **Download Link:** You can download our collected dataset from this [link](https://drive.google.com/file/d/1Vu42TTZYcUL01MYITh0vsiLTfeBPvW_W/view?usp=sharing).

> **⚠️ License Restriction:**
> This dataset is **only for academic use** and **cannot be used for commercial purposes**. Please refer to the accompanying license file (e.g., `LICENSE.txt`) for complete terms of use.


### Run
- **Stage1** *Pre-traing Image-based BERT Model*
  Pre-train an image-based Bidirectional Encoder Representations from Transformers (BERT) on large-scale data for eye movement encoding.
  ```
  python run_beit_pretraining.py
  ```
- **Stage2** *Fine-tune Model*
  Fine-tune the model for downstream tasks.
  ```
  python run_class_pretraining.py
  ```
