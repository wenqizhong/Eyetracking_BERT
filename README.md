# Eye Movement Classification via Global Semantic Representation with Fixation-Guided BERT
This repo will contains the code for our paper *"Eye Movement Classification via Global Semantic Representation with Fixation-Guided BERT"*.
> Previous eye-tracking studies have typically focused on predicting common mechanisms and behaviors in visual attention among diverse subjects. However, leveraging eye movements as biometric traits for differential recognition, such as group recognition or personalized identity recognition, remains an important yet understudied issue. To address this, we propose an eye movement recognition framework for distinct eye movement classification tasks, which effectively integrates images and eye movement data to learn global semantic representation. We pre-train an image-based Bidirectional Encoder Representations from Transformers (BERT) on large-scale data for eye movement encoding. To enhance foreground learning and better capture the global context of scenes, we construct a fixation-based masking method for eye-tracking-guided encoding. Finally, we learn the mappings from eye movement representations to image-based results for comprehensive classification of each subject. In the experiments, we constructed a database for identifying children with autism spectrum disorders (ASD) comprising the data from 126 children, and a gender recognition database with the eye-tracking data from 30 university students. The proposed method outperforms state-of-the-art algorithms across five databases under four eye movement classification tasks. Notably, this method is the first to apply image-related high-level information to the challenging task of multi-class identity recognition, achieving significant improvements in both quantitative metrics and qualitative performance.

## &#x1F527; Usage
### Dependencies
'''
pip install -r requirements.txt
'''

### Dataset
- Download [Saliency4ASD](https://saliency4asd.ls2n.fr/datasets/) dataset.

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
