# SSR-SAM: Retrieval-Style Segment Anything for UHR Segmentation

Official implementation of **SSR-SAM**  
*Retrieval-Style Segment Anything Model for Semi-Supervised Ultra-High-Resolution Image Segmentation*  

📄 [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37566) | 🧠 [Project Page](optional)

---

## 📌 Overview

Ultra-High-Resolution (UHR) image segmentation is critical in domains such as remote sensing and medical imaging, but dense annotations are expensive.

We propose **SSR-SAM**, a novel **retrieval-style semi-supervised segmentation framework** built upon Segment Anything Model (SAM), which:

- Treats **annotated regions as semantic prompts**
- Retrieves **semantically consistent regions across the image**
- Introduces **prompt-level perturbation** for consistency regularization
- Supports **zero-shot segmentation**

**SSR-SAM significantly improves segmentation performance under sparse annotations.**

---

## 🚀 Key Features

- 🔍 Retrieval-style segmentation
- 🧩 Mask-induced Semantic Prompt Generator (MSPG)
- 🔄 Prompt-level consistency regularization
- 🧪 Semi-supervised learning
- 🌍 Zero-shot segmentation
