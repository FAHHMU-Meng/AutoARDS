# AutoARDS

Official repository for the manuscript:  
**“AutoARDS: A Foundation Model for Acute Respiratory Distress Syndrome (ARDS) Analysis and Diagnosis via Transforming Chest CT into a Quantitative Tool.”**

---

## 📖 Introduction

Acute respiratory distress syndrome (ARDS) remains a major concern in intensive care units (ICUs), with mortality rates exceeding **40%**. Although chest CT is routinely performed in ARDS evaluation, its role has traditionally been **qualitative and subjective**.

We present **AutoARDS**, the first **CT-based foundation model** that transforms chest CT into a **quantitative tool** for ARDS evaluation and management. AutoARDS integrates lesion segmentation, multimodal representation learning, and prognosis modeling into a single framework.

### 🔑 Key Contributions
1. **Self-supervised lesion segmentation**  
   - Dice score: **75.62%**  
   - Establishes the first reproducible CT-derived quantitative biomarker for ARDS burden and progression tracking (>85% accuracy).

2. **Stepwise diagnosis**  
   - Acute respiratory failure (ARF): **AUC 0.9692–0.9805**  
   - ARDS within ARF patients: **AUC 0.8498–0.9249**, surpassing human diagnostic performance.

3. **Quantitative physiology from CT**  
   - Direct estimation of P/F ratio (Pearson r: **0.793–0.878**)  
   - Enables Berlin-definition severity stratification with **72–75% accuracy**.  
   - Outperforms conventional SpO₂-based surrogates.

4. **Extended clinical endpoints**  
   - ARDS-associated right ventricular dysfunction (RVD) estimation: **AUC 0.6715–0.8564**  
   - Longitudinal 28-day survival prediction: **time-averaged AUC 0.7872**

5. **Biological aging analysis**  
   - ARDS patients show an accelerated pulmonary aging shift (**Δ = +6.044**).

By consolidating multiple diagnostic procedures into a **single CT-driven workflow**, AutoARDS provides a scalable paradigm for **quantitative medical imaging**.

<p align="center">
  <img src="https://github.com/FAHHMU-Meng/AutoARDS/blob/main/figures/Fig_1.png" alt="Overview of AutoARDS" width="700">
</p>

---

## ⚙️ Reproduction

- All training scripts are provided in the **`train/`** folder for reproducing experiments and baseline comparisons.  
- The **official checkpoint** used in AutoARDS is under **patent protection** and cannot be released. However, all **implementation details** are described in the **Methods** section of the manuscript to support replication with non-proprietary libraries.

---

## 📊 ARDS Lesion Segmentation and Quantification
<p align="center">
  <img src="https://github.com/FAHHMU-Meng/AutoARDS/blob/main/figures/Fig_2.png" alt="Segmentation and quantification results" width="700">
</p>

---

## 🔬 Transferring CT for Pathology Quantification
<p align="center">
  <img src="https://github.com/FAHHMU-Meng/AutoARDS/blob/main/figures/Fig_3.png" alt="CT transformation for pathology quantification" width="700">
</p>

---

## 🙏 Acknowledgements

We acknowledge the released codes of:  
- [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP)  
- [MAE](https://github.com/pengzhiliang/MAE-pytorch)  

For further inquiries, please contact: **Xianglin Meng (mengzi@163.com)**

---
