# AutoARDS
This is the official repository for the manuscript "AutoARDS: A Foundation Model for Acute Respiratory Distress Syndrome (ARDS) Analysis and Diagnosis via Transforming Chest CT into a Quantitative Tool".

## Introduction
Acute respiratory distress syndrome (ARDS) remains a serious concern in ICUs, with mortality rates exceeding 40\%. Although chest CT is performed routinely in ARDS evaluation, its role has remained largely qualitative and subjective. Here, we present AutoARDS, the first all-in-one CT-based foundation model transforming chest CT into a comprehensive quantitative tool for ARDS evaluation. AutoARDS first incorporates a self-supervised lesion-segmentation for objective disease quantification, and employs a task-unified multimodal pretraining framework that integrates lesion, language, and metadata representations. An adversarial vision-language strategy further enhances sensitivity to ARDS-specific textual perturbations, enabling fine-grained feature alignment. Pretrained on over 50,000 CT volumes and fine-tuned on a multicenter cohort of 6,835 CT volumes from six institutions, AutoARDS delivers an integrated chain of disease quantification and prognosis for ARDS: 
(1) Precise self-supervised lesion segmentation (Dice: 75.62\%), providing the first reproducible CT-derived quantitative biomarker for ARDS burden and progression tracking with $>$85\% accuracy; 
(2) Diagnosis of acute respiratory failure (ARF) and ARDS with AUCs of 0.9692-0.9805 and 0.8498-0.9249, surpassing human performance; 
(3) Direct estimation of P/F ratio from CT (Pearson r: 0.793-0.878), substantially surpassing conventional SpO\textsubscript{2}‑based monitoring precision, and facilitating Berlin severity stratification (accuracy: 72-75\%); and 
(4) Robust ARDS-associated right ventricular dysfunction (RVD) estimation (AUC: 0.6715-0.8564) and time-dependency prognosis prediction (time-averaged AUC: 0.7872). 

Further analysis reveals a positive shift in biological age for ARDS patients ($\Delta = +6.044$), indicating accelerated pulmonary aging. By unifying multiple diagnostic procedures into a single CT-driven workflow, AutoARDS provides a scalable blueprint for transforming medical imaging from morphology to quantitative physiology.

<img src="https://github.com/FAHHMU-Meng/AutoARDS/blob/main/figures/Fig_1.png" alt="image">

## ARDS lesion segmentation and quantification
<img src="https://github.com/FAHHMU-Meng/AutoARDS/blob/main/figures/Fig_2.png" alt="image">

## Transferring CT for pathology quantification
<img src="https://github.com/FAHHMU-Meng/AutoARDS/blob/main/figures/Fig_3.png" alt="image">
