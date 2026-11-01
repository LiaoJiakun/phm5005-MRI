# OASIS-1 + OASIS-2: fixed vs Canny slices, axial / coronal / dual-view (team note)

> **中文版见文末 / Chinese version at the bottom:** [中文](#中文版)

**Question.** Can a 2D network detect dementia (CDR>0 vs CDR=0) from single MRI slices, and does the slice choice matter?

**Experiments (same subjects, folds, seeds, hyper-parameters)**

| | Axial | Coronal | Network |
|---|---|---|---|
| A | fixed (80) | – | single-view |
| B | – | fixed (105) | single-view |
| C | fixed (80) | fixed (105) | dual-view |
| D | Canny | Canny | dual-view |

Models: small 2D CNN (from scratch) and ImageNet-pretrained ResNet18 (full fine-tune). Evaluation: repeated stratified 5-fold CV (2 repeats), subject level, stratified by dataset × label.

## Data
- **OASIS-1** (age ≥ 60): 198 subjects (98 CDR=0 / 100 CDR>0).
- **OASIS-2** (PART1 + PART2, **first visit only**): 150 subjects → 149 registered to atlas space (1 failed QC) → **111 are the same people as OASIS-1 participants and were removed** → **38 unique** (21 / 17).
- **Final cohort: 236 subjects.**

> **Data-overlap warning.** Different IDs ≠ different people. 75% of the OASIS-2 first-visit subjects have a near-identical scan in OASIS-1 (image similarity 0.95–0.997 vs ≤ 0.92 between different OASIS-1 people; same sex/age/eTIV, same CDR in 104/111 pairs). Keeping them gave a spurious AUC of 0.85–0.90. All results from the OASIS-1 + OASIS-2 runs *before* de-duplication are in folders named `*_LEAKED_overlap` and **must not be used**. See `results/oasis_overlap_check.png`.

## Final results (236 subjects, pooled out-of-fold AUC, 95% bootstrap CI)

| | Small CNN | ResNet18 |
|---|---|---|
| A fixed axial 80 | 0.673 (0.60–0.74) | 0.659 (0.59–0.72) |
| B fixed coronal 105 | 0.582 (0.51–0.65) | 0.705 (0.64–0.77) |
| C fixed axial + coronal | 0.607 (0.53–0.68) | 0.685 (0.62–0.75) |
| D Canny axial + coronal | 0.563 (0.49–0.64) | 0.609 (0.53–0.68) |

- Modest discrimination overall (AUC ≈ 0.6–0.7); most differences are within noise (CI half-width ≈ 0.06).
- Canny-selected slices are lowest for both models (they sit in the cerebellum/posterior fossa).
- A second view does not clearly beat the best single view (ResNet18: C − A = +0.03, CI −0.02…+0.08).
- Only nominal positive paired difference: ResNet18 B − A = +0.054 (CI 0.006–0.102) – treat as a hypothesis.
- The small CNN mostly predicts CDR=0 at threshold 0.5 (poor calibration) – compare by AUC.

## 5-slice comparison (Method 1: 2.5D vs Method 2: late fusion; fixed 78–82 vs Canny c±2)
Same 236-subject cohort and folds. ResNet18: late fusion on fixed slices is best (AUC 0.740, CI 0.68–0.80) vs 0.673 (2.5D) and 0.679 (single slice); small CNN: no difference (0.59–0.65). Fixed beats Canny everywhere. Details: `results/five_slice/summary_en.md`, `five_slice_results.png`.

## Mean-pooling trial (late fusion with mean pooling vs 2.5D; fixed 78–82 vs Canny c±2)
No-overlap cohort (236), repeated stratified 5-fold CV. ResNet18 accuracy: mean pooling 0.661 (fixed) / 0.661 (Canny) vs 2.5D 0.623 / 0.555 vs concat 0.669 / 0.619. Mean pooling ≥ 2.5D in all cells (only ResNet18-Canny AUC clearly > 0: +0.078, CI 0.014–0.142); ≈ concatenation. Notebook: `notebooks/OASIS_5slice_meanpool_trial.ipynb`; tables/figures: `results/meanpool/`.

## Best model vs the same model with Canny slices (teacher-style notebook)
`notebooks/OASIS_best_model_vs_canny.ipynb`: ResNet18 late fusion (concat) on fixed 78–82 (AUC 0.740, acc 0.669) vs identical model on Canny c±2 (AUC 0.657, acc 0.619); paired AUC difference −0.088 (95% CI −0.177…+0.001, p = 0.064): suggestive, not conclusive. Code follows the course notebooks (explicit loop, checkpoint file, early stopping, torchmetrics, Grad-CAM); fold-0 results reproduce the cross-validation exactly.

## MRI data augmentation and square-padded axial slices (best model: ResNet18, late fusion by concatenation, fixed 78–82)
- **Aspect ratio:** axial slices (208 × 176) used to be resized straight to 128 × 128 (squashed by ~18%); now zero-padded to 208 × 208 first. No reliable effect: fixed 0.740 → 0.726 (paired Δ −0.009, p = 0.82), Canny 0.657 → 0.688 (Δ +0.027, p = 0.43). The Canny-selected slice changed for 56% of subjects.
- **Augmentation** (training only, one parameter set per subject for all 5 slices; same folds/seeds): rotation ±5° baseline vs G (rotation ±10°, shift ±5%, scale ±5%), I (+ bias field, gamma, noise, blur) and GI. None helped (Δ AUC vs baseline: G −0.02…−0.03, I −0.04…−0.05, GI +0.00…−0.05); intensity augmentation was slightly worse in both runs (p = 0.037 once, not conclusive with six comparisons).
- Details: `results/augmentation_square/summary_en.md`. The other experiments were **not** re-run on the square-padded slices. Slice axis / orientation / index checks: `scripts/verify_slices.py`.

## Layout
```
data/raw/            OASIS-1 disc archives (symlinks), oasis_cross-sectional.xlsx, OASIS-2 RAW tars + demographics
data/extracted/      OASIS-1 T88 volumes;  data/oasis2/extracted|registered/  OASIS-2 first-visit scans (raw / atlas space)
data/combined_full/  final cohort (236): combined_cohort.csv, combined_slices.npz, oasis2_overlap_report.csv
notebooks/           OASIS1_OASIS2_classification.ipynb  (executed, course-template format)
scripts/             oasis_lib.py (shared code), register_oasis2.py, build_combined.py, final_cv.py, final_analyze.py, ...
results/final_cv/    summary.csv, paired.csv, confusion.csv, final_results.png, final_confusion.png, discussion.md
results/*_LEAKED_overlap/  invalid (duplicated people) – do not use
```

## Reproduce
```bash
cd scripts
python register_oasis2.py                       # OASIS-2 first visit -> atlas space (+ --robust --only <ID> if QC fails)
python build_combined.py                        # cohort + image-level de-duplication + slices
python final_cv.py cnn & python final_cv.py resnet   # ~1 h on CPU
python final_analyze.py && python overlap_figure.py
python build_notebook.py                        # then execute the notebook
```
Requires Python 3.10 with `torch torchvision scikit-learn nibabel SimpleITK opencv-python-headless(4.11) pandas scipy matplotlib openpyxl grad-cam` and `numpy<2`.

**Note on the commit history.** The history was assembled after the work was finished, in the order in which the work was actually done. Commit timestamps therefore do not reflect when each step was performed.

---

# 中文版

> English version at the top of this file. 数字、结果和上面的英文版完全一致。

# OASIS-1 + OASIS-2：固定切片与 Canny 切片，轴位 / 冠状位 / 双视角（团队说明）

**问题。** 2D 网络能否仅凭单张 MRI 切片检测痴呆（CDR>0 对 CDR=0），切片的选择是否重要？

**实验（受试者、折、随机种子、超参数都相同）**

| | 轴位 | 冠状位 | 网络 |
|---|---|---|---|
| A | 固定（80） | – | 单视角 |
| B | – | 固定（105） | 单视角 |
| C | 固定（80） | 固定（105） | 双视角 |
| D | Canny | Canny | 双视角 |

模型：小型 2D CNN（从零训练）和 ImageNet 预训练的 ResNet18（全部层微调）。评估：重复分层 5 折交叉验证（重复 2 次），按受试者划分，按“数据集 × 标签”分层。

## 数据
- **OASIS-1**（年龄 ≥ 60）：198 人（98 个 CDR=0 / 100 个 CDR>0）。
- **OASIS-2**（PART1 + PART2，**只用第一次就诊**）：150 人 → 149 人配准到图谱空间（1 人配准质检失败）→ **其中 111 人和 OASIS-1 里的参与者是同一个人，已去除** → **38 人独有**（21 / 17）。
- **最终队列：236 人。**

> **数据重叠警告。** ID 不同不等于人不同。OASIS-2 第一次就诊的受试者里有 75% 在 OASIS-1 中有几乎相同的扫描（图像相似度 0.95–0.997，而 OASIS-1 内部不同人之间 ≤ 0.92；性别、年龄、eTIV 相同，111 对中 104 对 CDR 相同）。保留它们会得到虚假的 AUC 0.85–0.90。所有去重之前的 OASIS-1 + OASIS-2 运行结果都放在名为 `*_LEAKED_overlap` 的文件夹里，**不可使用**。见 `results/oasis_overlap_check.png`。

## 最终结果（236 人，汇总折外 AUC，95% 自助法置信区间）

| | 小 CNN | ResNet18 |
|---|---|---|
| A 固定轴位 80 | 0.673（0.60–0.74） | 0.659（0.59–0.72） |
| B 固定冠状位 105 | 0.582（0.51–0.65） | 0.705（0.64–0.77） |
| C 固定轴位 + 冠状位 | 0.607（0.53–0.68） | 0.685（0.62–0.75） |
| D Canny 轴位 + 冠状位 | 0.563（0.49–0.64） | 0.609（0.53–0.68） |

- 整体区分能力有限（AUC ≈ 0.6–0.7），大多数差异在噪声范围内（置信区间半宽 ≈ 0.06）。
- 两个模型上，Canny 选出的切片都是最低的（它们位于小脑 / 后颅窝）。
- 第二个视角并不明确优于最好的单视角（ResNet18：C − A = +0.03，置信区间 −0.02…+0.08）。
- 名义上唯一为正的配对差异：ResNet18 的 B − A = +0.054（置信区间 0.006–0.102）——只能当作假设。
- 小 CNN 在 0.5 阈值下大多预测 CDR=0（概率校准差）——应按 AUC 比较。

## 5 切片对比（方法 1：2.5D 对方法 2：晚期融合；固定 78–82 对 Canny c±2）
同样的 236 人队列和折。ResNet18：固定切片的晚期融合最好（AUC 0.740，置信区间 0.68–0.80），2.5D 为 0.673，单切片为 0.679；小 CNN：没有差别（0.59–0.65）。固定切片在所有设置下都优于 Canny。详情：`results/five_slice/summary_en.md`、`five_slice_results.png`。

## 均值池化试验（均值池化晚期融合对 2.5D；固定 78–82 对 Canny c±2）
无重叠队列（236 人），重复分层 5 折交叉验证。ResNet18 准确率：均值池化 0.661（固定）/ 0.661（Canny），2.5D 为 0.623 / 0.555，拼接为 0.669 / 0.619。均值池化在所有格子里都不低于 2.5D（只有 ResNet18-Canny 的 AUC 明显大于 0：+0.078，置信区间 0.014–0.142）；与拼接基本相当。notebook：`notebooks/OASIS_5slice_meanpool_trial.ipynb`；表格和图：`results/meanpool/`。

## 最好模型与换用 Canny 切片的同一模型（课程风格 notebook）
`notebooks/OASIS_best_model_vs_canny.ipynb`：固定 78–82 的 ResNet18 晚期融合（拼接）（AUC 0.740，准确率 0.669）对完全相同的、使用 Canny c±2 的模型（AUC 0.657，准确率 0.619）；AUC 配对差 −0.088（95% 置信区间 −0.177…+0.001，p = 0.064）：有提示意义，但不能下结论。代码遵循课程 notebook 的写法（显式训练循环、检查点文件、提前停止、torchmetrics、Grad-CAM）；第 0 折的结果与交叉验证完全一致。

## MRI 数据增强和补成正方形的轴位切片（最好模型：ResNet18，拼接晚期融合，固定 78–82）
- **纵横比：** 轴位切片（208 × 176）原来直接缩放到 128 × 128（被压扁约 18%）；现在先补零到 208 × 208 再缩放。没有可靠的影响：固定 0.740 → 0.726（配对差 −0.009，p = 0.82），Canny 0.657 → 0.688（差 +0.027，p = 0.43）。56% 的受试者的 Canny 选层发生了变化。
- **数据增强**（只用于训练，每个受试者的 5 张切片共用一组随机参数；折和随机种子相同）：旋转 ±5° 的基线对比 G（旋转 ±10°、平移 ±5%、缩放 ±5%）、I（再加偏置场、伽马、噪声、模糊）和 GI。都没有帮助（AUC 相对基线的变化：G −0.02…−0.03，I −0.04…−0.05，GI +0.00…−0.05）；强度增强在两次实验里都略差（有一次 p = 0.037，在六次比较下不能下结论）。
- 详情：`results/augmentation_square/summary_en.md`。其他实验**没有**用补成正方形的切片重跑。切片的轴、方向和层号检查：`scripts/verify_slices.py`。

## 目录结构
```
data/raw/            OASIS-1 压缩包（软链接）、oasis_cross-sectional.xlsx、OASIS-2 原始 tar 包和人口学表
data/extracted/      OASIS-1 的 T88 体积；  data/oasis2/extracted|registered/  OASIS-2 第一次就诊扫描（原始 / 图谱空间）
data/combined_full/  最终队列（236 人）：combined_cohort.csv、combined_slices.npz、oasis2_overlap_report.csv
notebooks/           OASIS1_OASIS2_classification.ipynb（已执行，课程模板格式）
scripts/             oasis_lib.py（共享代码）、register_oasis2.py、build_combined.py、final_cv.py、final_analyze.py 等
results/final_cv/    summary.csv、paired.csv、confusion.csv、final_results.png、final_confusion.png、discussion.md
results/*_LEAKED_overlap/  无效（含重复的人）——不要使用
```

## 复现
```bash
cd scripts
python register_oasis2.py                       # OASIS-2 第一次就诊 -> 图谱空间（质检失败时加 --robust --only <ID>）
python build_combined.py                        # 队列 + 图像级去重 + 切片
python final_cv.py cnn & python final_cv.py resnet   # CPU 上约 1 小时
python final_analyze.py && python overlap_figure.py
python build_notebook.py                        # 然后执行 notebook
```
需要 Python 3.10，以及 `torch torchvision scikit-learn nibabel SimpleITK opencv-python-headless(4.11) pandas scipy matplotlib openpyxl grad-cam` 和 `numpy<2`。

**关于提交历史的说明。** 提交历史是在工作全部完成后，按照实际工作的先后顺序整理出来的，因此提交时间不代表每一步实际完成的时间。
