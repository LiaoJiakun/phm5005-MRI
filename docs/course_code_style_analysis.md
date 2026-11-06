# Course notebook code-style analysis (05-4, 06-1, 06-2, 10-2)

Written before the new notebook. Cell numbers are 0-based positions in the files read from `~/Downloads`
(05-4, 06-1 and 10-2 are the "副本" copies; 06-2 is `06_2_DL_for_Image_Data_Classification.ipynb`;
a second copy of 06-2 with the same 141 cells exists and was not diffed line by line).
05-4 and 10-2 were read as outline plus the data, model, training and fusion cells; 06-2 was read in full.
06-1 has 8 markdown cells and no code (CNN overview, tasks, explainability, torchvision/MONAI API tables).

## 1. What the course notebooks look like

| Aspect | Course practice | Where |
|---|---|---|
| Layout | `# Title` → `## Workflow` → `## Dataset` (`### Loading/Transforming`, `### Setting DataLoader`, `### Inspecting Images`, `### Checking Shape`) → `## Architecture` → `### Training` (bold labels **Initial Set-up**, **Model**, **Training Loop Set-up**, **Training/Validation**, **Inspecting Performance**) → `### Evaluation` → **Inspecting Predictions** → confusion matrix | 06-2 cells 0–54 |
| Cell size | One idea per cell. Short set-up cells (2–20 lines); the whole epoch loop is one cell (≈105 lines) with banner comments `# ---- Training ----`, `# ---- Validation ----`, `# ---- Scheduler step ----`, `# ---- Checkpointing + Early Stopping ----`, `# ---- Report ----` | 06-2 cell 43; 05-4 cell 68 (79 lines, same shape, no checkpointing yet) |
| Imports | Local: each import sits in the cell where it is first needed | every cell |
| Constants | Upper-case block: `SEED = 42`, `DEVICE`, `LEARNING_RATE`, `NUM_EPOCHS`, `BATCH_SIZE`, `CHECKPOINT_FILE`; seeds set for `random`, `numpy`, `torch` | 05-4 cell 26; 06-2 cell 37 |
| Data | `DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)`; val/test `shuffle=False`; one batch inspected with `next(iter(...))` and shapes printed | 05-4 cells 19–21; 06-2 cells 12–20 |
| Split | `train_test_split(..., stratify=y, random_state=SEED)` twice (60/20/20), then `TensorDataset` | 05-4 cells 12–15 |
| Model | `class SimpleCNN(nn.Module)` with `self.features` / `self.classifier`, shape comments in `forward` (`# [B, in_channels, H, W]`), then `print(model)` and a `torchvista` trace | 06-2 cells 26–32 |
| Pretrained | `models.resnet18(weights=None)` (scratch) and `models.resnet18(weights="DEFAULT")` with `model.fc = nn.Linear(model.fc.in_features, n_classes)`; fine-tuning uses `LEARNING_RATE = 1e-4` | 06-2 cells 59–63, 83–86 |
| Training loop | `nn.CrossEntropyLoss`, `optim.Adam`, `torchmetrics Accuracy`, `ReduceLROnPlateau`, `tqdm`; `running_train_loss / running_val_loss / running_*_metric` lists; best checkpoint saved as a dict (`epoch`, `best_val_loss`, `model_state`, `optimizer_state`); early stopping with `patience = 5`, `min_delta = 1e-4` on validation loss | 06-2 cells 41–43; 05-4 cells 73–92 |
| Curves | Two separate cells: loss curve, then metric curve (train vs validation, `plt.grid(True)`) | 06-2 cells 45–46 |
| Evaluation | Restore best checkpoint, loop over the test loader under `torch.no_grad()`, then collect `all_labels` / `all_preds`, `confusion_matrix` + `ConfusionMatrixDisplay`, plot one example with true and predicted label | 06-2 cells 48–54 |
| Fusion vocabulary | Level of fusion: **early** = combine features before prediction, **late** = combine predictions. Mechanism: concatenation, summation/averaging, gating, attention. `ConcatFusion`: encoders → `torch.cat([zA, zB], dim=1)` → `fusion_head` | 10-2 cells 1–7, 21; late fusion `LateFusionWeightedLogits` |

## 2. Naming: what this project computes versus the course words

In the course (10-2) fusion of encoder *features* by concatenation is called **early fusion by concatenation**; combining *predictions* is **late fusion**.
In the new notebook the computation is described explicitly
(shared encoder per slice → 512-d feature per slice → concatenated 5120-d → dropout → linear classifier) and the course term is added in brackets
(feature-level fusion by concatenation). Predictions are never averaged across slices.

## 3. Mapping to the new project notebook

| Course element | New notebook | Deviation and reason |
|---|---|---|
| `SEED`, `DEVICE`, hyper-parameter block | Same block, plus `N_REPEATS`, `N_FOLDS` | Repeated subject-level CV instead of one 60/20/20 split (198 subjects are too few for one split) |
| `Dataset` / `DataLoader` | Subject-level dataset returning a tensor `[10, H, W]` and a label | The course datasets return one image; here one subject = 10 slices |
| `SimpleCNN` class | `SimpleCNN` kept in the course shape, 1 input channel, BatchNorm after every convolution | Grey slices; one shared encoder per slice, features concatenated. BatchNorm added because the course version did not train on 128 x 128 MRI slices |
| `resnet18(weights="DEFAULT")` + new `fc` | Shared `resnet18` encoder (`fc = Identity`), slice features concatenated, `Dropout` + `Linear(5120, 2)` | Required by the task; all layers are fine-tuned |
| Epoch loop with banner comments | Kept in one cell with the same banners | Validation AUC logged next to loss (course logs accuracy) |
| `torchmetrics.Accuracy(multiclass)` | `torchmetrics` binary metrics plus `sklearn` for ROC-AUC, F1, sensitivity, specificity | Binary screening task, AUC is the main metric |
| Checkpoint dict | Same dict, written per (model, selection, repeat, fold) into a checkpoint index | Many folds |
| Test loop, `confusion_matrix`, `ConfusionMatrixDisplay` | Same, run on pooled out-of-fold predictions | CV |
| `/content/...` paths | `ROOT`-relative paths from a config cell | Local VS Code run |
| `Adam`, `ReduceLROnPlateau` | Open decision: the previous best model used AdamW (lr 1e-4, weight decay 1e-4) and no scheduler | Kept as in the previous project unless told otherwise |

Core code (model classes, dataset, training loop, evaluation, Grad-CAM) stays in notebook cells. Only the slow data-building step and the CV driver data
(prepared arrays, fold table) are read from files.
