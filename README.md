# AI Video Detector (Computer Vision Project)

Deepfake and AI-generated/manipulated video detection project built for a Computer Vision class.

The project uses a spatio-temporal model (`ResNet50 + LSTM`) and includes an explainability workflow (Grad-CAM, temporal importance, frequency analysis) to justify why a video was detected or missed.

## 1. Repository Contents

- `app.py`: Streamlit demo app.
- `src/`: model, dataset, and Google Drive model download utility.
- `scripts/`: training, evaluation, threshold tuning, inference, and CV analysis scripts.
- `results/`: saved evaluation outputs and analysis figures.
- `report.md`: final LaTeX report source.

## 2. Environment Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

If you are on Windows with local venv:

```powershell
venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Model Weights

The project expects `models/best_model.pth`.

- If not present, scripts use `src/gdrive.py` to download from Google Drive (requires internet access).
- You can override model source with environment variables:
  - `MODEL_FILE_ID`
  - `MODEL_URL`

## 4. Run the App

```bash
streamlit run app.py
```

## 5. Reproduce Core Pipeline

### Data preprocessing

```bash
python scripts/preprocess_data.py
```

### Train

```bash
python scripts/train.py
```

### Evaluate test set

```bash
python scripts/evaluate_test.py
```

### Threshold optimization

```bash
python scripts/optimize_threshold_f1.py
python scripts/optimize_threshold_recall_constrained.py
```

## 6. Inference

```bash
python scripts/infer.py path/to/video.mp4 --mode f1
python scripts/infer.py path/to/video.mp4 --mode recall
```

## 7. Explainability / CV Analysis

Run detailed analysis for one video:

```bash
python scripts/cv_analysis.py scripts/testvid1.mp4 --true_label AI-generated --mode f1 --output_dir results/cv_analysis/f1
```

Run side-by-side analysis:

```bash
python scripts/cv_analysis.py scripts/testvid0.mp4 --true_label AI-generated --mode f1 --compare_with scripts/testvid1.mp4 --compare_with_label AI-generated
```

Outputs are saved as:

- `*_cv_analysis.png` (temporal + frequency + Grad-CAM figure)
- `*_analysis.json` (structured metrics and conclusion)
- `results/cv_analysis/all_testvid_comparison.csv`
- `results/cv_analysis/all_testvid_comparison.png`

## 8. Report Artifacts

- Main report source: `report.md`
- CV analysis figures embedded in report:
  - `results/cv_analysis/focus_overview_page1.png`
  - `results/cv_analysis/focus_overview_page2.png`
  - `results/cv_analysis/all_testvid_comparison.png`

## 9. Notes

- This is a class project and should be interpreted as a research prototype, not forensic proof software.
- Predictions should be read with threshold mode and decision margin context.
