# Fusion 2026 deepfake-resistant onboarding demo

This is a local prototype scaffold. It samples short videos, detects a face, filters very poor-quality frames, trains a small classifier from labeled face crops, scores a video, exposes a FastAPI endpoint, and provides a Streamlit mock onboarding screen.

The score is a demo signal only. Grad-CAM and the classifier do not prove manipulation or identity. Do not use this scaffold to approve or deny a real person.

## 1. Install

Use the same Python environment for every command. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU-enabled PyTorch, choose the matching command from the official [PyTorch install selector](https://pytorch.org/get-started/locally/). CPU inference/training is supported, though training will be slower.

## 2. Prepare training data

The classifier consumes cropped face images, not MP4 files. First create a CSV manifest with these columns:

```csv
path,label,group_id
original/000.mp4,real,000
Deepfakes/000_001.mp4,fake,000
```

`group_id` identifies the target original video. FF++ manipulated filenames use `target_source.mp4`, so the builder pairs `Deepfakes/000_003.mp4` with `original/000.mp4` under group `000`.

Build the manifest from the downloaded per-collection CSVs. This uses `original.csv` and the five core manipulation CSVs; it leaves `DeepFakeDetection.csv` out until its separate source-video mapping is verified. The shuffled and summary CSVs are not needed.

```powershell
python .\scripts\build_ffpp_manifest.py --dataset-root "C:\Users\samir\Downloads\archive\FaceForensics++_C23" --output .\raw_data\videos.csv
```

Then create a balanced subset of face crops. This selects one real and one fake video per target group and splits groups before extracting frames:

```powershell
python .\scripts\prepare_data.py --manifest .\raw_data\videos.csv --root "C:\Users\samir\Downloads\archive\FaceForensics++_C23" --output .\data --max-groups 300
```

`--max-groups 300` is an optional time-saving subset; omit it to process all groups. Confirm the resulting counts and inspect some crops before training.

The output layout is:

```text
data/
  train/
    fake/   # .jpg/.png face crops
    real/
  val/
    fake/
    real/
```

Use only correctly labeled clips. Never infer labels from a guessed filename rule if the dataset metadata provides them.

## 3. Train

```powershell
python .\scripts\train.py --data .\data --epochs 5
```

The trainer saves the best validation-loss checkpoint to `models/`. The first run may download MobileNetV3 ImageNet weights. Add `--no-pretrained` to skip that download, with a likely quality tradeoff.

To evaluate validation performance and confusion matrix:

```powershell
python .\scripts\evaluate_model.py --data .\data
```

## 4. Score one video

```powershell
python .\scripts\score_video.py .\samples\1_vid.mp4
```

## 5. Run the mock onboarding UI

Start the API in one terminal:

```powershell
python -m uvicorn api:app --reload
```

Start Streamlit in another terminal:

```powershell
python -m streamlit run .\streamlit_app.py
```

Open the local URL printed by Streamlit. The API returns a retry/review response when too few usable face frames are present.

The UI displays a Grad-CAM overlay for the median-scored usable face frame. It visualizes regions that influenced the classifier score; it is not a pixel-level manipulation mask.

## 6. QR Code Verification System

The demo includes a QR code verification pipeline for reviewers and compliance auditors:

1. **Session & Token Generation**:
   - Each onboarding submission generates an unpredictable, expiring security token stored in SQLite (`kyc_sessions.db`).
   - **Security**: Never stores or exposes private identity data inside the QR matrix. Only an ephemeral token reference is encoded.
2. **Reviewer QR Scanning**:
   - **Smartphone**: Scanning the QR code with a phone camera opens the mobile-responsive report page at `/report/{token}`.
   - **Streamlit Scanner**: The **Reviewer QR verification** tab uses OpenCV (`cv2.QRCodeDetector`) to scan live webcam frames or uploaded image badges.
3. **Audit Trail**:
   - Displays session ID, deepfake risk score, timestamp, usable face frames, and attribution heatmap.
   - Auditors can submit final decisions (`APPROVED`, `FLAGGED FOR FRAUD AUDIT`, `REJECTED`) directly into SQLite.

## 7. Dataset use

Follow the dataset owner's terms. FaceForensics++ access is request-based and its published terms restrict the data to non-commercial research and educational use. A third-party mirror does not itself prove redistribution permission. Credit the dataset and do not redistribute its videos or derived crops unless the applicable terms allow it.
