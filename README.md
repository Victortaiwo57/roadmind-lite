# RoadMind Lite — Road Scene Understanding

**RoadMind Lite** is an interactive baseline application for road-scene object detection. It uses a pretrained YOLO11n model to detect objects in an uploaded image and reports class counts, confidence scores, bounding boxes, and downloadable CSV results.

## Research motivation

The application is a first step toward road-user behaviour understanding. Single-frame object detection can locate road users, but it cannot reliably infer intent, driver attention, or future behaviour. Those require additional annotations and temporal evaluation.

### Research questions to develop
1. Which road-user classes are detected reliably in varied road scenes?
2. How robust are detections to lighting, weather, camera viewpoint, and geographic domain shift?
3. Does adding temporal context improve road-user behaviour classification over single-frame visual features?

## Run locally

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The pretrained `yolo11n.pt` weights download automatically on first use, so the first inference may take longer. A network connection is required for the initial model download.

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository, for example `roadmind-lite`.
2. Upload `app.py`, `requirements.txt`, and this `README.md` to the repository root.
3. Visit https://share.streamlit.io/ and choose **Create app**.
4. Select the repository, branch (`main`), and entrypoint `app.py`, then deploy.
5. Wait for dependency installation and model startup. If deployment fails due to memory or dependency limits, inspect the app logs and consider a smaller or separately hosted inference setup.

Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Evaluation plan (do not invent results)

To turn this prototype into a research project:
- Select a dataset with clear research-use terms, such as BDD100K.
- Define a held-out test split before tuning.
- Compare against a documented baseline.
- Report precision, recall, and mAP where ground-truth bounding boxes are available.
- Break down performance by class and relevant conditions such as daytime/nighttime.
- Document missed detections, false positives, limitations, and reproducibility details.

Dataset reference: https://github.com/bdd100k/bdd100k

## Limitations

- This prototype runs inference on one image at a time.
- It does not perform tracking, gaze estimation, intention prediction, or simulation.
- Outputs are not intended for driving decisions or safety-critical use.
- Check the model and library licence terms before commercial use or redistribution.
