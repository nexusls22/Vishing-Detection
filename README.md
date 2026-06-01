# Callya ID — Multimodal Vishing Detection

Detects voice phishing in call recordings by combining the audio (Wav2Vec2) with the
transcript (DistilBERT). Trained and evaluated on ASVspoof 2019 LA. Includes a small
Gradio web demo.

## Setup

Python 3.10. Install the requirements for your OS:

```
pip install -r requirements-win.txt     # Windows
pip install -r requirements-mac.txt     # macOS
```

Create a `.env` file in the project root:

```
HF_TOKEN=<your HuggingFace token>
ASV_DATA_ROOT=<path to the ASVspoof2019 LA folder>
TRANSCRIPT_PATH=<path to transcripts.csv>
```

The trained weights (`best_model.pth`, `best_attack_head.pth`, `prototypes.pt`) must be
in `src/models/`.

## Running (from the project root)

```
python src/app.py                 # web demo, opens at http://127.0.0.1:7860
python reconstruct_eer.py         # prints the EER on the dev set
python build_prototypes.py        # rebuild the prototype vectors
```

Retraining (optional): `train_binary.py` then `train_attack_head.py` in `src/models/`.
Transcripts are made once with `src/data/pre_transcribe.py`.

## Notes

- The encoders (`wav2vec2-base`, `distilbert-base-uncased`) download from HuggingFace on
  first use, then cache locally.
- `reconstruct_eer.py` sets `TRANSFORMERS_OFFLINE=1`, so it loads the encoders from the
  local cache only. This works once the encoders are cached. On a fresh machine, run
  `app.py` once first (it downloads them), or comment out that line for the first run.
- Uses the GPU (`cuda`) if available, otherwise CPU. Runs on Mac too, just slower.
