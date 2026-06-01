"""
Gradio interface for the vishing detector. Audio is preprocessed and
transcribed with Whisper, both modalities go through the detector, and the
256-dim embedding is scored by cosine similarity to the spoof prototype.
"""

import os
import sys
import numpy as np
import librosa
import torch
import whisper
import gradio as gr
from transformers import AutoTokenizer

# Lets `from src.models.models import ...` resolve when launched from the root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.models import MultimodalVishingDetector

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pth')
TARGET_SR = 16_000
TARGET_LEN = 48_000   # 3 s at 16 kHz, matching the training fixed length
MAX_TEXT_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IDX_TO_ATTACK = {i: f'A{i+1:02d}' for i in range(19)}

# These stay module-level so the inference helpers can read them, but are only
# filled in by load_models() inside the __main__ guard. Keeping the heavy loading
# out of import means a child process transformers may spawn on Windows (which
# re-imports this module as __mp_main__) won't re-run it and crash.
detector = None
PROTO_BONAFIDE = PROTO_SPOOF = None
THRESHOLD = 0.6297
whisper_model = None
tokenizer = None


def load_models():
    global detector, PROTO_BONAFIDE, PROTO_SPOOF, THRESHOLD, whisper_model, tokenizer

    print(f"[app] Device: {DEVICE}")

    print("[app] Loading MultimodalVishingDetector...")
    detector = MultimodalVishingDetector(embed_dim=256).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    result = detector.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"  [warn] missing keys: {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"  [warn] unexpected keys: {result.unexpected_keys[:5]}")
    detector.eval()
    print("[app] Detector loaded.")

    PROTO_PATH = os.path.join(os.path.dirname(__file__), 'models', 'prototypes.pt')
    if os.path.exists(PROTO_PATH):
        _protos = torch.load(PROTO_PATH, map_location=DEVICE, weights_only=True)
        PROTO_BONAFIDE = _protos['bonafide'].to(DEVICE)
        PROTO_SPOOF = _protos['spoof'].to(DEVICE)
        THRESHOLD = float(_protos.get('threshold', 0.6297))
        print(f"[app] Prototypes loaded, threshold: {THRESHOLD}")
    else:
        print("[app] WARNING: prototypes.pt not found, run build_prototypes.py first.")

    print("[app] Loading Whisper (small)...")
    whisper_model = whisper.load_model("small", device=DEVICE)
    print("[app] Whisper loaded.")

    print("[app] Loading DistilBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    print("[app] Tokenizer loaded.")


def preprocess_audio(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads and normalises audio to match the training preprocessing in collate_fn."""
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True, dtype=np.float32)
    y = y[:TARGET_LEN] if len(y) >= TARGET_LEN else np.pad(y, (0, TARGET_LEN - len(y)))
    y = (y - y.mean()) / (y.std() + 1e-9)
    input_values = torch.from_numpy(y).unsqueeze(0).to(DEVICE)
    attention_mask = torch.ones_like(input_values)
    return input_values, attention_mask


def transcribe(path: str) -> str:
    """Transcribes an audio file using Whisper. Loads via librosa to bypass ffmpeg."""
    audio_np, _ = librosa.load(path, sr=16000, mono=True, dtype=np.float32)
    result = whisper_model.transcribe(audio_np, language='en', fp16=DEVICE.type == 'cuda')
    return result['text'].strip()


def tokenize(transcript: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenises a transcript string for the DistilBERT text encoder."""
    enc = tokenizer(
        transcript, truncation=True, max_length=MAX_TEXT_LEN,
        padding='max_length', return_tensors='pt'
    )
    return enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE)


def run_inference(audio_path: str) -> dict:
    """End-to-end inference for one file, scored against the spoof prototype."""
    input_values, audio_mask = preprocess_audio(audio_path)
    transcript = transcribe(audio_path)
    transcript_ids, text_mask = tokenize(transcript)

    with torch.no_grad():
        with torch.amp.autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda'):
            _, aux_logits, embedding, _, _ = detector(
                input_values, audio_mask, transcript_ids, text_mask,
                return_embeddings=True
            )

        emb_norm = torch.nn.functional.normalize(embedding.float(), dim=1)
        spoof_cos = torch.nn.functional.cosine_similarity(emb_norm, PROTO_SPOOF).item()
        is_spoof = spoof_cos > THRESHOLD

        attack_idx = aux_logits.argmax(dim=-1).item()
        attack_type = IDX_TO_ATTACK.get(attack_idx, f'Unknown ({attack_idx})')

    return {
        'spoof_cos': spoof_cos,
        'is_spoof': is_spoof,
        'attack_type': attack_type,
        'transcript': transcript,
    }


def predict(audio):
    if audio is None:
        return "No audio provided.", ""

    try:
        out = run_inference(audio)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Inference error: {type(e).__name__}: {e}", ""

    if out['is_spoof']:
        verdict = (
            f"**SPOOFING DETECTED**\n\n"
            f"- Cosine score: `{out['spoof_cos']:.4f}` (threshold `{THRESHOLD}`)\n"
            f"- Predicted attack type: **{out['attack_type']}**"
        )
    else:
        verdict = (
            f"**Likely genuine call**\n\n"
            f"- Cosine score: `{out['spoof_cos']:.4f}` (threshold `{THRESHOLD}`)"
        )

    transcript_md = (
        f"**Transcript:**\n\n> {out['transcript']}"
        if out['transcript'] else "_No speech detected._"
    )

    return verdict, transcript_md


DESCRIPTION = (
    "Upload a call audio file. The system transcribes it with Whisper and "
    "feeds both the audio and the transcript into a multimodal Wav2Vec2 + DistilBERT "
    "classifier trained on the ASVspoof 2019 LA dataset."
)

with gr.Blocks(title="Vishing Detector", css=".output-box { min-height: 80px; }") as demo:

    gr.Markdown("# Vishing Detection System")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="Call recording",
            )
            run_btn = gr.Button("Analyse", variant="primary")

        with gr.Column(scale=1):
            verdict_out = gr.Markdown(
                label="Verdict",
                value="*Upload a file and click Analyse.*",
                elem_classes="output-box",
            )
            transcript_out = gr.Markdown(
                label="Transcript",
                value="",
                elem_classes="output-box",
            )

    run_btn.click(
        fn=predict,
        inputs=[audio_input],
        outputs=[verdict_out, transcript_out],
        show_progress="full",
    )


if __name__ == '__main__':
    load_models()
    demo.queue(max_size=4)
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        debug=False,
        theme=gr.themes.Soft(),
    )
