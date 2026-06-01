# Viva Cheat-Sheet — Callya ID (10 min)

## Intro slides (2 min)

**Slide 1 — The problem**
- Vishing (voice phishing) is one of the fastest-growing social-engineering attacks
- AI voice cloning makes fake calls sound real
- Audio-only detectors weaken as synthetic speech improves

**Slide 2 — The idea (my contribution)**
- Add a second, independent signal: *what is said*, not just *how real it sounds*
- Multimodal: audio (Wav2Vec2) + transcript (DistilBERT), fused
- Content-based detection still works even when the audio is flawless

**Slide 3 — What I built**
- Callya ID: Wav2Vec2 + DistilBERT → 256-dim fused embedding
- Trained on ASVspoof 2019 LA, deployed as a Gradio web app
- Result: ~2.79% EER (dev set), 66% attack-type accuracy

## The 6 core points

1. **Why multimodal** — audio judges realism (weakens as fakes improve); text adds an independent content signal. Audio + text fused.
2. **Architecture** — audio→Wav2Vec2→768; text→DistilBERT→768; concat→1536→fusion MLP→256-dim embedding. Two heads: binary (on embedding) + attack-type A01–A19 (on raw concat). Backbones frozen except top 4 Wav2Vec2 layers.
3. **Why AAM-Softmax** — adds an angular margin so each class sits in a tight angular cone; this matches cosine-similarity scoring, aligning training objective with the evaluation metric.
4. **Two-stage training** — joint training made the tasks interfere (~20% attack acc). Stage 1 = binary; Stage 2 = freeze all, train only attack head → 66%, no loss to detection.
5. **EER computation** — AAM-Softmax module wasn't saved, so prototypes were reconstructed as the normalised mean embedding per class from train, then dev scored by cosine to the spoof prototype. Valid because AAM-Softmax drives embeddings toward their weight vectors. ~2.79% EER, separation 0.31.
6. **Limitations (say first)** — (a) dev = known attacks A01–A06, so known-attack performance, not generalisation; eval set A07–A19 is future work. (b) Augmentation was inadvertently on during eval (eval reused collate_fn), so the EER is *conservative*, not inflated. (c) Prototype reconstruction is an approximation.

## Code tour order (open as tabs)

`app.py` → `models.py` → `train_binary.py` (the `AAMSoftmax` class) → `reconstruct_eer.py`
(Follows one audio file from upload to verdict, then how it was trained and evaluated.)

## Runsheet (timed)

| Time | What | Where |
|------|------|-------|
| 0:00–0:40 | Problem: vishing + voice cloning | Slide 1 |
| 0:40–1:20 | Idea: multimodal audio + text | Slide 2 |
| 1:20–2:00 | What I built + headline numbers | Slide 3 |
| 2:00–4:30 | **Live demo**: spoof sample then genuine sample (verdict, score, attack type, transcript) | Web app |
| 4:30–6:00 | Trace pipeline: preprocess → Whisper → tokenize → forward; show fusion architecture | app.py + models.py |
| 6:00–7:30 | Two-stage training + AAM-Softmax; 20%→66% | train_binary.py |
| 7:30–9:00 | Prototype reconstruction + printed EER; confusion matrix | reconstruct_eer.py + PNG |
| 9:00–10:00 | Honesty: known-attack scope, conservative EER, eval set future work | — |

## Demo checklist (do BEFORE presenting)

- [ ] Launch `app.py` ahead of time — it's already running on http://127.0.0.1:7860 (loading is slow; don't do it live)
- [ ] 2–3 audio files ready and pre-tested (≥1 spoof, ≥1 bonafide) — know the verdicts in advance
- [ ] `reconstruct_eer.py` pre-run, terminal output visible or screenshotted (takes a few min; don't run live)
- [ ] Confusion matrix PNG open in a viewer
- [ ] Editor tabs in tour order, font size up for the projector
- [ ] Thesis open to the results table

## Likely questions (crisp answers)

- **"2.79% looks too good — overfit/leak?"** → Known-attack dev performance, expected to be strong; real generalisation test (eval set, unseen attacks) is future work. Also conservative because augmentation was on during eval.
- **"Was augmentation applied during evaluation?"** → Yes, inadvertently — eval reused the training collate_fn. It degrades the signal, so the EER is pessimistic, not inflated. Disclosed in Sections 5.2 and 6.1.
- **"Reconstructed prototypes — is that fair?"** → AAM-Softmax drives embeddings toward their weight vectors, so the per-class mean is a close approximation; small discrepancy possible, stated as a limitation.
- **"Why freeze the backbones?"** → Transfer learning on modest hardware; top-4-layer fine-tuning balances adaptation vs overfitting and compute.
- **"Could the text branch leak the label?"** → Possible; the right check is an audio-only vs text-only ablation, flagged as follow-up. (Only raise if pushed.)
- **"Why cosine EER, not accuracy?"** → EER is the standard anti-spoofing metric, threshold-independent, comparable to the ASVspoof baselines.

## One-liners to memorise

- "Audio asks *does it sound real*; text asks *does it make sense*. I use both."
- "Angular-margin training is aligned with cosine scoring by design."
- "The number is conservative, not inflated — augmentation was on during eval."
- "This is known-attack performance; unseen attacks are the next step."
