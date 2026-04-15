import gradio as gr


# === IHRE BESTEHENDE VISHING-FUNKTION ===
# Diese Funktion nehmen wir an, existiert bereits und nutzt CUDA.
# def detect_vishing(audio_filepath):
#     ... Ihre Logik ...
#     return {"Vishing": True, "Confidence": 0.95}

# Für das Beispiel simulieren wir sie:
def detect_vishing(audio_filepath):
    # Hier später: Aufruf Ihres echten CUDA-Modells
    return {"Vishing": "Yes", "Probability": 0.92}


# === GRADIO WRAPPER ===
def predict_vishing(audio):
    if audio is None:
        return "Please upload an audio file."

    # Gradio gets the path for the file
    result = detect_vishing(audio)

    # Formatting the output
    return f"### Result\n\nVishing detected: **{result['Vishing']}**\n\nconfidence: **{result['Probability']:.2%}**"


# === BUILD INTERFACE ===
demo = gr.Interface(
    fn=predict_vishing,
    inputs=gr.Audio(type="filepath", label="Call recording (FLAC)"),
    outputs=gr.Markdown(),
    title="Vishing Detector (Demo)",
    description="Upload an audio file. The CUDA accelerated model analysis the recording on Voice Phishing.",
    #allow_flagging="never"  # No feedback buttons for the demo
)

# === START (NICHT BLOCKIEREND IN JUPYTER) ===
# Attention: queue() makes for stability with CUDA load
demo.queue(max_size=1)
demo.launch(
    share=False,  # True only, if public link is required
    debug=True,  # Not being blocked by jupyter cell
    #inline=True  # For display directly in notebook
)