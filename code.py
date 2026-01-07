import gradio as gr
import whisper
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
from PIL import Image
import tempfile
import torch

# ---------------- LOAD MODELS ---------------- #

# Whisper (Speech → Text)
whisper_model = whisper.load_model("base")

# BLIP (Image → Caption)
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# FLAN-T5 (Reasoning)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# ---------------- FUNCTIONS ---------------- #

def speech_to_text(audio_path):
    if audio_path is None:
        return ""
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def describe_image(image):
    inputs = blip_processor(image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

def answer_question(image, question):
    if image is None or question.strip() == "":
        return "Please upload an image and ask a question."

    image_desc = describe_image(image)

    prompt = f"""
    Image description: {image_desc}
    Question: {question}
    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = qa_model.generate(
        **inputs,
        max_length=200
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp.name)
    return temp.name

def vision_talk(image, text_q, voice_q):
    question = text_q
    if voice_q is not None:
        question = speech_to_text(voice_q)

    answer = answer_question(image, question)
    audio = text_to_speech(answer)

    return question, answer, audio

# ---------------- UI ---------------- #

with gr.Blocks(title="Vision Talk – Offline Multimodal AI") as demo:
    gr.Markdown("## 👁️ Vision Talk – Offline Image Question Answering")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")

        with gr.Column():
            text_input = gr.Textbox(label="Ask a question (Text)")
            voice_input = gr.Audio(type="filepath", label="Ask a question (Voice)")
            btn = gr.Button("Ask")

    recognized_q = gr.Textbox(label="Recognized Question")
    answer_text = gr.Textbox(label="Answer")
    answer_audio = gr.Audio(label="Answer (Speech)", autoplay=True)

    btn.click(
        fn=vision_talk,
        inputs=[image_input, text_input, voice_input],
        outputs=[recognized_q, answer_text, answer_audio]
    )

demo.launch()
