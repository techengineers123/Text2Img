import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import speech_recognition as sr

st.set_page_config(page_title="Text2Img_Generator", layout="wide")

@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu") 
    return pipe

pipe = load_model()

def generate_image(prompt):
    image = pipe(prompt, num_inference_steps=40).images[0]  # You can adjust the steps for quality
    return image

st.title("Text-to-Image Generator")
st.write("Enter a text prompt and generate an AI-based image!")

prompt = st.text_input("Enter your text prompt",placeholder="Enter your prompt here")

# Microphone input for speech recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        st.write("Recognizing...")
        try:
            prompt_text = recognizer.recognize_google(audio)
            st.success(f"You said: {prompt_text}")
            return prompt_text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None

# Use microphone button
if st.button("Use Microphone"):
    mic_prompt = recognize_speech()
    if mic_prompt:
        prompt = mic_prompt
        with st.spinner('Generating image...'):
            generated_image = generate_image(prompt)
            st.image(generated_image, caption="Generated Image", use_column_width=True)
            img_bytes = generated_image.save("generated_image.png")
            st.success("Image Generated Successfully!")
            with open("generated_image.png", "rb") as file:
                st.download_button("Download Image", file, file_name="generated_image.png", mime="image/png")
    else:
        st.warning("Please enter a prompt or use the microphone.")

# Button to trigger image generation
if st.button("Generate Image"):
    if prompt:
        with st.spinner('Generating image...'):
            generated_image = generate_image(prompt)
            st.image(generated_image, caption="Generated Image", use_column_width=True)
            img_bytes = generated_image.save("generated_image.png")
            st.success("Image Generated Successfully!")
            with open("generated_image.png", "rb") as file:
                st.download_button("Download Image", file, file_name="generated_image.png", mime="image/png")
    else:
        st.warning("Please enter a prompt or use the microphone.")