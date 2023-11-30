import streamlit as st
import openai
import base64
import os
from dotenv import load_dotenv
from PIL import Image
import io
import cv2
import tempfile

# 환경 변수 로드 및 OpenAI API 키 설정
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    openai_api_key = 'your-openai-api-key'
openai.api_key = openai_api_key

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_byte = buffered.getvalue()
    img_base64 = base64.b64encode(img_byte).decode()
    return img_base64

def extract_frames(video_bytes, every_n_frame=30):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_bytes)
        video_path = tmpfile.name

    video = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        if frame_count % every_n_frame == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_count += 1
    return frames

def analyze_frames(frames):
    encoded_images = [encode_image_to_base64(frame) for frame in frames]

    system_prompt = "As a dog trainer, you specialize in understanding canine behavior and emotions."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the dog's behavior and emotional state throughout these frames, focusing on the changes and progression in its body language, facial expressions, and emotions. Consider these images as telling a story and provide a cohesive narrative of what you think is happening to the dog over time."}
            ],
        }
    ]

    for encoded_image in encoded_images:
        messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}})

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=200,
    )
    return response.choices[0].message.content

def main():
    st.set_page_config(page_title="DOG Analyzer", page_icon="🐶")

    # UI 구성
    st.title("DOG Analyzer")
    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        frames = extract_frames(uploaded_file.read())
        if frames:
            st.success(f"Extracted {len(frames)} frames from the video.")
            if st.button('Analyze'):
                with st.spinner('Analyzing...'):
                    analysis_result = analyze_frames(frames)
                    st.write("Analysis Result:")
                    st.write(analysis_result)
        else:
            st.error("Could not extract frames from the video.")

if __name__ == '__main__':
    main()

