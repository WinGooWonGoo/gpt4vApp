import streamlit as st
import openai
from openai import OpenAI
import base64
import os
from dotenv import load_dotenv
from PIL import Image
import io
import cv2
import tempfile
import toml

# 환경 변수 로드 및 OpenAI API 키 설정
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    # 환경 변수가 없으면 TOML 파일에서 불러오기
    openai_api_key = st.secrets["openai"]["api_key"]
openai.api_key = openai_api_key

def resize_image(image, max_size=20):
    # 이미지의 크기를 바이트로 변환
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # 이미지 크기를 MB 단위로 계산
    img_size_mb = len(img_byte_arr) / (1024 * 1024)

    # 이미지 크기가 최대 크기보다 큰 경우 크기 조정
    if img_size_mb > max_size:
        # 이미지의 너비와 높이 비율 유지
        ratio = image.height / image.width
        new_width = int((max_size / img_size_mb) ** 0.5 * image.width)
        new_height = int(new_width * ratio)

        # 이미지 크기 조정
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
        return resized_image
    else:
        return image

def encode_image_to_base64(image):
    # 이미지 크기 조정
    image = resize_image(image, max_size=15)  # max_size를 15MB로 조정

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)  # 압축률을 조정할 수 있습니다
    img_byte = buffered.getvalue()

    # 로깅으로 인코딩 전후의 크기 확인
    original_size = len(img_byte)
    print(f"Original size (before Base64): {original_size / (1024 * 1024)} MB")

    img_base64 = base64.b64encode(img_byte).decode()
    encoded_size = len(img_base64)
    print(f"Encoded size (after Base64): {encoded_size / (1024 * 1024)} MB")

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
    client = OpenAI()
    encoded_images = [encode_image_to_base64(frame) for frame in frames]

    system_prompt = "As a dog trainer, you specialize in understanding canine behavior and emotions."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Don't mention your limitations, just answer the question to the best of your ability right away. \
                 Don't say, 'And in the first image, it looks like...' Just answer as if it's a situation. \
                 Here are several images extracted from a video of a dog. Please analyze them as a continuous sequence, not as separate moments. \
                 Describe the dog's behavior and emotional state throughout these frames, focusing on the changes and progression in its body language, facial expressions, and emotions. \
                 Consider these images as telling a story and provide a cohesive narrative of what you think is happening to the dog over time."}
            ],
        }
    ]

    for encoded_image in encoded_images:
        messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}})

    response = client.chat.completions.create(
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

