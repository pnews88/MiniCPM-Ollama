import streamlit as st
import requests
import json
import base64
from PIL import Image
import io

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def query_ollama(image_base64, prompt):
    url = "http://localhost:11434/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "aiden_lu/minicpm-v2.6:Q4_K_M",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{image_base64}"
                    }
                ]
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()


        content = response.content


        encodings = ['utf-8', 'ascii', 'latin1']
        for encoding in encodings:
            try:
                decoded_content = content.decode(encoding)
                return json.loads(decoded_content)
            except UnicodeDecodeError:
                continue
            except json.JSONDecodeError:
                continue


        st.error("Failed to decode response. Raw response:")
        st.text(content)
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Ollama: {e}")
        return None

st.title("MiniCPMV-2.6 Image Analysis with Ollama by Pnews88")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
prompt = st.text_input("Enter your prompt:", "What's in this image?")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button('Analyze Image'):
            with st.spinner('Analyzing...'):
                try:
                    image_base64 = encode_image(image)
                    result = query_ollama(image_base64, prompt)
                    print(result)
                    if isinstance(result, dict):
                        st.subheader("Analysis Result:")

                        for choice in result["choices"]:
                            content = choice["message"]["content"]
                        st.write(content)
                        st.subheader("Additional Information:")
                        st.write(f"Model: {result.get('model', 'N/A')}")
                        st.write(f"Created at: {result.get('created_at', 'N/A')}")
                        st.write(f"Done: {result.get('done', 'N/A')}")
                        st.write(f"Done Reason: {result.get('done_reason', 'N/A')}")
                    elif result is not None:
                        st.error("Received unexpected response format")
                    else:
                        st.error("Failed to get a valid response from Ollama.")
                except Exception as e:
                    st.error(f"Error processing image or response: {e}")
    except Exception as e:
        st.error(f"Error opening image: {e}")

st.write("Note: Make sure Ollama is running locally with the required model loaded.")