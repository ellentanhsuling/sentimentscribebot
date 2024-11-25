import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import speech_recognition as sr
import datetime
import numpy as np
import queue
import threading
import av
from transformers import pipeline

# Initialize queues and sentiment analyzer
audio_queue = queue.Queue()
result_queue = queue.Queue()

@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis")

sentiment_analyzer = load_sentiment_analyzer()

# Risk keywords for escalation
RISK_KEYWORDS = [
    "suicide", "kill", "hurt", "harm", "die", "end my life",
    "self harm", "cut myself", "overdose", "pills"
]

def initialize_session_state():
    if 'conversations' not in st.session_state:
        st.session_state.conversations = []
    if 'speaker_count' not in st.session_state:
        st.session_state.speaker_count = 0
    if 'risk_level' not in st.session_state:
        st.session_state.risk_level = "Normal"
    if 'current_text' not in st.session_state:
        st.session_state.current_text = ""
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'audio_frames_received' not in st.session_state:
        st.session_state.audio_frames_received = False
    if 'using_fallback' not in st.session_state:
        st.session_state.using_fallback = False

def detect_risk_level(text):
    sentiment_result = sentiment_analyzer(text)[0]
    sentiment_score = sentiment_result['score']
    sentiment_label = sentiment_result['label']
    
    text_lower = text.lower()
    for keyword in RISK_KEYWORDS:
        if keyword in text_lower:
            return "High", sentiment_score

    if sentiment_label == 'NEGATIVE' and sentiment_score > 0.8:
        return "Medium", sentiment_score
    elif sentiment_label == 'NEGATIVE' and sentiment_score > 0.95:
        return "High", sentiment_score
    
    return "Normal", sentiment_score

def save_conversation(conversations):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.txt"
    
    with open(filename, "w") as f:
        for conv in conversations:
            f.write(f"{conv['timestamp']} - {conv['speaker']}: {conv['text']}\n")
            f.write(f"Risk Level: {conv['risk_level']} - Sentiment: {conv['sentiment_score']:.2f}\n\n")
    return filename

def process_audio(frame):
    sound = frame.to_ndarray()
    audio_queue.put(sound)
    st.session_state.audio_frames_received = True
    return av.AudioFrame.from_ndarray(sound, layout='mono')

def process_fallback_audio(audio_data, speaker):
    try:
        recognizer = sr.Recognizer()
        audio = sr.AudioData(audio_data.getvalue(), sample_rate=44100, sample_width=2)
        text = recognizer.recognize_google(audio)
        if text:
            risk_level, sentiment_value = detect_risk_level(text)
            conversation_entry = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "speaker": speaker,
                "text": text,
                "risk_level": risk_level,
                "sentiment_score": sentiment_value
            }
            st.session_state.conversations.append(conversation_entry)
            return text, risk_level, sentiment_value
    except Exception as e:
        st.error(f"Error processing fallback audio: {e}")
    return None, None, None

def main():
    st.title("Conversation Transcription & Sentiment Monitor")
    
    initialize_session_state()

    st.sidebar.header("Controls")
    if st.sidebar.button("Add New Speaker"):
        st.session_state.speaker_count += 1
        st.sidebar.success(f"Added Person{st.session_state.speaker_count}")

    # Audio status container
    audio_status = st.container()
    with audio_status:
        st.markdown("### Audio Input Status")
        status_indicator = st.empty()
        level_indicator = st.empty()

    # Fallback recorder container
    fallback_container = st.container()
    
    try:
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": False, "audio": True},
            on_change=lambda state: setattr(st.session_state, 'is_recording', state.playing),
            audio_frame_callback=process_audio
        )

        if webrtc_ctx.state.playing:
            if st.session_state.audio_frames_received:
                status_indicator.markdown("🎤 **Audio Detected and Processing**")
                level_indicator.progress(0.8)
            else:
                status_indicator.markdown("🔴 **Waiting for Audio Input...**")
                level_indicator.progress(0.1)
                if not st.session_state.using_fallback:
                    with fallback_container:
                        st.warning("WebRTC audio not detected. Using fallback recorder...")
                        audio_data = st.audio_recorder("Record audio here (Fallback)")
                        if audio_data:
                            st.session_state.using_fallback = True
                            speaker = st.selectbox(
                                "Who is speaking?",
                                [f"Person{i+1}" for i in range(st.session_state.speaker_count)]
                            )
                            text, risk_level, sentiment_value = process_fallback_audio(audio_data, speaker)
                            if text:
                                st.markdown(f"**Transcribed:** {text}")
                                st.markdown(f"**Sentiment Score:** {sentiment_value:.2f}")
                                if risk_level == "High":
                                    st.error("⚠️ High Risk Detected")
                                elif risk_level == "Medium":
                                    st.warning("⚠️ Medium Risk Detected")
                                else:
                                    st.success("✓ Normal Risk Level")
        else:
            status_indicator.markdown("⚫ **Recording Inactive - Press START to begin**")
            level_indicator.progress(0)

    except Exception as e:
        st.error(f"WebRTC Error: {e}")
        st.warning("Using fallback recorder due to WebRTC error.")
        with fallback_container:
            audio_data = st.audio_recorder("Record audio here (Fallback)")
            if audio_data:
                speaker = st.selectbox(
                    "Who is speaking?",
                    [f"Person{i+1}" for i in range(st.session_state.speaker_count)]
                )
                text, risk_level, sentiment_value = process_fallback_audio(audio_data, speaker)
                if text:
                    st.markdown(f"**Transcribed:** {text}")
                    st.markdown(f"**Sentiment Score:** {sentiment_value:.2f}")

    if st.session_state.conversations:
        st.markdown("### Conversation History")
        for conv in st.session_state.conversations:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{conv['speaker']}** ({conv['timestamp']}): {conv['text']}")
            with col2:
                st.write(f"Sentiment: {conv['sentiment_score']:.2f}")
            with col3:
                if conv['risk_level'] == "High":
                    st.error(f"Risk: {conv['risk_level']}")
                elif conv['risk_level'] == "Medium":
                    st.warning(f"Risk: {conv['risk_level']}")
                else:
                    st.success(f"Risk: {conv['risk_level']}")

    if st.button("Save Conversation"):
        if st.session_state.conversations:
            filename = save_conversation(st.session_state.conversations)
            st.success(f"✅ Conversation saved to {filename}")
        else:
            st.warning("No conversation to save yet")

if __name__ == "__main__":
    main()
