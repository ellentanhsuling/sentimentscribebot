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

def detect_risk_level(text):
    # Get sentiment analysis
    sentiment_result = sentiment_analyzer(text)[0]
    sentiment_score = sentiment_result['score']
    sentiment_label = sentiment_result['label']
    
    # Check keywords
    text_lower = text.lower()
    for keyword in RISK_KEYWORDS:
        if keyword in text_lower:
            return "High", sentiment_score

    # Determine risk level based on sentiment
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
    return av.AudioFrame.from_ndarray(sound, layout='mono')

def audio_frame_callback(frame):
    try:
        recognizer = sr.Recognizer()
        audio_data = frame.to_ndarray().tobytes()
        audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)
        text = recognizer.recognize_google(audio)
        if text:
            st.session_state.current_text = text
            return text
    except Exception as e:
        return None

def main():
    st.title("Conversation Transcription & Sentiment Monitor")
    
    st.markdown("""
    ### How to Use This App
    1. **Add Speakers**: Use the 'Add New Speaker' button in the sidebar
    2. **Start Audio**: Click the 'START' button in the audio widget
    3. **Select Speaker**: Choose the current speaker when text appears
    4. **Monitor Risk & Sentiment**: Real-time analysis of conversation
    5. **Save**: Store the conversation with sentiment data
    """)
    
    initialize_session_state()

    st.sidebar.header("Controls")
    if st.sidebar.button("Add New Speaker"):
        st.session_state.speaker_count += 1
        st.sidebar.success(f"Added Person{st.session_state.speaker_count}")

    # Create placeholders for live updates
    status_placeholder = st.empty()
    transcript_placeholder = st.empty()
    sentiment_placeholder = st.empty()
    
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    if webrtc_ctx.state.playing:
        status_placeholder.markdown("🔴 **Recording in Progress**")
        transcript_placeholder.markdown("### Live Transcript")

    if webrtc_ctx.audio_receiver:
        if webrtc_ctx.state.playing:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                for audio_frame in audio_frames:
                    text = audio_frame_callback(audio_frame)
                    if text:
                        transcript_placeholder.markdown(f"🎤 **Current Speech:** {text}")
                        
                        speaker = st.selectbox(
                            "Who is speaking?",
                            [f"Person{i+1}" for i in range(st.session_state.speaker_count)]
                        )
                        
                        risk_level, sentiment_score = detect_risk_level(text)
                        sentiment_placeholder.markdown(f"Sentiment Score: {sentiment_score:.2f}")
                        
                        conversation_entry = {
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "speaker": speaker,
                            "text": text,
                            "risk_level": risk_level,
                            "sentiment_score": sentiment_score
                        }
                        
                        st.session_state.conversations.append(conversation_entry)
                        
                        if risk_level == "High":
                            st.error("⚠️ High Risk Detected - Immediate Action Required")
                            if st.button("Contact Psychologist"):
                                st.info("Connecting to emergency response system...")
                        elif risk_level == "Medium":
                            st.warning("⚠️ Medium Risk Detected - Monitor Closely")
                        else:
                            st.success("✓ Normal Risk Level")
            except queue.Empty:
                pass

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
