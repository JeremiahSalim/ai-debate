import streamlit as st
import google.generativeai as genai
import os
import whisper
import re
import html
import asyncio
import edge_tts
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv

# --- CONFIGURATION ---
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
load_dotenv()

st.set_page_config(page_title="AI Debate Trainer", page_icon="🎙️", layout="wide")

# --- CSS STYLING ---
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* --- HIDE DEFAULT STREAMLIT ELEMENTS --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* --- LAYOUT ADJUSTMENTS --- */
    .block-container { 
        padding-top: 2rem; 
        padding-bottom: 12rem; 
        max_width: 900px;
    }

    /* --- LANDING PAGE HERO --- */
    .hero-container {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 1rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #475569;
        margin-bottom: 2rem;
    }

    /* --- CHAT BUBBLES --- */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .chat-card { 
        padding: 25px; 
        border-radius: 16px; 
        margin-bottom: 24px; 
        position: relative;
        animation: slideIn 0.5s ease-out forwards;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* USER STYLES */
    .user-card { 
        background: #eff6ff; /* Light Blue */
        border-left: 6px solid #3b82f6; 
        margin-left: 2rem;
    }
    
    /* AI STYLES */
    .ai-card { 
        background: #fff7ed; /* Light Orange */
        border-left: 6px solid #f97316; 
        margin-right: 2rem;
    }

    .card-header { 
        font-size: 0.85rem; 
        font-weight: 700; 
        color: #64748b; 
        margin-bottom: 12px; 
        text-transform: uppercase; 
        letter-spacing: 0.05em;
        display: flex; 
        justify-content: space-between; 
        align-items: center;
    }
    
    .card-content { 
        font-size: 1.1rem; 
        color: #334155; 
        line-height: 1.7; 
        font-weight: 400;
    }

    /* --- STICKY AUDIO PLAYER --- */
    .audio-sticky-wrapper {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        max-width: 800px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(15px);
        border: 1px solid #e2e8f0;
        border-radius: 50px;
        padding: 10px 30px;
        z-index: 99999;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .audio-container {
        width: 100%;
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .audio-label {
        font-weight: 700;
        color: #f97316;
        font-size: 0.9rem;
        white-space: nowrap;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* --- SIDEBAR CUSTOMIZATION --- */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
</style>
""",
    unsafe_allow_html=True,
)

# --- STATE INITIALIZATION ---
if "debate_started" not in st.session_state:
    st.session_state.debate_started = False
if "current_round" not in st.session_state:
    st.session_state.current_round = 1
if "debate_history" not in st.session_state:
    st.session_state.debate_history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY")
if "user_input_text" not in st.session_state:
    st.session_state.user_input_text = ""
if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None
if "audio_to_play" not in st.session_state:
    st.session_state.audio_to_play = None
if "evaluation_report" not in st.session_state:
    st.session_state.evaluation_report = None


# --- HELPER FUNCTIONS ---
def configure_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configuring API: {e}")
        return False

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def clean_text_content(text):
    clean = re.sub(r'<[^>]*>', '', text)
    return clean.strip()

async def generate_speech_async(text, voice="en-US-ChristopherNeural"):
    communicate = edge_tts.Communicate(text, voice)
    filename = "temp_output.mp3"
    await communicate.save(filename)
    return filename

def generate_speech(text):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        filename = loop.run_until_complete(generate_speech_async(text))
        return filename
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

def get_ai_response(topic, user_role, ai_role, debate_history, user_argument):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        history_context = "\n".join(
            [f"Round {h['round']} - {h['speaker']}: {clean_text_content(h['argument'])}" for h in debate_history]
        )
        prompt = f"""You are in a debate about: "{topic}"
Role: {ai_role} | Opponent: {user_role}

History:
{history_context}

Opponent's latest argument:
{user_argument}

Reply with a counter-argument.
1. Be persuasive and logical.
2. Keep it under 150 words.
3. Address the specific point made.
4. STRICTLY OUTPUT PLAIN TEXT ONLY. DO NOT USE MARKDOWN OR HTML.
"""
        response = model.generate_content(prompt)
        return clean_text_content(response.text)
    except Exception as e:
        return f"Error: {e}"

def evaluate_debate_performance(topic, user_role, debate_history):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        user_args = [h for h in debate_history if h["speaker"] == "You"]
        context = "\n\n".join(
            [f"Round {h['round']}:\n{h['argument']}" for h in user_args]
        )
        prompt = f"""Act as a strict debate coach.
Topic: {topic}
Side: {user_role}

Here are the user's arguments:
{context}

Provide a performance review. 
1. Assign a score (0-100) for EACH round based on logic.
2. Calculate the 'Overall Score' by taking the AVERAGE of the round scores.
3. STRICTLY OUTPUT PLAIN TEXT ONLY.

Format:
Overall Score: [Average Score]/100

Round 1:
Score: [0-100]
Feedback: [Your feedback]
"""
        response = model.generate_content(prompt)
        return clean_text_content(response.text)
    except Exception as e:
        return f"Error: {e}"

def process_debate_turn():
    user_text = st.session_state.user_input_text
    
    if user_text and user_text.strip():
        st.session_state.debate_history.append({
            "round": st.session_state.current_round, "speaker": "You", "role": st.session_state.user_role, "argument": user_text
        })

        try:
            ai_reply = get_ai_response(st.session_state.topic, st.session_state.user_role, st.session_state.ai_role, st.session_state.debate_history, user_text)
            
            st.session_state.debate_history.append({
                "round": st.session_state.current_round, "speaker": "AI", "role": st.session_state.ai_role, "argument": ai_reply
            })
            
            audio_file = generate_speech(ai_reply)
            if audio_file:
                with open(audio_file, "rb") as f:
                    st.session_state.audio_to_play = f.read()
            
            st.session_state.current_round += 1
            st.session_state.user_input_text = ""

            if st.session_state.current_round > 3:
                with st.spinner("Debate complete! Coach is grading your performance..."):
                    report = evaluate_debate_performance(
                        st.session_state.topic,
                        st.session_state.user_role,
                        st.session_state.debate_history
                    )
                    st.session_state.evaluation_report = report
            
        except Exception as e:
            st.error(f"Error AI: {e}")
    else:
        st.warning("Argument cannot be empty!")


# --- MODAL: EVALUATION ---
@st.dialog("📊 Debate Evaluation", width="large")
def show_review_dialog():
    if st.session_state.evaluation_report:
        st.info(st.session_state.evaluation_report, icon="📝")
    else:
        st.error("Report not found. Please try finishing the round again.")

    if st.button("Finish & Start Over", type="primary", use_container_width=True):
        st.session_state.debate_started = False
        st.session_state.debate_history = []
        st.session_state.current_round = 1
        st.session_state.user_input_text = ""
        st.session_state.audio_to_play = None
        st.session_state.evaluation_report = None
        st.rerun()


# --- SIDEBAR (SETTINGS) ---
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    if st.session_state.api_key:
        configure_gemini(st.session_state.api_key)
        st.success("API Connected", icon="🟢")
    else:
        st.warning("⚠️ API Key Missing")
        st.session_state.api_key = st.text_input("Gemini API Key", type="password")
    
    st.divider()

    # --- TOPIC & ROLE INPUTS ---
    if not st.session_state.debate_started:
        st.markdown("#### 📝 Debate Configuration")
        topic_input = st.text_input("Debate Topic", "Social Media does more harm than good")
        role_input = st.selectbox("Your Position", ["Pro (Agree)", "Con (Disagree)"])
        first_speaker_input = st.selectbox("First Speaker", ["User", "AI"])

        st.markdown("") # Spacer
        if st.button("🚀 Start Debate", type="primary", use_container_width=True):
            if st.session_state.api_key:
                st.session_state.debate_started = True
                st.session_state.topic = topic_input
                st.session_state.user_role = role_input
                st.session_state.first_speaker = first_speaker_input
                
                st.session_state.ai_role = "Con" if "Pro" in role_input else "Pro"
                
                st.session_state.current_round = 1
                st.session_state.debate_history = []
                st.session_state.audio_to_play = None
                st.session_state.evaluation_report = None
                st.rerun()
            else:
                st.error("Please enter API Key")
    else:
        st.info(f"**Topic:** {st.session_state.topic}")
        st.info(f"**Side:** {st.session_state.user_role}")
        
        st.markdown("") # Spacer
        if st.button("🔄 End / Reset", type="secondary", use_container_width=True):
            st.session_state.debate_started = False
            st.session_state.user_input_text = ""
            st.session_state.last_audio_bytes = None
            st.session_state.audio_to_play = None
            st.session_state.evaluation_report = None
            st.rerun()


# --- MAIN CONTENT ---
if st.session_state.debate_started:
    
    # Title with styling
    st.markdown(f"## 🎙️ Debate: <span style='color:#3b82f6'>{html.escape(st.session_state.topic)}</span>", unsafe_allow_html=True)
    
    # Progress Bar / Round indicator styled
    progress = min(st.session_state.current_round / 3, 1.0)
    st.progress(progress)
    
    if st.session_state.current_round <= 3:
        st.caption(f"Round {st.session_state.current_round} / 3 | You are: **{st.session_state.user_role}**")
    else:
        st.caption("🏁 Debate Finished")

    st.markdown("---")

    # 4. CHAT HISTORY
    chat_container = st.container()
    
    with chat_container:
        for entry in st.session_state.debate_history:
            is_user = entry["speaker"] == "You"
            css_class = "user-card" if is_user else "ai-card"
            icon = "👤 You" if is_user else "🤖 AI Opponent"
            safe_arg = html.escape(entry["argument"])

            st.markdown(
                f"""
            <div class="chat-card {css_class}">
                <div class="card-header">
                    <span>{icon}</span>
                    <span style="opacity:0.6">Round {entry["round"]}</span>
                </div>
                <div class="card-content">{safe_arg}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # 5. AI OPENING LOGIC
    if (st.session_state.first_speaker == "AI" 
        and st.session_state.current_round == 1 
        and len(st.session_state.debate_history) == 0):
        
        with st.spinner("🤖 AI is preparing opening statement..."):
            ai_res = get_ai_response(st.session_state.topic, st.session_state.user_role, st.session_state.ai_role, [], "Opening Statement")
            
            audio_file = generate_speech(ai_res)
            if audio_file:
                with open(audio_file, "rb") as f:
                    st.session_state.audio_to_play = f.read()

            st.session_state.debate_history.append({
                "round": 1, "speaker": "AI", "role": st.session_state.ai_role, "argument": ai_res
            })
            st.rerun()

    # 6. INPUT AREA OR VIEW REPORT BUTTON
    if st.session_state.current_round <= 3:
        st.write(f"### 🗣️ Your Turn")
        
        c1, c2 = st.columns([7, 1])
        
        with c2:
            st.write("**Record**")
            audio_bytes = audio_recorder(text="", recording_color="#ef4444", neutral_color="#3b82f6", icon_name="microphone", icon_size="2x")

        # Transcription logic
        if audio_bytes and audio_bytes != st.session_state.get("last_audio_bytes"):
            st.session_state.last_audio_bytes = audio_bytes
            try:
                with st.spinner("Transcribing audio..."):
                    temp_filename = "temp_audio.wav"
                    with open(temp_filename, "wb") as f: f.write(audio_bytes)
                    whisper_model = load_whisper_model()
                    result = whisper_model.transcribe(temp_filename)
                    if result["text"].strip():
                        st.session_state.user_input_text = result["text"].strip()
                    if os.path.exists(temp_filename): os.remove(temp_filename)
                    st.rerun()
            except Exception as e: st.error(f"Error: {e}")

        with c1:
            # ADDED PLACEHOLDER HERE
            st.text_area("Draft your argument here...", key="user_input_text", height=120, label_visibility="collapsed", placeholder="Give your arguments...")

        # BUTTON IS NOW FULL WIDTH OUTSIDE COLUMNS
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Submit Argument 📤", use_container_width=True, type="primary", on_click=process_debate_turn)
            
        st.markdown('</div>', unsafe_allow_html=True) # End Input Container
    
    else:
        # ROUNDS FINISHED
        st.markdown("---")
        st.success("✅ The debate has concluded! The Coach has generated your report.")
        
        if st.button("📊 View Evaluation Report", type="primary", use_container_width=True):
            show_review_dialog()

    # 7. STICKY AUDIO PLAYER
    if st.session_state.audio_to_play:
        st.markdown('<div class="audio-sticky-wrapper">', unsafe_allow_html=True)
        
        st.markdown('<div class="audio-container">', unsafe_allow_html=True)
        st.markdown('<span class="audio-label">🔊 AI Speaking:</span>', unsafe_allow_html=True)
        st.audio(st.session_state.audio_to_play, format="audio/mp3", autoplay=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # LANDING PAGE
    st.markdown(
        """
    <div class="hero-container">
        <h1 class="hero-title">🎙️ AI Debate Trainer</h1>
        <p class="hero-subtitle">
            Master the art of persuasion against an AI opponent.<br>
            Real-time audio, text transcription, and scoring.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🗣️ Speak Freely")
        st.caption("Use your microphone to argue your points. Whisper AI transcribes your voice instantly.")
    with col2:
        st.markdown("#### 🤖 Smart Opposition")
        st.caption("Powered by Gemini 2.0 Flash, the AI counters your logic with precision and distinct personality.")
    with col3:
        st.markdown("#### 📊 Instant Feedback")
        st.caption("Get a round-by-round score and a final coaching report after the debate concludes.")