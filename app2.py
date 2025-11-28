import streamlit as st
import google.generativeai as genai
import os
import whisper
import re
import html  # <--- CRITICAL: Used to escape HTML characters
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv

# --- WINDOWS FFMPEG CONFIGURATION ---
# Ensure this path points to your actual FFmpeg bin folder
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

# 1. Load Environment Variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="AI Debate Trainer + Voice", page_icon="🎙️", layout="wide")

# --- CSS STYLING ---
st.markdown(
    """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 5rem; margin-top: 60px; }
    .sticky-header {
        position: fixed; top: 0; left: 0; width: 100%;
        background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px);
        z-index: 9999; border-bottom: 1px solid #e0e0e0;
        padding: 10px 20px; display: flex; justify-content: space-between; align-items: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .header-title { font-size: 1.2rem; font-weight: 700; color: #333; display: flex; align-items: center; gap: 10px; }
    .round-badge { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.9rem; font-weight: 600; }
    .status-badge-finished { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: #1a5c48; padding: 5px 15px; border-radius: 20px; font-size: 0.9rem; font-weight: 800; }
    .chat-card { padding: 20px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #f0f0f0; transition: transform 0.2s; }
    .user-card { background-color: #F8FBFF; border-left: 5px solid #2196F3; }
    .ai-card { background-color: #FFF9F0; border-left: 5px solid #FF9800; }
    .card-header { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 8px; color: #555; display: flex; justify-content: space-between; }
    .card-content { font-size: 1.05rem; line-height: 1.6; color: #2c3e50; }
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
    """
    1. Removes HTML tags (like </div>, <br>) using Regex.
    2. Strips whitespace.
    """
    clean = re.sub(r'<[^>]*>', '', text)
    return clean.strip()

def get_ai_response(topic, user_role, ai_role, debate_history, user_argument):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Prepare context (Clean history before sending)
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
4. STRICTLY OUTPUT PLAIN TEXT ONLY. DO NOT USE MARKDOWN OR HTML TAGS.
"""
        response = model.generate_content(prompt)
        # Clean response before returning to UI
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
1. Start with an 'Overall Score' from 0 to 100 based on logic, persuasion, and relevance.
2. Provide a breakdown for each round.
3. STRICTLY OUTPUT PLAIN TEXT ONLY. DO NOT USE HTML TAGS.

Format:
Overall Score: [0-100]/100

Round 1:
Score: [0-100]
Feedback: [Your feedback]
"""
        response = model.generate_content(prompt)
        # Clean response before returning
        return clean_text_content(response.text)
    except Exception as e:
        return f"Error: {e}"


# --- CORE LOGIC CALLBACK ---
def process_debate_turn():
    # Capture text from the bound widget state
    user_text = st.session_state.user_input_text
    
    if user_text and user_text.strip():
        # 1. Save User Argument
        st.session_state.debate_history.append(
            {
                "round": st.session_state.current_round,
                "speaker": "You",
                "role": st.session_state.user_role,
                "argument": user_text,
            }
        )

        # 2. Get AI Response
        try:
            ai_reply = get_ai_response(
                st.session_state.topic,
                st.session_state.user_role,
                st.session_state.ai_role,
                st.session_state.debate_history,
                user_text,
            )
            
            # 3. Save AI Argument
            st.session_state.debate_history.append(
                {
                    "round": st.session_state.current_round,
                    "speaker": "AI",
                    "role": st.session_state.ai_role,
                    "argument": ai_reply,
                }
            )
            
            # 4. Update Game State
            st.session_state.current_round += 1
            st.session_state.user_input_text = ""  # Clear textbox
            # NOTE: We DO NOT reset last_audio_bytes here to prevent ghost inputs
            
        except Exception as e:
            st.error(f"Error AI: {e}")
    else:
        st.warning("Argument cannot be empty!")


# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")

    if st.session_state.api_key:
        configure_gemini(st.session_state.api_key)
    else:
        st.warning("⚠️ API Key Missing")
        st.session_state.api_key = st.text_input("Gemini API Key", type="password")

    st.divider()

    if not st.session_state.debate_started:
        topic = st.text_input("Topic", "Social Media does more harm than good")
        user_role = st.selectbox("Your Side", ["Pro (For)", "Con (Against)"])
        first_speaker = st.selectbox("Who Speaks First?", ["User", "AI"])

        if st.button("Start Debate", type="primary", use_container_width=True):
            if st.session_state.api_key:
                st.session_state.debate_started = True
                st.session_state.topic = topic
                st.session_state.user_role = user_role
                st.session_state.ai_role = "Con" if "Pro" in user_role else "Pro"
                st.session_state.first_speaker = first_speaker
                st.session_state.current_round = 1
                st.session_state.debate_history = []
                st.rerun()
            else:
                st.error("Please enter API Key")
    else:
        st.info(f"Topic: **{st.session_state.topic}**")
        if st.button("End / Reset", type="secondary", use_container_width=True):
            st.session_state.debate_started = False
            st.session_state.user_input_text = ""
            st.session_state.last_audio_bytes = None
            st.rerun()

# --- MAIN LAYOUT ---

# 1. HEADER
if st.session_state.debate_started:
    if st.session_state.current_round > 3:
        round_display = '<span class="status-badge-finished">🏁 Debate Complete</span>'
    else:
        round_display = f'<span class="round-badge">Round {st.session_state.current_round} / 3</span>'

    st.markdown(
        f"""
        <div class="sticky-header">
            <div class="header-title">🎙️ AI Debate Trainer (Voice Enabled)</div>
            {round_display}
        </div>
    """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="sticky-header">
            <div class="header-title">🎙️ AI Debate Trainer</div>
            <span style="color:gray; font-size:0.9rem;">Not Started</span>
        </div>
    """,
        unsafe_allow_html=True,
    )

# 2. GAMEPLAY AREA
if st.session_state.debate_started:
    # A. Display History
    for entry in st.session_state.debate_history:
        is_user = entry["speaker"] == "You"
        css_class = "user-card" if is_user else "ai-card"
        icon = "👤" if is_user else "🤖"

        # --- FIX: ESCAPE HTML CONTENT ---
        # This converts characters like < and > into safe text so they don't break the layout
        safe_argument = html.escape(entry["argument"])

        st.markdown(
            f"""
        <div class="chat-card {css_class}">
            <div class="card-header">
                <span>{icon} {entry["speaker"]}</span>
                <span>Round {entry["round"]}</span>
            </div>
            <div class="card-content">
                {safe_argument}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # B. AI Opening (If AI first & Round 1)
    if (
        st.session_state.first_speaker == "AI"
        and st.session_state.current_round == 1
        and len(st.session_state.debate_history) == 0
    ):
        with st.spinner("🤖 AI is formulating opening argument..."):
            ai_res = get_ai_response(
                st.session_state.topic,
                st.session_state.user_role,
                st.session_state.ai_role,
                [],
                "Opening Statement",
            )
            st.session_state.debate_history.append(
                {
                    "round": 1,
                    "speaker": "AI",
                    "role": st.session_state.ai_role,
                    "argument": ai_res,
                }
            )
            st.rerun()

    # C. Input / Evaluation Area
    if st.session_state.current_round > 3:
        st.markdown("### 📊 Performance Review")
        if st.button("Generate Coach Feedback", type="primary"):
            with st.spinner("Coach is analyzing your logic..."):
                eval_text = evaluate_debate_performance(
                    st.session_state.topic,
                    st.session_state.user_role,
                    st.session_state.debate_history,
                )
                st.success("Analysis Complete")
                
                # --- SAFEST DISPLAY METHOD ---
                # Using st.info prevents any HTML layout breakage
                st.info(eval_text, icon="📝")
                
    else:
        st.markdown("---")
        st.write(f"### 🗣️ Your Turn (Round {st.session_state.current_round})")

        c1, c2 = st.columns([8, 1])

        # --- STEP 1: AUDIO RECORDER (Right Col) ---
        with c2:
            st.write("Record:")
            audio_bytes = audio_recorder(
                text="",
                recording_color="#e74c3c",
                neutral_color="#3498db",
                icon_name="microphone",
                icon_size="2x",
            )

        # --- STEP 2: TRANSCRIPTION PROCESS (Before UI Render) ---
        if audio_bytes and audio_bytes != st.session_state.get("last_audio_bytes"):
            st.session_state.last_audio_bytes = audio_bytes
            
            try:
                with st.spinner("Transcribing audio..."):
                    # Save temp file
                    temp_filename = "temp_audio.wav"
                    with open(temp_filename, "wb") as f:
                        f.write(audio_bytes)

                    # Transcribe (Auto Detect)
                    whisper_model = load_whisper_model()
                    result = whisper_model.transcribe(temp_filename)
                    transcribed_text = result["text"].strip()

                    # Update State
                    if transcribed_text:
                        st.session_state.user_input_text = transcribed_text
                        st.success("Audio transcribed! You can edit before submitting.")
                    else:
                        st.warning("No speech detected.")

                    # Cleanup
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
                    
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error transcription: {e}")

        # --- STEP 3: TEXT AREA (Left Col) ---
        with c1:
            st.text_area(
                "Argumen Anda",
                key="user_input_text", 
                height=100,
                placeholder="Tekan mic di kanan untuk bicara, atau ketik disini...",
                label_visibility="collapsed",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # --- STEP 4: SUBMIT BUTTON (Callback) ---
        st.button(
            "Submit Argument 📤", 
            use_container_width=True, 
            type="primary",
            on_click=process_debate_turn 
        )

else:
    # Welcome Screen
    st.markdown(
        """
    <div style="text-align: center; padding: 50px;">
        <h1>👋 Welcome to AI Debate Trainer</h1>
        <p style="color: gray; font-size: 1.2rem;">Improve your argumentation skills against a Gemini-powered opponent.</p>
        <br>
        <div style="background: #f9f9f9; padding: 20px; border-radius: 10px; display: inline-block; text-align: left;">
            <strong>How to use:</strong>
            <ol>
                <li>Choose a controversial topic.</li>
                <li>Pick your side (Pro/Con).</li>
                <li>Click <b>Start Debate</b>.</li>
                <li>🗣️ <b>Voice:</b> Record audio -> Check text -> Edit -> Submit.</li>
            </ol>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )