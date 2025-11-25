import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="AI Debate Trainer", page_icon="⚖️", layout="wide")

# --- MODERN STYLING & STICKY HEADER ---
st.markdown(
    """
<style>
    /* Remove default top padding from Streamlit */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        margin-top: 60px; /* Make room for sticky header */
    }
    
    /* Sticky Header Styling */
    .sticky-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        z-index: 9999;
        border-bottom: 1px solid #e0e0e0;
        padding: 10px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .header-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #333;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .round-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .status-badge-finished {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: #1a5c48;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 800;
    }

    /* Argument Card Styling */
    .chat-card {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
        transition: transform 0.2s;
    }
    
    .chat-card:hover {
        transform: translateY(-2px);
    }

    .user-card {
        background-color: #F8FBFF;
        border-left: 5px solid #2196F3;
    }

    .ai-card {
        background-color: #FFF9F0;
        border-left: 5px solid #FF9800;
    }
    
    .card-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #555;
        display: flex;
        justify-content: space-between;
    }
    
    .card-content {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #2c3e50;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "debate_started" not in st.session_state:
    st.session_state.debate_started = False
if "current_round" not in st.session_state:
    st.session_state.current_round = 1
if "debate_history" not in st.session_state:
    st.session_state.debate_history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY")


# --- FUNCTIONS ---
def configure_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configuring API: {e}")
        return False


def get_ai_response(topic, user_role, ai_role, debate_history, user_argument):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        history_context = "\n".join(
            [
                f"Round {h['round']} - {h['speaker']}: {h['argument']}"
                for h in debate_history
            ]
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
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"


def evaluate_debate_performance(topic, user_role, debate_history):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        user_args = [h for h in debate_history if h["speaker"] == "You"]
        context = "\n\n".join(
            [f"Round {h['round']}:\n{h['argument']}" for h in user_args]
        )

        prompt = f"""Act as a debate coach.
Topic: {topic}
Side: {user_role}
Arguments:
{context}

Provide feedback in this specific format:
**Score:** [0-100]
**Strength:** [One sentence]
**Weakness:** [One sentence]
**Advice:** [One sentence]
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"


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
            st.rerun()

# --- MAIN CONTENT ---

# 1. RENDER STICKY HEADER
if st.session_state.debate_started:
    # Logic to handle "Round 4" bug
    if st.session_state.current_round > 3:
        round_display = '<span class="status-badge-finished">🏁 Debate Complete</span>'
    else:
        round_display = f'<span class="round-badge">Round {st.session_state.current_round} / 3</span>'

    st.markdown(
        f"""
        <div class="sticky-header">
            <div class="header-title">
                ⚖️ AI Debate Trainer
            </div>
            {round_display}
        </div>
    """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="sticky-header">
            <div class="header-title">⚖️ AI Debate Trainer</div>
            <span style="color:gray; font-size:0.9rem;">Not Started</span>
        </div>
    """,
        unsafe_allow_html=True,
    )


if st.session_state.debate_started:
    # 2. DISPLAY HISTORY (Cards)
    for entry in st.session_state.debate_history:
        is_user = entry["speaker"] == "You"
        css_class = "user-card" if is_user else "ai-card"
        icon = "👤" if is_user else "🤖"
        align = "right" if is_user else "left"

        st.markdown(
            f"""
        <div class="chat-card {css_class}">
            <div class="card-header">
                <span>{icon} {entry["speaker"]}</span>
                <span>Round {entry["round"]}</span>
            </div>
            <div class="card-content">
                {entry["argument"]}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # 3. AI OPENING MOVE (If AI starts)
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

    # 4. INPUT AREA OR EVALUATION
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
                st.markdown(
                    f"""
                <div style="background-color:#f0f7ff; padding:20px; border-radius:10px; border:1px solid #cce5ff;">
                    {eval_text}
                </div>
                """,
                    unsafe_allow_html=True,
                )
    else:
        # Floating input area at bottom
        st.markdown("---")
        with st.form(key="input_form", clear_on_submit=True):
            col1, col2 = st.columns([6, 1])
            with col1:
                user_input = st.text_area(
                    "Type your argument here...",
                    height=100,
                    label_visibility="collapsed",
                    placeholder=f"Enter your Round {st.session_state.current_round} argument...",
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # spacer
                submit_btn = st.form_submit_button(
                    "Submit 📤", use_container_width=True
                )

            if submit_btn and user_input:
                # Add User Input
                st.session_state.debate_history.append(
                    {
                        "round": st.session_state.current_round,
                        "speaker": "You",
                        "role": st.session_state.user_role,
                        "argument": user_input,
                    }
                )

                # Get AI Response
                with st.spinner("Thinking..."):
                    ai_reply = get_ai_response(
                        st.session_state.topic,
                        st.session_state.user_role,
                        st.session_state.ai_role,
                        st.session_state.debate_history,
                        user_input,
                    )
                    st.session_state.debate_history.append(
                        {
                            "round": st.session_state.current_round,
                            "speaker": "AI",
                            "role": st.session_state.ai_role,
                            "argument": ai_reply,
                        }
                    )

                    st.session_state.current_round += 1
                    st.rerun()

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
            </ol>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
