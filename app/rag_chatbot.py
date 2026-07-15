import streamlit as st
import os
import io
import json
import base64
from groq import Groq
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder
from agent_tools import TOOLS_SPEC, dispatch_tool_call

MODEL_NAME = "llama-3.3-70b-versatile"
MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = """You are the IntelliRisk AI analyst assistant — an agent embedded in an \
enterprise loan-approval / insurance-fraud platform. You have four tools available:

- search_policy_docs: for policy/rule questions — thresholds, definitions, "why would X be
  rejected/flagged", "what counts as suspicious".
- query_portfolio_data: for questions about the REAL data (307,511 loan applicants,
  15,420 fraud claims) — counts, rates, averages, breakdowns, comparisons, trends. Write
  SQL yourself using the schema you're given in the tool description.
- score_loan_applicant / score_fraud_claim: for hypothetical "what if" scenarios — run the
  live trained models on a described applicant or claim.

Decide which tool(s) the question actually needs — call none, one, or several. Don't call
query_portfolio_data for policy questions, and don't call search_policy_docs for questions
about the real data's statistics. After getting tool results, answer the user directly and
concisely (2-5 sentences unless a breakdown/table is genuinely needed), citing concrete
numbers from the tool output. If you ran SQL or scored a scenario, briefly say so. This may
be read aloud, so avoid bullet points, markdown tables, or heavy formatting in the final
answer."""


# ── Groq client ───────────────────────────────────────────────────────────────
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    return Groq(api_key=api_key), api_key


# ── Speech to Text via Groq Whisper ──────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes) -> str:
    """Send audio bytes to Groq Whisper for transcription."""
    client, _ = get_groq_client()
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.wav"
    transcription = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=audio_file,
        response_format="text"
    )
    return transcription.strip()


# ── Text to Speech via gTTS ───────────────────────────────────────────────────
def text_to_speech(text: str) -> str:
    """Convert text to speech and return base64 audio string."""
    tts = gTTS(text=text, lang='en', slow=False)
    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)
    audio_base64 = base64.b64encode(mp3_buffer.read()).decode('utf-8')
    return audio_base64


def autoplay_audio(audio_base64: str):
    """Play audio using Streamlit's native audio player.

    Older Streamlit versions (pre-1.29) don't support the `autoplay` kwarg on
    st.audio() and raise a TypeError — fall back to a manual-play player
    instead of crashing the whole page.
    """
    import base64 as b64lib
    audio_bytes = b64lib.b64decode(audio_base64)
    buf = io.BytesIO(audio_bytes)
    try:
        st.audio(buf, format="audio/mp3", autoplay=True)
    except TypeError:
        buf.seek(0)
        st.audio(buf, format="audio/mp3")
        st.caption("🔊 Your Streamlit version doesn't support autoplay — press play above.")


# ── Agentic loop: LLM decides which tool(s) to call, we execute them ─────────
def ask_agent(question: str, history: list | None = None) -> tuple:
    """Run the tool-calling agent loop for one user question.

    Returns:
        (answer_text, tool_trace) where tool_trace is a list of
        {"name": str, "args": dict, "result": dict} for every tool the
        model actually invoked, in order.
    """
    client, _ = get_groq_client()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    tool_trace = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=800,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return (msg.content or "").strip(), tool_trace

        # Record the assistant's tool-call turn, then execute each tool.
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = dispatch_tool_call(tc.function.name, args)
            print(f"[agent] tool call: {tc.function.name}({args}) -> {str(result)[:300]}")
            tool_trace.append({"name": tc.function.name, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": json.dumps(result, default=str)[:4000],
            })

    # Ran out of tool-call rounds — force a final answer with no more tools.
    response = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, temperature=0.2, max_tokens=500,
    )
    return (response.choices[0].message.content or "").strip(), tool_trace


def _history_from_session() -> list:
    """Turn the displayed chat history into plain role/content pairs the
    agent loop can use as conversation context (tool-call details aren't
    replayed — only what was actually said)."""
    hist = []
    for m in st.session_state.rag_messages[-8:]:
        hist.append({"role": m["role"], "content": m["content"]})
    return hist


def _render_tool_trace(tool_trace: list):
    """Show which tool(s) fired and what they returned — the visible,
    auditable part of 'agentic'."""
    if not tool_trace:
        return
    labels = {
        "search_policy_docs": "📚 Policy search",
        "query_portfolio_data": "🗄️ Live SQL query",
        "score_loan_applicant": "🏦 Loan model run",
        "score_fraud_claim": "🛡️ Fraud model run",
    }
    names = [labels.get(t["name"], t["name"]) for t in tool_trace]
    with st.expander(f"🔧 Tools used: {', '.join(names)}", expanded=False):
        for t in tool_trace:
            st.markdown(f"**{labels.get(t['name'], t['name'])}**")
            if t["name"] == "query_portfolio_data":
                st.code(t["result"].get("sql", ""), language="sql")
                if t["result"].get("ok"):
                    st.dataframe(t["result"].get("rows", []), use_container_width=True)
                else:
                    st.error(t["result"].get("error", "query failed"))
            elif t["name"] == "search_policy_docs":
                for src in t["result"].get("results", []):
                    st.markdown(
                        f"<div class='chat-source'><strong>Source</strong><br>{src['title']}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.json(t["result"])


# ── Main Chatbot UI ───────────────────────────────────────────────────────────
def render_rag_chatbot():

    st.markdown("""
    <style>
    .chat-source {
        background: rgba(29, 158, 117, 0.08);
        border-left: 3px solid #1D9E75;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 12px;
        color: #8899aa;
    }
    .chat-source strong {
        color: #1D9E75;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .powered-badge {
        display: inline-block;
        background: rgba(83,74,183,0.15);
        border: 1px solid rgba(83,74,183,0.3);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 11px;
        color: #AFA9EC;
        margin-left: 8px;
        vertical-align: middle;
    }
    .voice-tip {
        background: rgba(29,158,117,0.08);
        border: 1px solid rgba(29,158,117,0.2);
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 12px;
        color: #8899aa;
        margin-bottom: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "## IntelliRisk AI Agent "
        "<span class='powered-badge'>⚡ Llama 3.3 · Agentic Tool-Calling · RAG + Live SQL</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#8899aa;margin-top:-10px;margin-bottom:16px'>"
        "Ask about policy rules, real portfolio stats, or test a what-if applicant/claim. "
        "The agent decides which tool(s) to use — policy search, live SQL over the real "
        "data, or the trained models. Type or use your voice.</p>",
        unsafe_allow_html=True
    )

    # ── API key check ─────────────────────────────────────────────────────────
    _, api_key = get_groq_client()
    if not api_key:
        st.error("GROQ_API_KEY not found.")
        st.markdown("Run this in your terminal then restart Streamlit:")
        st.code('export GROQ_API_KEY="gsk_your-key-here"', language="bash")
        return

    # ── Session state ─────────────────────────────────────────────────────────
    if "rag_messages"   not in st.session_state:
        st.session_state.rag_messages = []
    if "rag_tool_trace" not in st.session_state:
        st.session_state.rag_tool_trace = {}
    if "voice_enabled"  not in st.session_state:
        st.session_state.voice_enabled = True
    if "last_audio_key" not in st.session_state:
        st.session_state.last_audio_key = None

    def handle_question(question: str):
        history = _history_from_session()
        st.session_state.rag_messages.append({"role": "user", "content": question})
        with st.spinner("🤖 Thinking — deciding which tools to use..."):
            answer, trace = ask_agent(question, history=history)
        msg_idx = len(st.session_state.rag_messages)
        st.session_state.rag_messages.append({"role": "assistant", "content": answer})
        st.session_state.rag_tool_trace[msg_idx] = trace
        if st.session_state.voice_enabled:
            audio_b64 = text_to_speech(answer)
            autoplay_audio(audio_b64)

    # ── Voice toggle + mic row ────────────────────────────────────────────────
    col_toggle, col_mic = st.columns([1, 3])

    with col_toggle:
        st.session_state.voice_enabled = st.toggle(
            "🔊 Read answers aloud",
            value=st.session_state.voice_enabled
        )

    with col_mic:
        st.markdown(
            "<div class='voice-tip'>"
            "🎤 Click the mic button below to speak your question — "
            "it will be transcribed automatically."
            "</div>",
            unsafe_allow_html=True
        )

    # ── Mic recorder ─────────────────────────────────────────────────────────
    audio = mic_recorder(
        start_prompt="🎤 Click to speak",
        stop_prompt="⏹ Stop recording",
        just_once=True,
        use_container_width=True,
        key="mic_recorder"
    )

    # ── Process voice input ───────────────────────────────────────────────────
    if audio and audio.get("bytes"):
        audio_key = audio.get("id", str(len(audio["bytes"])))
        if audio_key != st.session_state.last_audio_key:
            st.session_state.last_audio_key = audio_key
            with st.spinner("🎤 Transcribing your voice..."):
                try:
                    transcribed = transcribe_audio(audio["bytes"])
                    if transcribed:
                        st.success(f'🎤 You said: *"{transcribed}"*')
                        handle_question(transcribed)
                        st.rerun()
                except Exception as e:
                    st.error(f"Transcription error: {e}")

    st.divider()

    # ── Suggested questions ───────────────────────────────────────────────────
    if not st.session_state.rag_messages:
        st.markdown(
            "<p style='font-size:13px;color:#8899aa;margin-bottom:8px'>"
            "Try asking:</p>",
            unsafe_allow_html=True
        )
        suggestions = [
            "Why would a loan be rejected?",
            "What % of loans default when the applicant owns no property?",
            "Would a 22 year old earning $18k asking for a $400k loan be approved?",
            "How does the anomaly detection score work?",
            "What's the fraud rate for third-party fault claims vs policy-holder fault?",
            "Is a claim with no witness, no police report, and 3 past claims suspicious?",
        ]
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            if cols[i % 3].button(suggestion, key=f"sug_{i}"):
                handle_question(suggestion)
                st.rerun()
        st.markdown("---")

    # ── Chat history ──────────────────────────────────────────────────────────
    for idx, message in enumerate(st.session_state.rag_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and idx in st.session_state.rag_tool_trace:
                _render_tool_trace(st.session_state.rag_tool_trace[idx])

    # ── Text chat input ───────────────────────────────────────────────────────
    if user_input := st.chat_input("Or type your question here..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            history = _history_from_session()
            st.session_state.rag_messages.append({"role": "user", "content": user_input})
            with st.spinner("🤖 Thinking — deciding which tools to use..."):
                answer, trace = ask_agent(user_input, history=history)
            st.markdown(answer)
            msg_idx = len(st.session_state.rag_messages)
            st.session_state.rag_messages.append({"role": "assistant", "content": answer})
            st.session_state.rag_tool_trace[msg_idx] = trace
            _render_tool_trace(trace)
            if st.session_state.voice_enabled:
                with st.spinner("🔊 Generating audio..."):
                    audio_b64 = text_to_speech(answer)
                autoplay_audio(audio_b64)

    # ── Clear chat ────────────────────────────────────────────────────────────
    if st.session_state.rag_messages:
        if st.button("🗑️ Clear chat", type="secondary"):
            st.session_state.rag_messages = []
            st.session_state.rag_tool_trace = {}
            st.rerun()
