import json
from io import BytesIO
import streamlit as st

from src.harryswarm.config import get_settings
from src.harryswarm.tools import build_tools
from src.harryswarm.agents import run_agent
from src.harryswarm.models import list_models, group_models
from src.harryswarm.audio import transcribe_bytes, synthesize_speech
from src.harryswarm.vector_store_cli import create_store_with_files

st.set_page_config(page_title="HarrySwarm", page_icon="🧩", layout="wide")
st.title("HarrySwarm 🧩")
st.caption("Agentic workflows on the OpenAI Responses API + built-in tools")

settings = get_settings()

@st.cache_data(ttl=300)
def _load_models():
    ids = list_models()
    return group_models(ids)

agentish, gpt5, all_models = _load_models()

tab_agent, tab_voice, tab_admin = st.tabs(["🤖 Agent", "🎤 Voice (STT/TTS)", "🗂️ Admin"])

with tab_agent:
    st.sidebar.header("Configuration")

    st.sidebar.markdown("#### Model selection")
    gpt5_choice = st.sidebar.selectbox(
        "GPT-5 (if accessible)",
        options=["(none)"] + (gpt5 or []),
        index=0,
        help="Shown only if the API key can see GPT-5 variants."
    )

    agent_choice = st.sidebar.selectbox(
        "Agent-friendly models",
        options=agentish if agentish else ["gpt-4o","gpt-4o-mini","gpt-4o-realtime-preview"],
        help="Good defaults for tool-use via Responses API."
    )

    manual_model = st.sidebar.text_input("Manual override (wins)", value=settings.model)
    model = (manual_model.strip() or (gpt5_choice if gpt5_choice != "(none)" else agent_choice))
    st.sidebar.markdown(f"**Using model:** `{model}`")

    enable_web = st.sidebar.checkbox("Enable Web Search", value=True)
    enable_file = st.sidebar.checkbox("Enable File Search", value=bool(settings.vector_store_ids))
    enable_code = st.sidebar.checkbox("Enable Code Interpreter", value=True)
    enable_computer = st.sidebar.checkbox("Enable Computer Use (Operator)", value=False)

    if enable_computer:
        st.sidebar.warning("Computer Use can click/type/navigate in a virtual browser. Keep this off unless needed.")

    temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.2, 0.05)

    sys_prompt = st.sidebar.text_area(
        "System prompt (optional)", height=120,
        value="You are HarrySwarm, a careful operator that uses built-in tools, cites sources when available, and confirms irreversible actions."
    )

    st.markdown("### Task")
    default_goal = "Create a concise daily market brief using fresh web sources. If needed, run quick calcs with Code Interpreter and cite sources."
    goal = st.text_area("Goal", height=160, value=default_goal)

    if st.button("Run Agent 🚀"):
        with st.spinner("Thinking with tools…"):
            tools, tool_resources = build_tools(
                enable_web=enable_web,
                enable_file=enable_file,
                enable_code=enable_code,
                enable_computer=enable_computer
            )
            output_text, raw = run_agent(
                prompt=goal, model=model, tools=tools,
                tool_resources=tool_resources,
                system_prompt=sys_prompt, temperature=temperature
            )
        st.success("Done")
        st.markdown("### Output")
        st.write(output_text)

        st.markdown("### Debug / Usage")
        try:
            usage = getattr(raw, "usage", None)
            if usage:
                st.json(usage.model_dump() if hasattr(usage, "model_dump") else usage.__dict__)
        except Exception:
            pass

        with st.expander("Raw API Response"):
            try:
                st.json(json.loads(raw.model_dump_json()))
            except Exception:
                st.write(raw)

with tab_voice:
    st.subheader("Speech ↔ Text utilities (Audio API)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Speech → Text**")
        up = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav","mp3","m4a"], key="stt_up")
        stt_model = st.selectbox("STT model", [settings.stt_model, "gpt-4o-mini-transcribe"], index=0)
        if up and st.button("Transcribe"):
            text = transcribe_bytes(up.read(), filename=up.name, model=stt_model)
            st.success("Transcribed")
            st.text_area("Transcript", value=text, height=200)

    with col2:
        st.markdown("**Text → Speech**")
        tts_text = st.text_area("Text to speak", height=160, value="Hello from HarrySwarm.")
        tts_voice = st.selectbox("Voice", [settings.tts_voice, "verse","aria","nova","sage","vital","coral","amber","flora","chorus","sonnet"], index=0)
        tts_model = st.selectbox("TTS model", [settings.tts_model], index=0)
        if st.button("Synthesize"):
            audio_bytes = synthesize_speech(tts_text, voice=tts_voice, model=tts_model, fmt="mp3")
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button("Download MP3", data=audio_bytes, file_name="speech.mp3", mime="audio/mpeg")

with tab_admin:
    st.subheader("Vector Store setup (for File Search)")
    st.caption("Create a vector store and upload a folder of docs, then paste the ID into your .env (FILE_SEARCH_VECTOR_STORE_IDS).")
    vs_name = st.text_input("Vector store name", value="HarrySwarm-Docs")
    folder = st.text_input("Local folder to upload", value="./docs")
    if st.button("Create / Upload"):
        try:
            vid = create_store_with_files(vs_name, folder)
            st.success(f"Vector store created: {vid}")
            st.code(vid)
            st.caption("Add this to .env as FILE_SEARCH_VECTOR_STORE_IDS (comma-separated if multiple).")
        except Exception as e:
            st.error(f"Failed: {e}")
            st.caption("Check your API key/permissions and that the folder exists.")
