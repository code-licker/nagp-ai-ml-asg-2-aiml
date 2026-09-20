"""
Streamlit Web UI for AI Travel Planning Assistant (Singapore).
Demonstrates:
- Conversational chat with retained multi-turn context
- Real-time MCP tool invocations (Weather and Currency)
- Semantic RAG knowledge base retrieval (Chroma DB)
- Explicit badges, tool execution details, and source citations
- Pre-built quick action buttons for one-click scenario evaluations
"""

import os
import sys
import json
import streamlit as st

# Ensure current directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from agent import SingaporeTravelAgent


# Set page configuration
st.set_page_config(
    page_title="Singapore AI Travel Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .badge-rag {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-mcp {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = SingaporeTravelAgent(model_name="gemini-2.5-flash")


# Sidebar Configuration
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1525625293386-3f8f99389edd?w=600", caption="Singapore Skyline & Marina Bay", use_container_width=True)
    st.title("Travel Assistant Config")
    
    st.markdown("### 🔌 Connected Services")
    st.markdown("✅ **RAG Knowledge Base**: Local Chroma DB (Wikivoyage & Visit Singapore)")
    st.markdown("✅ **MCP Tool 1**: Live Weather Forecast (`Open-Meteo API`)")
    st.markdown("✅ **MCP Tool 2**: Currency Conversion (`ExchangeRate-API`)")
    
    st.divider()
    st.markdown("### 🤖 LLM Provider")
    model_choice = st.selectbox(
        "Select Active Model",
        ["gemini-2.5-flash (Google AI Pro)", "gemini-1.5-pro", "gemini-1.5-flash", "Offline Demonstration Mode"],
        index=0
    )
    
    api_key_input = st.text_input("Gemini API Key (Optional if set in .env)", type="password")
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        if st.button("Apply API Key"):
            st.session_state.agent = SingaporeTravelAgent(model_name=model_choice.split(" ")[0])
            st.success("API Key applied to agent!")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent.reset_conversation()
        st.rerun()

    st.markdown("---")
    st.caption("Developer Assignment: Context-Aware Travel Planning Assistant using LangChain, RAG & MCP.")


# Main Application Header
st.markdown('<div class="main-title">🇸🇬 Singapore AI Travel Planning Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent destination planning powered by <b>Document-based RAG</b> + live <b>Model Context Protocol (MCP)</b> tools.</div>', unsafe_allow_html=True)


# Quick Action Scenario Buttons
st.markdown("##### ⚡ Quick Scenarios (Evaluator Shortcuts):")
col1, col2, col3, col4 = st.columns(4)

scenario_to_run = None

with col1:
    if st.button("🌦️ 3-Day Weather Itinerary", use_container_width=True):
        scenario_to_run = "Plan a three-day trip to Singapore and adjust the activities based on the weather forecast."

with col2:
    if st.button("💰 Budget & 60k INR Convert", use_container_width=True):
        scenario_to_run = "I have a budget of INR 60,000. Convert it to SGD and suggest a 3-day itinerary."

with col3:
    if st.button("🏛️ Best Indoor Attractions", use_container_width=True):
        scenario_to_run = "What are the best indoor cultural and museum attractions to visit when it rains?"

with col4:
    if st.button("👨‍👩‍👧 Family Trip with Kids", use_container_width=True):
        scenario_to_run = "Suggest outdoor attractions and replace them with indoor options if rain is expected, suitable for a family with children."


# Render Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # If assistant message has tool execution metadata, display expander
        if msg.get("tools_used"):
            with st.expander("🔍 View MCP Tools & RAG Retrieval Details", expanded=False):
                for t in msg["tools_used"]:
                    t_name = t.get("tool", "")
                    if "weather" in t_name:
                        st.markdown(f"**🌦️ MCP Weather Tool Invocation** (`{t_name}`)")
                        st.json(t.get("output"))
                    elif "currency" in t_name:
                        st.markdown(f"**💰 MCP Currency Tool Invocation** (`{t_name}`)")
                        st.json(t.get("output"))
                    else:
                        st.markdown(f"**📚 RAG Knowledge Retrieval** (`{t_name}`)")
                        st.text(f"Query: {t.get('args')}")
                        st.write(t.get("output"))


# Handle User Query (from Quick Button or Chat Input)
chat_input_val = st.chat_input("Ask about Singapore itineraries, attractions, weather, or budgets...")
user_prompt = scenario_to_run or chat_input_val

if user_prompt:
    # 1. Append User Message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing destination knowledge & querying live MCP tools..."):
            result = st.session_state.agent.chat(user_prompt)
            response_text = result["response"]
            tools_used = result.get("tools_used", [])

            st.markdown(response_text)

            if tools_used:
                with st.expander("🔍 View MCP Tools & RAG Retrieval Details", expanded=False):
                    for t in tools_used:
                        t_name = t.get("tool", "")
                        if "weather" in t_name:
                            st.markdown(f"**🌦️ MCP Weather Tool Invocation** (`{t_name}`)")
                            st.json(t.get("output"))
                        elif "currency" in t_name:
                            st.markdown(f"**💰 MCP Currency Tool Invocation** (`{t_name}`)")
                            st.json(t.get("output"))
                        else:
                            st.markdown(f"**📚 RAG Knowledge Retrieval** (`{t_name}`)")
                            st.text(f"Args: {t.get('args')}")
                            st.write(t.get("output"))

    # 3. Store Assistant Message in Session State
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "tools_used": tools_used
    })
