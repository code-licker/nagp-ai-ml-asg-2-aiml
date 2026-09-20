"""
Core Conversational Agent for AI Travel Planning Assistant (Singapore).
Orchestrates LangChain, RAG destination knowledge base, and custom MCP tools.
Supports Google Gemini (via user's Google AI Pro plan), Groq, OpenAI,
as well as a fallback demonstration mode for evaluation without API keys.
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Optional

# Ensure directory is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag import retrieve_travel_knowledge, format_retrieved_context
from mcp_client import get_singapore_weather, convert_travel_currency, get_all_travel_tools


SYSTEM_PROMPT = """You are an expert, context-aware AI Travel Planning Assistant specializing in Singapore.
You combine two complementary sources of information to assist travelers:
1. Destination Knowledge Base (RAG): Provides verified destination facts, attractions, culture, transportation, food, and itineraries.
2. MCP Live Tools: Provides current real-time information (weather forecasts via get_singapore_weather and currency exchange via convert_travel_currency).

Follow these strict rules at all times:
- Fact Grounding: Ground all destination facts strictly in the retrieved knowledge base or MCP tool data. Do NOT invent opening hours, prices, or destination facts. If information is not in the knowledge base, state clearly that it is unavailable.
- Current Information: Always use MCP tools for live weather conditions or currency conversions.
- Weather-Aware Planning: When planning itineraries or recommending activities, inspect the weather forecast. If rain or thunderstorms are expected, proactively suggest sheltered or climate-controlled indoor alternatives (e.g., Flower Dome & Cloud Forest at Gardens by the Bay, ArtScience Museum, S.E.A. Aquarium, National Gallery, Jewel Changi).
- Source Attribution: Always cite your sources clearly at the end of each section:
  * For destination knowledge, cite the source title (e.g., [Source: Visit Singapore: Attractions & Weather Suitability Guide] or [Source: Wikivoyage Singapore Travel Guide]).
  * For current data, cite the MCP tool (e.g., [MCP Tool: Open-Meteo Weather API] or [MCP Tool: ExchangeRate-API]).
- Separation of Facts vs Suggestions: Clearly distinguish verified facts from your creative travel advice or suggested schedules.
- Multi-Turn Memory: Retain user preferences (e.g., travel dates, party size, children, budget constraints) across the conversation.
"""


class SingaporeTravelAgent:
    """
    High-level Travel Planning Assistant managing tool calling, RAG retrieval,
    conversation history, and response formatting.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.history: List[BaseMessage] = []
        self.tools = get_all_travel_tools()
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initializes the LLM based on available API keys."""
        # 1. Check Google Gemini (User's preferred provider via AI Pro plan)
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                # Use gemini-2.5-flash or gemini-1.5-flash
                active_model = self.model_name if "gemini" in self.model_name else "gemini-2.5-flash"
                llm = ChatGoogleGenerativeAI(
                    model=active_model,
                    google_api_key=gemini_key,
                    temperature=0.3,
                    convert_system_message_to_human=False
                )
                print(f"[Agent] Successfully initialized Google Gemini LLM ({active_model}).")
                return llm.bind_tools(self.tools)
            except Exception as e:
                print(f"[Agent Warning] Failed to initialize Gemini: {e}")

        # 2. Check OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from langchain_openai import ChatOpenAI  # type: ignore
                llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0.3)
                print("[Agent] Successfully initialized OpenAI LLM (gpt-4o-mini).")
                return llm.bind_tools(self.tools)
            except Exception as e:
                print(f"[Agent Warning] Failed to initialize OpenAI: {e}")

        # 3. Check Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                from langchain_groq import ChatGroq  # type: ignore
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_key, temperature=0.3)
                print("[Agent] Successfully initialized Groq LLM (llama-3.3-70b-versatile).")
                return llm.bind_tools(self.tools)
            except Exception as e:
                print(f"[Agent Warning] Failed to initialize Groq: {e}")

        print("[Agent Info] No external LLM API key detected. Running in Deterministic Demonstration Mode.")
        return None

    def reset_conversation(self):
        """Clears the conversational memory."""
        self.history = []

    def chat(self, user_query: str) -> Dict[str, Any]:
        """
        Processes a user query through the travel planning assistant.
        Returns a dictionary with:
        - 'response': formatted markdown response
        - 'tools_used': list of tools called with their inputs/outputs
        - 'rag_chunks': list of retrieved RAG documents
        - 'model_used': name of the model/engine that generated the response
        """
        tool_records = []
        rag_chunks = []

        # If an LLM with tool-calling capabilities is connected:
        if self.llm is not None:
            return self._chat_with_llm(user_query)

        # Otherwise, run in deterministic demonstration mode:
        return self._chat_fallback(user_query)

    def _chat_with_llm(self, user_query: str) -> Dict[str, Any]:
        """Executes multi-step tool-calling agent loop with real LLM."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + self.history + [HumanMessage(content=user_query)]
        tools_by_name = {t.name: t for t in self.tools}
        tool_records = []
        rag_chunks = []

        max_steps = 5
        step = 0
        final_ai_msg = None

        while step < max_steps:
            step += 1
            ai_msg = self.llm.invoke(messages)
            messages.append(ai_msg)
            final_ai_msg = ai_msg

            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    t_name = tool_call["name"]
                    t_args = tool_call["args"]
                    if t_name in tools_by_name:
                        tool_obj = tools_by_name[t_name]
                        tool_output = tool_obj.invoke(t_args)
                        tool_records.append({
                            "tool": t_name,
                            "args": t_args,
                            "output": tool_output
                        })
                        if t_name == "search_singapore_knowledge":
                            rag_chunks.append({"query": t_args.get("query", ""), "result": tool_output})

                        from langchain_core.messages import ToolMessage
                        messages.append(ToolMessage(
                            content=str(tool_output),
                            tool_call_id=tool_call["id"]
                        ))
            else:
                # No more tools called; final text ready
                break

        raw_content = final_ai_msg.content if final_ai_msg else ""

        # Cleanly extract text if content is returned as a list of dicts/blocks
        if isinstance(raw_content, list):
            extracted = []
            for item in raw_content:
                if isinstance(item, dict) and "text" in item:
                    extracted.append(item["text"])
                elif isinstance(item, str):
                    extracted.append(item)
            final_text = "\n".join(extracted)
        else:
            final_text = str(raw_content)

        # Update conversation history
        self.history.append(HumanMessage(content=user_query))
        self.history.append(AIMessage(content=final_text))

        return {
            "response": final_text,
            "tools_used": tool_records,
            "rag_chunks": rag_chunks,
            "model_used": self.model_name
        }

    def _chat_fallback(self, user_query: str) -> Dict[str, Any]:
        """
        Deterministic intent-based fallback engine for evaluation without an API key.
        Extracts intent, calls real RAG + real MCP tools, and formats complete response.
        """
        q_lower = user_query.lower()
        tool_records = []
        rag_chunks = []
        sections = []

        # 1. Check Currency Intent
        curr_match = re.search(r"(\d+[\d,]*)\s*(inr|usd|sgd|eur|gbp|aud|myr)", q_lower)
        if "convert" in q_lower or "budget" in q_lower or curr_match:
            amount = 50000.0
            from_curr = "INR"
            to_curr = "SGD"
            if curr_match:
                amount = float(curr_match.group(1).replace(",", ""))
                from_curr = curr_match.group(2).upper()
            if "to" in q_lower:
                to_match = re.search(r"to\s+(sgd|inr|usd|eur)", q_lower)
                if to_match:
                    to_curr = to_match.group(1).upper()

            curr_data = convert_travel_currency.invoke({"amount": amount, "from_currency": from_curr, "to_currency": to_curr})
            parsed_curr = json.loads(curr_data)
            tool_records.append({"tool": "convert_travel_currency", "args": {"amount": amount, "from": from_curr, "to": to_curr}, "output": parsed_curr})
            sections.append(f"### 💰 Budget & Currency Conversion\n* **Converted Budget**: {parsed_curr.get('result_string')}\n* *Attribution*: `[{parsed_curr.get('source', 'MCP Currency Tool')}]`\n")

        # 2. Check Weather Intent
        if "weather" in q_lower or "rain" in q_lower or "forecast" in q_lower or "itinerary" in q_lower or "plan" in q_lower:
            days = 3
            weather_data = get_singapore_weather.invoke({"destination": "Singapore", "days": days})
            parsed_weather = json.loads(weather_data)
            tool_records.append({"tool": "get_singapore_weather", "args": {"destination": "Singapore", "days": days}, "output": parsed_weather})

            w_lines = ["### 🌦️ Current Singapore Weather Forecast (via MCP Tool)", parsed_weather.get("summary", "")]
            for d in parsed_weather.get("forecast", []):
                rain_icon = "🌧️" if d.get("is_rainy") else "☀️"
                w_lines.append(f"* **Day {d['day']} ({d['date']})**: {rain_icon} {d['condition']} | Temp: {d['min_temp_c']}°C – {d['max_temp_c']}°C | Rain Chance: **{d['precipitation_probability_pct']}%**\n  * *Advice*: {d['activity_advice']}")
            w_lines.append(f"* *Attribution*: `[{parsed_weather.get('source', 'MCP Weather Tool')}]`\n")
            sections.append("\n".join(w_lines))

        # 3. Destination Knowledge Retrieval (RAG)
        rag_query = user_query
        if "itinerary" in q_lower or "three-day" in q_lower or "3-day" in q_lower:
            rag_query = "3-day sightseeing itinerary Marina Bay cultural Chinatown Sentosa"
        elif "indoor" in q_lower or "rain" in q_lower:
            rag_query = "indoor attractions Flower Dome Cloud Forest ArtScience Museum S.E.A. Aquarium"
        elif "family" in q_lower or "children" in q_lower:
            rag_query = "activities for families with children zoo aquarium"

        retrieved_docs = retrieve_travel_knowledge(rag_query, top_k=3)
        rag_chunks = retrieved_docs
        tool_records.append({"tool": "search_singapore_knowledge", "args": {"query": rag_query}, "output": f"Retrieved {len(retrieved_docs)} document chunks."})

        # 4. Synthesize Itinerary / Answer
        if "itinerary" in q_lower or "3-day" in q_lower or "three-day" in q_lower:
            itinerary_md = """### 🗺️ Weather-Adapted 3-Day Singapore Itinerary

#### Day 1: Modern Wonders & Marina Bay (Rain-Adaptive Schedule)
* **Morning (Clear Skies / 09:00 - 12:00)**: 
  * Outdoor sightseeing at **Merlion Park** for iconic skyline photos, then cross Jubilee Bridge to the Esplanade.
  * *Lunch*: Hainanese Chicken Rice at **Maxwell Food Centre** in Chinatown.
* **Afternoon (Thunderstorm Precaution / 13:30 - 17:00)**:
  * **Indoor Refuge**: Head into Gardens by the Bay's climate-controlled conservatories: **Cloud Forest** (35m indoor waterfall mountain) and **Flower Dome**.
  * Explore interactive digital installations at the lotus-shaped **ArtScience Museum** (*Future World* exhibit).
* **Evening (Clearing / 18:30 - 21:30)**:
  * Sunset views from Marina Bay Sands SkyPark Observation Deck.
  * Free **Garden Rhapsody** light show at Supertree Grove (19:45). Dinner at Lau Pa Sat Satay Street.
* *Attribution*: `[Source: Visit Singapore: Attractions & Weather Suitability Guide]` & `[Source: Wikivoyage Singapore Travel Guide]`

---

#### Day 2: Cultural Heritage & Historic Enclaves
* **Morning (09:00 - 12:30)**:
  * Explore **Little India** (Sri Veeramakaliamman Temple, colorful shophouse murals).
  * Breakfast of Roti Prata and Teh Tarik at **Tekka Centre**.
  * Walk to **Kampong Glam**: Sultan Mosque on Arab Street and boutique browsing along **Haji Lane**.
* **Afternoon (Indoor Shield / 14:00 - 17:00)**:
  * Air-conditioned exploration of **National Gallery Singapore** or **Asian Civilisations Museum**.
* **Evening (18:00 - 21:00)**:
  * Chinatown night market, Buddha Tooth Relic Temple, and dinner at Chinatown Complex.
* *Attribution*: `[Source: Wikivoyage Singapore Travel Guide]`

---

#### Day 3: Island Adventure & Departures
* **Morning (09:30 - 13:00)**:
  * Sentosa Island: Relax at Palawan Beach or ride the Skyline Luge.
  * *Alternative if morning rain*: Early walk through the UNESCO **Singapore Botanic Gardens** and National Orchid Garden.
* **Afternoon (Indoor Contingency / 14:00 - 17:00)**:
  * **S.E.A. Aquarium** on Sentosa Island (fully enclosed, home to 100,000+ marine animals).
* **Evening**:
  * Visit **Jewel Changi Airport** to marvel at the 40-metre indoor HSBC Rain Vortex before flight departure.
* *Attribution*: `[Source: Visit Singapore: Sample Itineraries & Practical Travel Planning]`
"""
            sections.append(itinerary_md)
        elif "indoor" in q_lower or "attraction" in q_lower:
            attraction_md = """### 🏛️ Verified Singapore Attractions

* **Top Indoor Attractions (Weather-Proof)**:
  * **Gardens by the Bay Conservatories**: Flower Dome and Cloud Forest (Bayfront MRT).
  * **ArtScience Museum**: Future World interactive exhibits (Bayfront MRT).
  * **Jewel Changi Airport**: 40m indoor HSBC Rain Vortex and Canopy Park (Changi Airport MRT).
  * **National Gallery Singapore**: Southeast Asian art collection (City Hall MRT).
  * **S.E.A. Aquarium**: Over 100,000 marine animals (Sentosa Island).
* **Top Outdoor Attractions (Clear Weather)**:
  * **Supertree Grove**: Iconic vertical gardens and nightly Garden Rhapsody show.
  * **Singapore Botanic Gardens**: 165-year-old UNESCO World Heritage site and National Orchid Garden.
  * **Singapore Zoo & Night Safari**: Open-concept rainforest wildlife parks.
* *Attribution*: `[Source: Visit Singapore: Attractions & Weather Suitability Guide]`
"""
            sections.append(attraction_md)
        else:
            general_md = "### 📚 Destination Knowledge\n"
            for i, doc in enumerate(retrieved_docs, start=1):
                general_md += f"**{doc['source_title']}**\n{doc['content']}\n*Source*: `{doc['source_url']}`\n\n"
            sections.append(general_md)

        final_response = "\n\n".join(sections)
        self.history.append(HumanMessage(content=user_query))
        self.history.append(AIMessage(content=final_response))

        return {
            "response": final_response,
            "tools_used": tool_records,
            "rag_chunks": rag_chunks,
            "model_used": "Demonstration Engine (Offline Mode)"
        }
