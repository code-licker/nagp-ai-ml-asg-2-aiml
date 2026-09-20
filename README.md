# AI Travel Planning Assistant (Singapore)

An intelligent, context-aware AI travel assistant that blends a **document-based RAG knowledge base** with live, real-time **Model Context Protocol (MCP) tools**.

---

## 1. Architecture & System Design

```
+--------------------------------------------------------------------------+
|                          Streamlit Web UI / CLI                          |
|             (Interactive Chat, MCP Badges, RAG Citations)               |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                         LangChain Travel Agent                           |
|        (Multi-Turn Memory, Grounding System Prompt, Tool Binder)         |
|              Powered by Google Gemini (or Offline Engine)                |
+-------------------+-----------------------------------+------------------+
                    |                                   |
                    v                                   v
+-----------------------------------+   +----------------------------------+
|      RAG Knowledge Retriever      |   |        Custom MCP Client         |
+-----------------+-----------------+   +-----------------+----------------+
                  |                                       |
                  v                                       v
+-----------------------------------+   +----------------------------------+
|      Local Chroma Vector DB       |   |        Custom FastMCP Server     |
| (32 Chunks with Metadata URLs)   |   |        (JSON-RPC / Direct)       |
+-----------------+-----------------+   +--------+----------------+--------+
                  |                              |                |
                  v                              v                v
+-----------------------------------+     +-------------+  +---------------+
| Markdown Travel Knowledge Docs    |     | Open-Meteo  |  | ExchangeRate  |
| - Wikivoyage Singapore            |     | Weather API |  | Currency API  |
| - VisitSG Attractions (In/Out)    |     | (Live Rain) |  | (INR <-> SGD) |
| - VisitSG Itineraries & Practical |     +-------------+  +---------------+
+-----------------------------------+
```

---

## 2. Knowledge Base & RAG Workflow

The application ingests three structured travel documents in `data/knowledge_base/` with complete YAML metadata (`source_title`, `source_url`, `last_updated`):

1. [`wikivoyage_singapore.md`](data/knowledge_base/wikivoyage_singapore.md): Districts (Marina Bay, Chinatown, Little India, Kampong Glam, Sentosa), local transportation (MRT/buses, EZ-Link), laws & cultural etiquette, and famous hawker centres (Maxwell, Lau Pa Sat).
2. [`visitsg_attractions_indoor_outdoor.md`](data/knowledge_base/visitsg_attractions_indoor_outdoor.md): Carefully categorized directory of **Indoor attractions** (Cloud Forest & Flower Dome, ArtScience Museum, S.E.A. Aquarium, National Gallery, Jewel Changi) versus **Outdoor attractions** (Supertree Grove, Botanic Gardens, Zoo, Southern Ridges), including rain shelter advice and nearest MRT stations.
3. [`visitsg_itineraries_practical.md`](data/knowledge_base/visitsg_itineraries_practical.md): Flagship 3-day sightseeing itinerary, family activities with children, daily budget benchmarks (SGD 60–90 budget vs SGD 180–300 mid-range), and emergency contacts.

### Ingestion & Vector Retrieval:
* Chunking is handled by a header-aware markdown splitter respecting section hierarchies (max ~700 characters).
* Embeddings are generated using the fast local **ONNX MiniLM-L6-v2** model (384 dimensions), which runs 100% locally with zero external network lag.
* Chunks are stored in a persistent local **Chroma DB** (`chroma_db/`) alongside source metadata so every answer can be traced back to its origin.

---

## 3. Custom MCP Server Tools

Per the assignment brief and instructor Manish Kumar's guidance, this is a **custom-built MCP server** (`mcp_server.py`), NOT a ready-to-use cloud server:

1. **`get_weather(destination, days)`**:
   * Connects to the free Open-Meteo API for Singapore (`1.3521° N, 103.8198° E`).
   * Returns daily maximum/minimum temperatures, precipitation probabilities, weather descriptions, and tailored activity advice.
   * If offline, provides realistic deterministic tropical forecasts (31°C, afternoon convective showers).
2. **`convert_currency(amount, from_currency, to_currency)`**:
   * Connects to ExchangeRate-API for real-time exchange rates (e.g. INR to SGD, USD to SGD).
   * If offline, relies on benchmark currency index fallbacks.

---

## 4. Prompt Engineering & Grounding Strategy

The agent's system prompt strictly enforces:
1. **Fact Grounding**: Destination facts must come from the retrieved knowledge base.
2. **Current Information**: Weather and currency conversions must come from MCP tools.
3. **Weather-Aware Adaptation**: If rain or thunderstorms are detected by the weather tool, the agent proactively switches outdoor plans to indoor climate-controlled alternatives (e.g., swapping outdoor Sentosa beaches for S.E.A. Aquarium, or Marina Bay walks for Cloud Forest & Flower Dome).
4. **Attribution**: Every recommendation cites its source (e.g., `[Source: Visit Singapore...]`, `[MCP Tool: Open-Meteo Weather API]`).
5. **No Hallucinations**: If knowledge is absent, the agent explicitly states that information is unavailable rather than fabricating facts.

---

## 5. Development Mode Setup Instructions

Follow these quick steps to run the application locally:

### Step 1: Clone the Repository & Navigate to Folder
```bash
cd "assignment/AI and ML/travel_assistant_project"
```

### Step 2: Set Up Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Configure API Key (Optional)
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
Add your free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey):
```env
GEMINI_API_KEY=AIzaSy...
```
*(Note: If no API key is provided, the assistant automatically runs in **Offline Demonstration Mode**, allowing evaluators to test all RAG, MCP, and combined scenarios immediately without any friction).*

### Step 4: Ingest the Knowledge Base into Chroma DB
```bash
python ingest.py
```
*Creates the persistent vector database in `chroma_db/` from the markdown travel documents.*

---

## 6. How to Run the Application

### Option A: Interactive Streamlit Web UI (Recommended)

**Quick 1-Click Launch (Windows):**
```powershell
# In PowerShell:
.\run.ps1

# Or in Command Prompt:
run.bat
```

**Or run via virtual environment directly:**
```bash
# If virtual environment is activated:
streamlit run app.py

# Or without activating:
..\..\..\.venv\Scripts\python.exe -m streamlit run app.py
```
Open your browser at `http://localhost:8501`. 
* Use the quick scenario buttons for instant 1-click evaluations.
* Expand the **"View MCP Tools & RAG Retrieval Details"** drawer to inspect raw MCP JSON outputs and RAG source chunks.

### Option B: Terminal CLI
```bash
# Interactive chat loop
python cli.py

# Single-query execution
python cli.py -q "Plan a three-day trip to Singapore and adjust the activities based on the weather forecast."
```

---

## 7. Verification Test Suite

You can run our automated unit tests to verify each component individually:

```bash
# 1. Test Custom MCP Server (Weather + Currency)
python test_mcp.py

# 2. Test RAG Semantic Retrieval & Metadata
python test_rag.py

# 3. Test End-to-End Agent with Combined Scenario
python test_agent.py
```

---

## 8. Sample Questions & Expected Responses

### Query 1: Combined Scenario (RAG + MCP Weather)
> **User**: *"Plan a three-day trip to Singapore and adjust the activities based on the weather forecast."*

* **MCP Tool Invoked**: `get_singapore_weather(destination="Singapore", days=3)`
  * *Result*: Day 1 has 97% rain probability (Thunderstorms); Day 2 partly cloudy (35% rain).
* **RAG Retriever Invoked**: `search_singapore_knowledge("3-day sightseeing itinerary Marina Bay cultural Chinatown Sentosa")`
* **Synthesized Output**:
  * **Day 1**: Clear morning at Merlion Park. During the heavy afternoon thunderstorm, shifts indoors to Gardens by the Bay's **Cloud Forest** (35m indoor waterfall) and **ArtScience Museum** (*Future World* exhibit). Evening light show at Supertree Grove.
  * **Day 2**: Cultural walking tour through Little India (Tekka Centre) and Kampong Glam (Haji Lane). Afternoon air-conditioned visit to National Gallery Singapore. Evening in Chinatown.
  * **Day 3**: Morning at Sentosa Island / Botanic Gardens. Afternoon indoor marine sanctuary at **S.E.A. Aquarium**. Evening departure at Jewel Changi Airport.
  * *Citations*: `[MCP Tool: Open-Meteo Weather API]`, `[Source: Visit Singapore: Attractions & Weather Suitability Guide]`, `[Source: Wikivoyage Singapore Travel Guide]`.

---

### Query 2: Combined Budget & Currency Scenario
> **User**: *"I have a budget of INR 60,000. Convert it to SGD and suggest a 3-day itinerary."*

* **MCP Tool Invoked**: `convert_travel_currency(amount=60000, from_currency="INR", to_currency="SGD")`
  * *Result*: `60,000.00 INR = 799.32 SGD (Rate: 1 INR = 0.0133 SGD)`.
* **RAG Retriever Invoked**: Daily budget benchmarks (Mid-range benchmark is SGD 180–300/day).
* **Synthesized Output**: Confirms that ~800 SGD comfortably supports a 3-day mid-range stay (SGD 260/day), covering 3-star boutique hotel accommodation in Chinatown/Little India, unlimited MRT transit via Singapore Tourist Pass (SGD 20 for 3 days), authentic hawker dining at Maxwell/Tekka Centre (SGD 15–25/day), and key paid attractions (Gardens by the Bay conservatories).

---

## 9. Demo Video Recording Checklist

For the short (~1 minute) demo video:
1. Show starting the Streamlit app: `streamlit run app.py`.
2. Click the quick button: **"🌦️ 3-Day Weather Itinerary"**.
3. Highlight the response showing how outdoor activities were adapted to indoor attractions (Flower Dome/ArtScience Museum) due to rain.
4. Expand the **"View MCP Tools & RAG Retrieval Details"** drawer to show the live Open-Meteo JSON payload and RAG chunks.
5. Ask a follow-up budget question (e.g. *"Convert 50,000 INR to SGD"*), highlighting the live exchange rate MCP tool invocation.
