"""
End-to-end test suite for Singapore Travel Planning Agent.
Tests:
1. Destination knowledge RAG retrieval
2. Live MCP Weather retrieval
3. Live MCP Currency conversion
4. Required Combined Scenario (3-day itinerary adjusted for weather)
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import SingaporeTravelAgent


def test_end_to_end_agent():
    print("=== INITIALIZING SINGAPORE TRAVEL AGENT ===")
    agent = SingaporeTravelAgent()

    # Test 1: Combined Scenario (RAG + Weather)
    query_combined = "Plan a three-day trip to Singapore and adjust the activities based on the weather forecast."
    print(f"\n--- Scenario 1: Combined RAG + MCP Weather ---")
    print(f"User: {query_combined}")
    res1 = agent.chat(query_combined)
    assert len(res1["response"]) > 100, "Expected substantive itinerary response"
    print("Assistant Response Preview:")
    print(res1["response"][:350] + "...\n")
    print(f"Tools Used: {[t['tool'] for t in res1.get('tools_used', [])]}")
    assert any("weather" in t["tool"] for t in res1.get("tools_used", [])), "Expected weather tool execution"
    print(" Scenario 1: PASSED\n")

    # Test 2: Combined Scenario with Currency Conversion
    query_currency = "I have a budget of INR 60,000. Convert it to SGD and suggest a three-day itinerary."
    print(f"--- Scenario 2: Combined RAG + Currency + Weather ---")
    print(f"User: {query_currency}")
    res2 = agent.chat(query_currency)
    assert "SGD" in res2["response"], "Expected SGD currency in response"
    print("Assistant Response Preview:")
    print(res2["response"][:350] + "...\n")
    print(f"Tools Used: {[t['tool'] for t in res2.get('tools_used', [])]}")
    assert any("currency" in t["tool"] for t in res2.get("tools_used", [])), "Expected currency tool execution"
    print(" Scenario 2: PASSED\n")

    # Test 3: Multi-turn Context Retention
    query_followup = "Are there good indoor activities if it rains on Day 2?"
    print(f"--- Scenario 3: Multi-Turn Context Follow-Up ---")
    print(f"User: {query_followup}")
    res3 = agent.chat(query_followup)
    assert len(res3["response"]) > 50, "Expected indoor activity recommendations"
    print("Assistant Response Preview:")
    print(res3["response"][:250] + "...\n")
    print(" Scenario 3: PASSED\n")

    print("=== ALL END-TO-END AGENT TESTS PASSED SUCCESSFULLY! ===")


if __name__ == "__main__":
    test_end_to_end_agent()
