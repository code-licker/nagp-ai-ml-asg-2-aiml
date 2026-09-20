"""
MCP Client Module for LangChain Agent.
Binds custom MCP server tools (Weather and Currency) into LangChain-compatible tools
with strict error handling, schema documentation, and clear source demarcation.
"""

import os
import sys
import json

# Ensure directory is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from langchain_core.tools import tool
from mcp_server import _fetch_weather_direct, _convert_currency_direct


@tool
def get_singapore_weather(destination: str = "Singapore", days: int = 3) -> str:
    """
    Retrieve live weather conditions and multi-day forecast for Singapore.
    Use this tool whenever the user asks about current weather, rain probabilities,
    forecasts for upcoming days, or needs activity recommendations tailored to weather conditions.
    Returns: Day-by-day temperature, rain probability, conditions, and activity advice.
    """
    try:
        data = _fetch_weather_direct(destination=destination, days=days)
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({
            "source": "MCP Weather Tool (Error Fallback)",
            "error": f"Failed to retrieve live weather: {str(e)}",
            "destination": destination,
            "status": "unavailable"
        })


@tool
def convert_travel_currency(amount: float, from_currency: str = "INR", to_currency: str = "SGD") -> str:
    """
    Convert an amount from one currency to another (e.g., INR to SGD, USD to SGD, SGD to INR).
    Use this tool whenever the user mentions travel budget amounts, price conversions,
    or asks how much their money is worth in local Singapore Dollars (SGD).
    Returns: Converted amount, current exchange rate, and source attribution.
    """
    try:
        data = _convert_currency_direct(amount=amount, from_currency=from_currency, to_currency=to_currency)
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({
            "source": "MCP Currency Tool (Error Fallback)",
            "error": f"Failed to convert currency: {str(e)}",
            "amount": amount,
            "status": "unavailable"
        })


def get_all_travel_tools():
    """Returns the list of all tools available to the LangChain travel agent."""
    from rag import search_singapore_knowledge
    return [
        search_singapore_knowledge,
        get_singapore_weather,
        convert_travel_currency
    ]


if __name__ == "__main__":
    print("Testing MCP client tools...")
    print("Weather tool:", get_singapore_weather.invoke({"destination": "Singapore", "days": 2}))
    print("Currency tool:", convert_travel_currency.invoke({"amount": 50000, "from_currency": "INR", "to_currency": "SGD"}))
