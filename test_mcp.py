"""
Verification test for the custom MCP Server tools.
Tests both Weather retrieval and Currency conversion.
"""

import os
import sys

# Ensure current project directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server import _fetch_weather_direct, _convert_currency_direct


def test_weather_tool():
    print("Testing MCP Tool 1: get_weather('Singapore', days=3)...")
    res = _fetch_weather_direct("Singapore", days=3)
    assert "forecast" in res, "Expected 'forecast' in weather response"
    assert len(res["forecast"]) == 3, "Expected 3 days of forecast"
    print(f" Source: {res['source']}")
    print(f" Destination: {res['destination']}")
    for day in res["forecast"]:
        print(f"  Day {day['day']} ({day['date']}): {day['condition']}, Temp: {day['min_temp_c']}C - {day['max_temp_c']}C, Rain: {day['precipitation_probability_pct']}%, Advice: {day['activity_advice']}")
    print(" Weather tool test PASSED!\n")


def test_currency_tool():
    print("Testing MCP Tool 2: convert_currency(50000, 'INR', 'SGD')...")
    res = _convert_currency_direct(50000, "INR", "SGD")
    assert "converted_amount" in res, "Expected 'converted_amount' in currency response"
    print(f" Source: {res['source']}")
    print(f" Result: {res['result_string']}")
    print(f" Rate: 1 INR = {res['exchange_rate']} SGD")
    print(" Currency tool test PASSED!\n")


if __name__ == "__main__":
    print("=== RUNNING MCP SERVER TOOL VERIFICATIONS ===")
    test_weather_tool()
    test_currency_tool()
    print("=== ALL MCP SERVER TESTS COMPLETED SUCCESSFULLY ===")
