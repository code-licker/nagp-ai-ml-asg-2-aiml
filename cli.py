"""
Command-Line Interface for AI Travel Planning Assistant (Singapore).
Allows interactive testing in the terminal or single-shot execution for testing.
"""

import os
import sys
import argparse

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure current directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from agent import SingaporeTravelAgent


def run_interactive_cli():
    print("\n" + "=" * 65)
    print(" 🇸🇬 Singapore AI Travel Planning Assistant (Terminal CLI)")
    print(" Powered by Document-based RAG + Custom MCP Tools")
    print(" Type 'exit' or 'quit' to stop.")
    print("=" * 65 + "\n")

    agent = SingaporeTravelAgent()

    while True:
        try:
            query = input("User > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye! Safe travels!")
                break

            print("\nThinking and querying tools...\n")
            result = agent.chat(query)
            print("-" * 65)
            print(result["response"])
            print("-" * 65)

            if result.get("tools_used"):
                print(f"[Tools Executed: {', '.join([t['tool'] for t in result['tools_used']])}]\n")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def run_single_query(query: str):
    agent = SingaporeTravelAgent()
    print(f"\nProcessing Query: '{query}'\n")
    result = agent.chat(query)
    print("=" * 65)
    print(result["response"])
    print("=" * 65)
    print(f"\nTools Invoked: {len(result.get('tools_used', []))}")
    for t in result.get("tools_used", []):
        print(f" - {t['tool']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Singapore Travel Planning Assistant CLI")
    parser.add_argument("--query", "-q", type=str, help="Run a single query and exit")
    args = parser.parse_args()

    if args.query:
        run_single_query(args.query)
    else:
        run_interactive_cli()
