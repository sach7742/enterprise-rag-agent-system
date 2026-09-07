import sys
from typing import Dict, Any, List
from src.agents.graph_router import app, retriever


def format_response(output: Dict[str, Any]) -> None:
    route = output.get("route", "unknown").upper()
    answer = output.get("answer", "").strip()
    docs = output.get("documents", [])

    print(f"\n[\033[94mROUTE\033[0m: {route}]")
    if docs:
        print(f"[\033[93mCONTEXT\033[0m: Used {len(docs)} document chunk(s)]")

    print("\n\033[1mAgent Response:\033[0m")
    print(f"{answer}\n")
    print("-" * 60)


def main():
    print("=" * 60)
    print("\033[1mEnterprise RAG System - Stateful Memory CLI\033[0m")
    print("Type 'exit' to quit | Type 'clear' to reset chat memory")
    print("=" * 60)

    chat_history: List[Dict[str, str]] = []

    try:
        while True:
            try:
                user_input = input("\n\033[92mUser > \033[0m").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting session...")
                break

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("Shutting down session...")
                break

            if user_input.lower() == "clear":
                chat_history.clear()
                print("\033[93mChat history cleared!\033[0m")
                continue

            state = {
                "query": user_input,
                "chat_history": chat_history,
                "route": "",
                "documents": [],
                "answer": "",
            }

            try:
                output = app.invoke(state)
                format_response(output)

                # Persist turn to chat history
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": output["answer"].strip()})

            except Exception as e:
                print(f"\n\033[91mError executing graph:\033[0m {e}\n")

    finally:
        if hasattr(retriever, "close"):
            retriever.close()
        print("Session ended.")


if __name__ == "__main__":
    main()