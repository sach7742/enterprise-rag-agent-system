import streamlit as st
import requests

st.title("Enterprise RAG Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If message contains sources, render them in an expander
        if message.get("sources"):
            with st.expander("📌 Sources & References"):
                for src in message["sources"]:
                    st.markdown(f"- [{src['title']}]({src['url']})")

# User query input
if prompt := st.chat_input("Ask a question..."):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/query",
                    json={"question": prompt},
                    timeout=60
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])

                    # Render answer
                    st.markdown(answer)

                    # Render citation expander if sources exist
                    if sources:
                        with st.expander("📌 Sources & References"):
                            for src in sources:
                                st.markdown(f"- [{src['title']}]({src['url']})")

                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")