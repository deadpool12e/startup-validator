import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


def main():
    # ---------------- LOAD ENV ----------------
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not groq_api_key or not tavily_api_key:
        st.error("GROQ_API_KEY or TAVILY_API_KEY missing in .env")
        st.stop()

    # ---------------- STREAMLIT CONFIG ----------------
    st.set_page_config(
        page_title="📈 Startup Idea Validator",
        layout="centered"
    )

    st.title("📈 Startup Idea Validator")
    st.write(
        "AI-powered evaluation of startup ideas using agentic reasoning and market grounding."
    )

    # ---------------- USER INPUT ----------------
    idea = st.text_area(
        "💡 Startup Idea Description",
        placeholder="Describe your startup idea..."
    )

    target_users = st.text_input(
        "🎯 Target Users",
        placeholder="e.g. Students, Small business owners"
    )

    market_info = st.text_area(
        "🌍 Market Information (optional)",
        placeholder="Any assumptions or known market details"
    )

    analyze_btn = st.button("🚀 Analyze Startup Idea")

    # ---------------- TOOLS ----------------
    tavily_tool = TavilySearchResults(
        api_key=tavily_api_key,
        max_results=5,
        include_answer=True
    )

    @tool
    def expand_target_users(user_group: str) -> str:
        """
        Expands a narrow group of target users into a wider audience.
        """
        mapping = {
            "students": "College students, High school students, Exam aspirants, Skill learners",
            "business owners": "Small business owners, Entrepreneurs, Startup founders",
            "developers": "Software developers, Programmers, Data scientists"
        }
        key = user_group.lower()
        return mapping.get(
            key,
            f"Expanded users for '{user_group}' could include a broader audience."
        )

    tools = [tavily_tool, expand_target_users]

    # ---------------- AGENT PROMPT ----------------
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a startup evaluation expert AI agent.

You have access to the following tools:
- expand_target_users: use ONLY if target users are vague or narrow.
- tavily_search_results_json: use ONLY if market info is missing, unclear, or needed for validation.

Guidelines:
- Do not call tools unnecessarily.
- Prefer reasoning before using tools.
- Base conclusions on evidence when tools are used.
- Provide strengths, weaknesses, risk analysis,
  monetization ideas, and a viability score out of 10.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # ---------------- AGENT CREATION ----------------
    def create_agent():
        llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.6,
            max_tokens=1024
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=3,
            handle_parsing_errors=True
        )
        return executor

    # ---------------- SESSION STATE ----------------
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = create_agent()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---------------- MAIN LOGIC ----------------
    if analyze_btn:
        if not idea.strip() or not target_users.strip():
            st.warning("⚠️ Please fill in both idea and target users fields.")
        else:
            with st.spinner("🔍 Evaluating your startup idea..."):
                user_input = f"""
Startup Idea: {idea}
Target Users: {target_users}
Market Info: {market_info or 'None provided'}
"""

                try:
                    response = st.session_state.agent_executor.invoke({
                        "input": user_input,
                        "chat_history": st.session_state.chat_history
                    })

                    output_text = response.get("output", str(response))

                    st.session_state.chat_history.append(("human", user_input))
                    st.session_state.chat_history.append(("assistant", output_text))

                    st.subheader("📊 Startup Evaluation Result")
                    for line in output_text.split("\n"):
                        if line.strip():
                            st.markdown(line)

                except Exception as e:
                    st.error(f"❌ Error running agent: {str(e)}")


# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    main()
