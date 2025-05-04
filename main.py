import streamlit as st

from rag_llm import rag_llm
from langChain import langchain
# from img_gen import img_gen  # image gen disabled

# Initialize models only once
if "models" not in st.session_state.keys():
    query_llm = rag_llm()
    langchain_llm = langchain()
    st.session_state["models"] = [query_llm, langchain_llm]

# Recover model instances
query_llm, langchain_llm = st.session_state["models"]

# Track PDF state
if "files" not in st.session_state.keys():
    st.session_state["files"] = False

# Banner HTML
st.set_page_config(layout="wide")  # keep this line

# ✅ Show banner image correctly using Streamlit
st.image("images/fun_bot.png", use_column_width=True)


# Sidebar Upload
st.sidebar.header("Tools")
st.sidebar.subheader("Upload Files")
uploaded_file = st.sidebar.file_uploader("Upload files to add to your knowledge data base", type=['pdf'], accept_multiple_files=True)

st.title("The Fun AI bot !")

# Handle file upload
if uploaded_file:
    st.session_state["files"] = True
    query_llm.get_chroma()
    query_llm.upload_data(uploaded_file)

if not uploaded_file and st.session_state["files"] is True:
    st.session_state["files"] = False
    query_llm.remove_chroma()
    query_llm.get_chroma()

# Initialize message history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 👉 Main interaction block
if question := st.chat_input("Ask your question here !"):
    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)

    query_type = langchain_llm.get_query_type(question)

    if query_type == "img":
        st.warning("⚠️ Image generation is disabled on this system (no GPU support).")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = query_llm.search_chroma(question, query_type)
                answer = langchain_llm.get_chatbot_answer(question, context=context, query_type=query_type)
                st.write(answer)

        st.session_state["messages"].append({"role": "assistant", "content": answer})
