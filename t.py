# import streamlit as st
# import pandas as pd
# import os

import streamlit as st
# import time
st.markdown(f"Version: [GIT]('https://github.com/sean1832/SumGPT/blob/master/SumGPT/app/sidebar_handler.py')")
st.markdown("### SumGPT")
# progress_text = "Operation in progress. Please wait."
# my_bar = st.progress(0, text=progress_text)
st.sidebar.markdown("Version: `567890`")
st.sidebar.markdown("---")
st.sidebar.markdown("Author: [PranavSingh Patil]('url')")
st.sidebar.markdown("[Report a bug]('https://github.com/pranavsinghpatil/ArxivLensAI/issues')")
st.sidebar.markdown("[GitHub repo]('https://github.com/pranavsinghpatil/ArxivLensAI')")
st.sidebar.markdown("License: [MIT]('https://github.com/pranavsinghpatil/ArxivLensAI/blob/main/LICENSE')")
# # Writing a DataFrame
# df = pd.DataFrame({'first column': [1, 2, 3, 4], 'second column': [10, 20, 30, 40]})
# st.write(df)

# # Writing multiple arguments
# st.write("1 + 1 = ", 2)

# # You can then add more elements to the sidebar
# default_paper_path = "D:\Gits\re\temp\Attention Is All You Need(default_research_paper).pdf"
# temp_dir = "D:\Gits\re\temp"
# # st.sidebar.write("1. Select options from the dropdown below")
# # st.sidebar.write("2. View results in the main area")
# available_papers = {}
# #-------------------------------------SIDEBAR--------------------------------


# # ✅ Load Default Paper if No Uploads
# if not available_papers:
#     st.markdown("""
#         <style>
#         [data-testid="stSidebarContent"] .stAlert {
#             font-size: 5px;  # Adjust this value to your desired font size
#         }
#         </style>
#     """, unsafe_allow_html=True)
#     st.sidebar.info("📌 Attention Is All You Need (Default)")
#     if os.path.exists(default_paper_path):
#         available_papers["Attention Is All You Need (Default)"] = default_paper_path
#     else:
#         st.error("⚠️ Default research paper is missing! Please upload a file.")
# # 📌 Sidebar - Multiple PDF Upload
# # st.sidebar.header("📄 Upload Research Papers")
# uploaded_files = st.sidebar.file_uploader("📄 Upload Research Papers", type="pdf", accept_multiple_files=True)

# # ✅ Store uploaded papers
# if uploaded_files:
#     for uploaded_file in uploaded_files:
#         pdf_path = os.path.join(temp_dir, uploaded_file.name)
#         with open(pdf_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         # ✅ Show notification instead of sidebar clutter
#         st.toast(f"✅ {uploaded_file.name} uploaded!", icon="📄")
# st.sidebar.markdown("---")
# st.sidebar.info("Enter both API keys to proceed.")
# # Google AI API key input
# gapi_key = st.sidebar.text_input(
#     "🔑 Google AI API key",
#     type="password",
#     value=st.session_state.get("gapi_key", ""),
#     help="Enter your Google AI API key. It will be securely stored.",
#     key="gapi_key_input"
# )

# # Hugging Face API key input (assuming 'h' stands for Hugging Face)
# hapi_key = st.sidebar.text_input(
#     "🔑 Hugging Face API key",
#     type="password",
#     value=st.session_state.get("hapi_key", ""),
#     help="Enter your Hugging Face API key. It will be securely stored.",
#     key="hapi_key_input"
# )

# # Store the API keys in session state
# if gapi_key:
#     st.session_state["gapi_key"] = gapi_key

# if hapi_key:
#     st.session_state["hapi_key"] = hapi_key

# # Optional: Validate the API keys
# if gapi_key and not gapi_key.startswith("AIza"):
#     st.sidebar.warning("Please enter a valid Google AI API key!", icon="⚠️")

# if hapi_key and not hapi_key.startswith("hf_"):
#     st.sidebar.warning("Please enter a valid Hugging Face API key!", icon="⚠️")

# # Process the API keys
# if gapi_key and hapi_key:
#     st.sidebar.success("Both API keys have been entered successfully!")
#     # Your processing logic here
# elif gapi_key:
#     st.sidebar.info("Google AI API key entered. Hugging Face API key is missing.")
# elif hapi_key:
#     st.sidebar.info("Hugging Face API key entered. Google AI API key is missing.")
# else:
#     pass
# st.sidebar.markdown("---")


# # ✅ Dropdown to Select Research Papers
# selected_papers = st.sidebar.multiselect(
#     "📂 Select Research Papers",
#     list(available_papers.keys()),
#     default=list(available_papers.keys())
# )
# st.session_state.selected_papers = [available_papers[p] for p in selected_papers]

# # left_col, right_col = st.columns([2, 1])

# # with left_col:
# #     st.write("Main content goes here")

# # with right_col:
# #     st.write("This content will appear on the right")