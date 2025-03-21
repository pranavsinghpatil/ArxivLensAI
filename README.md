# ArxivLensAI 🚀 

![ArxivLensAI](https://img.shields.io/badge/Status-Active-brightgreen) ![License](https://img.shields.io/badge/License-MIT-blue) [![GitHub Release](https://img.shields.io/github/v/release/pranavsinghpatil/ArxivLensAI.svg)](https://github.com/pranavsinghpatil/ArxivLensAI/releases)
 ![Contributions](https://img.shields.io/badge/PRs-Welcome-orange) [![Issues](https://img.shields.io/github/issues/pranavsinghpatil/ArxivLensAI)](https://github.com/pranavsinghpatil/ArxivLensAI/issues) [![Last Commit](https://img.shields.io/github/last-commit/pranavsinghpatil/ArxivLensAI)](https://github.com/pranavsinghpatil/ArxivLensAI/commits/main)  

[![Stars](https://img.shields.io/github/stars/pranavsinghpatil/ArxivLensAI?style=social)](https://github.com/pranavsinghpatil/ArxivLensAI/stargazers)  [![Forks](https://img.shields.io/github/forks/pranavsinghpatil/ArxivLensAI?style=social)](https://github.com/pranavsinghpatil/ArxivLensAI/network/members)  


> **ArxivLensAI: The Future of Research Interaction**  
> ArxivLensAI is an innovative AI-powered research assistant designed to revolutionize the way researchers, students, and tech enthusiasts interact with academic content. By transforming static research papers into dynamic, interactive Q&A experiences, ArxivLensAI empowers users to delve deeper into complex topics with ease and precision.


## 🌟 Key Features
- 📄 Upload and process multiple PDFs  
- 🔍 Semantic search using FAISS  
- 🤖 AI-powered question answering  
- 📊 Extracts tables and figures from PDFs  
- 🖼️ Supports image extraction from research papers
  
- **Interactive Q&A System:** Pose questions and receive immediate, context-aware answers.
- **Intelligent PDF Processing:** Effortlessly extract and analyze content from research papers.
- **High-Performance FAISS Storage:** Leverage advanced similarity search for rapid retrieval.
- **Scalable Architecture:** Built to handle extensive research databases and evolving user needs.
- **User-Centric Design:** Enjoy a sleek, intuitive interface crafted for a seamless research experience.


---
## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/pranavsinghpatil/ArxivLensAI.git
cd ArxivLensAI
```

### 2️⃣ Set Up the Environment

#### Using Virtualenv (pip)
```bash
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On Mac/Linux
```

#### Using Conda
```bash
conda create --name arxivlensai python=3.9
conda activate arxivlensai
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Launch the Application

```bash
streamlit run app.py
```

💡 **Experience the future of research. Upload your PDF and let ArxivLensAI guide your discoveries!**

---
### System Workflow

1. **PDF Upload & Processing**
![PDF Upload & Processing](static/screenshots/sw1.png)

2. **Query Processing**
![Query Processing](static/screenshots/sw2.png)

---

## 📸 Visual Showcase

Explore the powerful interface through our curated screenshots:

### 📂 Upload Interface
<div align="center">
  <img src="static/screenshots/UI.png" alt="Upload" width="600"/>
</div>

### ❓ Interactive Query
<div align="center">
  <img src="static/screenshots/querying.png" alt="Query" width="600"/>
</div>

### 💡 Answer Display
<div align="center">
  <img src="static/screenshots/response.png" alt="Answer" width="600"/>
</div>


---
## 📂 Project Architecture

A modular design ensures scalability and ease of maintenance:

```
ArxivLensAI/
├── app.py                # Main application entry
├── main.py               # Primary execution file
├── extract_text.py       # PDF content extraction module
├── qa_system.py          # AI-driven Q&A engine
├── vector_store.py       # FAISS vector management
├── utils.py              # Utility functions
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
├── static/
│   ├── icons/            # UI assets and icons
│   └── screenshots/      # Visual documentation
├── extracted_images/     # Extracted images from PDFs
├── faiss_indexes/        # FAISS index storage
└── temp/                 # Temporary file storage
```

---

## 🔧 Configuration

### API Keys
- Set your API keys in `utils.py` or use environment variables
- Required APIs:
  - Google AI API (for Gemini)
  - Hugging Face API

### Model Configuration
- Default embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- QA model: `google/flan-t5-large`
- Retrieval model: `deepset/roberta-base-squad2`

---
## 🎯 Features in Detail

### PDF Processing
- Extracts plain text using PyMuPDF
- Identifies and extracts tables using pdfplumber
- Processes images with OCR using Tesseract
- Builds FAISS indexes for efficient searching

### Question Answering
- Combines multiple AI models for comprehensive answers
- Uses conversation history for context
- Supports full-context queries for detailed analysis
- Implements query expansion for better search results

### Vector Search
- Uses FAISS for efficient similarity search
- Implements dynamic thresholding for result filtering
- Supports batch processing for better performance


---
## 🔮 Roadmap & Future Enhancements

We're continuously evolving ArxivLensAI. Upcoming features include:

- **LangChain Integration:** Elevate NLP capabilities with cutting-edge technologies.
- **Advanced Content Parsing:** Extract and interpret tables, images, and figures.
- **Enhanced User Dashboard:** Build a comprehensive web UI using Streamlit.
- **Customizable AI Modules:** Tailor answer generation for domain-specific research.
- **Community-Driven Innovations:** Incorporate user feedback and contributions to shape future releases.


---
## 🤝 Contributing

We welcome contributions from developers and researchers alike. Here’s how you can join our journey:

1. **Fork the Repository:** Create your personal copy.
2. **Create a Feature Branch:** Work on your ideas without affecting the main branch.
3. **Commit Your Changes:** Ensure your commits are descriptive.
4. **Submit a Pull Request:** Let’s collaborate to innovate together!

For more details, please refer to our [Contribution Guidelines](CONTRIBUTING.md).

---

## 📜 License

ArxivLensAI is released under the **MIT License**. See the [LICENSE](LICENSE) file for complete details.

---

## 💬 Join the Conversation

Stay updated and get involved:

[![GitHub](https://img.shields.io/badge/Visit-GitHub-black?style=for-the-badge&logo=github)](https://github.com/pranavsinghpatil/ArxivLensAI) [![Twitter](https://img.shields.io/badge/Follow-Twitter-blue?style=for-the-badge&logo=twitter)](https://x.com/syntaxnomad)

---

## 📚 Additional Resources

- **Project Technical Report:** Explore our detailed [Project Documentation](docs/TechnicalReport.md).
- **User Guide:** Refer to the [User Guide](docs/UserGuide.md) for a complete walkthrough.
- **API Reference:** Check out the [API Reference](docs/APIReference.md) for integration details.
- **Release Notes:** Stay updated with our [Release Notes](docs/ReleaseNotes.md).

---

## 🛠️ Setup & Maintenance Tips

For a seamless setup experience, note the following:

- **Environment Management:** Whether using virtualenv or Conda, ensure your environment is activated before installing dependencies.
- **Dependency Updates:** Regularly update your dependencies to stay compatible with the latest features.
- **Community Support:** Engage with our community for troubleshooting and feature discussions.

---
## Project Files Descriptions:

`utils.py`: Contains utility functions for handling API keys, generating filenames, and expanding queries.

`main.py`: Handles the primary execution flow, including PDF processing and FAISS index creation.

`extract_text.py`: Manages the extraction of text, tables, and images from PDFs using PyMuPDF, pdfplumber, and Tesseract.

`vector_store.py`: Implements FAISS index creation and management for efficient similarity search.

`qa_system.py`: Integrates multiple AI models to provide comprehensive answers to user queries.

`app.py`: The main application file that sets up the Streamlit interface and manages user interactions.

---

*Made with ❤️ and passion for transforming the way research is experienced.*


