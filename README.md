# ArxivLensAI 🚀

![ArxivLensAI](https://img.shields.io/badge/Status-Active-brightgreen) ![License](https://img.shields.io/badge/License-MIT-blue) ![Contributions](https://img.shields.io/badge/PRs-Welcome-orange)

> **ArxivLensAI: The Future of Research Interaction**  
> Transform static research papers into a dynamic, interactive Q&A experience. Whether you're a researcher, student, or tech enthusiast, our AI-powered assistant empowers you to dive deeper into academic content with ease and precision.

---

## 🌟 Key Features

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
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate
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

## 📸 Visual Showcase

Explore the powerful interface through our curated screenshots:

### 📂 Upload Interface
![Upload](static/screenshots/upload.png)

### ❓ Interactive Query
![Query](static/screenshots/query.png)

### 💡 Answer Display
![Answer](static/screenshots/answer.png)

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

- **Documentation:** Explore our detailed [Project Documentation](docs/DOCUMENTATION.md).
- **User Guide:** Refer to the [User Guide](docs/USER_GUIDE.md) for a complete walkthrough.
- **API Reference:** Check out the [API Reference](docs/API_REFERENCE.md) for integration details.

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

```
