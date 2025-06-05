# Document Q&A with OpenAI

An intelligent document analysis and question-answering system powered by OpenAI's GPT models.

## Features

- **Multi-format Support**: Upload and process TXT, PDF, and DOCX files
- **Document Analysis**:
  - Automatic text summarization
  - Reading time estimation
  - Text complexity scoring
  - Keyword extraction and visualization
- **Interactive Q&A**: Ask questions about your documents and get AI-powered answers
- **Document Management**: Organize and manage multiple documents
- **Chat History**: Keep track of all questions and answers
- **Export Functionality**: Export chat history to CSV

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vikas-indexnine/Q-A-OpenAI-.git
cd Q-A-OpenAI-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install NLTK data:
```bash
python -m textblob.download_corpora
```

## Usage

1. Run the application:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to http://localhost:8501

3. Enter your OpenAI API key in the sidebar

4. Upload documents and start asking questions!

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

## Project Structure

```
.
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
└── test_imports.py     # Import testing utility
```

## Configuration

The application uses the following configurations:
- OpenAI GPT-3.5 Turbo model for Q&A
- Streamlit for the web interface
- TextBlob for text analysis
- PyPDF2 for PDF processing
- python-docx for DOCX processing

## Contributing

Feel free to open issues or submit pull requests for any improvements.

## License

This project is open source and available under the MIT License.

## Author

Vikas Gupta 