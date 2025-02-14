# RAG Tariff Calculator with Groq LLaMA 70B

This project implements a **Retrieval-Augmented Generation (RAG) system** using **Groq’s LLaMA 3 70B**.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo.git
   cd your-repo
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Add your Groq API Key to `.env`:
   ```sh
   echo "GROQ_API_KEY=your_api_key_here" > .env
   ```

4. Run the application:
   ```sh
   streamlit run app.py
   ```

## Usage

1. Open the Streamlit application in your browser.
2. Select an existing Vector DB or upload a new PDF document.
3. Enter your question in the input box and click "Submit Question".
4. The application will process the question and display the results.

## Docker

### Build and Run the Docker Container

1. Build the Docker image:
   ```sh
   docker build -t rag-tariff-calculator .
   ```

2. Run the Docker container:
   ```sh
   docker run -p 8501:8501 -e GROQ_API_KEY=your_actual_api_key rag-tariff-calculator
   ```
   Replace `your_actual_api_key` with your actual Groq API key.

### Additional Notes

- **Poppler Path**: Ensure that the path to Poppler is correctly set in your application. The Dockerfile installs Poppler in the default location, so you might need to adjust the path in your code if necessary.
- **Environment Variables**: The `GROQ_API_KEY` is set as an environment variable in the Docker container.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License.