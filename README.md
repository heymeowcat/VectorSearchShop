# VectorSearchShop

This app allows users to search for products by either entering text or uploading an image, and retrieves relevant products from a database

### Running the App

**Prerequisites:**

- Python 3.7 or later

**Installation:**

1.  **Clone the Repository:**

    ```
    git clone https://github.com/heymeowcat/VectorSearchShop.git
    cd mobileAppScreentoTestsGemini
    ```

2.  **Install Dependencies:**

    ```
    pip install -r requirements.txt
    ```

3.  **Set Up Gemini API Key:**

    - Create a project and obtain an API key from Google AI Platform:
    - Create a `.env` file in the project root directory and add the following line, replacing `YOUR_GEMINI_API_KEY` with your actual key:

      ```
      GEMINI_API_KEY=YOUR_GEMINI_API_KEY
      ```

4.  **Run the App:**

    ```
    streamlit run main.py
    ```

    This will launch the app in your web browser, usually at `http://localhost:8501`.
