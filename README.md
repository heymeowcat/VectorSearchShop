# VectorSearchShop

This app allows users to search for products by either entering text or uploading an image, and retrieves relevant products from a database
<img width="1512" alt="Screenshot 2024-03-26 at 5 54 27 PM" src="https://github.com/heymeowcat/VectorSearchShop/assets/40495273/5de95b98-1124-4fb4-a48e-357c7da372a0">


### Running the App

**Installation:**

1.  **Clone the Repository:**

    ```
    git clone https://github.com/heymeowcat/VectorSearchShop.git
    cd VectorSearchShop
    ```

2.  **Install Dependencies:**

    ```
    pip install -r requirements.txt
    ```

3.  **Set Up Gemini API Key:**

    - Create a project and obtain an API key from http://aistudio.google.com/app/apikey:
    - Create a `.env` file in the project root directory and add the following line, replacing `YOUR_GEMINI_API_KEY` with your actual key:

      ```
      GEMINI_API_KEY=YOUR_GEMINI_API_KEY
      ```

4.  **Set the Environment Variable:**

    ```
    export KMP_DUPLICATE_LIB_OK=TRUE
    ```

    This variable allows the program to continue executing despite the OpenMP conflict,

5.  **Set Up Qdrant:**

    - Install Docker (if not already installed)
    - Run the following command to start a Qdrant instance:

    ```
    docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
    ```

6.  **Run the App:**

    ```
    streamlit run app.py
    ```

    This will launch the app in your web browser, usually at `http://localhost:8501`.

### **Features**

- **Text-based Search**: Users can search for products by entering a text query, and the app will retrieve relevant products based on their names and descriptions using a vector similarity search.

- **Image-based Search**: Users can search for products by uploading an image, and the app will retrieve the most similar products based on their image embeddings using a pre-trained image-to-image similarity model.

- **Add Products**: Users can add new products to the database by uploading an image, providing a name and description, and optionally generating a description using the Gemini Pro Vision model.

- **Display Products**: The app displays the retrieved products in a grid layout, showing the product image, name, and description.

- **Search Term and Similarity Chart**: For semantic searches, the app displays the search term and a bar chart showing the similarity scores of the top retrieved products.

### **Implementation Details**

The app uses the following key components and libraries:

- **Streamlit**: A Python library for building interactive web applications.
- **SQLite**: A lightweight, file-based database for storing product data.
- **LangChain**: A framework for building applications with large language models, used for text-based vector similarity search and generating product descriptions.
- **Google Generative AI API**: The Gemini Pro Vision model is used for generating product descriptions from images.
- **Sentence Transformers**: A library for generating high-quality sentence embeddings, used for semantic similarity search.

The application flow is as follows:

1. Users can search for products by entering a text query or uploading an image.

2. For text-based searches, the app generates text embeddings using the `GoogleGenerativeAIEmbeddings` from LangChain and performs a similarity search on the product data using a FAISS vector store.

3. For semantic searches, the app uses the `sentence-transformers` library to generate embeddings for the search term and product texts, computes the cosine similarities, and retrieves the top relevant products.

4. For image-based searches, the app generates image search tags using the Gemini Pro Vision model. It then performs a similarity search using the generated image captions and the FAISS vector store.

5. The retrieved products are displayed in a grid layout, showing the product image, name, and description.

6. For semantic searches, the app displays the search term and a bar chart showing the similarity scores of the top retrieved products.

7. Users can add new products by uploading an image, providing a name and description, and optionally generating a description using the Gemini Pro Vision model.

The app stores product data, including images, names, descriptions, and their respective embeddings, in an SQLite database for persistence.
