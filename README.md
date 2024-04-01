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

- **Text-based Search**: Users can search for products by entering a text query, and the app will retrieve relevant products based on their names, descriptions, and image captions using a vector similarity search.

- **View Similar Products**: Users can view similar products to a specific product based on image embeddings, allowing for visual similarity search.

- **Add Products**: Users can add new products to the database by uploading an image, providing a name, description, and price, and optionally generating a description using the Gemini Pro Vision model.

- **Display Products**: The app displays the retrieved products in a grid or list layout, showing the product image, name, description, price, and similarity score.

- **View Product Details**: Users can view detailed information about a product, including the image, name, description, price, and similar products based on image embeddings.

### **Implementation Details**

The app uses the following key components and libraries:

- **Streamlit**: A Python library for building interactive web applications.
- **SQLite**: A lightweight, file-based database for storing product data.
- **Qdrant**: A vector similarity search engine used for storing and searching product data and embeddings.
- **Google Generative AI API**: The Gemini Pro Vision model is used for generating product descriptions from images.
- **Sentence Transformers**: A library for generating high-quality sentence embeddings, used for encoding product texts and images.

### **App Flow Diagram**

![vectorsearchshopmermaid](https://github.com/heymeowcat/VectorSearchShop/assets/40495273/78d022dc-f13a-42ec-ad55-df3f4a6e7767)


The application flow is as follows:

1. Users can search for products by entering a text query.

2. The app generates text embeddings using the `GoogleGenerativeAIEmbeddings` from the Google Generative AI API and performs a similarity search on the product data stored in Qdrant.

3. The retrieved products are displayed in a grid or list layout, showing the product image, name, description, price, and similarity score.

4. Users can view detailed information about a product, including the image, name, description, price, and similar products based on image embeddings.

5. For displaying similar products, the app generates image embeddings using the `clip-ViT-L-14` model from the `sentence-transformers` library and performs a similarity search on the image embeddings stored in Qdrant.

6. Users can add new products by uploading an image, providing a name, description, and price, and optionally generating a description using the Gemini Pro Vision model.

The app stores product data, including images, names, descriptions, prices, and their respective text and image embeddings, in an SQLite database for persistence. The Qdrant vector store is used for efficient similarity search and retrieval of products based on text and image embeddings.
