# VectorSearchShop

Vector Search Shop is a Streamlit application that enables users to search for products based on text and image queries. It leverages Qdrant, a vector similarity search engine, to store and retrieve product data and embeddings efficiently. The app also allows users to add new products, view product details, and explore similar products based on visual similarity.

![Screenshot 2024-04-10 at 4 18 03 PM](https://github.com/heymeowcat/VectorSearchShop/assets/40495273/2bf6a712-40e1-4b70-928b-6cbcf94579c9)


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

- **View Similar Products**: Users can upload an image to search for visually similar products using image embeddings.

- **Multimodal Search**: The app supports combined text and image queries, enabling users to search for products based on both textual and visual information.

- **View Similar Products**: Users can view similar products to a specific product based on image embeddings, allowing for visual similarity search.

- **Add Products**: Users can add new products to the database by providing a name, description, image URL, and price.

- **Display Products**: The app displays the retrieved products in a grid layout, showing the product image, name, description, price, and similarity score.

- **Add Products**: Users can view detailed information about a product, including the image, name, description, price, and similar products based on image embeddings.


### **Implementation Details**

The app uses the following key components and libraries:

- **Streamlit**: A Python library for building interactive web applications.
- **SQLite**: A lightweight, file-based database for storing product data.
- **Qdrant**: A vector similarity search engine used for storing and searching product data and embeddings.
- **Hugging Face Transformers**: The CLIPModel and CLIPProcessor from the Transformers library are used for generating text and image embeddings.

### **Architecture Diagram**

![Untitled diagram-2024-04-16-072131](https://github.com/heymeowcat/VectorSearchShop/assets/40495273/baa46e18-9c29-4356-a5df-0818abfec03b)

### Components

#### Frontend

-   **Streamlit**: A Python library for building interactive web applications. It provides the user interface for the Vector Search Shop application, allowing users to interact with the system, perform searches, view results, and add new products.

#### Backend

-   **App Server**: The server-side component of the application, responsible for handling user requests, interacting with the database and vector store, and generating embeddings using the CLIP model.
-   **SQLite Database**: A lightweight, file-based database used for storing product data, such as names, descriptions, image URLs, prices, and other relevant information.
-   **Qdrant Vector Store**: A vector similarity search engine used for storing and searching product data and embeddings. It enables efficient retrieval of relevant products based on the generated text and image embeddings.

#### Models

-   **CLIP Model**: The  `openai/clip-vit-large-patch14`  model from the Transformers library, a pre-trained CLIP (Contrastive Language-Image Pre-training) model used for generating text and image embeddings. This model is loaded from the Hugging Face Hub.

#### External

-   **Hugging Face Hub**: A platform for hosting and distributing pre-trained models, including the  `openai/clip-vit-large-patch14`  model used in this application.

### Data Flow
1.  Users interact with the Streamlit frontend, entering text queries, uploading images, or providing both text and image inputs.
2.  The Streamlit frontend sends requests to the App Server, including the user's search queries and product information (for adding new products).
3.  The App Server processes the requests and interacts with the SQLite Database and Qdrant Vector Store as needed:
    -   For search queries, the App Server generates text and image embeddings using the CLIP Model, performs similarity searches in the Qdrant Vector Store, and retrieves relevant product data from the SQLite Database.
    -   For adding new products, the App Server generates combined text and image embeddings using the CLIP Model, stores the embeddings in the Qdrant Vector Store, and persists the product data in the SQLite Database.
4.  The App Server sends the search results, product details, and similar products back to the Streamlit frontend for display.
5.  The Streamlit frontend renders the retrieved information, allowing users to view and interact with the search results, product details, and similar products.

### **App Flow Diagram**

![Untitled diagram-2024-04-16-072205](https://github.com/heymeowcat/VectorSearchShop/assets/40495273/35548422-91c8-40a6-8b1d-05d8d885aaec)


The application flow is as follows:
1.  Users can search for products by entering a text query, uploading an image, or providing both text and image input.
2.  The app generates text embeddings using the `openai/clip-vit-large-patch14` model from the Transformers library for the provided text input.

	-   The `openai/clip-vit-large-patch14` is a pre-trained CLIP (Contrastive Language-Image Pre-training) model from OpenAI, which utilizes the ViT (Vision Transformer) architecture with a large patch size of 14x14 pixels.
	-   This specific variant of the CLIP model has been pre-trained on a massive dataset of image-text pairs, allowing it to understand the relationship between visual and textual data effectively.
	-   The CLIPProcessor is used to preprocess the text input before passing it to the `openai/clip-vit-large-patch14` model for generating text embeddings.

3.  If an image is provided, the app generates image embeddings using the same `openai/clip-vit-large-patch14` model and CLIPProcessor.

	-   The `openai/clip-vit-large-patch14` model is capable of generating both text and image embeddings, enabling multimodal representation.
	-   The CLIPProcessor is used to preprocess the image input before passing it to the model for generating image embeddings.

4.  For combined text and image queries, the app combines the text and image embeddings using a weighted sum.
5.  The app performs a similarity search on the product data stored in Qdrant using the generated embeddings.
6.  The retrieved products are displayed in a grid layout, showing the product image, name, description, price, and similarity score.
7.  Users can view detailed information about a product, including the image, name, description, price, and similar products based on image embeddings.
8.  For displaying similar products, the app generates image embeddings for the selected product using the `openai/clip-vit-large-patch14` model and performs a similarity search on the image embeddings stored in Qdrant.
9.  Users can add new products by providing a name, description, image URL, and price. The app generates combined text and image embeddings for the new product using the `openai/clip-vit-large-patch14` model and stores them in Qdrant and the SQLite database.

The app stores product data, including images, names, descriptions, prices, and their respective text and image embeddings, in an SQLite database for persistence. The Qdrant vector store is used for efficient similarity search and retrieval of products based on combined text and image embeddings.
