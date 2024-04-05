import os
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import sqlite3
from io import BytesIO
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from typing import List, Optional
from sentence_transformers import SentenceTransformer

load_dotenv()
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model_vision = genai.GenerativeModel('gemini-pro-vision')
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()
    
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create a new Qdrant client instance
client = QdrantClient("localhost", port=6333)
collection_name = "products"

if not client.collection_exists(collection_name):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )   


model = SentenceTransformer("clip-ViT-L-14")

# Connect to SQLite database
conn = sqlite3.connect("products.db")
c = conn.cursor()

# Create products table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS products
             (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, image BLOB, image_captions TEXT, price REAL)''')

# Load products from the database
def load_products():
    c.execute("SELECT * FROM products")
    products = []
    for row in c.fetchall():
        product = {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "image": bytes_to_image(row[3]),
            "image_captions": row[4],
            "price": row[5],
        }
        products.append(product)
    return products

# Convert image bytes to PIL Image
def bytes_to_image(img_bytes):
    return Image.open(BytesIO(img_bytes))

# Convert PIL Image to bytes
def image_to_bytes(image):
    img_bytes = None
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
    return img_bytes

# Generate image captions using the Gemini Pro Vision model
def generate_image_captions(image):
    try:
        prompt = [
            "List up to 15 relevant search tags for the given image, separated by commas, including product category, brand, colors, patterns, materials, and distinctive features, considering multiple products if present.",
            image,
        ]
        response = model_vision.generate_content(prompt)
        image_captions = response.text
        return image_captions
    except Exception as e:
        st.error(f"Error generating image captions: {str(e)}")
        return None

# Add product to the database and update the vector store
def add_product(name, description, price, image):
    image_bytes = image_to_bytes(image)
    image_captions = generate_image_captions(image)
    c.execute("INSERT INTO products (name, description, image, image_captions, price) VALUES (?, ?, ?, ?, ?)", (name, description, image_bytes, image_captions, price))
    conn.commit()
    update_vector_store()

# Image Embedding
def calculate_embedding(img_bytes) -> Optional[List[float]]:
    try:
        embedding = model.encode(img_bytes).tolist()
        return embedding
    except Exception as e:
        st.error(f"Error calculating embedding for image: {e}")
        return None 

# Update the vector store with the new product data
def update_vector_store():
    products = load_products()

    if products:
        texts = []
        image_embeddings = []

        for product in products:
            name = product["name"]
            description = product["description"]
            image_captions = product["image_captions"]
            product_text = f"{name} {description} {image_captions}"
            product_text = product_text.replace("-", "").strip()
            texts.append(product_text)

            # Extract image embedding
            image = product["image"] 
            image_embedding = calculate_embedding(image)
            image_embeddings.append(image_embedding)

        # Embed all product texts
        results = [
            genai.embed_content(
                model="models/embedding-001",
                content=sentence,
                task_type="retrieval_document",
                title="Qdrant x Gemini",
            )
            for sentence in texts
        ]

        # Upsert all product texts and image embeddings into Qdrant
        client.upsert(
            collection_name="products",
            points=[
                PointStruct(
                    id=idx,
                    vector=list(result["embedding"]) + list(image_embedding),
                    payload={"text": text, "image_embedding": image_embedding},
                )
                for idx, (result, text, image_embedding) in enumerate(
                    zip(results, texts, image_embeddings)
                )
            ],
        )

# Search products using Qdrant
def search_products_qdrant(search_term, k=5):
    # Generate text embedding for the search term
    query_text_embedding = genai.embed_content(
        model="models/embedding-001",
        content=search_term,
        task_type="retrieval_query",
    )["embedding"]

    padded_query_vector = query_text_embedding + [0.0] * 768 
    search_result = client.search(
        collection_name='products',
        query_vector=padded_query_vector,
    )
    
    filtered_texts = [(result.payload["text"], result.score) for result in search_result]

    products = load_products()
    filtered_products = []

    # Retrieve the actual product objects using the filtered texts and scores
    for text, score in filtered_texts:
        for product in products:
            name = product["name"]
            description = product["description"]
            image_captions = product["image_captions"]
            product_text = f"{name} {description} {image_captions}"
            product_text = product_text.replace("-", "").strip()

            if product_text == text:
                filtered_products.append((product, score))
                break

    return filtered_products[:k]

# Display product card
def display_product_card(product, score, is_main=True):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(product["image"], use_column_width=True)
    with col2:
        st.subheader(product["name"])
        if product['price']:
            st.write(f"Price: ${product['price']:.2f}")
        description = product["description"]
        if len(description) > 100:
            description = f"{description[:100]}..."
        st.write(f"Description: {description}")
        st.progress(score)
        st.text(f"Score: {score:.2f}")

    key_prefix = "main_product" if is_main else "similar_product"
    view_details_button = st.button("View Details", key=f"{key_prefix}_{product['id']}")
    if view_details_button:
        display_product_details(product)

# Display product details modal/page
def display_product_details(product):
    st.subheader(f"Product Details: {product['name']}")
    st.image(product["image"], use_column_width=True)
    if product['price']:
        st.write(f"Price: ${product['price']:.2f}")
        st.write(f"Description: {product['description']}")

    # Find similar products
    query_image = product["image"]
    query_embedding =  [0.0] * 768 + model.encode(query_image).tolist()
    results = client.search(
        collection_name="products",
        query_vector=query_embedding,
        with_payload=True,
        limit=5,
    )

    # Display similar products
    st.subheader("Similar Products")
    similar_products = []
    for result in results:
        payload = result.payload
        for other_product in load_products():
            if payload["text"] == f"{other_product['name']} {other_product['description']} {other_product['image_captions']}".replace("-", "").strip():
                similar_products.append((other_product, result.score))
                break

    if similar_products:
        for similar_product, score in similar_products:
            if similar_product["id"] != product["id"]:
                display_product_card(similar_product, score, is_main=False)
    else:
        st.write("No similar products found.")

# Add product form
def add_product_form():
    with st.sidebar:
        st.subheader("Add Product")
        file = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"])
        name = st.text_input("Product Name")
        if "default" not in st.session_state:
            st.session_state["default"] = ""
        description = st.text_area(
            "Product Description", value=st.session_state["default"]
        )
        price = st.number_input("Price", min_value=0.0, step=0.01)
        if st.button("Generate Description"):
            if file:
                image = Image.open(file)
                try:
                    with st.spinner("Generating description..."):
                        prompt_parts_vision = [
                            "Generate a Item description for the Item for a oneline store in very few words in one line",
                            image,]
                        response = model_vision.generate_content(prompt_parts_vision)
                        generated_description = response.text
                        description = generated_description
                        st.session_state["default"] = description
                        st.rerun()
                except Exception as e:
                    st.error(f"Error generating description: {str(e)}")
            else:
                st.warning("Please upload an image to generate a description.")
        if st.button("Add Product"):
            if name and description and price and file:
                image = Image.open(file)
                add_product(name, description, price, image)
                st.success("Product added successfully!")
                st.rerun()
            else:
                st.error("Please fill in all fields and upload an image.")


def main():
    st.set_page_config(
        page_title="VectorSearchShop"
    )
    st.title("Product Shop")

    # Search Section
    search_term = st.text_input("Search Products")
    k_value = st.number_input("Number of results", min_value=1, value=5, step=1)
    search_button = st.button("Search")
    update_vector_store_button = st.button("Update Vector Store manually")

    if update_vector_store_button:
        update_vector_store()

    if search_button:
        search_results = search_products_qdrant(search_term, k=k_value)

        if search_results:
            for product, score in search_results:
                display_product_card(product, score, is_main=True)
        else:
            st.warning("No products found.")
    else:
        products = load_products()
        for product in products:
            display_product_card(product, 1.0, is_main=True)

    add_product_form()

if __name__ == "__main__":
    main()