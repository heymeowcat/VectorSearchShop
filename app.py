import os
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import sqlite3
from io import BytesIO
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from sentence_transformers import SentenceTransformer
import torch
import matplotlib.pyplot as plt

load_dotenv()
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model_vision = genai.GenerativeModel('gemini-pro-vision')
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = None

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
            "Generate up to 15 relevant search tags for the given image, separated by commas, including product category, brand, colors, patterns, materials, and distinctive features, considering multiple products if present.",
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

# Update the vector store with the new product data
def update_vector_store():
    global vector_store

    products = load_products()

    if products:
        product_data = []
        product_texts = []

        for product in products:
            name = product["name"]
            description = product["description"]
            image_captions = product["image_captions"]

            product_text = f"{name} {description} {image_captions}"
            product_text = product_text.replace("-", "").strip()

            product_texts.append(product_text)
            product_data.append({
                "id": product["id"],
                "name": name,
                "description": description,
                "image": image_to_bytes(product["image"]),
                "image_captions": image_captions,
                "price": product["price"],
                "product_text": product_text
            })

        vector_store = FAISS.from_texts(product_texts, embeddings, metadatas=product_data)
        vector_store.save_local("faiss_index")

# Search products using text and image embeddings
def search_products_faiss(search_term, k=5):
    global vector_store
    if vector_store is None:
        update_vector_store()

    search_results = vector_store.similarity_search(search_term, k=k)
    filtered_products = [result.metadata for result in search_results]

    return filtered_products

# Search products using semantic search
def search_products_semantic(search_term, k=5):
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Encode the search term
    search_term_embedding = model.encode([search_term])[0]

    products = load_products()
    product_texts = [f"{p['name']} {p['description']} {p['image_captions']}" for p in products]
    product_embeddings = model.encode(product_texts)

    # Compute cosine similarities
    similarities = torch.nn.functional.cosine_similarity(torch.tensor(product_embeddings), torch.tensor(search_term_embedding).unsqueeze(0))

    # Sort products by similarity and select top k
    sorted_indices = torch.argsort(similarities, descending=True)
    sorted_products = [products[idx] for idx in sorted_indices]
    sorted_similarities = similarities[sorted_indices]

    # Display search term and similarity chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(sorted_products)), sorted_similarities.tolist())
    ax.set_xticks(range(len(sorted_products)))
    ax.set_xticklabels([f"{p['name'][:20]}..." for p in sorted_products], rotation=45, ha='right')
    ax.set_xlabel("Products")
    ax.set_ylabel("Similarity")
    st.pyplot(fig)

    filtered_products = [products[idx] for idx in sorted_indices[:k]]

    return filtered_products


# Display product card
def display_product_card(product, is_main=True, is_grid=False):
    if is_grid:
        st.image(product["image"], use_column_width=True)
        st.subheader(product["name"])
        if product['price']:
            st.write(f"Price: ${product['price']:.2f}")
    else:
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

    # Similar Products
    st.subheader("Similar Products")
    search_term = f"{product['name']} {product['description']} {product['image_captions']}"
    similar_products = search_products_semantic(search_term, k=5)
    for similar_product in similar_products:
        if similar_product["id"] != product["id"]:
            display_product_card(similar_product, is_main=False)

# Generate search tag from image
def generate_search_tag(image):
    try:
        prompt = [
            "Generate relevant search tags for the given image, separated by commas, including product category, brand, model, specifications, colors, patterns, materials, and distinctive features, considering multiple products if present.",
            image,
        ]
        response = model_vision.generate_content(prompt)
        search_tag = response.text.strip()
        return search_tag
    except Exception as e:
        st.error(f"Error generating search tag: {str(e)}")
        return None

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
                            "Generate a Item description for the Item for a oneline store in very few words",
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

def display_products_grid(products):
    cols = st.columns(3)
    for i, product in enumerate(products):
        with cols[i % 3]:
            display_product_card(product, is_main=True, is_grid=True)

def main():
    st.title("Product Shop")

    # Search Section
    search_term = st.text_input("Search Products")
    search_image = st.file_uploader("Search by Image", type=["jpg", "png", "jpeg"])
    k_value = st.number_input("Number of results", min_value=1, value=5, step=1)
    search_button = st.button("Search")
    search_method = st.radio("Search Method", ["Semantic Search", "Search by Image", "FAISS Similarity Search"])
    view_mode = st.radio("View Mode", ["Grid", "List"], index=1)

    if search_button:
        if search_method == "FAISS Similarity Search":
            search_results = search_products_faiss(search_term, k=k_value)
        elif search_method == "Semantic Search":
            search_results = search_products_semantic(search_term, k=k_value)
        elif search_method == "Search by Image":
            if search_image:
                image = Image.open(search_image)
                search_tag = generate_search_tag(image)
                if search_tag:
                    search_results = search_products_semantic(search_tag, k=k_value)
                else:
                    search_results = []
            else:
                st.warning("Please upload an image to search.")
                search_results = []

        if search_results:
            if view_mode == "Grid":
                display_products_grid(search_results)
            else:
                for product in search_results:
                    display_product_card(product, is_main=True)
        else:
            st.warning("No products found.")
    else:
        products = load_products()
        if view_mode == "Grid":
            display_products_grid(products)
        else:
            for product in products:
                display_product_card(product, is_main=True)

    add_product_form()

if __name__ == "__main__":
    main()