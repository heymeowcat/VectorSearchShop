import streamlit as st
from PIL import Image
import sqlite3
import requests
from io import BytesIO
from helper import QdrantHelper

qdrant_helper = QdrantHelper()

# Connect to SQLite database
conn = sqlite3.connect("products.db")
c = conn.cursor()

# Create products table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS products
             (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, image_url TEXT, image_captions TEXT, price REAL)''')

c.execute("SELECT COUNT(*) FROM products")
count = c.fetchone()[0]

# Load products from the database
def load_products():
    c.execute("SELECT * FROM products ORDER BY id DESC")
    products = []
    for row in c.fetchall():
        product = {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "image_url": row[3],
            "image_captions": row[4],
            "price": row[5],
        }
        products.append(product)
    return products

# Download image from URL
def download_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_data = BytesIO(response.content)
        return Image.open(image_data)
    except (requests.exceptions.RequestException):
        return None

# Display product card
def display_product_card(product, score, is_main=True):
    with st.container(border=1):
        col1, col2 = st.columns([1, 2])
        with col1:
            image = download_image(product["image_url"])
            if image:
                st.image(image, use_column_width=True)
            else:
                st.warning("Failed to load image.")
        with col2:
            st.subheader(product["name"])
            if product['price']:
                st.write(f"Price: {product['price']}")
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
    image = download_image(product["image_url"])
    if image:
        st.image(image, use_column_width=True)
    else:
        st.warning("Failed to load image.")
    if product['price']:
        st.write(f"Price: {product['price']}")
        st.write(f"Description: {product['description']}")
    find_similar_products(product)

# Find similar products
def find_similar_products(product):
    query_image_url = product["image_url"]
    results = qdrant_helper.find_similar_items_by_image(query_image_url, 5)
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
    st.divider()

def main():
    st.set_page_config(
        page_title="VectorSearchShop",
        page_icon=":shopping_cart:",
        layout="wide"
    )

    st.title("Vector Search Shop")

    # Search Section
    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            search_term = st.text_input("Search Products")
        with col2:
            uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        search_button = st.button("Search")

    products = load_products()
    if len(products) == 0:
        st.warning("No products found.")

    if search_button:
        if uploaded_image:
            search_results = qdrant_helper.find_similar_items_by_image(uploaded_image, 5)
        else:
            search_results = qdrant_helper.find_similar_items_by_text(products, search_term, k=5)
        if search_results:
            with st.container():
                st.write("Search Results:")
                with st.container():
                    product_grid = st.columns(3)
                    for i, (product, score) in enumerate(search_results):
                        with product_grid[i % 3]:
                            display_product_card(product, score, is_main=True)
        else:
            st.warning("No products found.")
    else:
        st.write(f"Total Products Count: {count}")
        with st.container():
            st.write("All Products:")
            with st.container():
                product_grid = st.columns(3)
                for i, product in enumerate(products):
                    with product_grid[i % 3]:
                        display_product_card(product, 1.0, is_main=True)


if __name__ == "__main__":
    main()