import streamlit as st
from PIL import Image
import sqlite3
from io import BytesIO
from handler import QdrantSearch

qdrant_search = QdrantSearch()

conn = sqlite3.connect("products.db")
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS products
             (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, image BLOB, image_captions TEXT, price REAL)''')

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

def bytes_to_image(img_bytes):
    return Image.open(BytesIO(img_bytes))

# Search products using Qdrant
def search_products_qdrant(search_term, k=5):
    if isinstance(search_term, str):
        search_results = qdrant_search.find_similar_text(search_term, num_results=k)
    else:
        search_results = qdrant_search.find_similar_image(search_term, num_results=k)

    products = load_products()
    filtered_products = []
    for result in search_results:
        payload = result.payload
        product_id = payload["id"]
        for product in products:
            if product["id"] == product_id:
                filtered_products.append((product, result.score))
                break

    return filtered_products

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

def display_product_details(product):
    st.subheader(f"Product Details: {product['name']}")
    st.image(product["image"], use_column_width=True)
    if product['price']:
        st.write(f"Price: ${product['price']:.2f}")
    st.write(f"Description: {product['description']}")

    query_image = product["image"]
    results = qdrant_search.find_similar_image(query_image, num_results=5)

    st.subheader("Similar Products")
    similar_products = []
    for result in results:
        payload = result.payload
        product_id = payload["id"]
        for other_product in load_products():
            if other_product["id"] == product_id:
                similar_products.append((other_product, result.score))
                break

    if similar_products:
        for similar_product, score in similar_products:
            if similar_product["id"] != product["id"]:
                display_product_card(similar_product, score, is_main=False)
    else:
        st.write("No similar products found.")

def main():
    st.set_page_config(
        page_title="VectorSearchShop"
    )
    st.title("Product Shop")

    search_term = st.text_input("Search Products")
    k_value = st.number_input("Number of results", min_value=1, value=5, step=1)
    search_button = st.button("Search")
    update_vector = st.button("Vector Store Update")

    if update_vector:
        products = load_products()
        qdrant_search.upsert_products(products)

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

if __name__ == "__main__":
    main()