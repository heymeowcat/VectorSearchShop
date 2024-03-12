import streamlit as st
from PIL import Image
import sqlite3
import base64
from io import BytesIO

# Connect to SQLite database
conn = sqlite3.connect("products.db")
c = conn.cursor()

# Create products table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS products
             (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, image BLOB)''')

# Load products from the database
def load_products():
    c.execute("SELECT * FROM products")
    products = []
    for row in c.fetchall():
        product = {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "image": bytes_to_image(row[3])
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

# Add product to the database
def add_product(name, description, image):
    image_bytes = image_to_bytes(image)
    c.execute("INSERT INTO products (name, description, image) VALUES (?, ?, ?)", (name, description, image_bytes))
    conn.commit()

# Display products
def display_products(products):
    cols = st.columns(3)
    for i, product in enumerate(products):
        with cols[i % 3]:
            st.image(product["image"], use_column_width=True)
            st.write(f"**{product['name']}**")
            st.write(product["description"])
    if not products:
        st.warning("No products found.")

# Add product form
def add_product_form():
    with st.sidebar:
        st.subheader("Add Product")
        name = st.text_input("Product Name")
        description = st.text_area("Product Description")
        file = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"])
        if st.button("Add Product"):
            if name and description and file:
                image = Image.open(file)
                add_product(name, description, image)
                st.success("Product added successfully!")
                st.experimental_rerun()
            else:
                st.error("Please fill in all fields and upload an image.")

def main():
    st.title("Product Shop")

    with st.sidebar:
        search_term = st.text_input("Search Products")

    products = load_products()
    filtered_products = [product for product in products if search_term.lower() in product["name"].lower() or search_term.lower() in product["description"].lower()]

    if filtered_products:
        display_products(filtered_products)
    else:
        st.warning("No products found.")

    add_product_form()

if __name__ == "__main__":
    main()