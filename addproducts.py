import os
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import sqlite3
from io import BytesIO
import google.generativeai as genai

from handler import QdrantSearch

load_dotenv()
qdrant_search = QdrantSearch()


try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model_vision = genai.GenerativeModel('gemini-pro-vision')
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()
    

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

def image_to_bytes(image):
    img_bytes = None
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
    return img_bytes

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


def add_product(name, description, price, image):
    image_bytes = image_to_bytes(image)
    image_captions = generate_image_captions(image)
    c.execute("INSERT INTO products (name, description, image, image_captions, price) VALUES (?, ?, ?, ?, ?)", (name, description, image_bytes, image_captions, price))
    conn.commit()
    products = load_products()
    qdrant_search.upsert_products(products)


def add_product_form():
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
    st.title("Add Products to Shop")
    add_product_form()

if __name__ == "__main__":
    main()