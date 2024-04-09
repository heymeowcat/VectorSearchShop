import csv
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv
import requests
import streamlit as st
import sqlite3
import google.generativeai as genai
import ast
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

from helper import QdrantHelper

load_dotenv()

qdrant_helper = QdrantHelper()

try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model_vision = genai.GenerativeModel('gemini-pro-vision')
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()
    
conn = sqlite3.connect("products.db")
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS products
             (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, image_url TEXT, image_captions TEXT, price TEXT)''')

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 32
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_path):
    images = [image_path.convert('RGB')]
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    pred = preds[0].strip()

    return pred

def load_products():
    c.execute("SELECT * FROM products")
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

def generate_image_captions(image, useVisionEncoderState):
    try:
        image_captions = None
        if useVisionEncoderState:
            response = requests.get(image)
            response.raise_for_status()
            query_image = Image.open(BytesIO(response.content))
            image_captions = predict_step(query_image)
        else:
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

def add_product(name, description, price, image_url, image_captions):
    c.execute("INSERT INTO products (name, description, image_url, image_captions, price) VALUES (?, ?, ?, ?, ?)", (name, description, image_url, image_captions, price))
    conn.commit()
    product_id = c.lastrowid
    product = {
        "id": product_id,
        "name": name,
        "description": description,
        "image_url": image_url,
        "image_captions": image_captions,
        "price": price,
    }
    qdrant_helper.add_product_to_vector_store(product)

def add_product_form():
    st.subheader("Add Product")
    file = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"])
    name = st.text_input("Product Name")
    if "default" not in st.session_state:
        st.session_state["default"] = ""
    description = st.text_area("Product Description", value=st.session_state["default"])
    price = st.text_input("Price")

    if st.button("Generate Description"):
        if file:
            image = Image.open(file)
            try:
                with st.spinner("Generating description..."):
                    prompt_parts_vision = [
                        "Generate a Item description for the Item for a oneline store in very few words in one line",
                        image,
                    ]
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
            image_captions = generate_image_captions(image, False)
            if image_captions:
                image_path = os.path.join("products_images", f"{name}.png")
                image.save(image_path)
                add_product(name, description, price, image_path, image_captions)
                st.success("Product added successfully!")
                st.rerun()
            else:
                st.error("Error generating image captions.")
        else:
            st.error("Please fill in all fields and upload an image.")

def load_products_from_csv(uploaded_file):
    products = []
    csv_data = uploaded_file.getvalue().decode("utf-8")
    reader = csv.DictReader(csv_data.splitlines())
    for row in reader:
        image_urls = ast.literal_eval(row["image"])
        
        valid_image_url = None
        image_captions = None
        for image_url in image_urls:
            try:
                response = requests.get(image_url)
                if response.status_code == 200:
                    valid_image_url = image_url
                    image_captions = generate_image_captions(image_url, True)
                    break
            except requests.exceptions.RequestException:
                continue 

        if valid_image_url:
            product = {
                "name": row["product_name"],
                "description": row["description"],
                "image_url": valid_image_url,
                "price": row["retail_price"]
            }
            add_product(
                product["name"],
                product["description"],
                product["price"],
                product["image_url"],
                image_captions
            )
            products.append(product)

    return products

def main():
    st.set_page_config(
        page_title="Add Products VectorSearchShop"
    )
    st.title("Add Products to Shop")

    option = st.radio("Select an option", ("Add Product Manually", "Batch Add Products from CSV"))

    if option == "Add Product Manually":
        add_product_form()
    else:
        st.subheader("Batch Add Products from CSV")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            try:
                products = load_products_from_csv(uploaded_file)
                st.success(f"{len(products)} products added successfully!")
            except Exception as e:
                st.error(f"Error adding products from CSV: {str(e)}")

if __name__ == "__main__":
    main()