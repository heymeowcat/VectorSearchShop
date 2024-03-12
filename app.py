from io import BytesIO
import streamlit as st
from PIL import Image
import base64

# Initialize the products list
products = []

def add_product():
    with st.sidebar:
        st.subheader("Add Product")
        name = st.text_input("Product Name")
        description = st.text_area("Product Description")
        file = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"])
        if st.button("Add Product"):
            if name and description and file:
                image = Image.open(file)
                image_bytes = image_to_bytes(image)
                products.append({"name": name, "description": description, "image": image_bytes})
                st.success("Product added successfully!")
            else:
                st.error("Please fill in all fields and upload an image.")

def image_to_bytes(image):
    img_bytes = None
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode()

def display_products():
    for product in products:
        st.subheader(product["name"])
        st.write(product["description"])
        st.image(bytes_to_image(product["image"]))
        st.write("---")

def bytes_to_image(img_bytes):
    img_bytes = base64.b64decode(img_bytes)
    img = Image.open(BytesIO(img_bytes))
    return img

def main():
    st.title("Product Shop")

    with st.sidebar:
        search_term = st.text_input("Search Products")
        add_product()

    filtered_products = [product for product in products if search_term.lower() in product["name"].lower() or search_term.lower() in product["description"].lower()]
    if filtered_products:
        display_products()
    else:
        st.warning("No products found.")

if __name__ == "__main__":
    main()