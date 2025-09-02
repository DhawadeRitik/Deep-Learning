from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image 
import streamlit as st 
import numpy as np
import os


# ------------------------------------ Streamlit title and page configuration -----------------------------------------

# App title
st.set_page_config(page_title="Image Augmentation Playground", page_icon="🎨")
st.title("🖼️ Image Augmentation Playground")
st.markdown("Upload an image and apply different augmentation techniques using **Keras ImageDataGenerator**.")


# ----------------------------------------- Image path ---------------------------------------------------------------
# File uploader
image_uploader = st.file_uploader("📂 Upload your image (e.g., cat, dog, etc.)", type=["jpg", "jpeg", "png"])

if image_uploader:
    # Load and resize image
    img = image.load_img(image_uploader, target_size=(200, 200))

    # Layout: Original on left, Augmented on right
    col1, col2 = st.columns(2)
    
# ------------------------------------- Show Original Image ---------------------------------------------------------
    with col1:
        st.subheader("🖼️ Original Image")
        st.image(img, caption="🖼️ Original Image", use_container_width=True)
        
# ------------------------------------- Side bar setting Parameters for ImageDataGenarator --------------------------------
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Augmentation Settings")
        width_range = st.number_input("➡️ Width Shift Range (e.g., 0.2 = 20%)", value=0.2, step=0.1, 
                                      help="Move the image left or right by a fraction of total width.")
        height_range = st.number_input("⬆️ Height Shift Range (e.g., 0.2 = 20%)", value=0.2, step=0.1, 
                                       help="Move the image up or down by a fraction of total height.")
        imag_rotation = st.number_input("🔄 Rotation Range (degrees)", value=40, step=5, 
                                        help="Rotate the image within the given degree range.")
        zoom = st.number_input("🔍 Zoom Range (e.g., 0.3 = 70%–130%)", value=0.3, step=0.1, 
                               help="Random zoom in/out.")
        horizontal = st.radio("↔️ Enable Horizontal Flip?", ["No", "Yes"])
        vertical = st.radio("↕️ Enable Vertical Flip?", ["No", "Yes"])
        augmented_image_volume = st.number_input("🖼️ Number of Augmented Images", value=10, step=1, min_value=1)

    # Convert flip options to boolean
    horizontal_flip = True if horizontal == "Yes" else False
    vertical_flip = True if vertical == "Yes" else False
    
# ---------------------------------  Generate the Augmented Imge ------------------------------------------------------
    # Create ImageDataGenerator
    datagen = ImageDataGenerator(
        rescale=1./255.0,
        fill_mode='reflect',
        rotation_range=imag_rotation,
        zoom_range=zoom,
        width_shift_range=width_range,
        height_shift_range=height_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip
    )

    # Convert image to array and create batch
    img_array = image.img_to_array(img)
    input_batch = np.expand_dims(img_array, axis=0)
    
# ------------------------- Show the augmented image ----------------------------------------------------------------
    # Show augmented images
    with col2:
        st.subheader("✨ Augmented Images")
        cols = st.columns(3) 
        i = 0
        for output in datagen.flow(input_batch, batch_size=1):
            col = cols[i % 3] # 5 image in each row 
            with col :
                st.image(output[0], use_container_width=True)
            i += 1
            if i == augmented_image_volume:
                break
            if i % 3 == 0: # after every 5 image create new row
                col = st.columns(3)
                
# ------------------------------ save the augmented image to specifird folder -----------------------------------------

        save_folder_path = st.text_input("Enter a folder path to save image", )    
        if st.button("save augmented image"):
            #choose folder
            if not save_folder_path:
                st.error("Please enter a valid folder path to save images")
            else:
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)

                    i = 0
                    for output in datagen.flow(input_batch, batch_size=1,
                                   save_to_dir=save_folder_path,
                                   save_prefix="aug",
                                   save_format="jpeg"):
                        i += 1
                        if i == augmented_image_volume:
                            break

                st.success(f"✅ {augmented_image_volume} images saved to `{save_folder_path}/`")
                
    
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #333;
        text-align: center;
        padding: 8px;
        font-size: 15px;
        border-top: 1px solid #d3d3d3;
    }
    </style>
    <div class="footer">
        💡 Note: Use the sidebar controls to change parameters and generate different augmented images.
    </div>
    """,
    unsafe_allow_html=True
)
