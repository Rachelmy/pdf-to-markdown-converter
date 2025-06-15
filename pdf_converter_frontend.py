import streamlit as st
import requests
import io
import tempfile
import os
from typing import Optional, Dict
import base64
from PIL import Image
import shutil

# Configure Streamlit page
st.set_page_config(
    page_title="PDF to Markdown Converter", page_icon="📄", layout="wide"
)

# Configuration
MARKER_API_URL = "http://localhost:8001"  # Adjust this to your FastAPI server URL


def upload_pdf_to_server(pdf_file) -> Optional[str]:
    """Upload PDF to the Marker FastAPI server and get conversion result."""
    try:
        # Prepare the file for upload
        files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}

        # Make request to the FastAPI server
        response = requests.post(f"{MARKER_API_URL}/marker/upload", files=files)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Server error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the Marker server. Please ensure the FastAPI server is running on localhost:8000"
        )
        return None
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None


def clear_session_state():
    """Clear all conversion-related session state."""
    keys_to_clear = ["markdown_result", "pdf_file", "conversion_done"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def initialize_state():
    """Initialize the session state variables for the PDF converter."""
    if "markdown_result" not in st.session_state:
        st.session_state["markdown_result"] = ""
    if "conversion_done" not in st.session_state:
        st.session_state["conversion_done"] = False
    if "pdf_file" not in st.session_state:
        st.session_state["pdf_file"] = None


def decode_base64_images(encoded_images: Dict[str, str]) -> Dict[str, Image.Image]:
    """Decode base64-encoded images back to PIL Image objects.
    
    Args:
        encoded_images: Dictionary mapping image names to base64-encoded strings
        
    Returns:
        Dictionary mapping image names to PIL Image objects
    """
    decoded_images = {}
    for name, encoded in encoded_images.items():
        try:
            # Decode base64 string to bytes
            image_bytes = base64.b64decode(encoded)
            # Create image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            decoded_images[name] = image
        except Exception as e:
            print(f"Error decoding image {name}: {str(e)}")
    return decoded_images


def prepare_markdown_with_images(markdown_text: str, decoded_images: Dict[str, str], temp_dir: str):
    """Prepare markdown text with local image paths for preview.
    
    Args:
        markdown_text: Original markdown text
        decoded_images: Dictionary of base64-encoded images
        temp_dir: Directory to store images
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), temp_dir), exist_ok=True)
    
    # Save images to directory
    for image_name, image in decoded_images.items():
        # Save image to directory
        image_path = os.path.join(os.getcwd(), temp_dir, f"{image_name}")
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(image_path, "JPEG", quality=95)
    
    # Save markdown file
    with open(os.path.join(os.getcwd(), temp_dir, "output.md"), "w", encoding="utf-8") as f:
        f.write(markdown_text)


def main():

    initialize_state()

    st.title("📄 PDF to Markdown Converter")
    st.markdown(
        "Convert your PDF documents to Markdown format using the Marker library"
    )
    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload PDF")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF file to convert to Markdown",
        )

        # Handle file upload
        if uploaded_file is not None:
            # Check if this is a new file
            if st.session_state.pdf_file != uploaded_file.name:
                # Clear previous results when new file is uploaded
                clear_session_state()
                initialize_state()
                st.session_state.pdf_file = uploaded_file.name

            # Display file info
            st.success(f"📄 File uploaded: {uploaded_file.name}")
            st.info(f"File size: {uploaded_file.size / 1024:.1f} KB")

            # Convert button
            if st.button("🔄 Convert to Markdown", type="primary"):
                with st.spinner("Converting PDF to Markdown... This may take a while."):
                    result = upload_pdf_to_server(uploaded_file)
                    print("Conversion result:", result)
                    markdown_result = result.get("output", "")
                    image_encoded = result.get("images", {})
                    
                    # Decode images
                    decoded_images = decode_base64_images(image_encoded)
                    
                    prepare_markdown_with_images(markdown_result, decoded_images, 'output')               
                    # Store decoded images in session state for later use
                    st.session_state.decoded_images = decoded_images

                    if markdown_result:
                        st.session_state.markdown_result = markdown_result
                        st.session_state.conversion_done = True
                        st.success("✅ Conversion completed successfully!")
                    else:
                        st.error("❌ Conversion failed. Please try again.")

        else:
            # Clear results when no file is uploaded
            if st.session_state.pdf_file is not None:
                clear_session_state()
                st.rerun()

    with col2:
        st.header("📋 Conversion Results")
        if st.session_state["conversion_done"] and st.session_state["markdown_result"]:
            # Preview section
            st.subheader("👀 Markdown Preview")

            # Create tabs for different views
            tab1, tab2 = st.tabs(["Rendered", "Raw Markdown"])

            with tab1:
                # Show rendered markdown
                with st.container():
                    if st.session_state.markdown_result:
                        st.markdown(st.session_state.markdown_result)
            with tab2:
                # Show raw markdown
                st.code(st.session_state.markdown_result[:5000], language="markdown")
                if len(st.session_state.markdown_result) > 5000:
                    st.info(
                        "Preview truncated. Download the full file to see complete content."
                    )

            # Download section
            st.subheader("💾 Download")

            # Prepare filename
            if st.session_state.pdf_file:
                base_name = os.path.splitext(st.session_state.pdf_file)[0]
                download_filename = f"{base_name}.md"
            else:
                download_filename = "converted_document.md"

            # Download button
            st.download_button(
                label="📥 Download Markdown",
                data=st.session_state.markdown_result,
                file_name=download_filename,
                mime="text/markdown",
                help="Download the converted markdown file",
            )

            # Statistics
            st.subheader("📊 Statistics")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Characters", len(st.session_state.markdown_result))
            with col_b:
                st.metric("Words", len(st.session_state.markdown_result.split()))
            with col_c:
                st.metric("Lines", st.session_state.markdown_result.count("\n") + 1)

        else:
            st.info(
                "Upload a PDF file and click 'Convert to Markdown' to see results here."
            )


if __name__ == "__main__":
    main()
