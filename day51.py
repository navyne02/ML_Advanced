import streamlit as st
import time
import concurrent.futures

st.set_page_config(page_title="Day 51: Fast OCR Pipeline", layout="centered")

print("--- Step 1: Simulating Heavy Model Loading ---")
# 1. @st.cache_resource is used for global components like ML Models or Database connections.
# It loads the model ONLY ONCE, preventing memory leaks when the user interacts with the app.
@st.cache_resource
def load_ocr_model():
    # Simulating a heavy model load (e.g., Tesseract or a Deep Learning Vision Model)
    time.sleep(3) 
    return "Loaded_Vision_Model_v2.1"

print("--- Step 2: Optimizing Data Processing ---")
# 2. @st.cache_data is used for storing the results of expensive functions.
# If the exact same image is uploaded again, it instantly returns the cached result!
@st.cache_data
def process_certificate_image(image_name):
    # Simulating the time taken to extract text from a high-res certificate
    time.sleep(1.5) 
    return f"Extracted Text from {image_name}: [Name: Valid, ID: 98765, Status: Authentic]"

def main():
    st.title("⚡ Ultra-Fast Certificate OCR Pipeline")
    st.markdown("Optimized with Streamlit Caching and Parallel Processing.")

    # Load Model (Will only take 3 seconds the FIRST time the app runs)
    with st.spinner("Loading OCR Model to VRAM..."):
        model = load_ocr_model()
    st.success(f"Model Ready: {model}")

    st.divider()

    # Simulating a batch upload of 5 certificates
    uploaded_files = ["cert_batch_01.png", "cert_batch_02.png", "cert_batch_03.png", "cert_batch_04.png", "cert_batch_05.png"]

    if st.button("Process Batch Documents"):
        start_time = time.time()
        
        st.write("Processing documents in parallel...")
        results = []
        
        # 3. Parallel Processing: Instead of doing them 1 by 1 (which takes 7.5s),
        # we process them simultaneously using multiple CPU threads.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map the function to all uploaded files
            future_to_image = {executor.submit(process_certificate_image, img): img for img in uploaded_files}
            
            for future in concurrent.futures.as_completed(future_to_image):
                results.append(future.result())
                
        # Display Results
        for res in results:
            st.info(res)
            
        end_time = time.time()
        st.success(f"✅ Batch processed in {end_time - start_time:.2f} seconds!")
        st.caption("Try clicking the button again. Notice how it takes 0.00 seconds because of @st.cache_data!")

if __name__ == "__main__":
    main()