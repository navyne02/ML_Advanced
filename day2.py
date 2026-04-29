from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

print("Loading Deep Learning Models... (First time konjam neram aagum ⏳)")

# 1. Define the images we want to compare
image1_path = "my_photo.jpg"
image2_path = "test_photo.jpg"

try:
    # 2. The AI Magic! Comparing two faces
    # DeepFace will automatically detect the face, extract features, and compare them.
    result = DeepFace.verify(img1_path=image1_path, img2_path=image2_path)

    # 3. Reading the Output
    print("\n--- 🤖 AI Face Verification Result ---")
    
    # Check the 'verified' boolean in the result dictionary
    if result['verified'] == True:
        print("✅ MATCH FOUND! Ithu orey aal thaan. Access Granted!")
    else:
        print("❌ MISMATCH! Ithu vera vera aatkal. Access Denied!")
        
    print(f"Facial Distance Score: {result['distance']:.4f}")
    print("--------------------------------------")

    # 4. Show the images side by side for us to see
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Convert BGR to RGB for matplotlib
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img1)
    ax[0].set_title("Image 1 (Database)")
    ax[0].axis('off')
    
    ax[1].imshow(img2)
    ax[1].set_title(f"Image 2 (Test) - Match: {result['verified']}")
    ax[1].axis('off')
    
    plt.show()

except Exception as e:
    print(f"Error: Make sure both photos have clear faces in them! Details: {e}")