def read_image(file_path):
    with open(file_path, "rb") as file:
        return file.read()

def save_modified_image(file_path, modified_bytes):
    with open(file_path, "wb") as file:
        file.write(modified_bytes)

def main():
    try:
        input_image_path = "data/raw/potw2049a.jpg"
        output_image_path = "data/raw/potw2049a_modified.jpg"
        
        # Read the image into bytes
        original_image_bytes = read_image(input_image_path)
        
        # Modify bytes (demo only)
        modified_image_bytes = original_image_bytes + b"Watermark"
        
        # Save the modified image
        save_modified_image(output_image_path, modified_image_bytes)
        
        print("Image read and modified successfully.")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()