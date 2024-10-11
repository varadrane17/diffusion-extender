from PIL import Image


def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path)
    width, height = image.size
    print(f"Original Image Size: {width}x{height}")

    # Get user input for the number of pixels to add
    left_add = int(input("Enter the number of pixels to add on the left: "))
    right_add = int(input("Enter the number of pixels to add on the right: "))
    top_add = int(input("Enter the number of pixels to add on the top: "))
    bottom_add = int(input("Enter the number of pixels to add on the bottom: "))

    # Calculate new dimensions
    new_width = width + left_add + right_add
    new_height = height + top_add + bottom_add

    # Create a new blank image (white background) for the preprocessed image
    preprocessed_image = Image.new('RGB', (new_width, new_height), color=(255, 255, 255))
    
    # Paste the original image in the center of the new image
    preprocessed_image.paste(image, (left_add, top_add))

    preprocessed_output_path = 'preprocessed_image.jpg'
    preprocessed_image.save(preprocessed_output_path)

    # Create the mask image (black for the original area, white for the added areas)
    mask_image = Image.new('L', (new_width, new_height), color=255)  # Start with white
    mask_image.paste(0, (left_add, top_add, left_add + width, top_add + height))  # Black in the original area

    # Save the mask image
    mask_output_path = 'mask_image.jpg'
    print(mask_image.size)
    mask_image.save(mask_output_path)   # (1780, 1054)

    print(f"Preprocessed image saved as: {preprocessed_output_path}")
    print(f"Mask image saved as: {mask_output_path}")

# Example usage
image_path = '/home/ubuntu/Varad/diffusion-extender/assets/beach.jpg'  
preprocess_image(image_path)