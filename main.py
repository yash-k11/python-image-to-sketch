import cv2
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\yashk\Downloads\WhatsApp Image 2024-12-30 at 13.53.11.jpeg")

if image is None:
    print("Error: Image not found or unable to load.")
    exit()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
inverted = 255 - gray_image

blur = cv2.GaussianBlur(inverted, (21, 21), 0)
invertedblur = 255 - blur

sketch = cv2.divide(gray_image, invertedblur, scale=256.0)

# Save the image
cv2.imwrite("sketch_image.png", sketch)
print("Sketch saved as sketch_image.png.")

# Display using matplotlib
plt.imshow(sketch, cmap="gray")
plt.title("Sketch")
plt.axis("off")
plt.show()
