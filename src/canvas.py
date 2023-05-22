import tkinter as tk
from PIL import Image, ImageTk
from torch.utils.data import DataLoader 
from torchvision.datasets import ImageFolder
from torchvision import models
from EmbeddingSpace import *
from Networks import SiameseNetwork
import pyautogui
import torchvision.transforms as transforms

'''
SCRIPT DESCRIPTION:
This script displays a GUI where you can draw sketches and obtain the corresponding images.
'''

dataset_paths = {'mini' : ["../Mini Dataset/photo", "../Mini Dataset/sketch"], 
                 'full' : ["../Full Dataset/256x256/photo", "../Full Dataset/256x256/sketch"]}
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"We're using {DEVICE}")




# ====================================
#               CONFIG
# ====================================

#Pick a Dataset (you can use the dictionary up here as reference)
DATASET_NAME = 'full'
PHOTO_DATASET_PATH, SKETCHES_DATASET_PATH = dataset_paths[DATASET_NAME]

#Pick an embedding size
#   Must coincide with the model weights
OUTPUT_EMBEDDING = 2

#Choose a Weight Path
#   After the training your weight are going to be saved here
WEIGHT_PATH = f"../weights/{DATASET_NAME}-{OUTPUT_EMBEDDING}-contrastive-resnet50.pth"

#Pick a K (for the K-Precision)
#   It is used show k retrieved images
K = 12

#Pick a Batch Size
BATCH_SIZE = 16

#Pick a Backbone
#   The backbone represents the neural network within the siamese network, 
#   after which several linear layers will be applied to produce an embedding of size EMBEDDING_SIZE.
backbone = models.resnet18()
net = SiameseNetwork(output = OUTPUT_EMBEDDING, backbone = backbone).to(DEVICE)
net.load_state_dict(torch.load(WEIGHT_PATH))


#Load Dataset
workers = 0
images_ds = ImageFolder(PHOTO_DATASET_PATH, transform = transforms.ToTensor())
images_loader = DataLoader(images_ds, shuffle = False, num_workers = workers, pin_memory = True, batch_size = BATCH_SIZE)





# ====================================
#                CODE
# ====================================

# Constants
CANVAS_SIZE = 256  # Size of the square canvas
COLUMNS = 6

def clear_canvas():
    canvas.delete("all")

def get_canvas_image():
    # Get the coordinates of the canvas relative to the screen
    canvas_x = root.winfo_rootx() + canvas.winfo_x()
    canvas_y = root.winfo_rooty() + canvas.winfo_y()

    # Capture the screen image within the canvas region
    image = pyautogui.screenshot(region=(canvas_x, canvas_y, CANVAS_SIZE, CANVAS_SIZE))

    # Convert the image to PIL format
    image_pil = Image.frombytes('RGB', image.size, image.tobytes())

    # Apply transformation to convert the image to a tensor
    transform = transforms.ToTensor()
    tensor = transform(image_pil)

    return tensor

def search_images(event):

    # Get the sketch image from the canvas
    sketch = get_canvas_image()

    # Get the top K images that are most similar to the sketch
    topk_distances, topk_indices = embedding_space.top_k(sketch[None, :].to(DEVICE), K)

    # Clear the previous images
    image_frame.delete("all")

    # Display the top K images and their corresponding distances
    for i, (idx, d) in enumerate(zip(topk_indices, topk_distances)):

        # Resize the image and convert to PhotoImage format
        t = transforms.ToPILImage()
        image = t(images_ds[idx][0])
        resized_image = image.resize((256, 256))
        resized_image_tk = ImageTk.PhotoImage(resized_image)

        label = tk.Label(image_frame, image = resized_image_tk)
        label.image = resized_image_tk  # Keep a reference to avoid garbage collection
        label.grid(row=i // COLUMNS, column=i % COLUMNS, padx=10, pady=10)

        # Create label for the distance and add to the frame
        text_label = tk.Label(image_frame, text=str(f'{i+1}° - {d.item():.4}'))
        text_label.grid(row=i // COLUMNS, column=i % COLUMNS, padx=10, pady=10, sticky='n')


# Load the model and embedding space
net.load_state_dict(torch.load(WEIGHT_PATH))
embedding_space = EmbeddingSpace(net, images_loader, DEVICE)

# Create the main window
root = tk.Tk()
root.title("Image Search")
root.state('zoomed')

# Create the canvas for drawing
canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", highlightbackground="black", highlightthickness=2)
canvas.pack(padx=20, pady=20)

# Bind mouse events to canvas
canvas.bind("<B1-Motion>", lambda event: canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill="black"))
canvas.bind("<ButtonRelease-1>", search_images)

# Create a frame for the buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Create the buttons
clear_button = tk.Button(button_frame, text="Clear Canvas", command=clear_canvas)
clear_button.pack(side=tk.LEFT, padx=10)

# Create a frame for the image display
image_frame = tk.Canvas(root)
image_frame.pack(pady=20)

# Run the main event loop
root.mainloop()
