# Face Detection with YOLOv8 (Fine-Tuned)

This project leverages the YOLOv8 object detection architecture, fine-tuned specifically for face detection tasks.  
Trained on a publicly available image dataset, the model can accurately **detect and draw bounding boxes** around faces in both single and multiple images.
Developed using Python and the Ultralytics YOLOv8 framework, this solution provides a fast, efficient, and lightweight approach to face detection in images.

## Project Overview

This project focuses on face detection using a fine-tuned YOLOv8 model. It takes images as input and accurately identifies faces by drawing bounding boxes around them. The model is trained on a publicly available dataset, making it reliable and efficient for detecting faces in both individual images and batches of images.

**Key features include:**

- Training (Fine-tuning) a state-of-the-art object detection model (YOLOv8) for face detection  
- Support for single or multiple image inputs  
- Fast and accurate bounding box predictions around faces  
- Easy integration with Python scripts for inference  

The project is designed for image analysis tasks that require robust face localization without real-time video processing.


## Project Architecture

```mermaid
flowchart TD
    A[Environment Setup] --> B[Data Preparation]
    B --> C[Model Training]
    C --> D[Evaluation & Metrics Visualization]
    D --> E[Model Saving]
    E --> F[Inference & Prediction]
    F --> G[Model Validation]



---

### ✅ Step 2: **Textual Description Below It**

Put your textual explanation **after** the Mermaid block, like this:

```markdown
The Face Detection project follows a structured pipeline leveraging the YOLOv8 model and Google Colab GPU for efficient training and inference:

1. **Environment Setup**  
   - Check GPU availability using `nvidia-smi`  
   - Install required libraries like `ultralytics` for YOLOv8  
   - Mount Google Drive to access dataset and save results  

2. **Data Preparation**  
   - Load and visualize sample training images using OpenCV  
   - Prepare the dataset YAML file specifying training and validation image paths, number of classes, and class names  

3. **Model Training**  
   - Fine-tune the pre-trained YOLOv8n model with the prepared dataset  
   - Configure training parameters such as epochs, batch size, image size, caching, and worker threads  
   - Monitor training progress through logs and loss curves  

4. **Evaluation and Metrics Visualization**  
   - Load training results from CSV files  
   - Plot key metrics: box loss, precision, recall, and mean Average Precision (mAP) across epochs  
   - Validate model performance on the validation dataset  

5. **Model Saving**  
   - Copy the best trained weights (`best.pt`) from the local run directory to Google Drive for later use

6. **Inference and Prediction**  
   - Load the saved model weights for inference  
   - Predict on single or multiple validation images, drawing bounding boxes around detected faces  
   - Display predictions using Matplotlib and OpenCV  
   - Save predicted output images back to Google Drive  

7. **Model Validation**  
   - Run a comprehensive validation using the YOLOv8 `val()` method to report final metrics on the dataset
