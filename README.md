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

**Project Architecture**
```mermaid
flowchart TD
    A[Environment Setup] --> B[Data Preparation]
    B --> C[Model Training]
    C --> D[Evaluation & Metrics Visualization]
    D --> E[Model Saving]
    E --> F[Inference & Prediction]
    F --> G[Model Validation]

    subgraph Environment_Setup
        A1["Check GPU with nvidia-smi"]
        A2["Install ultralytics library"]
        A3["Mount Google Drive"]
        A --> A1 --> A2 --> A3
    end

    subgraph Data_Preparation
        B1["Load & visualize sample images"]
        B2["Create dataset YAML file"]
        B --> B1 --> B2
    end

    subgraph Model_Training
        C1["Fine-tune YOLOv8n model"]
        C2["Set epochs, batch size, img size, etc."]
        C3["Monitor training logs and loss curves"]
        C --> C1 --> C2 --> C3
    end

    subgraph Evaluation_Visualization
        D1["Load training results CSV"]
        D2["Plot box loss, precision, recall, mAP"]
        D3["Validate on validation dataset"]
        D --> D1 --> D2 --> D3
    end

    subgraph Model_Saving
        E1["Copy best.pt weights to Drive"]
        E --> E1
    end

    subgraph Inference_Prediction
        F1["Load saved model weights"]
        F2["Predict on images"]
        F3["Display & save output images"]
        F --> F1 --> F2 --> F3
    end

    subgraph Model_Validation
        G1["Run YOLOv8 val() method"]
        G --> G1
    end




