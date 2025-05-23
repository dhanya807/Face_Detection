# Face_Detection
graph TD
    subgraph "1. Dataset Preparation"
        A[Collect face images] --> B[Annotate images (bounding boxes)]
    end

    subgraph "2. Data Organization"
        C[Split into 'train' and 'val' folders] --> D[Structure images/]
        D --> E[Structure labels/]
        D -- images/train/ --> F[images/val/]
        E -- labels/train/ --> G[labels/val/]
    end

    subgraph "3. YAML Configuration"
        H[Create data.yaml] --> I[Paths to train/val images]
        H --> J[Class names (e.g., ["face"])]
        H --> K[Number of classes (nc)]
    end

    subgraph "4. Model Training"
        L[Use YOLOv8n] --> M[Set hyperparameters]
        M --> N[epochs, batch, imgsz, etc.]
        L --> O[Train using CLI]
        O -- "!yolo task=detect mode=train ..." --> P[Training in progress]
    end

    subgraph "5. Model Evaluation"
        Q[Analyze metrics] --> R[mAP, precision, recall]
        Q --> S[Use training logs and charts]
    end

    subgraph "6. Save Trained Model"
        T[Automatically saved in runs/detect/<exp_name>/weights/] --> U[Use best.pt or last.pt]
    end

    subgraph "7. Inference on New Data"
        V[Load model with YOLO("best.pt")] --> W[Run predictions on unseen images]
    end

    subgraph "8. Results Visualization"
        X[Save annotated images] --> Y[Display or store results]
    end

    B --> C
    K --> L
    S --> T
    P --> Q
    U --> V
    W --> X
