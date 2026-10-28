This script addresses the task of predicting the viewing angle from images.

The core model used is EfficientNet-B0, a type of Convolutional Neural Network (CNN). This model is initialized with pre-trained weights from ImageNet, indicating a fine-tuning approach where a pre-trained model is adapted for a specific task.

The training process utilizes Distributed Data Parallel (DDP) for multi-GPU training. It employs a custom loss function, Mean Absolute Angular Error (MAAE), specifically designed for the circular nature of angles (0-360 degrees). The training includes standard practices like data augmentation, AdamW optimizer, and a StepLR scheduler. The script also includes functionality for generating a submission file in the specified format after training.
