This script focuses on the task of predicting the latitude and longitude coordinates from images, framing it as a regression problem.

It employs the EfficientNet-B0 model, a Convolutional Neural Network (CNN), leveraging weights pre-trained on ImageNet. The model is then fine-tuned for the regression task of predicting latitude and longitude.

The training utilizes Distributed Data Parallel (DDP) for distributed training across multiple GPUs. A custom loss function, Average Lat/Lon MSE, is used, which calculates the mean squared error for latitude and longitude predictions separately. The script incorporates data normalization for latitude and longitude targets and applies inverse normalization during submission generation. Data cleaning steps, including filtering based on angle, IQR for lat/lon, and region-wise DBSCAN, are applied to the training data. The script also includes the process for generating a submission CSV file.
