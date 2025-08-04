# Ayna-ML-Polygon-Coloring

## Project Overview

This project implements a Conditional UNet model in PyTorch to color polygons based on an input outline image and a specified color. The model learns to fill the polygon shape with the target color while preserving the shape's structure. This task is a form of image-to-image translation where the output is conditioned on an additional input (the desired color), making the Conditional UNet a suitable architecture.

## Hyperparameters

The following key hyperparameters were used during the final training phase:

*   **Number of Epochs:** 250. Initially, the model was trained for 75 epochs as per the task requirements. However, preliminary results showed limitations in color accuracy and edge sharpness. Based on these observations, the training was extended to 150 epochs, and then further to 250 epochs to allow for better convergence and refinement of the generated images.
*   **Batch Size:** 16. A batch size of 16 was used for both the training and validation DataLoaders. This size was chosen as a balance between utilizing GPU memory effectively and providing a reasonable number of samples for gradient updates during training.
*   **Learning Rate:** 0.0002. This learning rate was used with the Adam optimizer. It was selected as a common starting point for training deep learning models and proved effective in allowing the loss to decrease steadily without significant oscillations. No learning rate scheduling was implemented in this project.
*   **Optimizer:** Adam. The Adam optimizer was chosen for its adaptive learning rate properties, which generally lead to faster convergence compared to standard SGD. It is a widely used and robust optimizer for various deep learning tasks.
*   **Loss Function:** L1Loss (Mean Absolute Error). L1Loss was used as the objective function to minimize the absolute difference between the pixel values of the generated output images and the ground truth images. This loss function is known to be less sensitive to outliers compared to L2 (Mean Squared Error) and often encourages sharper results in image generation tasks.
*   **Color Vector Dimension:** 8. This dimension is determined by the total number of unique colors present across the entire dataset (training and validation combined). The dataset contained 8 distinct colors, leading to an 8-dimensional one-hot encoding for each color.

*Rationale for Hyperparameter Choices and Evolution:*
The initial hyperparameters (75 epochs, batch size 16, learning rate 0.0002, Adam, L1Loss) served as a foundational setup. The primary modification was the significant increase in the number of epochs. This decision was directly driven by the qualitative assessment of the generated images after initial training, which indicated that the model had learned the basic task but needed more iterations to refine the details and achieve better fidelity in color and edges. The chosen batch size and learning rate appeared to facilitate stable training, so they were not modified.

## Architecture

The model implemented for this task is a **Conditional UNet** architecture, built from scratch using PyTorch. The UNet architecture is particularly well-suited for image-to-image translation tasks due to its symmetric encoder-decoder structure and the incorporation of skip connections. The conditional aspect is introduced to guide the image generation based on the desired color.

*   **Encoder:** The encoder pathway consists of a series of convolutional blocks and max-pooling layers. Each convolutional block comprises two sequential layers, each with a 3x3 convolutional layer, Batch Normalization, and a ReLU activation function. The number of feature maps progressively increases with each downsampling step: starting with 3 input channels, the encoder features are 64, 128, 256, and 512. Max pooling with a kernel size of 2 and stride of 2 is applied after each convolutional block to reduce the spatial dimensions.
*   **Bottleneck:** The bottleneck is the central part of the UNet, connecting the encoder and decoder. It consists of a convolutional block similar to those in the encoder, transforming the 512 feature maps from the last encoder layer into 1024 feature maps. This layer captures the most compressed and abstract representation of the input image.
*   **Color Conditioning:** The color conditioning is a key element of this Conditional UNet. A linear layer `self.color_linear` is used to project the 8-dimensional one-hot color vector into a higher-dimensional embedding that matches the number of feature maps in the bottleneck (1024). This color embedding is then reshaped to have spatial dimensions of 1x1 and added element-wise to the bottleneck feature map. This injection of color information at the bottleneck allows the model to modulate the global features based on the target color before upsampling begins in the decoder. This is a simple yet effective method for conditioning in image generation. No other methods like concatenating the color vector spatially or using Conditional Batch Normalization were explored.
*   **Decoder:** The decoder pathway mirrors the encoder, using transposed convolutional layers (`ConvTranspose2d`) to upsample the feature maps and recover the spatial resolution. Each transposed convolutional layer is followed by a convolutional block. The number of feature maps decreases with each upsampling step, corresponding to the encoder features in reverse order (512, 256, 128, 64).
*   **Skip Connections:** Skip connections are implemented by concatenating the feature maps from the end of each encoder convolutional block with the output of the corresponding transposed convolutional layer in the decoder. These connections provide the decoder with access to the higher-resolution features from the encoder, which is crucial for reconstructing fine details and sharp edges in the generated image. The concatenation happens along the channel dimension. A cropping mechanism was included to handle potential minor discrepancies in spatial dimensions between the transposed convolution output and the encoder skip connection due to padding or stride effects, although with the chosen kernel sizes and strides, this was primarily a safeguard.
*   **Output Layer:** The final layer is a 1x1 convolutional layer that maps the 64 feature maps from the last decoder block to 3 output channels, representing the RGB color channels of the generated image. A Tanh activation function is implicitly applied by the denormalization step in the inference, as the model output is in the range [-1, 1] due to the normalization applied to the input and ground truth images.

## Training Dynamics

The training process involved iterating over the training dataset for 250 epochs, minimizing the L1Loss between the model's output and the ground truth images. The model's performance was monitored using the validation dataset and tracked using Weights & Biases (wandb).

*   **Loss/Metric Curves:** The training loss showed a rapid decrease in the initial epochs and continued to decrease steadily throughout the 250 epochs, indicating that the model was consistently learning. The validation loss followed a similar trend, decreasing alongside the training loss and remaining close to it, which suggests that the model was generalizing well to unseen validation data and was not significantly overfitting. The continuous decrease in validation loss even in later epochs supported the decision to extend training beyond the initial 75 epochs.
*   **Qualitative Output Trends:** Visual inspection of the generated images logged to wandb each epoch provided crucial insights into the training progress:
    *   **Epochs 1-50:** Generated images were very noisy and did not resemble the target colored polygons. Colors were random and inconsistent.
    *   **Epochs 50-150:** The model started to grasp the task. The general shape of the polygon appeared, and the generated color was closer to the target color, but often with significant variations and inaccuracies (e.g., greenish output for a blue target). Edges were blurry and had noticeable artifacts, particularly reddish or orange halos.
    *   **Epochs 150-250:** A significant improvement was observed. Color accuracy became much higher, with generated colors closely matching the ground truth. Edge sharpness improved dramatically, and the artifacts around the polygons were substantially reduced, resulting in cleaner and more visually appealing outputs.
*   **Typical Failure Modes and Fixes Attempted:**
    *   **Shape Mismatch Error (`RuntimeError: mat1 and mat2 shapes cannot be multiplied`):** This was the most significant training failure mode. The traceback pointed to the linear layer for color conditioning. Debugging with print statements revealed that the color vector input to this layer had an incorrect shape (`[batch_size, 4]` instead of `[batch_size, 8]`) specifically for the last batch in the validation loop. This was traced back to the `PolygonDataset`'s `_create_onehot_encoding` method, which was generating color mappings based only on the colors present in the individual dataset splits. The fix involved loading all unique colors from both training and validation data *before* creating the dataset instances and passing this complete list to the `PolygonDataset` constructor. This ensured a consistent 8-dimensional one-hot encoding for all samples, resolving the error.
    *   **Poor Output Quality (Initial Epochs):** The blurry edges and inaccurate colors after 75 epochs were considered a failure mode in terms of achieving high-quality results. The fix attempted and successfully implemented was to simply train the model for a significantly longer duration (250 epochs). This allowed the model more time to optimize the weights and learn the intricate details required for accurate color filling and sharp edges.

## Key Learnings

*   **Data Consistency is Non-Negotiable:** The debugging process for the shape mismatch error underscored the absolute necessity of ensuring consistent data preprocessing and representation across all dataset splits. Even subtle inconsistencies can lead to fundamental errors during training. Thorough data analysis and validation before model training are crucial.
*   **Conditional UNet's Adaptability:** The project reaffirmed the power and adaptability of the Conditional UNet for tasks requiring conditioned image generation. The ability to inject external information (color) into the network effectively guided the generation process.
*   **Training Duration Matters for Refinement:** Achieving high-quality generative results often requires substantial training. While a model might learn the basic task relatively quickly, refining the output to achieve accuracy in details like color fidelity and edge sharpness can necessitate training for many more epochs. Monitoring validation metrics and qualitative outputs is key to determining sufficient training duration.
*   **Debugging with Intermediate Outputs:** Using print statements or debugger tools to inspect the shapes and values of tensors at different stages of the data loading, batching, and model forward pass was essential for pinpointing the source of the shape mismatch error. This iterative debugging process is a critical skill in deep learning development.
*   **Wandb for Experiment Tracking and Visualization:** Weights & Biases proved to be an invaluable tool for this project. Tracking loss curves provided quantitative evidence of learning, while logging sample images offered critical qualitative feedback, allowing for visual assessment of the model's progress and the impact of changes like extended training.

<img width="1479" height="511" alt="download (7)" src="https://github.com/user-attachments/assets/22e56cc9-9489-4ba7-89c6-1d02674ab05d" />


<img width="1479" height="511" alt="download (8)" src="https://github.com/user-attachments/assets/3c40d331-441f-4afb-9b57-39ca1909f645" />


<img width="1479" height="511" alt="download (9)" src="https://github.com/user-attachments/assets/f00b7a78-d3f7-46ee-b4ea-9c956b81667c" />



This project provided hands-on experience with implementing and training a Conditional UNet, highlighting common challenges like data consistency and the importance of sufficient training for achieving high-quality generative outputs in image synthesis tasks. The iterative process of identifying failure modes, debugging, and refining the training strategy was a key learning experience.
