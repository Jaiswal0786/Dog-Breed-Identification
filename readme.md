# Introduction
This code is a Jupyter notebook that is designed to make predictions on custom dog images using a pre-trained model. The model is a MobileNetV2 architecture, trained on the Stanford Dogs Dataset, to classify 120 dog breeds. The notebook also includes functions to load the saved model, process the custom images, and display the predicted breed for each image.

## Requirements
This code requires TensorFlow and TensorFlow Hub libraries to be installed. It also needs pandas, numpy, and matplotlib libraries to process the data and display the results.

## Usage
* Clone the repository and open the final.ipynb file in Jupyter notebook or Google Colab.
* Upload the labels.csv and the pre-trained model files to the appropriate directory. The model files should be located in Dog Vision/models/.
* Place your custom dog images in the Dog Vision/my-dog-photos/ directory.
* Run the notebook cells to load the pre-trained model, process the custom images, make predictions, and display the predicted breed for each image.

## Functions
* process_image(image_path): This function processes the image located at image_path and returns the resized and normalized image tensor.
* create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False): This function creates batches of data for predictions. If the data is test data, it may not have labels. The function returns a data batch.
* get_pred_label(prediction_probabilities): This function takes in the predicted probabilities of each class and returns the label of the predicted breed.
* load_model(model_path): This function loads the saved model from the specified model_path and returns the loaded model.
* plt.imshow(image): This function displays the image.

## Inputs
* labels.csv: This file contains the labels of the 120 dog breeds in the Stanford Dogs Dataset.
* Pre-trained model files: The pre-trained model files should be located in the Dog Vision/models/ directory.
* Custom dog images: The custom dog images should be placed in the Dog Vision/my-dog-photos/ directory.
## Outputs
* The code displays the predicted breed for each custom dog image. The predicted breed is displayed above the image in a plot.

Note that the loaded model used here is trained for dog breed classification, and was trained separately. For the code used to train the model, please refer to my other repository
