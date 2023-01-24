# Pneumonia Classification on chest X-rays

**Pneumonia** is a respiratory infection caused by bacteria or viruses; it affects many individuals, especially in developing and underdeveloped nations, where high levels of pollution, unhygienic living conditions, and overcrowding are relatively common, together with inadequate medical infrastructure. Pneumonia causes pleural effusion, a condition in which fluids fill the lung, causing respiratory difficulty. It accounts for more than **14% of deaths in children** under the age of five years [Ref:WHO](https://www.who.int/news-room/fact-sheets/detail/pneumonia)

Early diagnosis of pneumonia is crucial to ensure curative treatment and increase survival rates. **Chest X-ray imaging** is the most frequently used method for diagnosing pneumonia. However, the examination of chest X-rays is a challenging task and is prone to subjective variability.

Deep learning models can be trained to automatically identify features on X-rays that are indicative of pneumonia, such as inflammation or consolidation in the lungs.

One common use case for these models is in a **clinical setting**, where they can assist radiologists in **interpreting X-rays and making a diagnosis**. This can help to improve the accuracy and efficiency of the diagnostic process, as well as reduce the workload on radiologists.

Another use case is in an **research setting**, where deep learning models can be trained on a large dataset of chest X-rays to classify pneumonia with high accuracy, which can be used as a benchmark for new models and also to improve the diagnosis process in developing countries where radiologists are scarce.

In addition, these models can also be integrated into `mobile applications` to be used at the **point of care**, allowing for quick and easy diagnosis of pneumonia in resource-limited settings.


- **Dataset:** https://data.mendeley.com/datasets/rscbjbr9sj/2
    - filename: `ChestXRay2017.zip`


- Dataset into:
```
    Number of x-ray image classes in train: 2
    ['NORMAL', 'PNEUMONIA']
    NORMAL : 1349
    PNEUMONIA : 3884

    Number of x-ray image classes in test: 2
    ['NORMAL', 'PNEUMONIA']
    NORMAL : 235
    PNEUMONIA : 391

    Class names: ['NORMAL', 'PNEUMONIA']
```

**Notes**:
- A system with GPU will be needed for experimentation and training the models


## Model

Transfer learning can be used for pneumonia classification on chest X-rays using models such as `Xception`. Transfer learning is a technique where a pre-trained model is used as a starting point to train a new model on a different dataset. This can be done by fine-tuning the pre-trained model on a dataset of chest X-rays labeled with pneumonia or normal.

The idea behind transfer learning is that the pre-trained model already has learned useful features from a large dataset, and can be fine-tuned to learn new features specific to the task of pneumonia classification on chest X-rays. This can help to improve the performance of the model compared to training it from scratch, especially when the dataset of chest X-rays is relatively small.

`Xception` is a deep convolutional neural network that has been pre-trained on a large dataset of images and is useful in various computer vision tasks. It uses depthwise separable convolutions which reduces the number of parameters while maintaining the performance.

So, we will be using pre-trained convolutional neural network model known as `Xception`.

Pre-trained convolutional neural networks:

- Imagenet dataset: https://www.image-net.org/
- Pre-trained models: https://keras.io/api/applications/

Other models for consideration:
- InceptionV3
- ResNet50
- DenseNet121

 
## Deployment to cloud - Hugging Face Spaces
- For testing, this model is currently deployed to - https://huggingface.co/spaces/ranga4all/pneumonia-classification-app

- Note: This deployment may be teared down after couple of weeks

[screen-capture.webm](https://user-images.githubusercontent.com/80430945/214222948-2efad2ac-0686-43f2-b692-1019a53881b8.webm)


## Files included in this repo
1. `README.md` - readme file with description of the problem and instructions on how to run the project
2. `requirements.txt` - dependencies that can be installed with `pip`
3. `notebook.ipynb` - dataset download, data cleaning, preprocessing, EDA, model selection and parameter tuning. This file also includes final model training, saving, loading and inference(prediction) testing
4. `train.py` - script for training and saving final model
5. `test-model.ipynb` - testing TF model
6. `test-model.py` - script for testing TF model 
7. `gradio-pneumonia-classification-app/requirements.txt` - Hugging Face Spaces dependency file
8. `gradio-pneumonia-classification-app/app.py` - gradio app for deploying to Hugging Face Spaces
9. `gradio-pneumonia-classification-app/*.jpeg` - sample example files for interence on Hugging Face Spaces deployment

## How to run this project?

**Prerequisites:**
- System with GPU for experimenting/training the model
- `anaconda` or `miniconda` with `conda` package manager
- Dataset downloaded, extracted and cleaned up. If needed, refer to `notebook.ipynb` section 1.1.1 Data Download

### **Steps**

### A) Setup local environment

1. Create a new conda environment and activate
```
conda create --name pneumonia-classification python=3.9
conda activate pneumonia-classification
```
2. Clone the repo
```
git clone https://github.com/ranga4all1/pneumonia-classification.git
```
3. Install dependencies
```
cd pneumonia-classification
pip install -r requirements.txt
```

### B) Model training

1. Run `train.py'. This would save a model file.
```
python train.py
```

2. Test TF model by running below command
```
python test-model.py
```
Output should look like this:
```
{'NORMAL': 0.0, 'PNEUMONIA': 1.0}
```

### Deploy to cloud - Hugging Face Spaces


1. Login to Hugging Face Spaces from browser:
    1) Click -> `Create new Space`
    2) Enter/select below parameters:
      Space name: `pneumonia-classification-app`
      Select the Space SDK: `Gradio`
      License: <your choice - e.g. apache-2.0>
    3) Click -> `Create Space`
    4) Within your space, Go to `Files and versions` tab and click -> `+Add file` -> Upload files
    
    Select below files
     ```
     xray_model.h5
     and
     All files in gradio-pneumonia-classification-app/
     ```
    in `commit changes` type `uploaded model and app files`

    Click -> Commit changes to main

2. Test
    1) In your browser - within Hugging Face Spaces, app GUI should be available now to test. Select sample image and submit to get prediction.
    Or use upload/drag files option and submit to get prediction
