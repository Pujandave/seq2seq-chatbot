# seq2seq-chatbot
Create chatbot with help of seq2seq model
A sequence2sequence chatbot implementation with TensorFlow.

See instructions to get started below, or check out some chat logs
Chatting with a trained model
To chat with a trained model from the model directory:

(Batch files are only available for windows as of now. For mac and linux users see instructions below for python console.)

    Make sure a model exists in the models directory (to get started, download and unzip trained_model_v2 into the seq2seq-chatbot/models/cornell_movie_dialog folder)

For console chat:

    From the model directory run chat_console_best_weights_training.bat or chat_console_best_weights_validation.bat

For web chat:

    From the model directory run chat_web_best_weights_training.bat or chat_web_best_weights_validation.bat

    Open a browser to the URL indicated by the server console, followed by /chat_ui.html. This is typically: http://localhost:8080/chat_ui.html

To chat with a trained model from a python console:

    Set console working directory to the seq2seq-chatbot directory. This directory should have the models and datasets directories directly within it.

    Run chat.py with the model checkpoint path:

run chat.py models\dataset_name\model_name\checkpoint.ckpt

For example, to chat with the trained cornell movie dialog model trained_model_v2:

    Download and unzip trained_model_v2 into the seq2seq-chatbot/models/cornell_movie_dialog folder

    Set console working directory to the seq2seq-chatbot directory

    Run:

run chat.py models\cornell_movie_dialog\trained_model_v2\best_weights_training.ckpt

The result should look like this:
![image](https://user-images.githubusercontent.com/78218075/144166813-4a6400ca-c0fe-438e-93a7-318187c503a8.png)



Training a model

To train a model from a python console:

    Configure the hparams.json file to the desired training hyperparameters

    Set console working directory to the seq2seq-chatbot directory. This directory should have the models and datasets directories directly within it.

    To train a new model, run train.py with the dataset path:

run train.py --datasetdir=datasets\dataset_name

Or to resume training an existing model, run train.py with the model checkpoint path:

run train.py --checkpointfile=models\dataset_name\model_name\checkpoint.ckpt

For example, to train a new model on the cornell movie dialog dataset with default hyperparameters:

    Set console working directory to the seq2seq-chatbot directory

    Run:

run train.py --datasetdir=datasets\cornell_movie_dialog

The result should look like this:
![image](https://user-images.githubusercontent.com/78218075/144166858-79903b10-5ddf-496f-b1cf-5b2a978f3b7c.png)


train

Visualizing a model in TensorBoard

TensorBoard is a great tool for visualizing what is going on under the hood when a TensorFlow model is being trained.

To start TensorBoard from a terminal:

tensorboard --logdir=model_dir

Where model_dir is the path to the directory where the model checkpoint file is. For example, to view the trained cornell movie dialog model trained_model_v2:

tensorboard --logdir=models\cornell_movie_dialog\trained_model_v2



Visualize word embeddings

TensorBoard can project the word embeddings into 3D space by performing a dimensionality reduction technique like PCA or T-SNE, and can allow you to explore how your model has grouped together the words in your vocabulary by viewing nearest neighbors in the embedding space for any word. 

When launching TensorBoard for a model directory and selecting the "Projector" tab, it should look like this: train
Adding a new dataset
![image](https://user-images.githubusercontent.com/78218075/144166465-070474d2-4e62-4393-b5a7-0fd4122f2156.png)



Dependencies

The following python packages are used in seq2seq-chatbot: (excluding packages that come with Anaconda)

    TensorFlow *Note - TF 2.x is not yet supported, use the latest TF 1.x version.

    pip install --upgrade tensorflow==1.*

    For GPU support: (See here for full GPU install instructions including CUDA and cuDNN)

    pip install --upgrade tensorflow-gpu==1.*

    jsonpickle

    pip install --upgrade jsonpickle

    click 6.7, flask 0.12.4 and flask-restful (required to run the web interface)

    pip install click==6.7
    pip install flask==0.12.4
    pip install --upgrade flask-restful


