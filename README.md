**Setting Up the Python Environment** <br>
1. Clone the repository: <br>
   ```git clone [repository_url]``` <br>
2. Create virtual env: <br>
   ```python -m venv my_env``` <br>
3. Activate the environment: <br>
   ```source my_env/bin/activate``` <br>
4. Install Dependencies: <br>
   Navigate to the project directory and install the required dependencies from the requirements.txt file: <br>
   ```pip install -r requirements.txt``` <br>
   This command will install all the necessary libraries and packages needed for the experiments. <br>

**Training:** <br>
The training script is responsible for training different linear layers for each task (Action, Object, and Location) on top of the BERT pre-trained model embeddings. All the hyperparameters and data paths for training the model are specified in the config.yaml file. <br>

   **How to Train** <br>
   To replicate the experiment and train the model, check the configuration in config_.yaml file and change the values if needed. Use the following command to train the model: <br>

   ```python train.py ``` <br>
   The trained model checkpoints will be saved to the location specified in the output_dir configuration in config_.yaml. Make sure that the folder specified for the output_dir exists. Tensorboard log files are stored in `runs` directory. Run ```tensorboard --logdir=runs``` to visualize the train logs.<br>

**Testing** <br>
To test the model on your own test dataset, you need to provide the path to the model checkpoint to be used for testing and the path to the test CSV file. <br>
```python test.py --model_path [path to model checkpoint] --test_csv [path to csv file] ``` <br>
The output of this command evaluates the F1 score on the test data. <br>

**Decoding on one custom command** <br>
You can use the decode.py script to test the model on a custom command. Follow these steps to decode a custom command: <br>
```python custom_decode.py --model_path [path to model checkpoint] --speech_file [path to speech file ]``` <br>
This command allows you to evaluate the model's performance on your own custom speech command using the specified model checkpoint and speech file. <br>
