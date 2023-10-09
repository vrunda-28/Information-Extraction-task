**Setting Up the Python Environment** <br>
1. Clone the repository:
   ```git clone [repository_url]'''
2. Create virtual env:
   '''python -m venv my_env```
3. Activate the environment:
   ```source my_env/bin/activate```
4. Install Dependencies:
   Navigate to the project directory and install the required dependencies from the requirements.txt file:
   ```pip install -r requirements.txt```
   This command will install all the necessary libraries and packages needed for the experiments.
**Training:** <br>
The training script is responsible for training different linear layers for each task (Action, Object, and Location) on top of the BERT pre-trained model embeddings. All the hyperparameters and data paths for training the model are specified in the config.yaml file. <br>

**How to Train** <br>
To replicate the experiment and train the model, use the following command:

```python train.py --config config.yaml ```
The trained model checkpoints will be saved to the location specified in the output_dir configuration in config.yaml.

**Testing** <br>
To test the model on your own test dataset, you need to provide the path to the model checkpoint to be used for testing and the path to the test CSV file.
```python test.py --model_path [path to model checkpoint] --test_csv [path to csv file] ```
The output of this command evaluates the F1 score on the test data.

**Decoding on one custom command** <br>
You can use the decode.py script to test the model on a custom command. Follow these steps to decode a custom command:
```python decode.py --model_path [path to model checkpoint] --speech_file [path to speech file ]```
This command allows you to evaluate the model's performance on your own custom speech command using the specified model checkpoint and speech file.
