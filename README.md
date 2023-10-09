**Training:**
Training script includes the training of different linear layers for each task [Action, Object and Location] on top of the BERT pre-trained model embeddings. config.yaml has all the hyperparameters and the data path for training the model. To replicate the experiment run the following command:
```python train.py --config config.yaml ```

