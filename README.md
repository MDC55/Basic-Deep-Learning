# Basic Autoencoder (MNIST)

Notebook: `Basic_Autoencoder.ipynb` — a minimal autoencoder walkthrough using the MNIST dataset and tf.keras.

## Summary
- Loads MNIST via tf.keras.datasets.mnist, scales pixels to [0,1].
- Encoder: Input(shape=(28,28,1)) → Flatten → Dense(64, relu).
- Decoder: Dense(64, relu) → Dense(784, relu) → Reshape((28,28,1)).
- Model: autoencoder = Model(encoder_input, decoder_output); compiled with tf.keras.optimizers.legacy.Adam and loss='mse'.
- Training: autoencoder.fit(x_train, x_train, batch_size=32, validation_split=0.10). 

## Key snippets (from notebook)
```python
# data
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/255.0
x_test  = x_test/255.0

# encoder
encoder_input = keras.Input(shape=(28,28,1), name='img')
x = keras.layers.Flatten()(encoder_input)
encoder_output = keras.layers.Dense(64, activation='relu')(x)

# decoder / autoencoder
decoder_input = keras.layers.Dense(64, activation='relu')(encoder_output)
x = keras.layers.Dense(784, activation='relu')(decoder_input)
decoder_output = keras.layers.Reshape((28,28,1))(x)
```

## Requirements
- Python 3.8+
- tensorflow (tested with TF 2.10+)
- numpy, matplotlib
- Optional: opencv-python (cv2) if using image I/O
- jupyter

Install (Windows):
```bash
python -m venv .venv
.venv\Scripts\activate
pip install "tensorflow>=2.10" numpy matplotlib jupyter
# optional: pip install opencv-python
```

## Run
- Open in VS Code or:
```bash
jupyter notebook "Basic_Autoencoder.ipynb"
```
- To run converted script:
```bash
jupyter nbconvert --to script "Basic_Autoencoder.ipynb"
python "Basic_Autoencoder.py"
```

- Example commit message: `docs: add README for Basic_Autoencoder.ipynb`

