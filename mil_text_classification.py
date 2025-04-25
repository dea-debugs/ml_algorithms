""""
Text classification using multiple instance learning (MIL) with Keras

Resources
- https://keras.io/examples/vision/attention_mil_classification/
- https://openaccess.thecvf.com/content/ACCV2020/papers/Wang_In_Defense_of_LSTMs_for_Addressing_Multiple_Instance_Learning_Problems_ACCV_2020_paper.pdf
- https://github.com/shadowwkl/LSTM-for-Multiple-Instance-Learning
- https://roboticsfaq.com/neural-networks-vs-logistic-regression-understanding-the-differences-and-benefits/
- https://arxiv.org/pdf/1802.04712"

Requirements (pip install)
- numpy
- tensorflow
- tf-keras
- setfit
"""


# imports =========================================================================================
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer

from setfit import SetFitModel
setfit_model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")


# functions =======================================================================================

def embed(text, model):
    """
    Produce embeddings for a given text.

    Inputs
    - text:     string
    - model:    model object
    
    Outputs
    - resulting embedding
    """

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return model.encode(sentences)


def preprocess(docs, setfit_model, max_inst, details=False):
    """
    Preprocess training data.
    
    Inputs
    - docs:             list of strings
    - setfit_model:     pretrained model for embeddings
    - max_inst:         integer, maximum number of sentences found in a document
    - details:          Boolean, whether to print docs and embeddings
    
    Outputs
    - array of bags ready for the model
    """

    all_bags = []
    for doc in docs:

        # split doc into sentences and get embeddings
        embeddings = embed(doc, setfit_model)

        if details == True:
            print(f"Doc: {doc}")
            if embeddings.shape[1] >= 6:
                print(f"Embeddings: {embeddings[0][1:6]}") # confirm embeddings are working
            else:
                print("Embeddings are smaller than anticipated. Recommending investigation.")

        bags = [embeddings[i:i + max_inst] for i in range(0, len(embeddings), max_inst)]

        # pad bags to account for different numbers of instances
        padded_bags = []
        for bag in bags:
            if len(bag) < max_inst:
                padding = np.zeros((max_inst - len(bag), embeddings.shape[-1]))
                bag = np.vstack((bag, padding))
            padded_bags.append(bag)
        
        all_bags.extend(padded_bags)

    return np.array(all_bags, dtype=np.float32)


def preprocess_new(doc, setfit_model, max_inst):
    """
    Preprocess unseen data so model can make predictions on it.

    Inputs
    - doc:          string
    - setfit_model: pretrained model for embeddings
    - max_inst:     maximum number of instances in a document (from training data)

    Outputs
    - array ready to be input to model
    """

    embeddings = embed(doc, setfit_model)

    if len(embeddings) < max_inst:
        padding = np.zeros((max_inst - len(embeddings), embeddings.shape[-1]))
        embeddings = np.vstack((embeddings, padding))
    else:
        # if too many instances, cut off to maintain consistent dimensions (this is a limitation of the approach)
        embeddings = embeddings[:max_inst]
    
    return np.array([embeddings], dtype=np.float32)


def predict_new(new, mil_model, setfit_model, max_inst, threshold=0.5):
    """
    Make prediction on new data.

    Inputs
    - new:          list of strings, unseen documents
    - mil_model:    trained MIL model
    - setfit_model: pretrained embeddings model
    - max_inst:     integer, maximum number of instances in a document (from training data)
    - threshold:    float, threshold for a positive prediction
    """
    
    for new_doc in new:
        new_processed = preprocess_new(new_doc, setfit_model, max_inst)
        prediction = mil_model.predict(new_processed)

        predicted_prob = prediction[0][0]

        if predicted_prob >= threshold:
            print(f"Document is predicted as positive with P={predicted_prob:.4f}")
        else:
            print(f"Document is predicted as negative with P={predicted_prob:.4f}")


class AttentionPooling(Layer):
    """
    Custom neural network layer to return weighted sum using learnt weights.
    """

    def call(self, inputs):
        instance_model, attention_weights = inputs
        return tf.reduce_sum(instance_model*attention_weights, axis=1)
    

def create_mil_model(input_shape, scaling_factor=1):
    """
    Define model architecture.
    
    Inputs
    - input_shape:    tuple consisting of (number of instances, dimension of embeddings)
    - scaling_factor: integer default parameter, can be used to make attention weights have greater impact
    
    Outputs
    - model:          model object
    """

    # input layer
    input_layer = layers.Input(shape=input_shape)

    # encode each instance using an LSTM
    instance_model = layers.LSTM(64, return_sequences=True, name="lstm_layer")(input_layer)
    instance_model = layers.Dropout(0.3)(instance_model)

    # attention mechanism
    # - create attention weights with dense layer
    # - use softmax so all attention weights add to 1
    attention_weights = layers.Dense(1, activation='tanh', kernel_initializer="glorot_uniform", name="first_attention")(instance_model)
    scaled_attention = attention_weights*scaling_factor
    attention_weights = layers.Activation("sigmoid", name='attention_layer')(scaled_attention)

    # apply attention
    # - perform weighted sum to aggregate instances into documents
    attended_instances = AttentionPooling()([instance_model, attention_weights])

    # output layer
    # - convert to range [0, 1]
    output_layer = layers.Dense(1, activation='sigmoid')(attended_instances)

    # create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model


class AttentionLogger(tf.keras.callbacks.Callback):
    """
    Logger to track model training.  Useful for debugging.
    """

    def __init__(self, model, dataset):
        super().__init__()
        self.dataset = dataset

        self.first_attention = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer("first_attention").output
        )

        self.attention_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer("attention_layer").output
        )

        self.lstm_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer("lstm_layer").output
        )

    def on_epoch_end(self, epoch, logs=None):
        random_index = np.random.randint(0, len(self.dataset))
        test_sample = self.dataset[random_index:random_index+1]  # select a single sample (batch size = 1)
        print(f"\nTest: {test_sample[0][0][:5]}")

        raw_lstm_output = self.lstm_extractor(test_sample)
        print(f"Epoch {epoch+1} - LSTM output:\n{raw_lstm_output[0][0][:5]}")

        raw_first = self.first_attention(test_sample)
        print(f"Epoch {epoch+1} - first attention output:\n{raw_first}")

        # extract raw attention weights before softmax
        raw_attention_output = self.attention_extractor(test_sample)
        print(f"Epoch {epoch+1} - Raw Attention Weights Sample:\n{raw_attention_output.numpy()[:5]}")
        
        # apply softmax manually for comparison
        attention_weights = tf.nn.softmax(raw_attention_output, axis=1)
        print(f"Epoch {epoch+1} - Softmaxed Attention Weights:\n{attention_weights.numpy()[:5]}")


# main ============================================================================================

# define training data
docs = ["Dear person. You are bad."
        , "Hello. You did a good thing."
        , "Good morning. I hate you."
        , "Hi.  You are great!"
        , "This is a test message."
        , "Hello. I have nothing to say."
        , "Hi. Bad. Bad. Bad."
        , "Hi. Good. Good. Good."
        , "Hi. Indeterminate."
        , "Today I ate a cabbage then I walked a dog then I patted a cat then I cooked a bag of cheese."
        , "Macarena."
        , "You lied to us."
        , "I went for a walk. I did a bad thing."
        ]

labels = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1], dtype=np.float32)

# preprocess training data
max_inst = max(len(re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', doc)) for doc in docs)
padded_bags = preprocess(docs, setfit_model, max_inst)

# define data input shape (number of instances, embedding dimension)
embedding_dim = setfit_model.encode(["test"]).shape[-1]
input_shape = (max_inst, embedding_dim)

# create MIL model
mil_model = create_mil_model(input_shape)

# compile model
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
mil_model.compile(optimizer=opt
                  , loss='binary_crossentropy'
                  , metrics=['accuracy']
                  )

attention_logger = AttentionLogger(mil_model, padded_bags)

# train model
labels = labels.reshape(-1, 1)
mil_model.fit(padded_bags, labels, epochs=10, batch_size=4, callbacks=[attention_logger])

# define testing data
new = ["Hello. You did well. Goodbye."
       , "That was wrong."
       , "Dear client. No evidence was there."
       , "Hello. No changes will be made."
       , "Potato potato potato."
       , "Bad."
       , "Today I saw a cloud. Then I became a thief."
]

# make predictions on new (unseen) data
predict_new(new, mil_model, setfit_model, max_inst)
