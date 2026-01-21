# Custom L1 Distance layer module 
# WHY DO WE NEED THIS: its needed to load the custom model

# Import dependencies
import tensorflow as tf
from keras.layers import Layer

# Custom L1 Distance Layer from Jupyter 
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
       
    def call(self, inputs):
        # 修正：確保處理的是 Tensor 而不是 list
        # 這是處理 [[tensor1], [tensor2]] 報錯的最穩寫法
        input_embedding = inputs[0]
        validation_embedding = inputs[1]
        
        if isinstance(input_embedding, list):
            input_embedding = input_embedding[0]
        if isinstance(validation_embedding, list):
            validation_embedding = validation_embedding[0]
            
        return tf.math.abs(input_embedding - validation_embedding)