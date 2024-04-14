import tensorflow as tf
import os
from datetime import datetime

class TestCallback(tf.keras.callbacks.Callback):
    '''
    Test Callback class to evaluate the model on the test data at the end of each epoch
    the test results are stored in a dictionary and can be accessed later
    args:
        :test_data: tf.data.Dataset, the test data
    '''
    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data
        self.test_results = {'test_loss': [], 'test_accuracy': [], 'epoch': []}

    def on_epoch_end(self, epoch, logs=None):
        loss, accuracy = self.model.evaluate(self.test_data, verbose=0)
        # Append the results to the dictionary under the corresponding lists
        self.test_results['epoch'].append(epoch)
        self.test_results['test_loss'].append(loss)
        self.test_results['test_accuracy'].append(accuracy)
        print(f"Epoch {epoch}: Testing loss: {loss}, accuracy: {accuracy}")
        if logs is not None:
            logs['epoch'] = epoch
            logs['test_loss'] = loss
            logs['test_accuracy'] = accuracy
    
    # def on_train_begin(self, logs=None):
    #     '''
    #     initialize/reset the test results dictionary at the beginning of the training
    #     '''
    #     self.test_results = {'test_loss': [], 'test_accuracy': [], 'epoch': []}
    #     return super().on_train_begin(logs)


class SaveModelWeights(tf.keras.callbacks.Callback):
    def __init__(self, n=2, save_path='trained_models/'):
        super().__init__()
        self.n = n
        self.save_path= save_path

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.n:
            os.makedirs(self.save_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") 
            self.model.save_weights(f'{self.save_path}{self.model.name}_{timestamp}_EP_{epoch}.h5')

        # return super().on_epoch_end(epoch, logs)