import tensorflow as tf

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