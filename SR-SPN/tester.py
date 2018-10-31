import tensorflow as tf
import numpy as np
import help_functions as helper
import parameter_configs as pc
import matplotlib.pyplot as plt


class Tester(object):
    """ Test the performance of an already trained network with methods from this class """
    round_digit = 6

    def __init__(self, session, test_data, batch_size, placeholder, network):
        self.sess = session
        self.test_data = test_data
        self.batch_size = batch_size
        self.network = network

        self.input_ph = placeholder["input_ph"]
        self.label_ph = placeholder["label_ph"]

    def spn_outlier_tests(self):
        """ Test outlier performance of spn """

        # CIFAR 100 Test
        input_arr = self.test_data["good_fake"]
        loss_arr_house = self.single_outlier_test(input_arr, "house set")

        # Random CIFAR 10 Test
        input_arr = self.test_data["random"]
        loss_arr_rand = self.single_outlier_test(input_arr, "random")

        # Standard CIFAR 10 Test
        input_arr = self.test_data["x"]
        loss_arr_norm = self.single_outlier_test(input_arr, "mnist")

        self.plot_losses(loss_arr_house, loss_arr_rand, loss_arr_norm)

    def single_outlier_test(self, input_arr, dataset_name):
        """
        Run a single test to test accuracy and loss on given input_arr
        :param input_arr: Array to test performance on
        :param dataset_name: Name of data set provided
        :return: Array with loss for each sample
        """

        print("Running standard test for " + dataset_name)
        batches = input_arr.shape[0] // self.batch_size
        cnn_out_arr = []
        loss_arr = []

        if helper.check_short_run("spn"):
            batches = 1

        graph = tf.get_default_graph()
        if pc.NORMALIZE:
            cnn_output = graph.get_tensor_by_name("spn/normalized_cnn/add:0")
        else:
            cnn_output = graph.get_tensor_by_name("cnn/cnn_output:0")

        for i in range(batches):
            x_batch = input_arr[i * self.batch_size: (i + 1) * self.batch_size]
            feed_dict = {self.input_ph: x_batch}

            probability, cnn_out_value = self.sess.run(
                [self.network.out_layer_spn, cnn_output],
                feed_dict=feed_dict)

            cnn_out_arr = np.append(cnn_out_arr, cnn_out_value)
            loss_single = -1 * tf.reduce_logsumexp(probability, axis=1)
            loss_single_values = self.sess.run(loss_single)

            loss_arr = np.append(loss_arr, loss_single_values)

        mean_total = np.mean(cnn_out_arr)
        stddv_total = np.std(cnn_out_arr)
        final_loss_mean = np.mean(loss_arr)
        final_loss_mean = np.round(final_loss_mean, self.round_digit)
        print("Normalization:{n}    Mean: {x:.{y}f}    Stddv: {z:.{y}f}"
              .format(n=pc.NORMALIZE, x=mean_total, z=stddv_total, y=self.round_digit))
        print(f"Mean Loss: {final_loss_mean:.{self.round_digit}f}")
        return loss_arr

    def performance_test(self, metrics, name):
        """
        Run a performance test to test the accuracy and the loss of the model
        :param metrics: dict with acc_op, acc and loss entries
        :param name: name of network to train (spn or fc)
        :return:
        """

        print("Performance testing for ", name)
        batches = self.test_data["x"].shape[0] // self.batch_size
        total_loss = 0
        acc_op = metrics["acc_op"]
        acc = metrics["acc"]
        loss = metrics["loss"]

        if helper.check_short_run(name):
            batches = 1

        for i in range(batches):
            x_batch = self.test_data["x"][i * self.batch_size: (i + 1) * self.batch_size]
            y_batch = self.test_data["y"][i * self.batch_size: (i + 1) * self.batch_size]

            print("x_batch, y_batch shape:", x_batch.shape, y_batch.shape)
            feed_dict = {self.input_ph: x_batch, self.label_ph: y_batch}

            _, loss_value = self.sess.run([acc_op, loss], feed_dict=feed_dict)

            total_loss += loss_value

        final_loss = round(total_loss/batches, self.round_digit)
        acc_value = self.sess.run(acc)
        acc_value = round(acc_value, self.round_digit)

        loss_string = " loss: {x:.{y}f} ".format(x=final_loss, y=self.round_digit)
        acc_string = " acc: {x:.{y}f} ".format(x=acc_value, y=self.round_digit)

        print("Test results: ", loss_string, acc_string)
        # helper.log_text(loss_string, "Evaluation", path=self.text_logs_path)

    @staticmethod
    def plot_losses(loss1, loss2, loss3):
        """ Plot losses in a very simple single graph """
        try:
            plt.hist(loss1, 100)
            plt.hist(loss2, 100)
            plt.hist(loss3, 100)
            plt.plot()
        except RuntimeError as e:
            print(e)
