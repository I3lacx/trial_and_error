import tensorflow as tf
import help_functions as helper
import datetime
import parameter_configs as pc
import os
import numpy as np
import shutil


class Trainer(object):
    """ Train a fc or an spn based on provided parameters """
    round_digit = 4

    def __init__(self, data, network, placeholder, run_params,
                 session_name, text_logs_path):
        self.data = data
        self.network = network
        self.run_config = run_params

        self.input_ph = placeholder["input_ph"]
        self.label_ph = placeholder["label_ph"]
        self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.batch_size = self.run_config["batch_size"]

        self.session_name = session_name
        self.text_logs_path = text_logs_path
        self.logs_path = helper.get_tensorflow_logs_path(self.session_name)
        self.writer_train = tf.summary.FileWriter(self.logs_path + "/train")
        self.writer_test = tf.summary.FileWriter(self.logs_path + "/test")

        # Todo: global step variable is 390 * epoch, not epoch number
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.highest_acc = 0
        self.lowest_loss = float('Inf')
        self.saver = {}

    def _train_fc(self, train_op, sess, acc, acc_op):
        """
        Train Fully Connected network on train_op, configs based on config file
        :param train_op: Optimizer for network
        :param sess: active session
        :param acc: accuracy metric to get current accuracy
        :param acc_op: accuracy metric to update current accuracy
        :return:
        """

        batches_per_epoch = self.run_config["batches_per_epoch"]
        epochs = self.run_config["num_epochs"]

        if helper.check_short_run("fc"):
            batches_per_epoch = 1
            epochs = 1

        summary_list = tf.get_collection(tf.GraphKeys.SUMMARIES, scope="fc_metrics")
        summary = tf.summary.merge(summary_list)

        # Define Acc initializer to reset accuracy
        acc_var = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="fc_metrics/accuracy")
        acc_initializer = tf.variables_initializer(var_list=acc_var)

        start_time = datetime.datetime.now()

        print("   ----------Training---------- | ----------Testing-----------")
        for epoch in range(epochs):
            feed_dict = {}

            # ----- Train -------
            for i in range(batches_per_epoch):
                x_batch = self.data["x_train"][i*self.batch_size:(i+1)*self.batch_size]
                y_batch = self.data["y_train"][i*self.batch_size:(i+1)*self.batch_size]
                learning_rate = self.get_cnn_learning_rate(epoch)
                feed_dict = {self.input_ph: x_batch,
                             self.label_ph: y_batch,
                             self.learning_rate_ph: learning_rate}
                _, _, out = sess.run([train_op, acc_op, self.network.out_layer_fc], feed_dict=feed_dict)

            # ----- Evaluate Train Performance ------
            summary_train, train_acc = sess.run([summary, acc], feed_dict=feed_dict)
            cur_step = sess.run(self.global_step)
            self.writer_train.add_summary(summary_train, cur_step)
            train_acc = round(train_acc, self.round_digit)
            train_results = "acc:  {z:.{y}f}".format(z=train_acc, y=self.round_digit)

            # Save model
            self._save_model(train_acc, sess)
            sess.run(acc_initializer)

            # ---- TEST ----
            summary_test = self._run_single_batch_test(sess, summary, "fc", acc_op)

            # ---- Evaluate Test Performance ----
            test_acc = sess.run(acc)
            self.writer_test.add_summary(summary_test, cur_step)
            test_acc = round(test_acc, self.round_digit)
            test_results = "acc:  {z:.{y}f}".format(z=test_acc, y=self.round_digit)

            print("{}:   {}        {}".format(epoch, train_results, test_results))
            sess.run(acc_initializer)

        helper.log_time(start_time, name="Training Time:", path=self.text_logs_path)

    def _train_spn(self, train_op, sess, loss, acc, acc_op):
        """
        Train SPN on train_op, configs based on config file
        :param train_op: Optimizer
        :param sess: active session
        :param loss: loss to train on
        :param acc: accuracy metric to get current accuracy
        :param acc_op: accuracy metric to update current accuracy
        :return:
        """

        batches_per_epoch = self.run_config["batches_per_epoch"]
        epochs = self.run_config["num_epochs"]

        if helper.check_short_run("spn"):
            batches_per_epoch = 1
            epochs = 1

        summary_list = tf.get_collection(tf.GraphKeys.SUMMARIES, scope="spn_metrics")
        summary = tf.summary.merge(summary_list)

        graph = tf.get_default_graph()
        if pc.NORMALIZE:
            cnn_output = graph.get_tensor_by_name("spn/normalized_cnn/add:0")
        else:
            cnn_output = graph.get_tensor_by_name("cnn/cnn_output:0")

        acc_var = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="spn_metrics/accuracy")
        acc_initializer = tf.variables_initializer(var_list=acc_var)

        start_time = datetime.datetime.now()
        print("   ----------Training---------- | ----------Testing-----------")
        for epoch in range(epochs):
            means_cnn_total = 0
            stddv_cnn_total = 0
            feed_dict = {}

            # ---- Train -----
            train_loss = 0
            for i in range(batches_per_epoch):
                x_batch = self.data["x_train"][i*self.batch_size:(i+1)*self.batch_size]
                y_batch = self.data["y_train"][i*self.batch_size:(i+1)*self.batch_size]
                learning_rate = self.get_spn_learning_rate(epoch)
                feed_dict = {
                        self.input_ph: x_batch,
                        self.label_ph: y_batch,
                        self.learning_rate_ph: learning_rate}

                _2, loss_value, cnn_out_value, _3 = sess.run(
                    [train_op, loss, cnn_output, acc_op], feed_dict=feed_dict)

                means_cnn_total += np.mean(cnn_out_value)
                stddv_cnn_total += np.std(cnn_out_value)

                train_loss += loss_value
                if i % 20 == 0:
                    print("Loss in {i}: {x:.{y}f}: "
                          .format(i=i, x=loss_value, y=self.round_digit))

            print("Total mean during run:  ", means_cnn_total/batches_per_epoch)
            print("Total stddv during run: ", stddv_cnn_total/batches_per_epoch)

            # ----- Evaluate Train Performance ------
            # TODO: summary is only the last thing the network trained on, not all!
            summary_train, train_acc_value = sess.run([summary, acc], feed_dict=feed_dict)
            cur_step = sess.run(self.global_step)

            train_acc = round(train_acc_value, self.round_digit)
            train_loss = round(train_loss / batches_per_epoch, self.round_digit)
            train_results = "loss: {x:.{y}f} acc:  {z:.{y}f}"\
                .format(x=train_loss, y=self.round_digit, z=train_acc)

            # Temporary solution to fix summary problem
            summary_two = tf.Summary()
            summary_two.value.add(tag="spn_metrics/accuracy", simple_value=train_acc)
            summary_two.value.add(tag="spn_metrics/optimized_loss", simple_value=train_loss)
            self.writer_train.add_summary(summary_two, cur_step)
            self.writer_train.add_summary(summary_train, cur_step)

            self._save_model(train_loss, sess)
            sess.run(acc_initializer)

            # ---- Test -----
            summary_test, test_loss = self._run_single_batch_test(sess, summary, "spn",
                                                                  acc_op, loss=loss)

            # ----- Evaluate Test Performance -----
            self.writer_test.add_summary(summary_test, cur_step)
            test_acc = sess.run(acc)
            test_acc = round(test_acc, self.round_digit)
            test_results = "loss: {x:.{y}f}   acc:{z:.{y}f}"\
                .format(x=test_loss, y=self.round_digit, z=test_acc)

            print("------------------------------------------------------------------------")
            print("{}:   {}        {}".format(epoch, train_results, test_results))
            print("------------------------------------------------------------------------")
            sess.run(acc_initializer)

        helper.log_time(start_time, name="Training Time:", path=self.text_logs_path)

    def _run_single_batch_test(self, sess, summary, name, acc_op, loss=None):
        """ Run a single batch test on the test set """
        random_ids = np.random.choice(self.data["x_test"].shape[0], self.batch_size)
        # Switches order around to test different test batches each epoch
        x_batch = self.data["x_test"][random_ids, :]
        y_batch = self.data["y_test"][random_ids, :]

        feed_dict = {self.input_ph: x_batch, self.label_ph: y_batch}

        if name == "spn":
            # First run only acc_op so acc is updated for summary
            sess.run(acc_op, feed_dict=feed_dict)
            summ_value, loss_value = sess.run([summary, loss], feed_dict=feed_dict)
            loss_value = round(loss_value, self.round_digit)
            return summ_value, loss_value
        elif name == "fc":
            # First run only acc_op so acc is updated for summary
            sess.run(acc_op, feed_dict=feed_dict)
            return sess.run(summary, feed_dict=feed_dict)

    def _run_batch_tests(self, sess, summary, acc_op, loss, count):
        """ run several batch tests on test set """
        # TODO: implement functionality and include in model

        random_ids = np.random.choice(self.data["x_test"].shape[0], self.batch_size)
        # Switches order around to test different test batches each epoch
        x_batch = self.data["x_test"][random_ids, :]
        y_batch = self.data["y_test"][random_ids, :]

        feed_dict = {self.input_ph: x_batch, self.label_ph: y_batch}

        total_loss = 0
        for _ in range(count):
            loss, _ = sess.run([acc_op, loss], feed_dict=feed_dict)
            total_loss += loss

        summ_value, loss_value = sess.run([summary, loss], feed_dict=feed_dict)
        loss_value = round(loss_value, self.round_digit)
        return summ_value, loss_value

    def get_normalization_values(self, layer, sess):
        """ Calculate Mean and Stddv of CNN output layer
        ;param layer: The layer to calculate mean and stddv from
        :return mean and stddv of layer
        """
        if not pc.NORMALIZE:
            return 0, 1

        batches_per_epoch = self.run_config["batches_per_epoch"]
        out_values_array = []

        if helper.check_short_run("spn"):
            batches_per_epoch = 1

        print("Calculating normalization values of layer: ", layer)
        for i in range(batches_per_epoch):
            x_batch = self.data["x_train"][i * self.batch_size:(i + 1) * self.batch_size]
            y_batch = self.data["y_train"][i * self.batch_size:(i + 1) * self.batch_size]
            feed_dict = {self.input_ph: x_batch, self.label_ph: y_batch}

            cnn_out_value = sess.run(layer, feed_dict=feed_dict)
            out_values_array = np.append(out_values_array, cnn_out_value)

        mean_total = np.mean(out_values_array)
        stddv_total = np.std(out_values_array)
        print("Mean {x:.{y}f}, Stddv {z:.{y}f}"
              .format(x=mean_total, y=self.round_digit, z=stddv_total))

        return mean_total, stddv_total

    def check_normalization_values(self, sess):
        graph = tf.get_default_graph()
        normalized_layer = graph.get_tensor_by_name("spn/normalized_cnn/add:0")
        self.get_normalization_values(normalized_layer, sess)

    def _save_model(self, sess, name):
        """ Test the performance then saves variable values into saver """
        # TODO: Add validation on test set before saving the model

        if pc.SAVE[name]:
            if helper.check_short_run(name):
                print("Not saving")
                return

            saver = self.saver[name]
            save_path = self.saver["path_" + name]
            saver.save(sess,
                       save_path=save_path,
                       latest_filename=None,
                       global_step=self.global_step)

    def restore_checkpoint(self, sess, name):
        """ Restores Checkpoint if RESTORE constant is True, else create new variables """
        if name == "spn":
            variables_to_save = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rat")
            saver = tf.train.Saver(var_list=variables_to_save)
        elif name == "fc":
            fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc")
            cnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cnn")
            variables_to_save = fc_vars + cnn_vars
            saver = tf.train.Saver(var_list=variables_to_save)
        else:
            print(f"NAME: {name} NOT FOUND")
            exit(0)
            return

        cwd = os.getcwd()
        save_path = cwd + "/logs/checkpoints/" + self.session_name + "/" + name

        self.saver[name] = saver

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            num_models = 0
        else:
            num_models = len(os.listdir(save_path))

        if num_models > 0 and pc.RESTORE[name]:
            if not pc.RESTORE[name+"_model_number"] == "":
                model_number = pc.RESTORE[name+"_model_number"]
                existing_path = save_path + "/model_" + model_number + "/"
            elif not pc.RESTORE["good_model_"+name] == "":
                model_number = pc.RESTORE["good_model_"+name]
                existing_path = cwd + "/good_models/" + name + "/model_" + model_number + "/"
            else:
                existing_path = save_path + "/model_{:02d}/".format(num_models - 1)
                print(f"\nRESTORE {name}: {pc.RESTORE[name]} --- Existing models: {num_models > 0}")

            self.saver["path_" + name] = existing_path
            try:
                print(f"Trying to restore checkpoint from: {existing_path}")
                last_checkpoint = tf.train.latest_checkpoint(existing_path)
                saver.restore(sess, save_path=last_checkpoint)
                print("Restored checkpoint from:", last_checkpoint)
                if pc.SAVE[name]:
                    print("Saving in: ", existing_path)
                return saver, existing_path
            except ValueError as e:
                print("Failed to restore checkpoint.\n" + str(e))

        if pc.SAVE[name]:
            save_path += "/model_{:02d}/".format(num_models)
            self.saver["path_" + name] = save_path
            os.makedirs(save_path)
            print("Saving in: ", save_path)
            return saver, save_path
        print("")
        return saver, None

    def spn_print_example(self, sess):
        """ print probabilities of one x_train example """
        x_batch = self.data["x_train"][0:self.batch_size]
        y_batch = self.data["y_train"][0:self.batch_size]
        feed_dict = {self.input_ph: x_batch, self.label_ph: y_batch}

        probability = sess.run(self.network.out_layer_spn, feed_dict=feed_dict)
        print("Probability: ", probability)

    def _add_metrics(self, out_layer):
        """
        Add accuracy metric for given out_layer
        :param out_layer: Output layer for which acc should be calculated
        :return: acc: used to get the current acc
        ;return: acc_op: used to update the current acc
        """
        argmax_prediction = tf.argmax(out_layer, axis=-1)
        argmax_labels = tf.argmax(self.label_ph, axis=-1)
        acc, acc_op = tf.metrics.accuracy(argmax_labels, argmax_prediction, name="accuracy")

        tf.summary.scalar("accuracy", acc)
        return acc, acc_op

    def get_loss_fc(self):
        """ Get loss for fc network """
        fc_output = self.network.out_layer_fc

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.label_ph, logits=fc_output))

        tf.summary.scalar("disc_loss", loss)

        return loss

    def get_loss_spn(self):
        """
         Selected and return loss for spn based on self.run_configs["loss"]
         Adds summary loss scalars to graph
         """
        spn_output = self.network.out_layer_spn

        labels = tf.cast(tf.argmax(self.label_ph, axis=1), dtype=tf.int32)
        label_idx = tf.stack([tf.range(self.batch_size), labels], axis=1)
        gen_loss = tf.reduce_mean(-1 * tf.gather_nd(spn_output, label_idx))

        disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.label_ph, logits=spn_output))

        very_gen_loss = -1 * tf.reduce_mean(tf.reduce_logsumexp(spn_output, axis=1))

        cnn_output_dim = self.network.cnn.out_layer_size
        print("CNN Output dimension: ", cnn_output_dim)
        very_gen_loss_norm = very_gen_loss / cnn_output_dim

        balance = 4/5
        mixed_loss = balance * disc_loss + (1 - balance) * very_gen_loss_norm
        loss_name = self.run_config["loss"]
        if loss_name == "very_gen_loss":
            loss = very_gen_loss
        elif loss_name == "gen_loss":
            loss = gen_loss
        elif loss_name == "disc_loss":
            loss = disc_loss
        elif loss_name == "mixed_loss":
            loss = mixed_loss
        else:
            raise Exception(f"Name {loss_name} not found in loss list")

        tf.summary.scalar("very_gen_loss", very_gen_loss)
        tf.summary.scalar("gen_loss", gen_loss)
        tf.summary.scalar("disc_loss", disc_loss)
        tf.summary.scalar("mixed_loss", mixed_loss)

        return loss

    def _get_optimizer(self, loss, name):
        """ Get optimizer for network, given the loss """
        print("Adding Optimizer to graph")
        if name == "fc":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
            optimizer = optimizer.minimize(loss, global_step=self.global_step)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
            optimizer = optimizer.minimize(loss, global_step=self.global_step)
        return optimizer

    def _compile_fc(self, sess):
        """ Adds metrics for fc to tf summary and calculates loss and train optimizer"""
        with tf.variable_scope("fc_metrics", reuse=tf.AUTO_REUSE):
            acc, acc_op = self._add_metrics(self.network.out_layer_fc)
            loss = self.get_loss_fc()
            metrics = {
                "acc": acc,
                "acc_op": acc_op,
                "loss": loss
            }

        train_op = self._get_optimizer(loss, sess)
        return train_op, metrics

    def _compile_spn(self, sess):
        """ Adds metrics for spn to tf summary and calculates loss and train optimizer"""
        with tf.variable_scope("spn_metrics", reuse=tf.AUTO_REUSE):
            acc, acc_op = self._add_metrics(self.network.out_layer_spn)
            loss = self.get_loss_spn()
            metrics = {
                "acc": acc,
                "acc_op": acc_op,
                "loss": loss
            }

        train_op = self._get_optimizer(loss, sess)

        return train_op, metrics

    def run_fc(self, sess):
        """ Initialize and run Fully connected network """
        train_op, metrics = self._compile_fc(sess)
        self._init_vars(sess, "fc")
        self.restore_checkpoint(sess, "fc")

        # Todo: Maybe add loss in fc network?
        if pc.TRAIN["fc"]:
            print("Start Training for fc")
            self._train_fc(train_op, sess, metrics["acc"], metrics["acc_op"])
        return metrics

    def run_spn(self, sess):
        """ Initialize and train spn network """
        train_op, metrics = self._compile_spn(sess)
        self._init_vars(sess, "spn")
        self.restore_checkpoint(sess, "spn")

        # Enable to check normalization values
        # self.check_normalized_values(sess)

        if pc.TRAIN["spn"]:
            print("Start Training for spn")
            self._train_spn(train_op, sess, metrics["loss"], metrics["acc"], metrics["acc_op"])

        return metrics

    def run_spn_with_grad(self, sess):
        """ Initialize and run spn network without the gradient """
        train_op, metrics = self._compile_spn(sess)
        # self.init_vars(sess, "fc") # global vars initiializer
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        self._restore_checkpoint_without_grad(sess)

        if pc.TRAIN["spn"]:
            print("Start Training for spn without grad")
            self._train_spn(train_op, sess, metrics["loss"], metrics["acc"], metrics["acc_op"])

    def _restore_checkpoint_without_grad(self, sess):
        """ """
        cwd = os.getcwd()
        save_path_spn = cwd + "/logs/tmp/spn/"
        save_path_fc_cnn = cwd + "/logs/tmp/fc_cnn/"

        variables_spn = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rat")
        saver_spn = tf.train.Saver(var_list=variables_spn)

        variables_fc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc")
        variables_cnn = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cnn")
        variables_to_save = variables_fc + variables_cnn
        saver_fc_cnn = tf.train.Saver(var_list=variables_to_save)

        last_checkpoint_spn = tf.train.latest_checkpoint(save_path_spn)
        saver_spn.restore(sess, save_path=last_checkpoint_spn)

        last_checkpoint_fc_spn = tf.train.latest_checkpoint(save_path_fc_cnn)
        saver_fc_cnn.restore(sess, save_path=last_checkpoint_fc_spn)

        save_path = cwd + "/logs/checkpoints/" + self.session_name + "/" + "spn"

        self.saver["spn"] = saver_spn
        self.saver["path_spn"] = save_path

        # Delete Folder since it is only temporary
        cwd = os.getcwd()
        shutil.rmtree(cwd + '/logs/tmp/')

    def save_model_tmp(self, sess):
        """ Saves current spn model in a temporary path """
        variables_spn = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rat")
        saver_spn = tf.train.Saver(var_list=variables_spn)

        variables_fc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc")
        variables_cnn = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cnn")
        variables_to_save = variables_fc + variables_cnn
        saver_fc_cnn = tf.train.Saver(var_list=variables_to_save)

        cwd = os.getcwd()
        save_path_spn = cwd + "/logs/tmp/spn/"
        save_path_fc_cnn = cwd + "/logs/tmp/fc_cnn/"
        if not os.path.exists(save_path_spn):
            os.makedirs(save_path_spn)
        else:
            shutil.rmtree(save_path_spn)
            os.makedirs(save_path_spn)
        if not os.path.exists(save_path_fc_cnn):
            os.makedirs(save_path_fc_cnn)
        else:
            shutil.rmtree(save_path_fc_cnn)
            os.makedirs(save_path_fc_cnn)

        saver_spn.save(sess,
                       save_path=save_path_spn,
                       latest_filename=None,
                       global_step=self.global_step)

        saver_fc_cnn.save(sess,
                          save_path=save_path_fc_cnn,
                          latest_filename=None,
                          global_step=self.global_step)

    @staticmethod
    def _init_vars(sess, name):
        """ tf initialization of variables """
        print("Initialize Variables...")
        if name == "fc":
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            return
        if name == "spn":
            # Not able to run global var init here since it destroys previous training
            print("Hard coded initialization, not very nice")

            # initialize adam variables
            adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name]
            sess.run(adam_initializers)

            # initialize trainable vars
            sess.run(tf.local_variables_initializer())
            train_vars_to_init = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rat")
            sess.run(tf.initializers.variables(train_vars_to_init))

            # initialize additional vars
            sess.run(sess.graph.get_tensor_by_name('beta1_power_1/Assign:0'))
            sess.run(sess.graph.get_tensor_by_name('beta2_power_1/Assign:0'))

            # check if there are any more left
            uninitialized_variables = sess.run(tf.report_uninitialized_variables())
            if list(uninitialized_variables):
                print("----- UNINITIALIZED VARIABLES -------")
                print("Uninitialized VARS:", (list(uninitialized_variables)))
        return

    @staticmethod
    def get_cnn_learning_rate(epoch):
        """ Return the learning rate for cnn given current epoch """
        learning_rate = 1e-3
        if epoch > 80:
            learning_rate *= 0.5e-3
        elif epoch > 60:
            learning_rate *= 1e-3
        elif epoch > 40:
            learning_rate *= 1e-2
        elif epoch > 20:
            learning_rate *= 1e-1
        return learning_rate

    @staticmethod
    def get_spn_learning_rate(epoch):
        """ Return the learning rate for spn given current epoch """
        learning_rate = 2e-3
        if epoch > 10:
            learning_rate *= 1e-1
        elif epoch > 20:
            learning_rate *= 1e-2
        elif epoch > 30:
            learning_rate *= 1e-3
        elif epoch > 40:
            learning_rate *= 1e-4
        elif epoch > 50:
            learning_rate *= 1e-5
        elif epoch > 60:
            learning_rate *= 1e-6
        return learning_rate

    @staticmethod
    def define_placeholder(input_size, num_classes, batch_size):
        """ Defines placeholder for input and labels """
        placeholder = {
            "input_ph": tf.placeholder(tf.float32, [batch_size, input_size]),
            "label_ph": tf.placeholder(tf.int32, [batch_size, num_classes])
        }

        return placeholder

    @staticmethod
    def create_session():
        """ Creates a session with additonal parameters """
        # TODO: Add configs in running configs
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=30,
            inter_op_parallelism_threads=30)
        sess = tf.Session(config=session_conf)
        return sess
