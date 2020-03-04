import numpy as np


class ProbModel(object):
    """
    Abstract Class about Probabilistic Model
    """

    def __init__(self):
        pass

    def prob(self, samples):
        """
        Calculates the probability of a array of samples
        :param samples:
        :return:
        """
        pass

    def log_prob(self, samples):
        """
        Calculates log probability of array of samples
        :param samples:
        :return:
        """


class BayesNet1(ProbModel):
    """
    BayesNet with 3 Nodes A,B,C in this structure:
    B <- A -> C
    """
    # This does not fit the ProbModel Idea....
    # because of marginalization and stuff

    def __init__(self, parameters):
        # parameters is a dict? with all the parameters and probs as keys
        self.params = parameters

    def p_a(self, b, c):
        # P(A|B=b, C=c)
        top_part = self.params["A"]*self.p_b_a(a=1, b=b)*self.p_c_a(a=1, c=c)
        bottom_part = top_part + (1-self.params["A"])*self.p_b_a(a=0, b=b)*self.p_c_a(a=0, c=c)
        result = top_part/bottom_part
        return result

    def p_b_a(self, a, b=1):
        # P(B|A)
        if a == 0:
            prob = self.params["B"][0]
        elif a == 1:
            prob = self.params["B"][1]
        else:
            raise ValueError("A: ", a)
        if b == 0:
            return 1 - prob
        else:
            return prob

    def p_c_a(self, a, c=1):
        # p(C|A)
        if a == 0:
            prob = self.params["C"][0]
        elif a == 1:
            prob = self.params["C"][1]
        else:
            raise ValueError("A: ", a)
        if c == 0:
            return 1 - prob
        else:
            return prob


class SimpleBernoulli(ProbModel):
    """
    Simple Naive Bernoulli Model with x parameter
    """

    def __init__(self, num_features):
        super().__init__()
        """
        :param num_features: defines number of features
        """
        self.features = None
        self.num_features = num_features
        self.random_init()

    def prob(self, samples):
        """
        :param samples: Array with arrays with Values as 0 or 1
        :return:
        """
        samples = np.array(samples)
        # Check Values are 0 or 1
        if not(np.all(np.where((samples == 0) | (samples == 1), True, False))):
            raise ValueError("Expected sample values 0 or 1 but found: ", samples)

        # Short notation for element wise exponent Only works with numpy arrays!
        prob = np.prod(self.features**samples * (1-self.features)**(1-samples))
        return prob

    def log_prob(self, samples):
        samples = np.array(samples)
        # Check Values are 0 or 1
        if not(np.all(np.where((samples == 0) | (samples == 1), True, False))):
            raise ValueError("Expected sample values 0 or 1 but found: ", samples)

        log_prob = np.sum(np.log(self.features)*samples + np.log(1-self.features)*(1-samples))
        return log_prob

    def random_init(self):
        self.features = np.random.random(self.num_features)

