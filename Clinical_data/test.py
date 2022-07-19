import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model
import numpy
import M_compare_auc_delong_xu
import unittest
import scipy.stats

class TestIris(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = sklearn.datasets.load_iris()
        x_train, x_test, y_train, cls.y_test = sklearn.model_selection.train_test_split(
            data.data, (data.target == 1).astype(numpy.int), test_size=0.8, random_state=42)
        cls.predictions = sklearn.linear_model.LogisticRegression(solver="lbfgs").fit(
            x_train, y_train).predict_proba(x_test)[:, 1]
        cls.predictions_2 = sklearn.linear_model.LogisticRegression(solver="liblinear").fit(
            x_train, y_train).predict_proba(x_test)[:, 1]
        cls.sklearn_auc = sklearn.metrics.roc_auc_score(cls.y_test, cls.predictions)
        print(cls.sklearn_auc)


    def test_variance_const(self):
        auc, variance = M_compare_auc_delong_xu.delong_roc_variance(self.y_test, self.predictions)
        result = M_compare_auc_delong_xu.delong_roc_test(self.y_test, self.predictions_2, self.predictions)

        numpy.testing.assert_allclose(self.sklearn_auc, auc)
        numpy.testing.assert_allclose(0.0015359814789736538, variance)
        print(variance)
