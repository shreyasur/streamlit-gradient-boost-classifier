
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split


class StHelper:

    def __init__(self,X,y):
        self.X = X
        self.y = y
        # Apply train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


    def train_gradient_boost_classifier(self, loss_function, learning_rate, n_estimators, subsample, criterion,
                                            min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_depth,
                                            min_impurity_decrease, min_impurity_split, init, random_state, max_features,
                                            verbose, max_leaf_nodes, warm_start, validation_fraction, n_iter_no_change,
                                            tol, ccp_alpha):
            gradientboost_clf = GradientBoostingClassifier(
                loss=loss_function, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample,
                criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth,
                min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, init=init,
                random_state=random_state, max_features=max_features, verbose=verbose, max_leaf_nodes=max_leaf_nodes,
                warm_start=warm_start, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change,
                tol=tol, ccp_alpha=ccp_alpha)

            gradientboost_clf.fit(self.X_train, self.y_train)
            y_pred = gradientboost_clf.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)

            return gradientboost_clf, accuracy



    def draw_main_graph(self,gradientboost_clf,ax):

        XX, YY, input_array = self.draw_meshgrid()
        labels = gradientboost_clf.predict(input_array)
        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')



    def draw_meshgrid(self):
        a = np.arange(start=self.X[:, 0].min() - 1, stop=self.X[:, 0].max() + 1, step=0.01)
        b = np.arange(start=self.X[:, 1].min() - 1, stop=self.X[:, 1].max() + 1, step=0.01)

        XX, YY = np.meshgrid(a, b)

        input_array = np.array([XX.ravel(), YY.ravel()]).T

        return XX, YY, input_array

        labels = gradientboost_clf.predict(input_array)
