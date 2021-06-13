import streamlit as st

import numpy as np
import data_helper
import matplotlib.pyplot as plt
from sthelper import StHelper
import webbrowser
from sklearn.datasets import make_moons
from sklearn.model_selection import  train_test_split

##st.title('Gradient Boost Classification')
#url='https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'


concentric,linear,outlier,spiral,ushape,xor = data_helper.load_dataset()


# configure matplotlib styling
plt.style.use('seaborn-bright')

# Dataset selection dropdown
st.sidebar.markdown("# Gradient Boost Classifier")

if st.sidebar.button('HOW TO RUN'):
    st.sidebar.text('1.All the parameters have been set to \n their default values,to increase/decrease\n/input the values use the interactive widgets given below. \n'
                     '2.Once done, click on Run Algorithm\nbutton.\n'
                     '3.If the accuracy is more than or equal to 0.90\nthen you will get a celebratory balloon\n show.\n'
                     '4.To again set the default values\n, reload the page.\n'
                     '5.For more details about the parameters\n click on Documentation button')

#if st.sidebar.button('DOCUMENTATION'):
#   webbrowser.open_new_tab(url)

# dataset
dataset_options=st.sidebar.radio('Choose Dataset',('use generated dataset','use toy dataset'))

if dataset_options=='use generated dataset':
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


    def draw_meshgrid():
        a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
        b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

        XX, YY = np.meshgrid(a, b)

        input_array = np.array([XX.ravel(), YY.ravel()]).T

        return XX, YY, input_array


    # Load initial graph
    fig, ax = plt.subplots()

    # Plot initial graph
    ax.scatter(X.T[0], X.T[1], c=y, cmap='Spectral')

    orig = st.pyplot(fig)

elif dataset_options=='use toy dataset':
    dataset = st.sidebar.selectbox(
        "Dataset",
        ("U-Shaped", "Linearly Separable", "Outlier", "Two Spirals", "Concentric Circles", "XOR"), index=3
    )
    st.header(dataset)
    fig, ax = plt.subplots()
    # Plot initial graph
    df = data_helper.load_initial_graph(dataset, ax)
    orig = st.pyplot(fig)

    # Extract X and Y
    X = df.iloc[:, :2].values
    y = df.iloc[:, -1].values



# loss function

loss_function= st.sidebar.radio("loss ", ('deviance', 'exponential'))

#learning rate

learning_rate = st.sidebar.slider("learning_rate", min_value=0., max_value=1., step=0.01,value=0.1)


st.sidebar.text('Selected: {}'.format(learning_rate))

#number of estimators

n_estimators = int(st.sidebar.number_input('n_estimators',min_value=0,value=100,step=10))

#number of subsamples

subsample=st.sidebar.slider("subsample",min_value=0.,max_value=1.0,step=0.1,value=1.0)

st.sidebar.text('Selected: {}'.format(subsample))

#criterion

criterion=st.sidebar.radio("criterion",('friedman_mse','mse'))

#min_samples_split
min_samples_split=st.sidebar.number_input('min_samples_split',value=2,step=1)

#min_samples_leaf
min_samples_leaf=st.sidebar.number_input('min_samples_leaf',value=1,step=1)

#min_weight_fraction_leaf
min_weight_fraction_leaf=st.sidebar.slider('min_weight_fraction_leaf',min_value=0.0,max_value=1.0,value=0.0,step=0.1)
st.sidebar.text('Selected: {}'.format(min_weight_fraction_leaf))
#max_depth

max_depth=int(st.sidebar.number_input('max_depth',value=3,step=1))

#min_impurity_decrease
min_impurity_decrease=float(st.sidebar.number_input('min_impurity_decrease',min_value=0.0,step=0.1))

#min_impurity_split
min_impurity_split_option=st.sidebar.radio('min_impurity_split',(None,'set a value'))
if min_impurity_split_option=='set a value':
    min_impurity_split=float(st.sidebar.number_input('set the value',min_value=0.0,step=0.1))
else:
    min_impurity_split=None

#init
init_option=st.sidebar.radio('init',(None,'zero','estimator'))
if init_option=='estimator':
   init=str(st.sidebar.text_input('enter the name of the estimator'))
elif(init_option=='zero'):

   init='zero'
else:
   init=None

#random_state
random_state_option=st.sidebar.radio('random_state',(None,'set a value'))

if random_state_option=='set a value':
    random_state=int(st.sidebar.number_input('set the value',min_value=0,step=1))
else:
    random_state=None

#max_features

max_feature_option=st.sidebar.radio("max_features",(None,'auto','sqrt','log2','set a value in int','set a value in float'))

if(max_feature_option=='set a value in int'):
    max_features=int(st.sidebar.number_input("choose number of feature",min_value=1,step=1,value=2))
    #st.sidebar.text('Selected: {}'.format(max_features))
elif(max_feature_option=='set a value in float'):
    max_features=float(st.sidebar.number_input("choose fraction of feature",min_value=0.0,max_value=1.0,step=0.1,value=1.0))
elif(max_feature_option=='auto'):
    max_features='auto'
elif(max_feature_option=='sqrt'):
    max_features='sqrt'
elif(max_feature_option=='log2'):
    max_features='log2'
else:
    max_features=None

#verbose
verbose=int(st.sidebar.number_input('verbose',min_value=0,max_value=2,step=1,value=0))

#max_leaf_nodes

max_leaf_nodes_options=st.sidebar.radio('max_leaf_nodes',(None,'set a value'))
if max_leaf_nodes_options=='set a value':

    max_leaf_nodes=int(st.sidebar.number_input('set the value',min_value=1,max_value=32,step=1))
else:
    max_leaf_nodes=None

#warm_start

warm_start=st.sidebar.radio("warm_start",('False','True'))

#validation_fraction

validation_fraction=st.sidebar.slider('validation_fraction',min_value=0.0,max_value=1.0,step=0.1,value=0.1)
st.sidebar.text('Selected: {}'.format(validation_fraction))



#n_iter_no_change
n_iter_no_change_options=st.sidebar.radio('n_iter_no_change',(None,'set a value'))
if n_iter_no_change_options=='set a value':
    n_iter_no_change=int(st.sidebar.number_input('set the value',min_value=1,step=1))
else:
    n_iter_no_change=None

#tol
tol=float(st.sidebar.number_input('tol',min_value=0.,value=0.0001,step=0.0001,format='%.4f'))

#ccp_alpha
ccp_alpha=float(st.sidebar.number_input('ccp_alpha',min_value=0.0,step=0.1))

# Create sthelper object
sthelper = StHelper(X, y)




# On button click
if st.sidebar.button("RUN ALGORITHM"):



    gradientboost_clf,accuracy= sthelper.train_gradient_boost_classifier(loss_function,learning_rate,n_estimators,subsample,criterion,min_samples_split,min_samples_leaf,min_weight_fraction_leaf,max_depth,min_impurity_decrease,min_impurity_split,init,random_state,max_features,verbose,max_leaf_nodes,warm_start,validation_fraction,n_iter_no_change,tol,ccp_alpha)

    sthelper.draw_main_graph(gradientboost_clf,ax)
    orig.pyplot(fig)






    # plot accuracies


    st.sidebar.header("Classification Metrics")
    st.sidebar.text("Gradient Boost Classifier accuracy:" + str(accuracy))


    if accuracy>=0.90:
       st.balloons()







