import numpy as np
import os

import amber
from amber import Amber, AmberSpecifications, DataToParse
from amber.architect import ModelSpace, Operation
import pickle

from tensorflow import keras
from sklearn import datasets, svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import tensorflow as tf
import warnings

tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")


def get_data(data_type="all"):
    digits = datasets.load_digits()
    # Model / data parameters
    num_classes = 10

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, 8, 8, 1))

    # Split data into 80% train and 20% test subsets
    X_trainvalid, X_test, y_trainvalid, y_test = train_test_split(
        data, digits.target, test_size=0.2, shuffle=False, random_state=111
    )

    # Scale images to the [0, 1] range
    X_test = X_test.astype("float32") / 16
    X_trainvalid = X_trainvalid.astype("float32") / 16

    # convert class vectors to binary class matrices
    y_trainvalid = keras.utils.to_categorical(y_trainvalid, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if data_type == "test":
        return X_test, y_test
    elif data_type == "trainvalid":
        return X_trainvalid, y_trainvalid

    # Split into 90% train 10% valid
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainvalid, y_trainvalid, test_size=0.2, shuffle=True, random_state=777
    )
    if data_type == "train":
        return X_train, y_train
    elif data_type == "valid":
        return X_valid, y_valid
    else:
        assert data_type == "all"
        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def get_model_space():
    conv_layers = lambda m: [
        {
            "Layer_type": "conv2d",
            "filters": 16 * m,
            "kernel_size": (1, 1),
            "activation": "relu",
        },
        {
            "Layer_type": "conv2d",
            "filters": 16 * m,
            "kernel_size": (3, 3),
            "activation": "relu",
        },
        {
            "Layer_type": "conv2d",
            "filters": 16 * m,
            "kernel_size": (1, 1),
            "activation": "tanh",
        },
        {
            "Layer_type": "conv2d",
            "filters": 16 * m,
            "kernel_size": (3, 3),
            "activation": "tanh",
        },
        {"Layer_type": "identity"},
    ]

    model_space = ModelSpace.from_dict(
        [
            # conv layer 1
            conv_layers(1),
            # conv layer 2
            conv_layers(2),
            # pooling
            [
                {"Layer_type": "maxpool2d", "pool_size": (2, 2)},
                {"Layer_type": "avgpool2d", "pool_size": (2, 2)},
                {"Layer_type": "identity"},
            ],
            # flatten
            [
                {"Layer_type": "flatten"},
                {"Layer_type": "globalmaxpool2d"},
                {"Layer_type": "globalavgpool2d"},
            ],
            # dropout
            [
                {"Layer_type": "dropout", "rate": 0.5},
                {"Layer_type": "dropout", "rate": 0.1},
                {"Layer_type": "identity"},
            ],
            # dense
            [
                {"Layer_type": "dense", "units": 32, "activation": "relu"},
                {"Layer_type": "dense", "units": 16, "activation": "relu"},
                {"Layer_type": "identity"},
            ],
        ]
    )
    return model_space


def accuracy(y_true, y_score):
    return np.mean(y_true.argmax(axis=1) == y_score.argmax(axis=1))


if os.path.isfile("conv2d_specs.pkl"):
    # reload config file
    with open("conv2d_specs.pkl", "rb") as f:
        amber_config = pickle.load(f)

    # NOTE: if you want to change the content, you can export the dict by d=amber_config.to_dict(), modify the dict,
    # then add it back by amber_config.from_dict(d)
else:
    # First, define the components we need to use
    type_dict = {
        "controller_type": "GeneralController",
        "modeler_type": "KerasModelBuilder",
        "knowledge_fn_type": "zero",
        "reward_fn_type": "SparseCategoricalReward",
        "manager_type": "GeneralManager",
        "env_type": "ControllerTrainEnv",
    }

    model_space = get_model_space()
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data()

    specs = {
        "model_space": model_space,
        "controller": {
            "share_embedding": {1: 0},
            "with_skip_connection": False,
            "num_input_blocks": 1,
            "lstm_size": 32,
            "lstm_num_layers": 1,
            "kl_threshold": 0.01,
            "train_pi_iter": 10,
            "optim_algo": "adam",
            "temperature": 2.0,
            "lr_init": 0.001,
            "tanh_constant": 1.5,
            "buffer_size": 5,
            "batch_size": 5,
        },
        "model_builder": {
            "batch_size": 128,
            "inputs_op": [Operation("input", shape=(8, 8, 1), name="input")],
            "outputs_op": [Operation("dense", units=10, activation="softmax")],
            "model_compile_dict": {
                "loss": "categorical_crossentropy",
                "optimizer": "adam",
            },
        },
        "knowledge_fn": {"data": None, "params": {}},
        "reward_fn": {"method": accuracy},
        "manager": {
            "data": {
                "train_data": DataToParse(obj=get_data, kws={"data_type": "train"}),
                "validation_data": DataToParse(
                    obj=get_data, kws={"data_type": "valid"}
                ),
            },
            "params": {
                "epochs": 25,
                "child_batchsize": 128,
                "store_fn": "minimal",
                "working_dir": "./outputs/",
                "verbose": 0,
            },
        },
        "train_env": {
            "max_episode": 30,
            "max_step_per_ep": 3,
            "working_dir": "./outputs/",
            "time_budget": "00:10:00",  # 5min
            "with_input_blocks": False,
            "with_skip_connection": False,
        },
    }

    amber_config = AmberSpecifications(name="digits", types=type_dict)
    amber_config.from_dict(specs)
    amber_config.save_pickle("conv2d_specs.pkl")

# finally, run program
amb = Amber(types=amber_config.types, specs=amber_config.to_dict())
if not amber.utils.run_from_ipython():
    amb.run()

# evaluation
X_test, y_test = get_data(data_type="test")
X_trainvalid, y_trainvalid = get_data(data_type="trainvalid")

# Reload best searched model
hist = amber.utils.io.read_history(
    [os.path.join(amber_config.manager["params"]["working_dir"], "train_history.csv")],
    metric_name_dict={"accuracy": 1},
)
best_trial = hist.tail(30).sort_values("accuracy", ascending=False).iloc[0]
model_best = keras.models.load_model(
    os.path.join(
        amber_config.manager["params"]["working_dir"],
        "weights",
        "trial_%i" % best_trial.ID,
        "bestmodel.h5",
    )
)
model_best.fit(X_trainvalid, y_trainvalid, batch_size=128, epochs=50, verbose=0)

models_rand = []
for seed in range(5):
    random_trial = hist.head(30).sample(random_state=seed)
    model_rand = keras.models.load_model(
        os.path.join(
            amber_config.manager["params"]["working_dir"],
            "weights",
            "trial_%i" % random_trial.ID,
            "bestmodel.h5",
        )
    )
    model_rand.fit(X_trainvalid, y_trainvalid, batch_size=128, epochs=50, verbose=0)
    models_rand.append(model_rand)


# Baseline - SVM
X_trainvalid = X_trainvalid.squeeze().reshape((len(X_trainvalid), 64))
y_trainvalid = y_trainvalid.argmax(axis=1)
svm = svm.SVC(gamma="auto", random_state=111)
svm.fit(X_trainvalid, y_trainvalid)

# Baseline - Logistic
lr = LogisticRegression(C=50.0 / len(X_trainvalid), penalty="l1", solver="saga", tol=0.1, random_state=111)
lr.fit(X_trainvalid, y_trainvalid)

# Predict the value of the digit on the test subset
print("baseline - logistic")
predicted = lr.predict(X_test.squeeze().reshape((len(X_test), -1)))
print(metrics.classification_report(y_test.argmax(axis=1), predicted))

print("baseline - SVM")
predicted = svm.predict(X_test.squeeze().reshape((len(X_test), -1)))
print(metrics.classification_report(y_test.argmax(axis=1), predicted))


print("-" * 10)
print("amber - rand ensemble 5 of first 30")
predicted = np.mean(np.array([m.predict(X_test) for m in models_rand]), axis=0)
print(metrics.classification_report(y_test.argmax(axis=1), predicted.argmax(axis=1)))

print("-" * 10)
print("amber - best")
predicted = model_best.predict(X_test)
print(metrics.classification_report(y_test.argmax(axis=1), predicted.argmax(axis=1)))
