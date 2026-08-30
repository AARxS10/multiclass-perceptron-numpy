import matplotlib.pyplot as plt


# RGB validation accuracy
rgb_results = {
    "none": 0.346875,
    "minmax": 0.339375,
    "zscore": 0.3825
}


# Grayscale validation accuracy
gray_results = {
    "none": 0.195625,
    "minmax": 0.210000,
    "zscore": 0.216875
}


# --------------------------------------------------
# Normalisation comparison
# --------------------------------------------------

methods = ["none", "minmax", "zscore"]

rgb_accuracy = [
    rgb_results[method]
    for method in methods
]

gray_accuracy = [
    gray_results[method]
    for method in methods
]


plt.figure()

plt.plot(
    methods,
    rgb_accuracy,
    marker="o",
    label="RGB"
)

plt.plot(
    methods,
    gray_accuracy,
    marker="o",
    label="Grayscale"
)

plt.xlabel("Normalisation method")
plt.ylabel("Validation accuracy")
plt.title("Effect of Normalisation on Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("normalisation_comparison.png")
plt.show()


# --------------------------------------------------
# Learning-rate comparison
# --------------------------------------------------

learning_rates = [0.01, 0.1, 1.0]

rgb_lr_accuracy = [0.346875, 0.346875, 0.346875]
gray_lr_accuracy = [0.195625, 0.195625, 0.195625]


plt.figure()

plt.plot(
    learning_rates,
    rgb_lr_accuracy,
    marker="o",
    label="RGB"
)

plt.plot(
    learning_rates,
    gray_lr_accuracy,
    marker="o",
    label="Grayscale"
)

plt.xlabel("Learning rate")
plt.ylabel("Validation accuracy")
plt.title("Effect of Learning Rate on Validation Accuracy")
plt.xscale("log")
plt.legend()
plt.grid(True)

plt.savefig("learning_rate_comparison.png")
plt.show()