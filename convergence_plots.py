import matplotlib.pyplot as plt


# RGB training/validation accuracy from the fixed-epoch experiment
rgb_train = [
    0.2456359102244389,
    0.39775561097256856,
    0.27556109725685785,
    0.46384039900249374,
    0.5411471321695761,
    0.5635910224438903,
    0.5997506234413965,
    0.6072319201995012,
    0.34538653366583544,
    0.57356608478803
]

rgb_val = [
    0.20202020202020202,
    0.25252525252525254,
    0.18686868686868688,
    0.2727272727272727,
    0.32323232323232326,
    0.29797979797979796,
    0.31313131313131315,
    0.29797979797979796,
    0.19696969696969696,
    0.3181818181818182
]


# Grayscale final validation accuracy
gray_val = [
    0.195625
]


epochs = range(1, len(rgb_train) + 1)


# ------------------------------------------------------------
# RGB convergence
# ------------------------------------------------------------

plt.figure()

plt.plot(
    epochs,
    rgb_train,
    marker="o",
    label="Training accuracy"
)

plt.plot(
    epochs,
    rgb_val,
    marker="o",
    label="Validation accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("RGB Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("rgb_convergence.png")
plt.show()


# ------------------------------------------------------------
# RGB validation accuracy
# ------------------------------------------------------------

plt.figure()

plt.plot(
    epochs,
    rgb_val,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.title("RGB Validation Accuracy Across Epochs")
plt.grid(True)

plt.savefig("rgb_validation_accuracy.png")
plt.show()
