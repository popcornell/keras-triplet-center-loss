from sklearn.manifold import TSNE
import numpy as np
import matplotlib.patheffects as PathEffects
import seaborn as sns
import matplotlib.pyplot as plt


def scatter(x, labels, subtitle=None):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    if subtitle != None:
        plt.suptitle(subtitle)

    plt.savefig(subtitle)


def tsne_plot(x_train, x_test, y_train, y_test, name):
    x_train_flat = x_train.reshape(-1, 784)
    x_test_flat = x_test.reshape(-1, 784)

    tsne = TSNE()
    train_tsne_embeds = tsne.fit_transform(x_train_flat[:512])
    scatter(train_tsne_embeds, y_train[:512], "Samples from {} Data".format(name))

    eval_tsne_embeds = tsne.fit_transform(x_test_flat[:512])
    scatter(eval_tsne_embeds, y_test[:512], "Samples from {} Data".format(name))