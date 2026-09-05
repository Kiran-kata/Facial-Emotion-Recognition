import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _as_numpy_array(values):
    if values is None:
        return None
    return np.asarray(values)


def _to_integer_labels(labels):
    labels = _as_numpy_array(labels)
    if labels is None:
        return None
    if labels.ndim == 1:
        return labels.astype(int)
    if labels.ndim > 1 and labels.shape[-1] > 1:
        return labels.argmax(axis = -1).astype(int)
    return labels.astype(int)


def _safe_class_names(class_mapping):
    if isinstance(class_mapping, dict):
        return list(class_mapping.values())
    return list(class_mapping)


def display_majority_predictions(images,
                                 labels,
                                 class_mapping,
                                 label_ps = None):
    '''Plots majority predictions for up to 24 images.
    Works with basis-vector labels and optional predicted distributions.
    '''
    images = _as_numpy_array(images)
    labels = _to_integer_labels(labels)
    label_ps = _as_numpy_array(label_ps)

    if images is None or labels is None:
        raise ValueError('images and labels are required')

    n_images = min(len(images), 24)
    class_names = _safe_class_names(class_mapping)

    plt.figure(figsize = (12, 8))
    for i in range(n_images):
        ax = plt.subplot(4, 6, i + 1)
        plt.imshow(images[i].squeeze(), cmap = 'gray')

        if label_ps is not None and len(label_ps) > i:
            predicted_label = int(np.argmax(label_ps[i]))
            color = 'green' if predicted_label == int(labels[i]) else 'red'
            plt.title(class_names[predicted_label], color = color)
        else:
            plt.title(class_names[int(labels[i])])

        plt.axis('off')

    plt.tight_layout()
    plt.show()


def display_cross_entropy_predictions(images,
                                      labels,
                                      class_mapping,
                                      label_ps = None):
    '''Plots cross-entropy predictions for image/label pairs.
    Supports both true distributions and majority labels.
    '''
    images = _as_numpy_array(images)
    labels = _as_numpy_array(labels)
    label_ps = _as_numpy_array(label_ps)
    class_names = _safe_class_names(class_mapping)

    if images is None or labels is None:
        raise ValueError('images and labels are required')

    max_items = min(len(images), 24)
    plt.figure(figsize = (10.75, 13))
    for i in range(0, max_items, 2):
        index = i // 2
        if index >= min(len(images), len(labels)):
            break

        image = images[index]
        true_label = labels[index]

        ax1 = plt.subplot(6, 4, i + 1)
        ax2 = plt.subplot(6, 4, i + 2)
        ax1.imshow(image.squeeze(), cmap = 'gray')
        ax1.axis('off')
        y_ticks = np.arange(len(class_names))

        if label_ps is not None and len(label_ps) > index:
            true_series = pd.Series(true_label, index = class_names)
            pred_series = pd.Series(label_ps[index], index = class_names)
            pd.DataFrame({'true': true_series, 'predicted': pred_series}).plot.barh(ax = ax2)
            ax2.legend(prop = {'size': 8}, loc = 'lower right')
        else:
            ax2.barh(y_ticks, true_label)

        ax2.set_aspect(0.12)
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels(class_names)
        ax2.set_xlim(0, 1)
        ax2.invert_yaxis()

    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    '''Plots training accuracy and loss from a history dict or Keras History object.'''
    if hasattr(history, 'history'):
        history = history.history

    training_accuracy = history.get('accuracy', history.get('acc', []))
    validation_accuracy = history.get('val_accuracy', history.get('val_acc', []))
    training_loss = history.get('loss', [])
    validation_loss = history.get('val_loss', [])

    epochs_range = range(len(training_accuracy) if training_accuracy else len(training_loss))

    plt.figure(figsize = (8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, training_accuracy, label = 'Training Accuracy')
    plt.plot(epochs_range, validation_accuracy, label = 'Validation Accuracy')
    plt.legend(loc = 'lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, training_loss, label = 'Training Loss')
    plt.plot(epochs_range, validation_loss, label = 'Validation Loss')
    plt.legend(loc = 'upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def plot_confusion_matrix(confusion_matrix, class_names):
    '''Plots confusion matrix.'''
    sns.set_theme(style = 'white')
    plt.figure(1, figsize = (10, 7))
    plt.title('Confusion Matrix')

    ax = sns.heatmap(data = confusion_matrix,
                     annot = True,
                     cmap = 'YlGnBu',
                     cbar_kws = {'label': 'Scale'},
                     fmt = '4d')

    ax.set_xticklabels(class_names, rotation = -30)
    ax.set_yticklabels(class_names, rotation = -30)
    ax.set(ylabel = 'True Label', xlabel = 'Predicted Label')

    plt.show()
