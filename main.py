import pickle
from PIL import Image
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def train_and_test_set():
    """
    upload the train and test sets
    :return: x train and test (images), y train and test (labels)
    """
    batch1 = unpickle('./cifar-10-batches-py/data_batch_1')
    batch2 = unpickle('./cifar-10-batches-py/data_batch_2')
    batch3 = unpickle('./cifar-10-batches-py/data_batch_3')
    batch4 = unpickle('./cifar-10-batches-py/data_batch_4')
    batch5 = unpickle('./cifar-10-batches-py/data_batch_5')
    test = unpickle('./cifar-10-batches-py/test_batch')

    images = np.concatenate([batch1[b'data'], batch2[b'data'], batch3[b'data'], batch4[b'data'], batch5[b'data']],
                            axis=0)
    labels = np.concatenate(
        [batch1[b'labels'], batch2[b'labels'], batch3[b'labels'], batch4[b'labels'], batch5[b'labels']], axis=0)

    test_images = np.concatenate([test[b'data']], axis=0)
    test_labels = np.concatenate([test[b'labels']], axis=0)

    x_train = np.empty((1024, images.shape[0]))
    for i in range(images.shape[0]):
        single_img = np.array(images[i])
        single_img_reshaped = np.transpose(np.reshape(single_img, (3, 32, 32)), (1, 2, 0))
        img = Image.fromarray(single_img_reshaped.astype('uint8'))
        grey_img = img.convert('L')
        single_grey_array = np.array(grey_img)
        x_train[:, i] = single_grey_array.flatten('F')
    mean_x_train = np.reshape(np.mean(x_train, axis=1), (-1, 1))
    x_train = x_train - mean_x_train

    x_test = np.empty((1024, test_images.shape[0]))
    for i in range(test_images.shape[0]):
        single_img = np.array(test_images[i])
        single_img_reshaped = np.transpose(np.reshape(single_img, (3, 32, 32)), (1, 2, 0))
        img = Image.fromarray(single_img_reshaped.astype('uint8'))
        grey_img = img.convert('L')
        single_grey_array = np.array(grey_img)
        x_test[:, i] = single_grey_array.flatten('F')
    mean_x_test = np.reshape(np.mean(x_test, axis=1), (-1, 1))
    x_test = x_test - mean_x_test
    return x_train, labels, x_test, test_labels


def knn_error(x_train, y_train, x_test, y_test, k_num):
    """
    calculates the error of knn for every k value in k_num between x_train and x_test
    :param x_train: the train images, every col is a different image
    :param y_train: the train labels
    :param x_test: the text images, every col is a different image
    :param y_test: the train labels
    :param k_num: an array of k value that represent amount of k-nearest neighbors we use
    :return: error of the knn for every k in k_num
    """

    distance = np.zeros((x_test.shape[1], x_train.shape[1]))
    errors = np.zeros(len(k_num))
    for j in range(x_test.shape[1]):
        dist = x_train - x_test[:, j].reshape(-1, 1)
        distance[j] = np.linalg.norm(dist, axis=0)
    for k_place, k in enumerate(k_num):
        predicted_labels = np.zeros(x_test.shape[1])
        for i, dist_for_z in enumerate(distance):
            sorted_idx = np.argsort(dist_for_z)
            nearest_k_idx = sorted_idx[:k]
            nearest_k_labels = y_train[nearest_k_idx]
            count_labels = np.bincount(nearest_k_labels) #works like histo
            predicted_labels[i] = np.argmax(count_labels)
        cnt = 0
        for j in range(len(predicted_labels)):
            if predicted_labels[j] != y_test[j]:
                cnt = cnt + 1
        errors[k_place] = cnt / len(predicted_labels)
    return errors


def pca(x_train, y_train, x_test, y_test, s_num, k_num):
    """
    calculate the projection matrix onto x_train and x_test separately and
    calculate the error of knn between x_train and x_test for every k value in k_num
    :param x_train: the train images, every col is a different image
    :param y_train: the train labels
    :param x_test: the text images, every col is a different image
    :param y_test: the train labels
    :param s_num: an array of s value, when s is the amount of the main components we need to take
    :param k_num: an array of k value that represent amount of k-nearest neighbors we use
    :return: error of knn for every k value in k_num
    """

    U, _, _ = np.linalg.svd(x_train, full_matrices=False, compute_uv=True)
    errors = np.zeros((len(s_num), len(k_num)))
    for i, s in enumerate(s_num):
        Us = U[:, :s]
        UsT = Us.T
        prj_x = np.matmul(UsT, x_train)
        prj_z = np.matmul(UsT, x_test)
        errors[i] = knn_error(prj_x, y_train, prj_z, y_test, k_num)
    return errors


def main():
    s_num = [5, 10, 50, 100, 200]
    k_num = [3, 5, 10, 50, 100]
    x_train, y_train, x_test, y_test = train_and_test_set()
    errors_with_pca = pca(x_train, y_train, x_test, y_test, s_num, k_num)
    errors_without_pca = knn_error(x_train, y_train, x_test, y_test, k_num)
    for i in range(len(s_num)):
        for j in range(len(k_num)):
            print(f'Error for  s = {s_num[i]} and k = {k_num[j]} is {errors_with_pca[i, j]}')
    print("without pca the errors are")
    for i in range(len(k_num)):
        print(f'Error for k = {k_num[i]} is {errors_without_pca[i]}')


if __name__ == "__main__":
    main()
