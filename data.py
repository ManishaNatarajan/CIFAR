import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys


def get_data_set(name="train", cifar=10):

    """
    Downloads, extracts, normalizes, transposes, and combines the training and testing CIFAR dataset.

    Input
    ------
    name: (string) "train" or "test" data
    cifar: (int) 10 

    Output:
    -------
    
    x (numpy array): of the flattened images (total number of images, 32*32*3)
                    total number of images = 50000 for train and 10000 for the test

    one_hot_encoding (total number of images, 10): one hot encoded labels
    
    l: all the possible ten labels in the order

    """
    x = None
    y = None
    l = None    

    #Download and extract the data in the specified location
    download_and_extract()

    folder_name = "cifar_10" if cifar == 10 else "cifar_100"

    f = open('./data_set/'+folder_name+'/batches.meta', 'rb')
    datadict = pickle.load(f, encoding='latin1')

    f.close()
    l = datadict['label_names']

    if name is "train":
        for i in range(5):
            f = open('./data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            #Normalizing, reshaping, transposing, and reshaping back training data
            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32]) # Each batch shape is (10000, 3, 32, 32)
            _X = _X.transpose([0, 2, 3, 1]) 
            _X = _X.reshape(-1, 32*32*3)


            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)


    elif name is "test":
        f = open('./data_set/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()


        x = datadict['data']
        y = np.array(datadict['labels'])

        #Normalizing, reshaping, transposing and reshaping back test dataset
        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)


    #One hot encoding
    def dense_to_one_hot(labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        print(index_offset)
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    return x, dense_to_one_hot(y), l


def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def download_and_extract():

    #Main directory 
    main_directory = "./data_set/"
    cifar_10_directory = main_directory+"cifar_10/"

    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory+"./cifar-10-batches-py", cifar_10_directory)
        os.remove(zip_cifar_10)