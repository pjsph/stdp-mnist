import gzip as gz

def get_labeled_data(only = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], train = True):
    """Get MNIST images and labels

    Parameters
    ----------
    only
        Array of numbers to sort the images
    train : bool
        Whether to load training or testing set

    Returns
    -------
    to_return_images
        Array of 28x28 2D arrays storing the grey-levels of the MNIST images
    to_return_images
        Array storing the actual numbers that the images are representing
    """
    if train:
        images_path = 'MNIST/train-images-idx3-ubyte.gz'
        labels_path = 'MNIST/train-labels-idx1-ubyte.gz'
    else:
        images_path = 'MNIST/t10k-images-idx3-ubyte.gz'
        labels_path = 'MNIST/t10k-labels-idx1-ubyte.gz'

    # Read labels
    with gz.open(labels_path, 'rb') as f:
        magic = f.read(4)
        magic = int.from_bytes(magic, 'big')
        print("Magic is:", magic)

        nblab = f.read(4)
        nblab = int.from_bytes(nblab, 'big')
        print("Number of labels:", nblab)

        labels = [f.read(1) for i in range(nblab)]
        labels = [int.from_bytes(label, 'big') for label in labels]

    # Read images
    with gz.open(images_path, 'rb') as f:
        magic = f.read(4)
        magic = int.from_bytes(magic, 'big')
        print("Magic is:", magic)

        nbimg = f.read(4)
        nbimg = int.from_bytes(nbimg, 'big')
        print("Number of images:", nbimg)

        nbrow = f.read(4)
        nbrow = int.from_bytes(nbrow, 'big')
        print("Number of rows:", nbrow)

        nbcol = f.read(4)
        nbcol = int.from_bytes(nbcol, 'big')
        print("Number of columns:", nbcol)

        images = []
        for i in range(nbimg):
            rows = []
            for r in range(nbrow):
                cols = []
                for c in range(nbcol):
                    cols.append(int.from_bytes(f.read(1), 'big'))
                rows.append(cols)
            images.append(rows)

    if images and labels:
        to_return_images = [images[i] for i in range(len(images)) if labels[i] in only]
        to_return_labels = [i for i in labels if i in only]

        return to_return_images, to_return_labels

    return [], []
