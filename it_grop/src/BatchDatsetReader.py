"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import cv2
from PIL import Image
from matplotlib.pyplot import imread
from skimage.transform import resize

class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self.__channels = True
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self.__channels = False
        self.annotations = np.array(
            [np.expand_dims(self._transform(filename['annotation']), axis=2) for filename in self.files])
        #print (self.images.shape)
        #print (self.annotations.shape)

    def _transform(self, filename):
        #image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        #image1 = image*255
        keyword = 'annotations'
        if keyword in filename:
            image = cv2.imread(filename, 0)
        else:
            image = cv2.imread(filename)
        #image = cv2.imread(filename)
        #cv2.imwrite(filename, image1)
        #image = cv2.imread(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            #print (filename)
            #print (image.shape)
            #resize_image = imresize(image,
            #                             [resize_size, resize_size], interp='nearest')
            #resize_image = Image.fromarray(image).resize((resize_size, resize_size), resample=Image.NEAREST)
            resize_image = cv2.resize(image, (resize_size, resize_size), interpolation = cv2.INTER_NEAREST)
            #print (resize_image.shape)
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
