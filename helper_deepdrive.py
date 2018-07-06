import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(img_dir, gt_dir, img_shape):
    """
    Generate function to create batches of training data
    :return:
    """
    image_shape = img_shape
    image_paths = glob(img_dir+'/*.jpg')
    label_paths = {os.path.basename(path)[:-13]+'.jpg': path
                   for path in glob(gt_dir+'/*.png')}
    
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(
                    scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(
                    scipy.misc.imread(gt_image_file), image_shape)
                    
                gt_bg = np.zeros(
                    [image_shape[0], image_shape[1]], dtype=bool)
                gt_list = []
                for label in [13,0]: #auto background
                    gt = gt_image == label
                    gt_list.append(gt)
                    gt_bg = np.logical_or(gt_bg, gt)

                gt_image = np.dstack(
                    [np.invert(gt_bg), *gt_list]).astype(np.float32)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn

def generate_output(sess, logits, keep_prob, image, image_pl, image_shape):
    road_color = [0,255,0,80] # last value is for transperancy
    car_color = [255,0,0,80]
    
    # Run image through graph and perform softmax activation on logits
    im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_pl: [image]})
    
    # Softmax currently has shape (batch, imshape[0]*imshape[1], numclasses) or (1, 92160, 2)
    # We reshape it into the same as the image in order to paste it
    im_softmax_car = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    im_softmax_road = im_softmax[0][:, 2].reshape(image_shape[0], image_shape[1])
    
    # Any value greater then 0.75 returns as True (1) and then is multiplied by an array of RGBA where 
    # A is the transperancy. Chosen color of green for the mask
    segmentation_road = (im_softmax_road > 0.75).reshape(image_shape[0], image_shape[1], 1)
    mask_road = np.dot(segmentation_road, np.array([road_color]))
    segmentation_car = (im_softmax_car > 0.75).reshape(image_shape[0], image_shape[1], 1)
    mask_car = np.dot(segmentation_car, np.array([car_color]))
    
    #RGB and with transparency mask
    mask_road = scipy.misc.toimage(mask_road, mode="RGBA")
    mask_car = scipy.misc.toimage(mask_car, mode="RGBA")
    street_im = scipy.misc.toimage(image) # Convert image numpy array to PIL image in order to paste mask
    
    street_im.paste(mask_road, box=None, mask=mask_road)
    street_im.paste(mask_car, box=None, mask=mask_car)
    
    return street_im

def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
        
    for image_file in glob(os.path.join(data_folder,'*.jpg')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        
        output_img = generate_output(sess, logits, keep_prob, image, image_pl, image_shape)

        yield os.path.basename(image_file), np.array(output_img)


def save_inference_samples(runs_dir, test_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, test_dir, image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
