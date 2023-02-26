import numpy as np
import cv2
import tensorflow as tf
import os

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_example(image,i):
  #  image = tf.io.decode_jpeg(tf.io.read_file(path))#[::4,::4,:]
   # ts = int(path.split('/')[-1].split('_')[0])
   #img = int(path.split('/')[-1].split('.')[0])



    feature = {
        "image": image_feature(image),
      #  "ts": int64_feature(ts),
        "i": int64_feature(i)
       # "label": bytes_feature(path)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        #"label": tf.io.FixedLenFeature([], tf.string)
        #"ts": tf.io.FixedLenFeature([], tf.int64),
        "i": tf.io.FixedLenFeature([], tf.int64)        
       # "lane": tf.io.FixedLenFeature([], tf.int64)

    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"])
    return example['image'],example['i']


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        dim_size = tf.shape(images)[-1]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        #patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [-1,self.patch_size,self.patch_size, dim_size])

        return patches


def patch(inp,patch_size,keep):

  patches=Patches(patch_size)(inp[0])

  x=tf.gather(patches,indices=keep)
  #c=tf.gather(coords.flatten(),indices=keep)
  
  return x, tf.repeat(inp[1],keep.shape[0])



def mov2tfrecs(path, tfrecs_dir, center, crop_size, imgs_per_rec=64):

  if not os.path.exists(tfrecs_dir):
    os.makedirs(tfrecs_dir)  # creating TFRecords output folder

  vidcap = cv2.VideoCapture(path)
  success,image = vidcap.read()


  crop=crop_size//2


  done=False


  print(crop)



  
  tfrec_count=0
  img_count=0

  while success:

    count = 0
    
    file=tfrecs_dir + 'train_'+str(tfrec_count).zfill(4)+".tfrec"
    print(file)

    with tf.io.TFRecordWriter(file) as writer:
    
      while success and count<imgs_per_rec:

        image=image[center[0]-crop:center[0]+crop,center[1]-crop:center[1]+crop,:]
        #patches=Patches(patch_size)([image])
        #print(patches.shape)

        
        example = create_example(image,img_count)
        writer.write(example.SerializeToString())
        count+=1
        img_count+=1
        success,image = vidcap.read()
        
    tfrec_count+=1

def donut(patch_size, img_size,
      lower_limit=0.55,upper_limit=0.95):

  
  gridsize=img_size//2//patch_size

  coords=np.array([[(i+0.5,j+0.5) for i in range(-gridsize,gridsize)] for j in range(-gridsize,gridsize)])

  norm=np.linalg.norm(coords,axis=2)

  keep_bool=((norm>(gridsize*lower_limit))*(norm<(gridsize*upper_limit)))

  keep=tf.constant(np.where(keep_bool.flatten())[0],dtype=tf.int32)
  

  return coords,keep

def create_mask(coords_valid,keep):
  coords_valid=coords[keep_bool]

  diff=coords_valid[np.newaxis,:,:]-coords_valid[:,np.newaxis,:]
  diff_norm=np.linalg.norm(diff,axis=2)
  mask=diff_norm<3


  return tf.constant(mask,dtype=tf.float32)

def show_patches(p):
  fig = plt.figure(figsize=(8, 80))
  #fig.set_facecolor('black')
  columns = 8
  rows = 8

  #plt.imshow(stack_imgs(p))

  for i in range(len(p)):
      img=p[i]
      
      fig.add_subplot(rows, columns, i+1)

      plt.imshow(img)
      #plt.title(np.where((b[i]==batch_matrix.numpy()).all(axis=1))[0])

      plt.axis('off')


  plt.show()

"""
#def tfrecs2ds(tfrecs,img_size,patch_size,limits):
  img_size=
  gridsize=img_size/2//patch_size

  coords=np.array([[(i+0.5,j+0.5) for i in range(-gridsize,gridsize)] for j in range(-gridsize,gridsize)])
  norm=np.linalg.norm(coords,axis=2)
  keep_bool=((norm>(gridsize*0.55))*(norm<(gridsize*0.95))).flatten()
  keep=tf.constant(np.where(keep_bool.flatten())[0],dtype=tf.int32)
  n_patches=keep_bool.sum()

"""





