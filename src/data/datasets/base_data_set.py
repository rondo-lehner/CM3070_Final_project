import os

from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
import tensorflow_datasets.public_api as tfds

class Builder(tfds.core.GeneratorBasedBuilder):
  """
    Sources: 
        * https://www.tensorflow.org/tutorials/load_data/images
        * https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet
"""

  VERSION = tfds.core.Version("4.9.2")

  def _info(self):
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(),
            "file_name": tfds.features.Text(),
            "segmentation_mask": tfds.features.Image(
                shape=(None, None, 1), use_colormap=True
            ),
        }),
        supervised_keys=("image", "label"),
    )

  def _split_generators(self, dl_manager):
    """Returns splits."""
    # Download images and annotations that come in separate archives.
    # Note, that the extension of archives is .tar.gz even though the actual
    # archives format is uncompressed tar.
    dl_paths = dl_manager.download_and_extract({
        "images": _BASE_URL + "/images.tar.gz",
        "annotations": _BASE_URL + "/annotations.tar.gz",
    })

    images_path_dir = os.path.join(dl_paths["images"], "images")
    annotations_path_dir = os.path.join(dl_paths["annotations"], "annotations")

    # Setup train and test splits
    train_split = tfds.core.SplitGenerator(
        name="train",
        gen_kwargs={
            "images_dir_path": images_path_dir,
            "annotations_dir_path": annotations_path_dir,
            "images_list_file": os.path.join(
                annotations_path_dir, "trainval.txt"
            ),
        },
    )
    test_split = tfds.core.SplitGenerator(
        name="test",
        gen_kwargs={
            "images_dir_path": images_path_dir,
            "annotations_dir_path": annotations_path_dir,
            "images_list_file": os.path.join(annotations_path_dir, "test.txt"),
        },
    )

    return [train_split, test_split]

  def _generate_examples(
      self, images_dir_path, annotations_dir_path, images_list_file
  ):
    with tf.io.gfile.GFile(images_list_file, "r") as images_list:
      for line in images_list:
        image_name, label, species, _ = line.strip().split(" ")

        trimaps_dir_path = os.path.join(annotations_dir_path, "trimaps")

        trimap_name = image_name + ".png"
        image_name += ".jpg"
        label = int(label) - 1
        species = int(species) - 1

        record = {
            "image": os.path.join(images_dir_path, image_name),
            "label": int(label),
            "species": species,
            "file_name": image_name,
            "segmentation_mask": os.path.join(trimaps_dir_path, trimap_name),
        }
        yield image_name, record



class BaseDataSet:
    def __init__(self, data_dir):
        
        self.data_dir = data_dir
        self.data = tf.keras.utils.image_dataset_from_directory(
            self.data_dir + '/train',   # Only the train folder has Ground Truth masks
            labels='inferred',
            label_mode='int',
            class_names=None,
            color_mode='rgb',
            batch_size=32,
            image_size=(256, 256),
            shuffle=True,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation='bilinear',
            follow_links=False,
            crop_to_aspect_ratio=False,
            **kwargs
        )

        
        # load data here

    def next_batch(self, batch_size):
        pass