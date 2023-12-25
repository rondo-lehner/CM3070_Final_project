"""deep_globe_2018 dataset."""

import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for deep_globe_2018 dataset."""

  
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Run `make download_deep_globe` in this project.
    Specify custom manual_dir: `tfds_build --manual_dir=/workspaces/CM3070_Final_project/data/external/` 
    """
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(deep_globe_2018): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(),
            "file_name": tfds.features.Text(),
            "segmentation_mask": tfds.features.Image(
                shape=(None, None, 1), use_colormap=True
            )
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # Makes use of manual path as specified in tf guide: https://www.tensorflow.org/datasets/add_dataset#manual_download_and_extraction
    # Specify paths similar to oxford_iiit_pet dataset: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/oxford_iiit_pet/oxford_iiit_pet_dataset_builder.py
    images_path_dir = dl_manager.manual_dir / "images"
    annotations_path_dir = dl_manager.manual_dir / "annotations"
    

    # TODO(deep_globe_2018): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'train_imgs'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(deep_globe_2018): Yields (key, example) tuples from the dataset
    for f in path.glob('*.jpeg'):
      yield 'key', {
          'image': f,
          'label': 'yes',
      }
