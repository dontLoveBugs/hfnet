import tensorflow as tf
from pathlib import Path

from .base_dataset import BaseDataset
from hfnet.settings import DATA_PATH


class Aachen(BaseDataset):
    default_config = {
        'load_db': True,
        'load_queries': True,
        'image_names': None,
        'grayscale': True,
        'resize_max': 640,
        'num_parallel_calls': 10,
    }
    dataset_folder = 'aachen/images_upright'

    def _init_dataset(self, **config):
        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
            fn, num_parallel_calls=config['num_parallel_calls'])
        base_path = Path(DATA_PATH, self.dataset_folder)

        if config['image_names'] is not None:
            paths = [Path(base_path, n) for n in config['image_names']]
        else:
            search = []
            if config['load_db']:
                search.append(Path(base_path, 'db'))
            if config['load_queries']:
                search.append(Path(base_path, 'query'))
            assert len(search) != 0
            paths = [p for s in search for p in s.glob('**/*.jpg')]

        data = {'image': [], 'name': []}
        for p in paths:
            data['image'].append(p.as_posix())
            rel = p.relative_to(base_path)
            data['name'].append(Path(rel.parent, rel.stem).as_posix())

        # print('testing data:', data)
        return data

    def _get_data(self, data, split_name, **config):
        def _read_image(path):
            image = tf.read_file(path)
            image = tf.image.decode_png(image, channels=3)
            return image

        def _resize_max(image, resize):
            # print('testing resize max #0')
            target_size = tf.to_float(tf.convert_to_tensor(resize))
            current_size = tf.to_float(tf.shape(image)[:2])
            scale = target_size / tf.reduce_max(current_size)
            new_size = tf.to_int32(current_size * scale)
            # print('testing resize max #1')
            # print('image:', image.shape, 'new size:', new_size)
            # return tf.image.resize_images(
            #     image, new_size, method=tf.image.ResizeMethod.BILINEAR)

            out = tf.image.resize_images(
                image, new_size, method=tf.image.ResizeMethod.BILINEAR)
            # print('testing !!!!!!!!!!!!!!!!!!!!')
            return out

        def _preprocess(data):
            # print('testing preprocess #0')
            image = data['image']
            original_size = tf.shape(image)
            tf.Tensor.set_shape(image, [None, None, 3])
            # print('testing preprocess #1')
            if config['grayscale']:
                image = tf.image.rgb_to_grayscale(image)
            # print('testing preprocess #2')
            if config['resize_max']:
                image = _resize_max(image, config['resize_max'])
            # print('testing preprocess #3')
            data['image'] = image
            data['original_size'] = original_size
            return data

        # print('testing get data #0')
        images = tf.data.Dataset.from_tensor_slices(data['image'])
        # print('testing get data #1')
        images = images.map_parallel(_read_image)
        # print('testing get data #2')
        names = tf.data.Dataset.from_tensor_slices(data['name'])
        # print('testing get data #3')
        dataset = tf.data.Dataset.zip({'image': images, 'name': names})
        # print('testing get data #4')
        dataset = dataset.map_parallel(_preprocess)
        # print('testing get data #5')
        return dataset
