"""
Module with visualization commands
"""

import invoke


@invoke.task
def visualize_mesh_predictions(_context, config_path):
    """
    Visualize mesh predictions

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import glob
    import numpy as np
    import os

    import box
    import tensorflow as tf
    import skimage.io
    import tqdm

    import photobridge.logging
    import photobridge.utilities
    import src.config
    import src.RunModel
    import demo

    print(src.config.PRETRAINED_MODEL)

    config = photobridge.utilities.read_yaml(config_path)

    model_config = src.config.get_config()
    model_config.load_path = src.config.PRETRAINED_MODEL
    model_config.batch_size = 1

    session = tf.Session()
    model = src.RunModel.RunModel(model_config, sess=session)

    logger = photobridge.utilities.get_logger(config["logging_path"])

    for image_path in tqdm.tqdm(sorted(glob.glob(os.path.join(config["test_data_dir"], "*.jpg")))):

        full_resolution_image = skimage.io.imread(image_path)

        model_input_image, preprocessing_parameters, img = demo.preprocess_image_v2(
            image=full_resolution_image, json_path=None, model_configuration=model_config)

        # Add batch dimension: 1 x D x D x 3
        model_input_image = np.expand_dims(model_input_image, 0)

        # Theta is the 85D vector holding [camera, pose, shape]
        # where camera is 3D [s, tx, ty]
        # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
        # shape is 10D shape coefficients of SMPL
        joints, vertices, cameras, joints3d, theta = model.predict(
            model_input_image, get_theta=True)

        photobridge.logging.log_mesh_prediction(
            logger=logger,
            image_path=image_path,
            full_resolution_image=full_resolution_image,
            model_input_image=img,
            preprocessing_parameters=preprocessing_parameters,
            joints=joints[0],
            vertices=vertices[0],
            camera=cameras[0],
            smpl_face_path=model_config.smpl_face_path
        )
