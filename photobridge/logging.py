"""
Module with logging utilities
"""

import logging
import os

import cv2
import vlogging

import src.util.renderer

def log_mesh_prediction(
        logger: logging.getLogger,
        image_path: src,
        full_resolution_image,
        model_input_image,
        preprocessing_parameters: dict,
        joints, vertices, camera, smpl_face_path):

    cam_for_render, vert_shifted, joints_orig = src.util.renderer.get_original(
        proc_param=preprocessing_parameters,
        verts=vertices,
        cam=camera,
        joints=joints,
        img_size=model_input_image.shape[:2]
    )

    renderer = src.util.renderer.FullResolutionSMPLRenderer(face_path=smpl_face_path)

    rend_img_overlay = renderer(
        vert_shifted,
        cam=cam_for_render,
        full_resolution_image=full_resolution_image,
        model_input_image=model_input_image,
        do_alpha=True)

    logger.info(
        vlogging.VisualRecord(
            image_path,
            imgs=[
                cv2.cvtColor(full_resolution_image, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(rend_img_overlay, cv2.COLOR_RGBA2BGR)
            ]
        )
    )
