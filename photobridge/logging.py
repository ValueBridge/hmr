"""
Module with logging utilities
"""

import logging

import cv2
import vlogging

import src.util.renderer


def log_mesh_prediction(
        logger: logging.getLogger,
        image_path: src,
        full_resolution_image,
        model_input_image,
        preprocessing_parameters: dict,
        joints,
        vertices,
        perturbed_vertices,
        camera, smpl_face_path):

    camera_for_render, shifted_vertices, _ = src.util.renderer.get_original(
        proc_param=preprocessing_parameters,
        verts=vertices,
        cam=camera,
        joints=joints,
        img_size=model_input_image.shape[:2]
    )

    renderer = src.util.renderer.FullResolutionSMPLRenderer(face_path=smpl_face_path)

    body_render_image_overlay = renderer(
        shifted_vertices,
        cam=camera_for_render,
        full_resolution_image=full_resolution_image,
        model_input_image=model_input_image,
        do_alpha=True)

    _, shifted_perturbed_vertices, _ = src.util.renderer.get_original(
        proc_param=preprocessing_parameters,
        verts=perturbed_vertices,
        cam=camera,
        joints=joints,
        img_size=model_input_image.shape[:2]
    )

    perturbed_body_render_image_overlay = renderer(
        shifted_perturbed_vertices,
        cam=camera_for_render,
        full_resolution_image=full_resolution_image,
        model_input_image=model_input_image,
        do_alpha=True)

    logger.info(
        vlogging.VisualRecord(
            image_path,
            imgs=[
                cv2.cvtColor(full_resolution_image, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(body_render_image_overlay, cv2.COLOR_RGBA2BGR),
                cv2.cvtColor(perturbed_body_render_image_overlay, cv2.COLOR_RGBA2BGR)
            ]
        )
    )
