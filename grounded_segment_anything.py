# Apache
# Based on https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/08c6b6118eb203bdeefd154203766f1053d03f5a/automatic_label_simple_demo.py
# python -m pip install -e segment_anything
# python -m pip install -e GroundingDINO

"""
Note: Had to modify line 243 of groundingdino/util/inference.py to:
    class_ids.append(0)
instead of:
    class_ids.append(none)

Because it was tagging some as None and hitting an error:
    "Traceback (most recent call last):
    File "D:\StableDiffusion\InvokeAI\invokeai\app\services\processor.py", line 70, in __process
        outputs = invocation.invoke(
    File "D:\StableDiffusion\InvokeAI\invokeai\app\extensions\iai_node_sam\grounded_segment_anything.py", line 90, in invoke
        labels = [
    File "D:\StableDiffusion\InvokeAI\invokeai\app\extensions\iai_node_sam\grounded_segment_anything.py", line 91, in <listcomp>
        f"{CLASSES[class_id]} {confidence:0.2f}"
    TypeError: list indices must be integers or slices, not NoneType
    "

So index 0 is now always "Unknown" and I think it happens when a box applies to multiple classes?
"""

from PIL import Image as im
from pydantic import Field
from typing import Literal

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision

from invokeai.app.invocations.baseinvocation import BaseInvocation, InvocationContext
from invokeai.app.invocations.image import ImageField, ImageOutput, ImageType
from .GroundingDINO.groundingdino.util.inference import Model
from .segment_anything.segment_anything import sam_model_registry, SamPredictor


class GroundedSegmentAnythingInvocation(BaseInvocation):
    """Use grounded segment anything to make a mask - https://github.com/IDEA-Research/Grounded-Segment-Anything"""
    #fmt: off
    type: Literal["grounded_segment_anything"] = "grounded_segment_anything"
    config_file: str = Field(default="E:\\StableDiffusion\\GroundingDINO\\GroundingDINO_SwinT_OGC.py", description="path to config file")  # change the path of the model config file
    grounded_checkpoint: str = Field(default="E:\\StableDiffusion\\GroundingDINO\\groundingdino_swint_ogc.pth", description="path to checkpoint filet")
    encoder_version: Literal[tuple(["vit_h", "vit_l", "vit_b"])] = Field(default="vit_h", description="description")
    sam_checkpoint: str = Field(default="E:\\StableDiffusion\\SegmentAnything\\sam_vit_h_4b8939.pth", description="path to checkpoint file")
    image: ImageField = Field(default=None, description="The image to run inference on.")
    classes: str = Field(default="wall, door", description="comma separated list of things to classify")
    box_threshold: float = Field(default=0.35, description="box threshold")
    text_threshold: float = Field(default=0.25, description="text treshold")
    nms_threshold: float = Field(default=0.5, description="NMS threshold")
    #fmt: on

    def invoke(self, context: InvocationContext) -> ImageOutput:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        CLASSES = ("unknown,"+self.classes).split(",")

        init_image1 = context.services.images.get(
            self.image.image_type, self.image.image_name
        )

        init_image = np.array(init_image1)

        grounding_dino_model = Model(
            model_config_path=self.config_file, model_checkpoint_path=self.grounded_checkpoint)
        sam = sam_model_registry[self.encoder_version](
            checkpoint=self.sam_checkpoint).to(device=device)
        sam_predictor = SamPredictor(sam)

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=init_image,
            classes=CLASSES,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        # NMS post process
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            self.nms_threshold
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # Prompting SAM with detected boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            return np.array(result_masks)

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # TODO: Save mask without labels separately
        # annotate image with detections
        mask_annotator = sv.MaskAnnotator()
        annotated_image = mask_annotator.annotate(
            scene=init_image.copy(), detections=detections)

        # loop through all the detections and make an image for each
        # TODO: NOT WORKING
        for i in range(len(detections)):
            # save the mask
            image_type = ImageType.INTERMEDIATE
            image_name = context.services.images.create_name(
                context.graph_execution_state_id, self.id +
                "_"+CLASSES[detections.class_id[i]]+"_"
            )

            ret_image = im.fromarray(detections.mask[i])
            context.services.images.save(
                image_type, image_name, ret_image)

        # save the annotated image
        image_type = ImageType.INTERMEDIATE
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )

        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )
        ret_image = im.fromarray(annotated_image)
        context.services.images.save(
            image_type, image_name, ret_image, metadata)
        return ImageOutput(image=ImageField(image_type=image_type, image_name=image_name), width=ret_image.width, height=ret_image.height)
