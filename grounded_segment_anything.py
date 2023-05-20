# Apache
# Based on https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/08c6b6118eb203bdeefd154203766f1053d03f5a/automatic_label_simple_demo.py
# Step 1 - Mostly copy/pasting things into the right places:
#    - Imports at the top
#    - Args into inputs
#    - Functions into the class
#    - __main__ into invoke()
#    - add self. to variables that need it.
# python -m pip install -e segment_anything
# python -m pip install -e GroundingDINO
# git submodule update --init --recursive

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

from typing import Literal
from pydantic import Field
from invokeai.app.invocations.baseinvocation import BaseInvocation, InvocationContext
from invokeai.app.invocations.image import ImageField, ImageOutput, ImageType

import cv2
import numpy as np
import supervision as sv
from typing import List

import torch
import torchvision

from .GroundingDINO.groundingdino.util.inference import Model
from .segment_anything.segment_anything import sam_model_registry, SamPredictor

class GroundedSegmentAnythingInvocation(BaseInvocation):
    """Use grounded segment anything to make a mask - https://github.com/IDEA-Research/Grounded-Segment-Anything"""
    #fmt: off
    type: Literal["grounded_segment_anything"] = "grounded_segment_anything"
    config_file: str = Field(default="E:\\StableDiffusion\\GroundingDINO\\GroundingDINO_SwinT_OGC.py", description="path to config file")  # change the path of the model config file
    tag2text_checkpoint: str = Field(default="E:\\StableDiffusion\\Tag2Text\\tag2text_swin_14m.pth", description="path to checkpoint file")
    grounded_checkpoint: str = Field(default="E:\\StableDiffusion\\GroundingDINO\\groundingdino_swint_ogc.pth", description="path to checkpoint filet")
    sam_checkpoint: str = Field(default="E:\\StableDiffusion\\SegmentAnything\\sam_vit_h_4b8939.pth", description="path to checkpoint file")
    image: ImageField = Field(default=None, description="The image to run inference on.")
    split: str = Field(default=",", description="split for text prompt")
    output_dir: str = Field(default="E:\\StableDiffusion", description="output directory")
    box_threshold: float = Field(default=0.25, description="box threshold")
    text_threshold: float = Field(default=0.2, description="text treshold")
    iou_threshold: float = Field(default=0.5, description="iou threshold")
    cpu_only: bool = Field(default=True, description="running on cpu only!, default=False")
    #fmt: on

    def invoke(self, context: InvocationContext) -> ImageOutput:        
        # make dir
        # load image
        initial_image = context.services.images.get(
            # self.image.image_type, self.image.image_name
            "results", "Background1.JPG"
        )


        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {DEVICE}")
        # GroundingDINO config and checkpoint
        GROUNDING_DINO_CONFIG_PATH = self.config_file
        GROUNDING_DINO_CHECKPOINT_PATH = self.grounded_checkpoint

        # Segment-Anything checkpoint
        SAM_ENCODER_VERSION = "vit_h"
        SAM_CHECKPOINT_PATH = self.sam_checkpoint

        # Building GroundingDINO inference model
        grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam_predictor = SamPredictor(sam)


        # Predict classes and hyper-param for GroundingDINO
        SOURCE_IMAGE_PATH = "D:\\StableDiffusion\\Outputs\\images\\results\\Background1.JPG"
        CLASSES = ['unknown', 'door', 'bookshelf', 'picture', 'toy']
        BOX_THRESHOLD = 0.35
        TEXT_THRESHOLD = 0.25
        NMS_THRESHOLD = 0.8

        # load image
        image = cv2.imread(SOURCE_IMAGE_PATH)

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _ 
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        # save the annotated grounding dino image
        cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")

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
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _ 
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # save the annotated grounded-sam image
        cv2.imwrite("grounded_sam_annotated_image.jpg", annotated_image)

        image_type = ImageType.INTERMEDIATE
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )

        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )

        context.services.images.save(image_type, image_name, annotated_image, metadata)
        return ImageOutput(image=ImageField(image_type=image_type, image_name=image_name))