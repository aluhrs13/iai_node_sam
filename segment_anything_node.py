# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb

import torch
import numpy as np

from PIL import Image
from pydantic import Field
from segment_anything import sam_model_registry, SamPredictor
from typing import Literal

from invokeai.app.invocations.baseinvocation import BaseInvocation, InvocationContext
from invokeai.app.invocations.image import ImageField, ImageOutput, ImageType

class SegmentAnythingInvocation(BaseInvocation):
    #fmt: off
    type: Literal["segment_anything"] = "segment_anything"
    x: int = Field(default=0, description="x coordinate of point")
    y: int = Field(default=0, description="y coordinate of point")
    include_exclude: Literal[tuple(["include", "exclude"])] = Field(default="include", description="include or exclude the point")
    encoder_version: Literal[tuple(["vit_h", "vit_l", "vit_b"])] = Field(default="vit_h", description="description")
    sam_checkpoint: str = Field(default="E:\\StableDiffusion\\SegmentAnything\\sam_vit_h_4b8939.pth", description="path to checkpoint file")
    image: ImageField = Field(default=None, description="The image to run inference on.")
    multimask_output: bool = Field(default=False, description="whether to output multiple masks")
    #fmt: on

    def invoke(self, context: InvocationContext) -> ImageOutput:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inc_exc = 1 if self.include_exclude == "include" else 0

        init_image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )

        sam = sam_model_registry[self.encoder_version](checkpoint=self.sam_checkpoint).to(device=device)
        sam_predictor = SamPredictor(sam)
        sam_predictor.set_image(np.array(init_image))

        """
        TODO: Multiple points
        input_point = np.array([[500, 375], [1125, 625]])
        input_label = np.array([1, 1])

        mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )
        """

        """
        TODO: Box Input
        input_box = np.array([425, 600, 700, 875])
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        """

        """
        TODO: Combo boxes and points
        input_box = np.array([425, 600, 700, 875])
        input_point = np.array([[575, 750]])
        input_label = np.array([0])
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
        )
        """

        input_point = np.array([[self.x, self.y]])
        input_label = np.array([inc_exc])

        masks, _, _ = sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=self.multimask_output
        )
        
        #TODO: Actually handle multiple masks
        mask = masks[0]
        mask_pil = Image.fromarray(mask)

        image_type = ImageType.INTERMEDIATE
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )

        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )

        context.services.images.save(image_type, image_name, mask_pil, metadata)
        return ImageOutput(image=ImageField(image_type=image_type, image_name=image_name), width=mask_pil.width, height=mask_pil.height)