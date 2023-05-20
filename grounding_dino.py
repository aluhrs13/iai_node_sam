# Step-by-step migrating GroundingDINO from https://github.com/IDEA-Research/GroundingDINO/blob/main/demo/inference_on_a_image.py
# Reqs:pip install nw-groundingdino
# Step 1 - Basic Files.
from typing import Literal, Union
from pydantic import Field
from .baseinvocation import BaseInvocation, InvocationContext
from .image import ImageField, ImageOutput, ImageType

# Step 2 - Take libraries straight from the demo. We don't need all of them, and we'll clean them up later.
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from torchvision import transforms


class SegmentedGrayscale(object):
    def __init__(self, image: Image, heatmap: torch.Tensor):
        self.heatmap = heatmap
        self.image = image

    def to_grayscale(self, invert: bool = False) -> Image:
        return self._rescale(
            Image.fromarray(
                np.uint8(255 - self.heatmap *
                         255 if invert else self.heatmap * 255)
            )
        )

    def to_mask(self, threshold: float = 0.5) -> Image:
        discrete_heatmap = self.heatmap.lt(threshold).int()
        return self._rescale(
            Image.fromarray(np.uint8(discrete_heatmap * 255), mode="L")
        )

    def to_transparent(self, invert: bool = False) -> Image:
        transparent_image = self.image.copy()
        # For img2img, we want the selected regions to be transparent,
        # but to_grayscale() returns the opposite. Thus invert.
        gs = self.to_grayscale(not invert)
        transparent_image.putalpha(gs)
        return transparent_image

    # unscales and uncrops the 352x352 heatmap so that it matches the image again
    def _rescale(self, heatmap: Image) -> Image:
        size = (
            self.image.width
            if (self.image.width > self.image.height)
            else self.image.height
        )
        resized_image = heatmap.resize(
            (size, size), resample=Image.Resampling.LANCZOS)
        return resized_image.crop((0, 0, self.image.width, self.image.height))


class GroundingDinoInvocation(BaseInvocation):
    """GroundingDINO - https://github.com/IDEA-Research/GroundingDINO"""
    #fmt: off
    type: Literal["grounding_dino"] = "grounding_dino"

    # Step 3 - The demo takes in a bunch of parameters, we'll take them in as inputs to our node. Future steps will adjust these to make more sense.
    config_file: str = Field(default="E:\\StableDiffusion\\GroundingDINO\\GroundingDINO_SwinT_OGC.py", description="Path to the model config file")
    checkpoint_path: str = Field(default="E:\\StableDiffusion\\GroundingDINO\\groundingdino_swint_ogc.pth", description="Path to the GroundingDINO checkpoint file.")
    image: ImageField = Field(default=None, description="The image to run inference on.")
    text_prompt: str = Field(default="The black cat.", description="The input prompt")
    output_dir: str = Field(default="E:\\StableDiffusion", description="Path to output image to")
    box_threshold: float = Field(default=0.3, description="Box threshold")
    text_threshold: float = Field(default=0.25, description="Text threshold")
    cpu_only: bool = Field(default=True, description="Run on CPU only")
    mask_count: int = Field(default=1, description="Number of masks to generate")
    #fmt: on

    # Step 2 - Take all the helper functions straight from the demo.

    def plot_boxes_to_image(self, image_pil, tgt):
        H, W = tgt["size"]
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        assert len(boxes) == len(
            labels), "boxes and labels must have same length"

        draw = ImageDraw.Draw(image_pil)
        mask = Image.new("L", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        drawn_count = 0

        # draw boxes and masks
        for box, label in zip(boxes, labels):
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            # draw
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
            # draw.text((x0, y0), str(label), fill=color)

            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), str(label), font)
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (x0, y0, w + x0, y0 + h)
            # bbox = draw.textbbox((x0, y0), str(label))
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), str(label), fill="white")

            if (drawn_count < self.mask_count):
                mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)
                drawn_count += 1

        return image_pil, mask

    def load_model(self, model_config_path, model_checkpoint_path, cpu_only=False):
        args = SLConfig.fromfile(model_config_path)
        args.device = "cuda" if not cpu_only else "cpu"
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model

    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True, cpu_only=False):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = "cuda" if not cpu_only else "cpu"
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(
                    pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    # Step 4 - copy everything else from __main__ into here. Fix a lot of "self." references.
    # Step 5 - Change to outputting a mask.
    # Step 6 - Change to inputting from a previous node.
    def invoke(self, context: InvocationContext) -> ImageOutput:
        # make dir
        os.makedirs(self.output_dir, exist_ok=True)

        # load image
        initial_image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )

        image_pil = initial_image.copy()

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)

        # load model
        model = self.load_model(self.config_file, self.checkpoint_path,
                                cpu_only=self.cpu_only)

        # visualize raw image
        image_pil.save(os.path.join(self.output_dir, "raw_image.jpg"))

        # run model
        boxes_filt, pred_phrases = self.get_grounding_output(
            model, image, self.text_prompt, self.box_threshold, self.text_threshold, cpu_only=self.cpu_only
        )

        # visualize pred
        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        # import ipdb; ipdb.set_trace()
        image_with_box, output_mask = self.plot_boxes_to_image(
            image_pil, pred_dict)

        image_with_box.save(os.path.join(self.output_dir, "pred.jpg"))

        # Step 5 - Save the mask to the output directory for debugging, and pass it to services.
        # Services stuff stolen from MaskFromAlphaInvocation in image.py
        output_mask.save(os.path.join(self.output_dir, "mask.jpg"))

        image_type = ImageType.INTERMEDIATE
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )

        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )

        context.services.images.save(
            image_type, image_name, output_mask, metadata)
        return ImageOutput(image=ImageField(image_type=image_type, image_name=image_name))
