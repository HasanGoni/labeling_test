# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/05_grounding_dino.ipynb.

# %% auto 0
__all__ = ['OpenCvImage', 'load_image', 'BoundingBox', 'DetectionResult', 'msk_to_polygon', 'polygon_to_msk', 'refine_masks',
           'get_boxes', 'detect', 'segment', 'grounding_dino_segmentation', 'get_annotated_img']

# %% ../nbs/05_grounding_dino.ipynb 4
import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple, NewType
from fastcore.basics import *


import cv2
import torch
import requests
import numpy as np
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

# %% ../nbs/05_grounding_dino.ipynb 5
OpenCvImage = NewType('OpenCvImage', np.ndarray)

# %% ../nbs/05_grounding_dino.ipynb 6
from cv_tools.core import *

# %% ../nbs/05_grounding_dino.ipynb 8
def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(
            requests.get(
                image_str, 
                stream=True
                ).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image

# %% ../nbs/05_grounding_dino.ipynb 11
@dataclass
class BoundingBox:

    xmin:int 
    ymin:int 
    xmax:int 
    ymax:int

    @property
    def xyxy(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]

# %% ../nbs/05_grounding_dino.ipynb 12
@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_dict(
        cls, 
        d: Dict[str, Any] # detection dict
        ) -> "DetectionResult":
        return cls(
            score=d["score"],
            label=d["label"],
            box=BoundingBox(
                xmin=d["box"]["xmin"],
                ymin=d["box"]["ymin"],
                xmax=d["box"]["xmax"],
                ymax=d["box"]["ymax"],
            )
        )

# %% ../nbs/05_grounding_dino.ipynb 13
def msk_to_polygon(
        msk: np.ndarray
        ) -> List[List[int]]:
    msk = msk.astype(np.uint8)
    contours, _ = cv2.findContours(
        msk, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour.reshape(-1,2).tolist()



# %% ../nbs/05_grounding_dino.ipynb 14
def polygon_to_msk(
        polygon: List[Tuple[int, int]],
        im_shape: Tuple[int, int]
        ) -> np.ndarray:
    mask = np.zeros(im_shape, dtype=np.uint8)
    pts = np.array(polygon, np.int32)
    cv2.fillPoly(mask, [pts], color=(255,))
    return mask

# %% ../nbs/05_grounding_dino.ipynb 15
def refine_masks(
   msks:torch.BoolTensor,
   polygon_refinement:bool=False
   )->List[np.ndarray]:
   'Generate numpy masks from torch masks'

   msks = msks.cpu().float()
   msks = msks.permute(0, 2, 3, 1)
   msks = msks.mean(axis=-1)
   msks = (msks > 0).int()
   msks = msks.numpy().astype(np.uint8)
   msks = list(msks)

   if polygon_refinement:
      for idx, msk in enumerate(msks):
         shape = msk.shape
         pol = msk_to_polygon(msk)
         poly = polygon_to_msk(pol, shape)
         msks[idx] = poly
   return msks


# %% ../nbs/05_grounding_dino.ipynb 16
def get_boxes(
    results: DetectionResult
             ) -> List[List[List[float]]]:
    'From Detection Result to List of Boxes'    

    return [[i.box.xyxy for i in results]]


# %% ../nbs/05_grounding_dino.ipynb 17
def detect(
        image: Image.Image,
        labels: List[str],
        detector_id:Optional[str],
        threshold: float = 0.5,
        device = "cuda",
        ) -> List[dict[str, Any]]:
    'Use Gronding DINO to detect set of labels in image with Zero shot'

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device
    detector_id = detector_id if detector_id is not None else "IDEA-Research/gronding-dino-tiny"

    object_detector = pipeline(
        model=detector_id,
        task="zero-shot-object-detection",
        device=device)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(
        image,
        candidate_labels=labels,
        threshold=threshold)

    return [DetectionResult.from_dict(i) for i in results]
    

    


# %% ../nbs/05_grounding_dino.ipynb 20
def segment(
        image: Image.Image,
        det_res: List[Dict[str, Any]], # detection results
        device: str = None,
        pol_ref:bool = False, # use polygon refinement
        model_id: Optional[str] = None
    ) -> List[DetectionResult]:
    ' Use SAM from image + bbox'

    # Sometimes I have gpu but memory is very small so I need to use cpu
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device

    segmenter_id = model_id if model_id is not None else "facebook/sam-vit-base"
    segmenter = AutoModelForMaskGeneration.from_pretrained(
                                                            segmenter_id
                                                            ).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)
    boxes = get_boxes(det_res)
    inputs = processor(
        images=image,
        input_boxes=boxes,
        return_tensors="pt"
        ).to(device)

    outputs = segmenter(**inputs)

    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(
        masks, 
        pol_ref)

    for d_r, msk in zip(det_res, masks):
        d_r.mask = msk

    return det_res






# %% ../nbs/05_grounding_dino.ipynb 23
def grounding_dino_segmentation(
        image:Union[Image.Image, str],
        labels:List[str], # labels to detect[what you want to detect],
        threshold:float=0.3,
        polygon_refinement:bool=False,# use polygon refinement
        detector_id:Optional[str]=None, # detector model id
        segmenter_id:Optional[str]=None, # segmenter model id
        device='cpu'
    )->Tuple[np.ndarray, List[DetectionResult]]:
    'Create segmented masks from image and labels'

    if isinstance(image, str):
        image = load_image(image)
    detections = detect(
        image=image,
        labels=labels,
        detector_id=detector_id,
        threshold=threshold,
        device=device
    )
    segmenations = segment(
        image=image,
        device=device,
        det_res=detections,
        pol_ref=polygon_refinement,
        model_id=segmenter_id
    )
    return np.array(image), segmenations




# %% ../nbs/05_grounding_dino.ipynb 26
def get_annotated_img(
    img:Union[Image.Image, OpenCvImage],
    detection_results:List[DetectionResult],    
    )->OpenCvImage:
    'Annotate image based on mask and bounding boxes'


    cv2_img = np.array(img) if isinstance(img, Image.Image) else img
    # don't know why but cv2 uses BGR
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)

    # loop through results
    for i in detection_results:
        label = i.label
        mask = i.mask
        bbox = i.box
        score = i.score

        # random color 
        clr = np.random.randint(0, 255, size=3)

        # adding bounding box
        cv2.rectangle(
            cv2_img,
            pt1=(bbox.xmin, bbox.ymin),
            pt2=(bbox.xmax, bbox.ymax),
            color=clr.tolist(), 
            thickness=2
        )

        # adding text
        cv2.putText(
            cv2_img,
            text=f"{label}:{score:.2f}",
            org=(bbox.xmin, bbox.ymin-10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=clr.tolist(),
            thickness=2
        )
        # adding mask

        if mask is not None:
            msk_uint8 = (mask* 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                msk_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(
                cv2_img, 
                contours=contours, 
                contourIdx=-1, 
                color=clr.tolist(), 
                thickness=2
            )
    return cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)



