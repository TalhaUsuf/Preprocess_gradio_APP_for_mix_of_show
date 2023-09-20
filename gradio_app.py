import gradio as gr
import torch
from PIL import Image
from controlnet_aux import OpenposeDetector
import cv2
import numpy as np



# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to('cuda')
open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
open_pose.to('cuda')


def scale_bbox(bbox, original_dims, new_dims):
    """
    Scale bounding box according to new dimensions.

    Parameters:
    - bbox: List of [xmin, ymin, xmax, ymax]
    - original_dims: Tuple of (original_width, original_height)
    - new_dims: Tuple of (new_width, new_height)

    Returns:
    - Scaled bounding box as [xmin', ymin', xmax', ymax']
    """
    w_ratio = new_dims[0] / original_dims[0]
    h_ratio = new_dims[1] / original_dims[1]

    return [
        bbox[0] * w_ratio,  # xmin'
        bbox[1] * h_ratio,  # ymin'
        bbox[2] * w_ratio,  # xmax'
        bbox[3] * h_ratio   # ymax'
    ]



def _generate(input_image : Image.Image):
    print(f"got {input_image}")
    input_image.save('inp.jpg')
    input_image = 'inp.jpg'
    # Inference of person detector, boxes are of the same size as the input image
    results = model([input_image])

    # get the processed pose image, it resizes the image so pose maps are not of the
    # same size as the input image
    original_image = Image.open(input_image).convert('RGB')
    processed_image_open_pose = open_pose(original_image)



    # original predictions of boxes (not scaled according to the
    df = results.pandas().xyxy[0]  # img1 predictions (pandas)

    # scale the bboxes according to the openpose mask image
    df['bbox_new'] = df.apply(
        lambda row: scale_bbox([row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                               (original_image.size[0], original_image.size[1]),
                               (processed_image_open_pose.size[0], processed_image_open_pose.size[1])), axis=1)
    # drop the previous columns
    df.drop(columns=['xmin', 'ymin', 'xmax', 'ymax'], inplace=True)

    # plot the boxes to see if the bboxes are scaled correctly
    processed_array = np.array(processed_image_open_pose)

    for k in range(df.shape[0]):
        bbox1 = df.iloc[k]['bbox_new']
        # plot the bbox1 [xmin, ymin, xmax, ymax]
        rectangled = cv2.rectangle(processed_array,
                                   (int(bbox1[0]), int(bbox1[1])),
                                   (int(bbox1[2]), int(bbox1[3])),
                                   (0, 255, 0),
                                   2)

    # get the boxes
    # Convert each float to its nearest integer
    data = df['bbox_new'].tolist()
    beautiful_data = [[int(round(val)) for val in sublist] for sublist in data]

    # Convert the list of lists to a string representation
    boxes_as_str = str(beautiful_data)


    return processed_image_open_pose, boxes_as_str, Image.fromarray(rectangled), df


gr.Interface(
    fn=_generate,
    inputs=[gr.Image(label="input image with multiple characters", type='pil', source='upload')],
    outputs=[
                gr.Image(label="Pose images use with MixofShow"),
                gr.Textbox(label="bboxes  use with MixofShow"),
                gr.Image(label="Pose with bbox for validation : FOR DEBUGGING"),
                gr.Dataframe(label="bboxes - scaled according to pose-image : FOR DEBUGGING")],
    examples=[
        ['4-people.jpg'],
        ['3people.jpg'],
    ],
    title='Generate the preprocessing masks for Mix Of Show Multi-Character generation',
    description="""
    
    # Mix of Show Multi-Character generation
    mix of show repo solves the problem of multiple character generation using single fused Lora with promising results hence
    user can guide the image generation by conditioning using:
     - pose image of each character
     - bounding box of each character given in the prompt so each character will be rendered within its bbox
    For more details see the Mix of Show repo.
    
    *This repo. is for genrating the preprocessing masks for use in inference*
    """

).queue().launch()