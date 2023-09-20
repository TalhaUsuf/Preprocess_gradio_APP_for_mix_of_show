# What this app does

**Mix of show** is used for multi character generation using **fused lora**. For generation of multiple characters the
model needs images to be preprocessed in a peculiar way:
 - openpose images of multiple characters within an image
 - detection bboxes of each person scaled according to the combined pose image


Each of the bboxes_i will be associated with the charcter lora trigger word (see the Mix of show repo.)
Poses will condition the image generation of each character in place

![](msedge_6upBIrn207.gif)
