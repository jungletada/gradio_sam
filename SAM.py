from segment_anything import SamPredictor, sam_model_registry
import cv2
# import gradio as gr
import numpy as np
import torch
# import matplotlib.pyplot as plt


class SAM:
    def __init__(self, path, model_type, points, ispos, mask_level) -> None:
        """
        path: the path of image
        model_type: one of ('vit_h', 'vit_l', 'vit_b')
        points: [[x1,y1],[x2,y2],]
        ispos:[1,0,]
        """

        assert path is not None
        self.path = path
        # load image
        img = cv2.imread(self.path)
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert model_type in ('vit_h', 'vit_l', 'vit_b')
        self.model_type = model_type
        self.points = np.array(points)
        self.ispos = np.array(ispos)
        assert self.points.shape[0] == self.ispos.shape[0]
        assert mask_level < 2 and mask_level >= 0
        self.mask_level = mask_level
        #load sam
        self.sam_predictor = self.setup_model()
        self.set_image()

    def setup_model(self,) -> SamPredictor:
        """Setup the model and predictor.

        Returns:
            SamPredictor: The predictor.
        """

        device = 'cuda'
        sam = sam_model_registry[self.model_type](checkpoint='/root/autodl-tmp/ReGe/rege_weights/sam_weights/sam_vit_h_4b8939.pth')
        sam.to(device=device)

        predictor = SamPredictor(sam)

        return predictor
    
    def set_image(self,):
        with torch.cuda.amp.autocast():
            self.sam_predictor.set_image(self.img)
    
    def segment_anything(self,):
        """Segment the selected region."""
        mask_level = 2 - self.mask_level
        points = np.array(self.points)
        label = np.array(self.ispos)

        with torch.cuda.amp.autocast():
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=points,
                point_labels=label,
                multimask_output=True,
            )
        
        # print(masks[2])
        
        # mask has a shape of [3, h, w]
        masks = masks[mask_level:mask_level + 1, ...]
        # print(masks.shape)

        mask_img = self.show_mask(masks)
        img_with_mask = self.show_img(self.img, masks)

        # print(mask_img.shape)
        # print(img_with_mask.shape)

        # plt.imshow(mask_img)
        # plt.show()

        return mask_img, img_with_mask

    def show_mask(self, mask: np.ndarray):
        """
        return: np.ndarry shape:[h,w,3]
        """
        # print(np.bincount(mask))
        mask = np.where(mask == 0, 1, 0)
        color = np.array([1, 1, 1])
        # color = np.array([0, 0, 0])
        h, w = mask.shape[-2:]
        # cv2.bitwise_not(mask)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255
        return mask_image
    
    def show_img(self, img: np.ndarray, mask: np.ndarray, scale = 1.):
        color = np.array([30 / 255, 144 / 255, 255 / 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255

        img = cv2.addWeighted(img, 1-scale, mask_image.astype('uint8'), scale, 0)
        return img
        

sam = SAM(path='/root/autodl-tmp/qyd/gradio_sam/Interactive-SAM-with-Gradio/ball.png', model_type='vit_h',points=[[260,256],[261,256]], ispos=[1,1],mask_level=0)

mask,img = sam.segment_anything()

print(mask.shape)
print(img.shape)

# cv2.imwrite('coffeemachine.png', img)
cv2.imwrite('ball_mask.png', mask)


