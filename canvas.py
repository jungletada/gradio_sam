import cv2
# import gradio as gr
import numpy as np

class Canvas:
    def __init__(self, img, mask, canvas_h, canvas_w, scale_factor, rotation_angle, translation_xy) -> None:
        """
        img: np.ndarray PNG RGB 
        mask: np.ndarray PNG RGB
        sacle_factor: a float in range of [0, 1.0]
        """
        assert img.shape == mask.shape
        self.img = img
        self.mask = mask
        h = img.shape[0]
        w = img.shape[1]
        assert h <= canvas_h and w <= canvas_w
        self.canvas_h = canvas_h
        self.canvas_w = canvas_w
        self.scale_factor = scale_factor
        self.rotation_angle = rotation_angle
        self.translation = translation_xy

    def create_canvas_and_transform_image(self, img_rgb):
        """
        return mask and img on canvas
        img:[h,w,4]
        mask:[h,w]
        """
        img_h, img_w = img_rgb.shape[:2]
        # print(f'Image size ={img_h}x{img_w}')
        # Step 1: Create a canvas
        canvas = np.zeros((self.canvas_h, self.canvas_w, 4), dtype=np.uint8)
        # Step 2: Place img_rgba in the center of the canvas
        start_y, start_x = (self.canvas_h - img_h) // 2, (self.canvas_w - img_w) // 2
        canvas[start_y:start_y+img_h, start_x:start_x+img_w, :] = img_rgb
        # cv2.imwrite('test3.png', canvas)
        # Step 3 & 4: Perform the geometric transformations and superimpose the image
        # Calculate the center of the original image on the canvas
        center = (start_x + img_w // 2, start_y + img_h // 2)
        # Scaling and rotation
        M = cv2.getRotationMatrix2D(center, self.rotation_angle, self.scale_factor)
        # Translation
        M[:, 2] += self.translation  # Adding the translation values to the transformation matrix
        # Apply affine transformation
        transformed_img = cv2.warpAffine(canvas, M, (self.canvas_w, self.canvas_h), borderMode=cv2.BORDER_CONSTANT)
        # cv2.imwrite('test3.png', transformed_img)
        # Step 5: Generate a mask from the alpha channel
        mask = transformed_img[:, :, 3] == 0
        return transformed_img, mask
    
    def find_border(self, mask):
        top = np.argmax(mask.sum(axis=1) > 0)   # 上边界
        bottom = len(mask) - np.argmax(mask[::-1].sum(axis=1) > 0)# 下边界
        left = np.argmax(mask.sum(axis=0) > 0)  # 左边界
        right = len(mask[0]) - np.argmax(mask[:,::-1].sum(axis=0) > 0)  # 右边界

        return top, bottom, left, right


    def get_crop_image_mask(self,):
        """
        return [h,w,4]
        """
        # 保存为带有透明度的四通道BGRA格式：
        # sample_image_np = sample_image_np.copy().astype(np.uint8)
        sample_image_np = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGRA)  # 转换为带有透明度的图像
        mask = np.where(self.mask, 255, 0).astype(np.uint8)
        mask = mask[:, :, 0]
        # cv2.imwrite('test2_mask.png', mask)
        # print(mask.shape)
        # print(np.max(mask, axis=0))
        sample_image_np[:,:,3] = mask  # 设置透明度
        # cv2.imwrite('test2.jpg', sample_image_np)
        # 找到蒙版的边界
        top, bottom, left, right = self.find_border(mask)
        # 根据边界裁剪图像
        cropped_image = sample_image_np[top:bottom, left:right]
        return cropped_image
    
    def get_canvas(self,):
        cropped_image = self.get_crop_image_mask()

        # cv2.imwrite('test3.png', cropped_image)
        print(cropped_image.shape)

        img_geo, mask = self.create_canvas_and_transform_image(cropped_image)

        print(mask.shape)

        mask = np.logical_not(mask)

        color = np.array([1, 1, 1])

        mask_image = mask.reshape(mask.shape[0], mask.shape[1], 1) * color.reshape(1, 1, -1) * 255

        return img_geo, mask_image



img = cv2.imread('./test_image/cup.jpg')
mask = cv2.imread('./cup_mask.png')

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)

# print(mask)

can = Canvas(img=img, mask=mask, canvas_h=800, canvas_w=800, scale_factor=1.0, rotation_angle=40, translation_xy=(0,0))
img, mask = can.get_canvas()

# print(img.shape)
# print(mask_rgb.shape)

cv2.imwrite('cup_rota40.png', img)
cv2.imwrite('cup_rota40_mask.png', mask)

# print(img.shape)
# print(mask_rgb.shape)

# print(mask)

