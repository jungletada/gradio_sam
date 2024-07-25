import cv2
import gradio as gr
import numpy as np
import torch
# import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from segment_anything import SamPredictor, sam_model_registry

MODEL_DICT = dict(
    vit_h='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',  # yapf: disable  # noqa
    vit_l='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',  # yapf: disable  # noqa
    vit_b='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',  # yapf: disable  # noqa
)




def find_border(mask):
    top = np.argmax(mask.sum(axis=1) > 0)   # 上边界
    bottom = len(mask) - np.argmax(mask[::-1].sum(axis=1) > 0)# 下边界
    left = np.argmax(mask.sum(axis=0) > 0)  # 左边界
    right = len(mask[0]) - np.argmax(mask[:,::-1].sum(axis=0) > 0)  # 右边界

    return top, bottom, left, right


def get_crop_image_mask(sample_image_np, mask):
    # 保存为带有透明度的四通道BGRA格式：
    sample_image_np = sample_image_np.copy().astype(np.uint8)
    sample_image_np = cv2.cvtColor(sample_image_np, cv2.COLOR_RGB2BGRA)  # 转换为带有透明度的图像
    mask = np.where(mask, 255, 0).astype(np.uint8)
    sample_image_np[:,:,3] = mask  # 设置透明度
    # 找到蒙版的边界
    top, bottom, left, right = find_border(mask)
    # 根据边界裁剪图像
    cropped_image = sample_image_np[top:bottom, left:right]
    # 保存裁剪后的结果为带有透明度的PNG格式
    return cropped_image


def show_points(coords: np.ndarray, labels: np.ndarray,
                image: np.ndarray) -> np.ndarray:
    """Visualize points on top of an image.

    Args:
        coords (np.ndarray): A 2D array of shape (N, 2).
        labels (np.ndarray): A 1D array of shape (N,).
        image (np.ndarray): A 3D array of shape (H, W, 3).
    Returns:
        np.ndarray: A 3D array of shape (H, W, 3) with the points
        visualized on top of the image.
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    for p in pos_points:
        image = cv2.circle(
            image, p.astype(int), radius=5, color=(0, 255, 0), thickness=-1)
    for p in neg_points:
        image = cv2.circle(
            image, p.astype(int), radius=5, color=(255, 0, 0), thickness=-1)
    return image


def setup_model() -> SamPredictor:
    """Setup the model and predictor.

    Returns:
        SamPredictor: The predictor.
    """
    model_type = 'vit_h'
    # model_type = 'vit_l'
    device = 'cuda'

    sam = sam_model_registry[model_type](checkpoint='/root/autodl-tmp/ReGe/rege_weights/sam_weights/sam_vit_h_4b8939.pth')
    # sam_weights = torch.load('/root/autodl-tmp/ReGe/rege_weights/sam_weights/sam_vit_l_0b3195.pth')
    # sam.load_state_dict(torch.utils.model_zoo.load_url(MODEL_DICT[model_type]))
    # sam.load_state_dict(sam_weights)
    # sam.half()
    sam.to(device=device)

    predictor = SamPredictor(sam)

    return predictor






def resize_with_aspect_ratio(image, max_length=500):
    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 确定较长的一边
    if height > width:
        new_height = max_length
        new_width = int((new_height / height) * width)
    else:
        new_width = max_length
        new_height = int((new_width / width) * height)

    # 使用cv2.resize函数调整图像大小
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_image



def create_canvas_and_transform_image(img_rgba, canvas_size, scale_factor, rotation_angle, translation_xy):
    H, W = canvas_size
    img_h, img_w = img_rgba.shape[:2]
    # print(f'Image size ={img_h}x{img_w}')
    # Step 1: Create a canvas
    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    # Step 2: Place img_rgba in the center of the canvas
    start_y, start_x = (H - img_h) // 2, (W - img_w) // 2
    canvas[start_y:start_y+img_h, start_x:start_x+img_w, :] = img_rgba
    # Step 3 & 4: Perform the geometric transformations and superimpose the image
    # Calculate the center of the original image on the canvas
    center = (start_x + img_w // 2, start_y + img_h // 2)
    # Scaling and rotation
    M = cv2.getRotationMatrix2D(center, rotation_angle, scale_factor)
    # Translation
    M[:, 2] += translation_xy  # Adding the translation values to the transformation matrix
    # Apply affine transformation
    transformed_img = cv2.warpAffine(canvas, M, (W, H), borderMode=cv2.BORDER_CONSTANT)
    # Step 5: Generate a mask from the alpha channel
    mask = transformed_img[:, :, 3] == 0
    return transformed_img, mask



def show_mask(height:int, width:int, mask: np.ndarray,
              image: np.ndarray,
              ) -> np.ndarray:
    """Visualize a mask on top of an image.

    Args:
        mask (np.ndarray): A 2D array of shape (H, W).
        image (np.ndarray): A 3D array of shape (H, W, 3).
        random_color (bool): Whether to use a random color for the mask.
    Returns:
        np.ndarray: A 3D array of shape (H, W, 3) with the mask
        visualized on top of the image.
    """

    mask = mask[0]
    sample_image_np = np.array(image)
    cropped_image = get_crop_image_mask(sample_image_np, mask)

    height = int(height)
    width = int(width)

    canvas_size = (height, width)  # Example canvas size
    print(canvas_size)
    scale_factor = 0.9  # Example scaling factor
    rotation_angle = 0  # Example rotation angle in degrees
    translation_xy = (10, 60)  # Example translation (x, y)
    img_geo, mask = create_canvas_and_transform_image(cropped_image, canvas_size, scale_factor, rotation_angle, translation_xy)
    img_geo = cv2.cvtColor(img_geo, cv2.COLOR_BGRA2RGBA)

    # cv2.imwrite('./temp.png', img_geo)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(img_geo)
    # plt.show()

    return img_geo, mask

    # if random_color:
    #     color = np.concatenate([np.random.random(3)], axis=0)
    # else:
    #     color = np.array([1, 1, 1])
    #     # color = np.array([30 / 255, 144 / 255, 255 / 255])
    # h, w = mask.shape[-2:]
    # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255

    # image = cv2.addWeighted(image, 0.0, mask_image.astype('uint8'), 1.0, 0)
    # return image



predictor = setup_model()

with gr.Blocks() as demo:

    # Define the UI
    mask_level = gr.Slider(minimum=0, maximum=2, value=1, step=1,\
                           label='Masking level',
                           info='(Whole - Part - Subpart) level')

    canvas_h = gr.Number(label="Enter Height")
    canvas_w = gr.Number(label="Enter Width")


    # with gr.Row():
    input_img = gr.Image(label='Input')
    output_img = gr.Image(label='image')
    output_img1 = gr.Image(label='mask')

    is_positive_box = gr.Checkbox(value=True, label='Positive point')
    reset = gr.Button('Reset Points')

    # Define the logic
    saved_points = []
    saved_labels = []

    def set_image(img) -> None:
        """Set the image for the predictor."""
        # img = resize_with_aspect_ratio(img)

        with torch.cuda.amp.autocast():
            predictor.set_image(img)

    def segment_anything(img, height: int, width: int, mask_level: int, is_positive: bool,
                         evt: gr.SelectData):
        """Segment the selected region."""
        mask_level = 2 - mask_level
        saved_points.append([evt.index[0], evt.index[1]])
        saved_labels.append(1 if is_positive else 0)
        input_point = np.array(saved_points)
        input_label = np.array(saved_labels)

        # Predict the mask
        with torch.cuda.amp.autocast():
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
        # mask has a shape of [3, h, w]

        masks = masks[mask_level:mask_level + 1, ...]

        # Visualize the mask

        # print(img.shape)
        # print(masks.shape)
        # img = resize_with_aspect_ratio(img)
        # print(img.shape)
        # mask_cv2 = np.transpose(masks, (1, 2, 0))
        # print(mask_cv2.shape)
        # mask_cv2 = resize_with_aspect_ratio(mask_cv2)
        # masks = np.transpose(mask_cv2, (2, 0, 1))

        # print(img.shape)
        # print(masks.shape)
        res, mask1 = show_mask(height, width, masks, img)
        print(mask1.shape)
        # res = cv2.cvtColor(res, cv2.COLOR_BGRA2RGBA)

        # mask1 = np.stack([mask1] * 3, axis=-1)

        # h, w = mask1.shape[-2:]
        # color = np.array([1, 1, 1])
        # mask_image = mask1.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255
        mask1 = np.where(mask1, 255, 0).astype(np.uint8)
        mask1 = Image.fromarray(mask1 , 'L')
        # mask = ImageOps.invert(mask)



        # mask_image = cv2.addWeighted(res, 0.0, mask_image.astype('uint8'), 1.0, 0)

        # print('========')
        # Visualize the points
        # res = show_points(input_point, input_label, res)
        # print(res.shape)
        # print(mask1.shape)
        # res = cv2.cvtColor(res, cv2.COLOR_BGRA2RGBA)
        return res, mask1

    def reset_points() -> None:
        """Reset the points."""
        global saved_points
        global saved_labels
        saved_points = []
        saved_labels = []

    # Connect the UI and logic
    input_img.upload(set_image, [input_img])
    input_img.select(segment_anything,
                     [input_img, canvas_h, canvas_w, mask_level, is_positive_box], [output_img, output_img1],)
    reset.click(reset_points)

if __name__ == '__main__':
    demo.launch()
