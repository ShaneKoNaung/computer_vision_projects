from pathlib import Path
import cv2

def get_overlay_images(fld):
    '''
    params:
        fld : folder path of the images
    Returns:
        list of opencv images
    '''
    img_fld = Path(fld)
    img_path_list = list(img_fld.iterdir())


    overlay_img_list = []
    for img_path in img_path_list:
        image = cv2.imread(str(img_path))
        overlay_img_list.append(image)

    return overlay_img_list