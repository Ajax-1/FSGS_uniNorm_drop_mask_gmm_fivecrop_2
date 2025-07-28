from PIL import Image

def remove_lower_diagonal(image_path, output_path):
    """
    以左下角为原点，按左上-右下连线分割，线下部分变为透明
    """
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size
    pixels = img.load()

    # 斜率
    k = -(height - 1) / (width - 1)
    b = height - 1

    for x in range(width):
        for y in range(height):
            # 注意：PIL坐标系y向下递增，左下角为(0, height-1)
            # 需要将y轴反向
            y_from_bottom = height - 1 - y
            if y_from_bottom <= k * x + b:
                r, g, b_, a = pixels[x, y]
                pixels[x, y] = (r, g, b_, 0)

    img.save(output_path)

if __name__ == "__main__":
    # input_path = "../dataset/dataset/LLFF_4/trex/depth_maps/depth_DJI_20200223_163654_571.png"
    input_path="./output/LLFF_res4/trex/train/ours_3100/beforegmm_midas_depth/DJI_20200223_163654_571.png"
    output_path = "./output/LLFF_res4/trex/train/ours_3100/beforegmm_midas_depth/processed_image.png"
    remove_lower_diagonal(input_path, output_path)