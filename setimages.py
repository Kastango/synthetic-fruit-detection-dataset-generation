import os
import random
from PIL import Image, ImageChops, ImageStat, ImageDraw
import blend_modes
import numpy
from tqdm import tqdm

class RandomParameters:
    def __init__(self, min_images=0, max_images=35, min_proportion=0.075, max_proportion=0.10, min_depth=75, min_visibility=0.25, hardlight_blend_power=1, overlay_blend_mode=True, num_outputs=2000, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0, dev_mode=True):
        self.min_images = min_images
        self.max_images = max_images
        self.min_proportion = min_proportion
        self.max_proportion = max_proportion
        self.min_depth = min_depth
        self.min_visibility = min_visibility
        # self.z_value = z_value
        self.hardlight_blend_power = hardlight_blend_power
        self.overlay_blend_mode = overlay_blend_mode
        self.num_outputs = num_outputs
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.dev_mode = dev_mode

random_parameters = RandomParameters()

image_folder = 'pictures_trimmed'
background_folder = 'backgrounds'
depth_map_folder = 'backgrounds_map'
light_folder = 'lights'
output_folder = 'new_experiment2_output'
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png'))]

progress_bars = {
    'train': tqdm(total=random_parameters.num_outputs * random_parameters.train_ratio, desc='Train Folder'),
    'val': tqdm(total=random_parameters.num_outputs * random_parameters.val_ratio, desc='Validation Folder'),
    'test': tqdm(total=random_parameters.num_outputs * random_parameters.test_ratio, desc='Test Folder')
}

for output_num in range(random_parameters.num_outputs):
    num_selected_images = random.randint(random_parameters.min_images, random_parameters.max_images)
    selected_images = random.sample(image_files, num_selected_images)
    
    if output_num < random_parameters.num_outputs * random_parameters.train_ratio:
        subfolder = 'train'
    elif output_num < random_parameters.num_outputs * (random_parameters.train_ratio + random_parameters.val_ratio):
        subfolder = 'val'
    else:
        subfolder = 'test'

    background_image_filenames = [f for f in os.listdir(background_folder) if f.endswith(('.jpg', '.JPG'))]
    background_image_filename = random.choice(background_image_filenames)
    background_image_path = os.path.join(background_folder, background_image_filename)
    background_image = Image.open(background_image_path).rotate(-90, expand=True)
    background_image = background_image.resize((1500, 2000), Image.LANCZOS) 

    depth_map_filename = os.path.splitext(background_image_filename)[0] + '_map.jpg'
    depth_map_path = os.path.join(depth_map_folder, depth_map_filename)
    depth_map_image = Image.open(depth_map_path)
    depth_map_image = depth_map_image.resize((1500, 2000), Image.LANCZOS)

    if random_parameters.dev_mode:
        # If dev_mode is enabled, overlay the depth map on the background for debugging
        depth_map_overlay = depth_map_image.convert("RGBA")
        background_with_opacity = Image.blend(background_image.convert("RGBA"), depth_map_overlay, 1).convert("RGB")
    else:
        background_with_opacity = background_image.copy()

    yolo_notation = []

    for image_filename in selected_images:
        random_image_path = os.path.join(image_folder, image_filename)
        random_image = Image.open(random_image_path)
        proportion_to_retain = random.uniform(random_parameters.min_proportion, random_parameters.max_proportion)
        new_width = int(random_image.width * proportion_to_retain)
        new_height = int(random_image.height * proportion_to_retain)
        random_image = random_image.resize((new_width, new_height), Image.LANCZOS)

        while True:
            x_position = random.randint(0, background_with_opacity.width - random_image.width)
            y_position = random.randint(0, background_with_opacity.height - random_image.height)
            depth_pixel = depth_map_image.getpixel((x_position, y_position))
            if depth_pixel[0] > random_parameters.min_depth:
                break

        # Calculate the average color of the region where the picture will be placed
        region = background_with_opacity.crop((x_position, y_position, x_position + random_image.width, y_position + random_image.height))
        stat = ImageStat.Stat(region)
        avg_color = tuple(map(int, stat.mean))

        # Create a new image with the average color
        avg_color_image = Image.new('RGB', random_image.size, avg_color)
        avg_color_image = avg_color_image.convert("RGBA")
        datas = avg_color_image.getdata()

        new_data = []
        for item in datas:
            # Apply the average color only to the not transparent pixels
            if item[3] > 0:
                new_data.append((avg_color[0], avg_color[1], avg_color[2], item[3]))
            else:
                new_data.append(item)
        avg_color_image.putdata(new_data)

        # Blend the image with the average color image
        blended_image = ImageChops.hard_light(random_image, avg_color_image)
        random_image_alpha = random_image.split()[3]
        blended_image.putalpha(random_image_alpha)

        # Blend the original image with the hard light blended image
        random_image = Image.blend(random_image, blended_image, random_parameters.hardlight_blend_power)

        # Apply overlay blend mode if flag is set
        if random_parameters.overlay_blend_mode and random.random() < 0.5:
            # Select a random light image
            light_files = [f for f in os.listdir(light_folder) if f.endswith(('.png'))]
            light_filename = random.choice(light_files)
            light_path = os.path.join(light_folder, light_filename)
            light_image = Image.open(light_path)
            light_image = light_image.resize(random_image.size)

            # Randomize the rotation of the light image
            rotation_angle = random.randint(0, 360)
            light_image = light_image.rotate(rotation_angle)

            # Convert images to numpy arrays
            random_image_np = numpy.array(random_image, dtype='float32')
            light_image_np = numpy.array(light_image, dtype='float32')

            # Apply the overlay filter to the light image with 50% opacity
            light_image_np = blend_modes.overlay(random_image_np, light_image_np, 0.75)

            # Convert back to Image from numpy array
            light_image_np = light_image_np.astype('uint8')
            light_image = Image.fromarray(light_image_np)

            # Paste the light image onto the random image
            random_image.paste(light_image, (0, 0), mask=light_image)

        # Calculate the average depth and standard deviation in the region where the image will be placed
        depth_region = depth_map_image.crop((x_position, y_position, x_position + random_image.width, y_position + random_image.height))
        depth_stat = ImageStat.Stat(depth_region)
        avg_depth = depth_stat.mean[0]
        std_depth = depth_stat.stddev[0]

        # Calculate z_value based on average depth and standard deviation to ensure partial hiding
        z_value = avg_depth + std_depth   # This will ensure that parts of the image are hidden

        total_pixels = random_image.width * random_image.height
        visible_pixels = 0

        # Convert images to numpy arrays for faster processing
        random_image_np = numpy.array(random_image)
        depth_map_np = numpy.array(depth_region)

        # Create a gradient based on the depth values
        visibility_gradient = 1 - ((depth_map_np[:,:,0] - avg_depth) / std_depth)
        visibility_gradient = numpy.clip(visibility_gradient, 0, 1)  # Normalize the gradient between 0 and 1

        # Calculate visibility based on z_value directly on numpy array
        visibility_mask = depth_map_np[:,:,0] <= z_value
        visible_pixels = numpy.sum(visibility_mask)

        # Apply the gradient mask to the alpha channel
        random_image_np[:,:,3] = (random_image_np[:,:,3] * visibility_gradient).astype(numpy.uint8)

        # Convert back to Image from numpy array
        random_image = Image.fromarray(random_image_np)

        if visible_pixels / total_pixels >= random_parameters.min_visibility:
            background_with_opacity.paste(random_image, (x_position, y_position), random_image)
            bx = (x_position + random_image.width / 2) / background_with_opacity.width
            by = (y_position + random_image.height / 2) / background_with_opacity.height
            bw = random_image.width / background_with_opacity.width
            bh = random_image.height / background_with_opacity.height
            yolo_notation.append(f'0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n')

    output_image_folder = os.path.join(output_folder, subfolder, 'image')
    os.makedirs(output_image_folder, exist_ok=True)
    output_image_path = os.path.join(output_image_folder, f'{subfolder}_output_{output_num}.jpg')
    background_with_opacity.save(output_image_path)

    output_txt_folder = os.path.join(output_folder, subfolder, 'labels')
    os.makedirs(output_txt_folder, exist_ok=True)
    output_txt_path = os.path.join(output_txt_folder, f'{subfolder}_output_{output_num}.txt')
    with open(output_txt_path, 'w') as txt_file:
        txt_file.writelines(yolo_notation)

    # Update the progress bar for the current subfolder
    progress_bars[subfolder].update(1)

# Close all progress bars
for pb in progress_bars.values():
    pb.close()

