import os
import numpy as np
from pil import Image

fname = 'galaga_sprite_sheet.png'
image_dir = 'images'
sprite_sheet = Image.open(os.path.join(image_dir, fname))
im_data = np.array(sprite_sheet)
assert im_data.shape == (383, 424, 3)

# Extract data for each sprite from sprite sheet
sprites = []
for yi in range(12):
    y = 55 + yi*24
    for xi in range(8):
        x = 16 + xi*24
        sprite = im_data[y:y+16, x:x+16, :]
        sprites.append(sprite)

# Add a border and save as individual files
for i, sprite in enumerate(sprites):
    img_data = np.zeros((19, 19, 3), dtype='uint8')
    img_data[2:18, 2:18, :] = sprite
    img = Image.fromarray(img_data)
    if sprite.sum() > 0:
        img.save(os.path.join(image_dir, f'galaga_{i:02d}.png'))
