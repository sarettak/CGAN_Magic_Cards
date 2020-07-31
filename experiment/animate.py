import imageio
import os

root = "experiment_data/"
for subdir, dirs, files in os.walk(root):
    if "fixed" in subdir:
        images = []
        print(subdir)
        for file_ in files:
            if file_.endswith(".jpg"):
                image_path = os.path.join(subdir, file_)
                image = imageio.imread(image_path)
                if "128" in subdir:
                    image = image[:910, :, :] 
                else:
                    image = image[:465, :, :]
                images.append((image_path, image))
        images.sort(key=lambda it: int(it[0].split("fixed_samples")[1].split(".jpg")[0]))
        images = [im for _, im in images]
        imageio.mimsave(os.path.join(subdir, "anim.gif"), images, duration=0.2)
        
