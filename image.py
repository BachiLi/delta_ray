import numpy as np
import OpenEXR
import Imath
import skimage
import skimage.io
import imageio

def imwrite(img, filename):
    if (filename[-4:] == '.exr'):
        img_r = img[:, :, 0]
        img_g = img[:, :, 1]
        img_b = img[:, :, 2]
        pixels_r = img_r.astype(np.float16).tostring()
        pixels_g = img_g.astype(np.float16).tostring()
        pixels_b = img_b.astype(np.float16).tostring()
        HEADER = OpenEXR.Header(img.shape[1], img.shape[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])
        exr = OpenEXR.OutputFile(filename, HEADER)
        exr.writePixels({'R': pixels_r, 'G': pixels_g, 'B': pixels_b})
        exr.close()
    else:
        skimage.io.imsave(filename, np.power(np.clip(img, 0.0, 1.0), 1.0/2.2))

def imread(filename):
    if (filename[-4:] == '.exr'):
        file = OpenEXR.InputFile(filename)
        dw = file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        redstr = file.channel('R', pt)
        red = np.fromstring(redstr, dtype = np.float32)
        red.shape = (size[1], size[0]) 
        greenstr = file.channel('G', pt)
        green = np.fromstring(greenstr, dtype = np.float32)
        green.shape = (size[1], size[0]) 
        bluestr = file.channel('B', pt)
        blue = np.fromstring(bluestr, dtype = np.float32)
        blue.shape = (size[1], size[0]) 
        return np.stack([red, green, blue], axis=-1).astype(np.float32)
    elif (filename[-4:] == '.hdr'):
        return imageio.imread(filename)
    else:
        im = skimage.io.imread(filename)
        if im.ndim == 2:
            im = np.stack([im, im, im], axis=-1)
        elif im.shape[2] == 4:
            im = im[:, :, :3]
        return np.power(\
            skimage.img_as_float(im).astype(np.float32), 2.2)