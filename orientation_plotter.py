import numpy as np
import tfplus


class OrientationPlotter(tfplus.utils.VideoPlotter):

    def __init__(self, filename=None, name=None, cmap='Greys',
                 max_num_frame=9, max_num_col=9):
        super(OrientationPlotter, self).__init__(filename=filename,
                                                 name=name, cmap=cmap,
                                                 max_num_frame=max_num_frame,
                                                 max_num_col=max_num_col)
        self.color_wheel = np.array([[255, 17, 0],
                                     [255, 137, 0],
                                     [230, 255, 0],
                                     [34, 255, 0],
                                     [0, 255, 213],
                                     [0, 154, 255],
                                     [9, 0, 255],
                                     [255, 0, 255]], dtype='uint8')
        pass

    def build_orientation_img(self, d, y):
        """
        Args:
            d: [B, T, H, W, 8] or [B, H, W, 8] or [H, W, 8]
            y: [B, T, H, W, 1] or [B, H, W, 1] or [H, W, 1]
        """
        cw = self.color_wheel
        did = np.argmax(d, -1)
        yshape = np.array(y.shape)
        yshape[-1] = 3
        c2 = cw[did.reshape([-1])].reshape(yshape)
        img = (c2 * y).astype('uint8')
        return img

    def listen(self, results):
        """Plot results.

        Args:
            images: [B, H, W] or [B, H, W, 3]
        """
        fg = results['foreground']
        orient = results['orientation']
        img = self.build_orientation_img(orient, fg)
        self.plot(img)
        self.register()
        pass
    pass

tfplus.utils.listener.register('orientation_video', OrientationPlotter)
