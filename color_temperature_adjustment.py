from scipy.interpolate import UnivariateSpline

def _create_LUT_8UC1(self, x, y):
    spl = UnivariateSpline(x, y)
    return spl(xrange(256))


class CoolingFilter:

    def __init__(self):
        self.incr_ch_lut = _create_LUT_8UC1([0, 64, 128, 192, 256],
          [0, 70, 140, 210, 256])
        self.decr_ch_lut = _create_LUT_8UC1([0, 64, 128, 192, 256],
          [0, 30,  80, 120, 192])

    def render(self, img_rgb):

        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.decr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.incr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # decrease color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.decr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)
        
        
class WarmingFilter:

    def __init__(self):
        self.incr_ch_lut = _create_LUT_8UC1([0, 64, 128, 192, 256],
          [0, 70, 140, 210, 256])
        self.decr_ch_lut = _create_LUT_8UC1([0, 64, 128, 192, 256],
          [0, 30,  80, 120, 192])

    def render(self, img_rgb):
        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.decr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))


        c_b = cv2.LUT(c_b, decrChLUT).astype(np.uint8)

        # increase color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, 
            cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.incr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), 
            cv2.COLOR_HSV2RGB)

        
        
#Apply the filter and adjust the temperature of the image