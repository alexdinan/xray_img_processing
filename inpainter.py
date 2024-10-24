# import required modules
import numpy as np
import cv2


# The Inpainter class code is adapted from
# https://github.com/igorcmoura/inpaint-object-remover
# Orginal code written by Igor Cescon de Moura, MacGregor Winegard
class Inpainter():
    def __init__(self, img, inpaint_mask, patch_size=9, optimisation_factor=3):
        # assign user-provided attributes
        self.img = np.copy(img)
        self.inpaint_mask = np.copy(inpaint_mask)
        self.patch_size = patch_size
        # float > 0 - smaller value => increased optimisation
        self.opt_factor = optimisation_factor

        # get image dimensions/resolution
        self.img_height, self.img_width = img.shape[:2]

        # create source region mask
        self.src_mask = 1 - inpaint_mask

        # initialise confidence/data terms
        self.confidence = 1 - inpaint_mask
        self.data = np.zeros_like(inpaint_mask)

        # assign non-initialised attributes
        self.fill_front = None
        self.priority = None

    def inpaint(self):
        while not self._finished():
            # find fill front -> binary mask: 1 => point on front
            self.fill_front = self._get_fill_front()

            # compute priority values for points on fill front
            self.confidence = self._compute_confidence()
            self.data = self._compute_data()
            self.priority = self.confidence * self.data * self.fill_front

            # get maximum/highest priority point
            target_point = np.unravel_index(np.argmax(self.priority),
                                            self.priority.shape)
            target_patch_box = self._get_patch_box(target_point)

            # find most similar patch from source region
            # self._find_best_src_patch(target_patch_box)
            src_patch_box = self._find_best_src_patch(target_patch_box)

            # fix confidence value for filling region (in-place)
            self._set_confidence(target_point, target_patch_box)
            # update image and masks (in-place)
            self._update_img(src_patch_box, target_patch_box)
        return self.img

    def _get_fill_front(self):
        # find edges of inpaint mask
        laplacian = cv2.Laplacian(self.inpaint_mask, cv2.CV_64F)
        # threshold to create binary edge mask (arbitrarily low number)
        min_val = 0.0001
        _, edge_mask = cv2.threshold(laplacian, min_val, 1, cv2.THRESH_BINARY)
        return edge_mask

    def _compute_confidence(self):
        # create copy of confidence values
        new_confidence = np.copy(self.confidence)

        # get list of points on fill front
        fill_front_pts = np.argwhere(self.fill_front == 1.0)

        for point in fill_front_pts:
            # compute patch box for points
            patch_box = self._get_patch_box(point)
            # get area of patch
            patch_area = ((patch_box[0][1] - patch_box[0][0] + 1) *
                          (patch_box[1][1] - patch_box[1][0] + 1))

            # get patch confidence data -> (n x n) float matrix
            patch_data = self._get_patch_data(self.confidence, patch_box)

            # compute new confidence value
            new_confidence[point[0], point[1]] = np.sum(patch_data)/patch_area
        return new_confidence

    def _compute_data(self):
        # calculate normal and gradient
        norm = self._calc_normal_matrix()
        grad = self._calc_gradient_matrix()

        norm_grad = norm * grad
        # ensure data term is non-zero everywhere
        return np.sqrt(norm_grad[:, :, 0]**2 + norm_grad[:, :, 1]**2) + 0.001

    def _calc_normal_matrix(self):
        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])
        # apply 2D convolution of kernel with inpaint mask to get normal
        x_norm = self.convolve_(self.inpaint_mask, x_kernel)
        y_norm = self.convolve_(self.inpaint_mask, y_kernel)
        normal = np.dstack((x_norm, y_norm))

        # calculate unit normal matrix
        h, w = normal.shape[:2]
        norm = np.sqrt(y_norm**2 + x_norm**2) \
                 .reshape(h, w, 1).repeat(2, axis=2)
        norm[norm == 0] = 1
        return normal / norm

    def _calc_gradient_matrix(self):
        # convert img to grey - set target region to NaN
        greyImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY).astype(float)
        greyImg[self.inpaint_mask == 1] = np.nan

        # calculate gradient of img
        grad = np.nan_to_num(np.array(np.gradient(greyImg)))
        grad_val = np.sqrt(grad[0]**2 + grad[1]**2)

        max_grad = np.zeros((self.img_height, self.img_width, 2))
        # get a list of points on fill front
        fill_front_pts = np.argwhere(self.fill_front == 1)
        for point in fill_front_pts:
            patch_box = self._get_patch_box(point)

            # retrieve gradient data for patch
            x_grad_patch = self._get_patch_data(grad[0], patch_box)
            y_grad_patch = self._get_patch_data(grad[1], patch_box)
            grad_val_patch = self._get_patch_data(grad_val, patch_box)

            # find position of max gradient val in patch
            max_grad_pos = np.unravel_index(
                grad_val_patch.argmax(), grad_val_patch.shape)

            # add data to max_grad matrix
            max_grad[point[0], point[1], 0] = y_grad_patch[max_grad_pos]
            max_grad[point[0], point[1], 1] = x_grad_patch[max_grad_pos]
        return max_grad

    def convolve_(self, input, kernel):
        # flip the kernel vertically / horizontally
        k_flipped = np.flipud(np.fliplr(kernel))

        # apply convolution along rows
        rows_conv = np.apply_along_axis(
            lambda row: np.convolve(row, k_flipped[0], mode="same"),
            axis=1, arr=input)
        # apply convolution across columns
        cols_conv = np.apply_along_axis(
            lambda col: np.convolve(col, k_flipped[:, 0], mode="same"),
            axis=0, arr=rows_conv)
        return cols_conv

    def _find_best_src_patch(self, target_patch_box):
        # get patch box for target
        patch_height = target_patch_box[0][1] - target_patch_box[0][0] + 1
        patch_width = target_patch_box[1][1] - target_patch_box[1][0] + 1

        # get lab version of img for perceptual uniformity
        lab_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)

        best_match = None
        best_diff = np.inf
        # for every possible source patch
        y = 0
        while y < (self.img_height - patch_height + 1):
            x = 0
            best_row_diff = np.inf
            while x < (self.img_width - patch_width + 1):
                # define source patch box
                src_patch_box = [[y, y+patch_height-1], [x, x+patch_width-1]]

                # if source overlaps with inpaint mask -> discard
                if np.sum(self._get_patch_data(self.inpaint_mask,
                                               src_patch_box)) > 0:
                    x += self.patch_size
                    continue

                # calculate SS difference between source/target patch
                diff = self._calc_patch_diff(src_patch_box, target_patch_box,
                                             lab_img)

                # update best patch found in row
                if diff < best_row_diff:
                    best_row_diff = diff
                # compare to global best patch
                if diff > (best_diff * self.opt_factor):
                    # skip adjacent patches - unlikely to be optimal sol
                    x += self.patch_size
                else:
                    if diff < best_diff:
                        best_diff = diff
                        best_match = src_patch_box
                    x += 1

            # update current row/y
            if best_row_diff > (best_diff * self.opt_factor/1.5):
                # no good patch in row => skip adjacent rows
                y += self.patch_size
            else:
                y += 1
        return best_match

    def _calc_patch_diff(self, src_patch_box, target_patch_box, lab_img):
        # get patch data from source mask + create rgb version
        src_mask_data = self._get_patch_data(self.src_mask, target_patch_box)
        src_mask_rgb = self._three_channel(src_mask_data)

        # get image data for target and source - only points in source region
        src_data = self._get_patch_data(lab_img, src_patch_box) * src_mask_rgb
        target_data = (self._get_patch_data(lab_img, target_patch_box) *
                       src_mask_rgb)

        # calculate sum of squared differences
        return np.sum((target_data - src_data) ** 2)

    def _set_confidence(self, point, target_patch_box):
        # get target positions in target region
        positions = np.argwhere(self._get_patch_data(self.inpaint_mask,
                                target_patch_box) == 1) + \
                    [target_patch_box[0][0], target_patch_box[1][0]]

        # fix all confidence values to that of target point
        conf = self.confidence[point[0], point[1]]
        for pos in positions:
            self.confidence[pos[0], pos[1]] = conf

    def _update_img(self, src_patch_box, target_patch_box):
        # mask for regions of patch already filled in.
        mask_rgb = self._three_channel(self._get_patch_data(self.inpaint_mask,
                                                            target_patch_box))
        # get patch image data
        src_data = self._get_patch_data(self.img, src_patch_box)
        target_data = self._get_patch_data(self.img, target_patch_box)

        # compute new data
        new_data = (target_data * (1 - mask_rgb)) + (src_data * mask_rgb)
        # replace data in image
        self._copy_to_patch(self.img, target_patch_box, new_data)

        # fill region in inpaint mask, remove from source mask
        self._copy_to_patch(self.inpaint_mask, target_patch_box, 0)
        self._copy_to_patch(self.src_mask, target_patch_box, 1)

    def _get_patch_box(self, point):
        increment = (self.patch_size - 1) // 2
        # compute (n x n) box centered at point,
        # represented by [[top, bottom],[left,right]]
        patch_box = [[max(0, point[0] - increment),
                      min(self.img_height, point[0] + increment)],
                     [max(0, point[1] - increment),
                      min(self.img_width, point[1] + increment)]]
        return patch_box

    def _finished(self):
        return np.sum(self.inpaint_mask) == 0

    @staticmethod
    def _get_patch_data(data_src, patch_box):
        return data_src[patch_box[0][0]:patch_box[0][1]+1,
                        patch_box[1][0]:patch_box[1][1]+1]

    @staticmethod
    def _three_channel(matrix):
        height, width = matrix.shape
        return matrix.reshape(height, width, 1).repeat(3, axis=2)

    @staticmethod
    def _copy_to_patch(dest, patch, data):
        dest[patch[0][0]:patch[0][1] + 1,
             patch[1][0]:patch[1][1] + 1] = data
