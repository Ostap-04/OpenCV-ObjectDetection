import copy

import numpy as np
import cv2

class SGM():
    def __init__(self):
        self.learning_rate = 0.05
        self.age_cap = 8
        self.diff_threshold = 5  # Threshold for difference to update alpha
        self.blur_ksize = (3, 3)  # Kernel size for GaussianBlur
        self.blur_sigma = 1.5  # Sigma for GaussianBlur

    def initialize_background_model(self, frame):
        self.mu = frame.astype(np.float32)
        self.sigma2 = np.ones_like(frame, dtype=np.float32) * 15
        self.alpha = np.ones_like(frame, dtype=np.float32)

    def update_background_model(self, frame):
        # Calculate M_t (current pixel value)
        M_t = frame

        # Calculate the absolute difference
        diff = np.abs(frame - self.mu)

        # Update age only if the difference is below the threshold
        alpha_increase = 1
        alpha_update_condition = diff < self.diff_threshold
        self.alpha = np.where(alpha_update_condition, np.minimum(self.alpha + alpha_increase, self.age_cap), self.alpha)

        # Update age
        # alpha = np.minimum(alpha + alpha_increase, age_cap)

        rho = self.learning_rate / self.alpha

        # Update mean (mu)
        self.mu = (1 - rho) * self.mu + rho * M_t

        # Update variance (sigma^2)
        V_t = (self.mu - M_t) ** 2
        self.sigma2 = (1 - rho) * self.sigma2 + rho * V_t

        return M_t

    def subtract_background(self, frame):
        frame = frame.astype(np.float32)
        self.update_background_model(frame)
        diff = np.abs(frame - self.mu)
        foreground_mask = diff > 2.5 * np.sqrt(self.sigma2)
        return foreground_mask.astype(np.uint8) * 255


def create_submatrices(matrix, size=(2,2)):
    submatrices = []
    row_count = len(matrix)
    col_count = len(matrix[0])

    for i in range(0, row_count, size[0]):
        for j in range(0, col_count, size[1]):
            submatrix = []
            for sub_i in range(size[0]):
                submatrix_row = []
                for sub_j in range(size[1]):
                    submatrix_row.append(matrix[i + sub_i][j + sub_j])
                submatrix.append(submatrix_row)
            submatrices.append(submatrix)

    return submatrices


def assemble_matrix(submatrices, matrix_height, matrix_width, submatrix_height, submatrix_width):
    # Initialize an empty matrix with black pixels (all zeros)
    matrix = [[0 for _ in range(matrix_width)] for _ in range(matrix_height)]

    # Calculate the number of submatrices per row and column
    submatrices_per_row = matrix_width // submatrix_width
    submatrices_per_col = matrix_height // submatrix_height

    # Iterate through the submatrices
    for index, submatrix in enumerate(submatrices):
        # Calculate the starting row and column for the current submatrix
        row = (index // submatrices_per_row) * submatrix_height
        col = (index % submatrices_per_row) * submatrix_width

        # Place submatrix elements back into the original matrix
        for i in range(submatrix_height):
            for j in range(submatrix_width):
                if row + i < matrix_height and col + j < matrix_width:
                    matrix[row + i][col + j] = submatrix[i][j]

    return matrix

# matrix = [
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
#     [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
#     [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
#     [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96],
#     [97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
#     [121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144],
#     [145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168],
#     [169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192],
#     [193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216],
#     [217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240],
#     [241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264],
#     [265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288],
#     [289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312],
#     [313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336],
#     [337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360]
# ]
#
# result = create_submatrices(matrix, size=(3,4))
# for submatrix in result:
#     print(submatrix)
#
# matrix1 = assemble_matrix2(result, len(matrix), len(matrix[0]), 3, 4)
# print(matrix1)

def resize_frame(frame, submatrix_size=(20,20)):
    height, width = frame.shape[:2]
    new_height = (height // submatrix_size[0]) * submatrix_size[0]
    new_width = (width // submatrix_size[1]) * submatrix_size[1]
    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame


def run_clusterized(cl_size=(20,20)):

    cap = cv2.VideoCapture('../Videos/foggy_mountain.mp4')
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        exit()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = resize_frame(frame, cl_size)
    models = []
    clusters = np.array(create_submatrices(resized_frame, cl_size))
    for cluster in clusters:
        sgm = SGM()
        sgm.initialize_background_model(np.array(cluster))
        models.append(sgm)


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = resize_frame(frame, cl_size)
        clusters = np.array(create_submatrices(resized_frame, cl_size))
        clusters_masks = []
        model_id = 0
        for cluster in clusters:
            mask = models[model_id].subtract_background(cluster)
            clusters_masks.append(mask)
            model_id += 1

        foreground_mask = np.array(assemble_matrix(clusters_masks, resized_frame.shape[0], resized_frame.shape[1], cl_size[0], cl_size[1]))
        resized = cv2.resize(foreground_mask, (1240, 720))
        cv2.imshow('Foreground Mask', resized)
        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


run_clusterized((60, 80))