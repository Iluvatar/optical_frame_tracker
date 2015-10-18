import cv2
import numpy as np
import time


class KalmanFilter:
    def __init__(self, confidence):
        self.raw_positions = []
        self.adjusted_positions = []
        self.confidence = confidence

    def add_raw(self, pos):
        self.raw_positions.append(pos)

    def add_adj(self, pos):
        self.adjusted_positions.append(pos)

    @staticmethod
    def _average(points):
        points = np.array(points)
        x = np.sum(points[:, 0]) / len(points)
        y = np.sum(points[:, 1]) / len(points)
        return np.array([x, y])

    def stability(self):
        if len(self.raw_positions) > 10:
            i = self.raw_positions[-10:]
            avg = self._average(i)
            diff = 0
            for p in i:
                diff += np.linalg.norm(np.array(p) - avg) ** 2
            return pow(diff, .5)
        else:
            return 0.

    def _is_still(self):
        if len(self.raw_positions) > 10:
            i = self.raw_positions[-10:]
            avg = self._average(i)
            for p in i:
                if np.linalg.norm(np.array(p) - avg) > 40:
                    return False
            return True
        else:
            return False

    def _get_predicted_position(self):
        if self._is_still():
            return self.raw_positions[-1]
        elif len(self.raw_positions) > 0:
            return self.raw_positions[-1]
        else:
            return 0, 0

    def next_pos(self, curr_pos):
        if curr_pos is not None:
            stability = min(self.stability(), 1000)
            conf_adjustment = (stability / 1000. + 2 * self.confidence) / 4
            p = self._get_predicted_position()
            ret = (int(round(p[0] * conf_adjustment + curr_pos[0] * (1 - conf_adjustment))),
                   int(round(p[1] * conf_adjustment + curr_pos[1] * (1 - conf_adjustment))))
        else:
            ret = tuple(np.round(self._get_predicted_position()).astype(np.uint8))

        self.add_adj(ret)
        return ret


class GrayImageFinder:
    def __init__(self, template, filter_confidence=.9, min_count=5):
        self.template = template

        self.orb = cv2.ORB_create()
        self.filter = KalmanFilter(filter_confidence)

        self.FLANN_INDEX_KDTREE = 0
        self.MIN_MATCH_COUNT = min_count

    @staticmethod
    def get_center(points):
        p = np.array(points)
        x = np.sum(p[:, 0, 0]) / len(points)
        y = np.sum(p[:, 0, 1]) / len(points)
        x = int(round(x))
        y = int(round(y))
        return x, y

    def find_image(self, scene):
        kp1, des1 = self.orb.detectAndCompute(self.template, None)
        kp2, des2 = self.orb.detectAndCompute(scene, None)

        index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        cv2.imshow('1', self.template)

        if des1 is None or des2 is None:
            return None

        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)

        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:

                h, w = self.template.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, m)

                self.filter.add_raw(self.get_center(dst))
                return self.filter.next_pos(self.get_center(dst))

        return None


class ImageFinder:
    def __init__(self, template, filter_confidence=.9, min_count=5):
        self.red_finder = GrayImageFinder(template[:, :, 2], filter_confidence, min_count)
        self.blue_finder = GrayImageFinder(template[:, :, 1], filter_confidence, min_count)
        self.green_finder = GrayImageFinder(template[:, :, 0], filter_confidence, min_count)

    @staticmethod
    def get_center(points):
        p = np.array(points)
        x = np.sum(p[:, 0]) / len(points)
        y = np.sum(p[:, 1]) / len(points)
        x = int(round(x))
        y = int(round(y))
        return x, y

    def find_image(self, scene):
        red_pos = self.red_finder.find_image(scene[:, :, 2])
        blue_pos = self.blue_finder.find_image(scene[:, :, 1])
        green_pos = self.green_finder.find_image(scene[:, :, 0])

        found_points = []
        found_points.append(red_pos) if red_pos is not None else False
        found_points.append(blue_pos) if blue_pos is not None else False
        found_points.append(green_pos) if green_pos is not None else False

        if len(found_points) > 0:
            return self.get_center(found_points)
        else:
            return None


def set_new_template(template):
    global finder1, finder2

    new = template.copy()
    gray_new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    finder1 = GrayImageFinder(gray_new, .7)
    # finder2 = ImageFinder(new, .7)


cam = cv2.VideoCapture(0)
finder1 = GrayImageFinder(np.fliplr(cv2.imread('test.jpg', 0)), .8, 10)
finder2 = ImageFinder(np.fliplr(cv2.imread('test.jpg')), .8)

time.sleep(2)

while True:
    e, scene = cam.read()
    # scene = cv2.resize(scene, (0, 0), fx=0.5, fy=0.5)
    gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    p1 = finder1.find_image(gray_scene)
    # p2 = finder2.find_image(scene)

    radius = 100
    if finder1.filter.stability() < 800 and p1 is not None and p1[0] - radius >= 0 and p1[1] - radius >= 0 \
            and p1[0] + radius < scene.shape[1] and p1[1] + radius < scene.shape[0]:
        tmp = scene[p1[1] - radius:p1[1] + radius, p1[0] - radius:p1[0] + radius, :]
        # cv2.rectangle(scene, (p1[0] - radius, p1[1] - radius), (p1[0] + radius, p1[1] + radius), (0, 255, 0))
        set_new_template(tmp)

    dim = np.array(scene.shape[1::-1])
    cv2.rectangle(scene, tuple((dim / 2) - (100, 100)), tuple((dim / 2) + (100, 100)), (0, 0, 0))

    if finder1.filter.stability() < 900:
        cv2.circle(scene, p1, 10, (0, 0, 0), -1)

    scene_disp = cv2.resize(scene, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('t', np.fliplr(scene_disp))

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord(' '):
        tmp = scene[dim[1] / 2 - 100:dim[1] / 2 + 100, dim[0] / 2 - 100:dim[0] / 2 + 100, :]
        set_new_template(tmp)
