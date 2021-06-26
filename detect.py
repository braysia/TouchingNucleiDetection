import numpy as np
import cv2


def cart2pol_angle(x, y):
    phi = np.arctan2(y, x)
    return phi


def calc_clockwise_degree(p, c, q):
    """Return an degree in clockwise if you give three points. c will be a center.
    >>> q = [10, 10]
    >>> c = [0, 0]
    >>> p = [-10, 10]
    >>> calc_closewise_degree(p, c, q)
    90.0
    """
    angle_r = cart2pol_angle(q[0]-c[0], q[1]-c[1]) - cart2pol_angle(p[0]-c[0], p[1]-c[1])
    angle = 180.0 * angle_r/np.pi
    if angle < 0:
        angle += 360.0
    return angle


def calc_neck_score(coords, edgelen=5):
    """Calculate the score (angle changes) and return the sorted score and the corresponding pixel
    coordinates. Pass the coordinates of outlines without border."""
    ordered_c = coords
    nc = np.vstack((ordered_c, ordered_c[:edgelen, :]))
    score = []
    for n, ci in enumerate(nc[:-edgelen, :]):
        score.append(calc_clockwise_degree(ordered_c[n-edgelen, :], nc[n, :], nc[n+edgelen, :]))
    idx = np.flipud(np.argsort(score))
    return np.array(score)[idx], ordered_c[idx]


def find_oriented_coords(outline):
    cnt = cv2.findContours(outline.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0][0]
    cnt = np.flipud(cnt)
    if not cv2.contourArea(np.float32(cnt), oriented=True) > 0:
        return None
    return np.fliplr(np.squeeze(cnt))


def calc_angle(seg, edgelen=5):
    coords = find_oriented_coords(seg)
    return calc_neck_score(coords, edgelen)


def detect_touch(seg, thres_angle=180, edgelen=5):
    """
    """
    angles, _ = calc_angle(seg, edgelen=5)
    return max(angles) > thres_angle


if __name__ == "__main__":
    from imageio import imsave
    nuc = np.load('data/nuc.npz')['arr_0']
    st = []
    for i in nuc:
        if detect_touch(i>0.1, thres_angle=170, edgelen=5):
            st.append(i)
    imsave('data/outliers.png', np.concatenate(st).T)
