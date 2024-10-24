import numpy as np

def get_angle_from_3points(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle


def get_fingers_angle_from_hand3d(hand3d):
    if hand3d is not None:
        return [
            get_angle_from_3points(hand3d[0], hand3d[1], hand3d[2]),
            get_angle_from_3points(hand3d[1], hand3d[2], hand3d[3]),
            get_angle_from_3points(hand3d[2], hand3d[3], hand3d[4]),
            get_angle_from_3points(hand3d[0], hand3d[5], hand3d[6]),
            get_angle_from_3points(hand3d[5], hand3d[6], hand3d[7]),
            get_angle_from_3points(hand3d[6], hand3d[7], hand3d[8]),
            get_angle_from_3points(hand3d[0], hand3d[9], hand3d[10]),
            get_angle_from_3points(hand3d[9], hand3d[10], hand3d[11]),
            get_angle_from_3points(hand3d[10], hand3d[11], hand3d[12]),
            get_angle_from_3points(hand3d[0], hand3d[13], hand3d[14]),
            get_angle_from_3points(hand3d[13], hand3d[14], hand3d[15]),
            get_angle_from_3points(hand3d[14], hand3d[15], hand3d[16]),
            get_angle_from_3points(hand3d[0], hand3d[17], hand3d[18]),
            get_angle_from_3points(hand3d[17], hand3d[18], hand3d[19]),
            get_angle_from_3points(hand3d[18], hand3d[19], hand3d[20]),
        ]
    else:
        return None
