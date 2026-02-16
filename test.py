import numpy as np

intrinsic_matrix = np.array([[686, 0.0, 640],
                            [0.0, 663, 360],
                            [0.0, 0.0, 1.0]])

R = np.diag([1, -1, -1])

K_inverse = np.linalg.inv(intrinsic_matrix)

P = np.array([1280, 0, 1])

R_Body_Cam = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

d_cam  = K_inverse @ P

d_body = R_Body_Cam @ d_cam

if d_body[2] <= 0:
    pass

t = 10 / d_body[2]

p_body = t * d_body

# d_cam = K_inverse @ P

# p_cam = z * d_cam

# p_body = R_Body_Cam @ p_cam

p_ENU = R @ p_body

x = p_ENU[0]
y = p_ENU[1]

print(f"X: {x}, Y: {y}")