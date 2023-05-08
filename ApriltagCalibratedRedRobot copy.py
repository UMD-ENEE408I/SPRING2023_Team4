from pupil_apriltags import Detector
import cv2
import numpy as np
import time
from robomaster import robot
from robomaster import camera

at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

def find_pose_from_tag(K, detection):
    m_half_size = tag_size / 2

    marker_center = np.array((0, 0, 0))
    marker_points = []
    marker_points.append(marker_center + (-m_half_size, m_half_size, 0))
    marker_points.append(marker_center + ( m_half_size, m_half_size, 0))
    marker_points.append(marker_center + ( m_half_size, -m_half_size, 0))
    marker_points.append(marker_center + (-m_half_size, -m_half_size, 0))
    _marker_points = np.array(marker_points)

    object_points = _marker_points
    image_points = detection.corners

    pnp_ret = cv2.solvePnP(object_points, image_points, K, distCoeffs=None,flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if pnp_ret[0] == False:
        raise Exception('Error solving PnP')

    r = pnp_ret[1]
    p = pnp_ret[2]

    return p.reshape((3,)), r.reshape((3,))


if __name__ == '__main__':
   
    cap = cv2.VideoCapture(1)

    tag_size=0.053975 # tag size in meters for small tags
    DIM=(640, 480)
    K=np.array([[208.55576928368237, 0.0, 328.33307886215096], [0.0, 209.14602301828555, 257.02413362090175], [0.0, 0.0, 1.0]])
    D=np.array([[0.4247825378689211], [0.7872431366499725], [-1.4964774199333926], [0.7148745575574587]])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    
    while True:

        try:
            ret, img = cap.read() 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray.astype(np.uint8)

            # K=np.array([[184.752, 0, 320], [0, 184.752, 180], [0, 0, 1]])
            
            
            gray = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            results = at_detector.detect(gray, estimate_tag_pose=False)


            for res in results:
                pose = find_pose_from_tag(K, res)
                rot, jaco = cv2.Rodrigues(pose[1], pose[1])
                # print(res.tag_id)
                # print("rot", rot)
                

                pts = res.corners.reshape((-1, 1, 2)).astype(np.int32)
                img = cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=5)
                cv2.circle(img, tuple(res.center.astype(np.int32)), 5, (0, 0, 255), -1)
                # print(res.tag_id)
                # print("pose", pose[0])
                # print("rotation", rot)
                cv2.waitKey(10)

                # For 461
                # print("pose\n", pose[0])
                # print("1\t",pose[0][0])
                # print("2\t",pose[0][1])
                # print("3\t",pose[0][2])
                if(res.tag_id == 3):
                    if(pose[0][0]< (-0.25118638)):
                        print("You are a bit too Right")
                    elif(pose[0][0]> -0.15 ):
                        print("You are a bit too Left")
                    else:
                        print("This is centered enough!")
                if(res.tag_id == 3):
                    if(pose[0][1]< (-0.15)):
                        print("Your tag is above camera level")
                    elif(pose[0][1]> -0.1 ):
                        print("Your tag is below camera level")
                    else:
                        print("This range is enough!")
                MyDistance = (pose[0][2]/0.0254) - 10
                print("Your camera is ", MyDistance," away from the tag.")

                # End Work for 461

            cv2.imshow("img", img)
            cv2.waitKey(10)

        except KeyboardInterrupt:
            # ep_camera.stop_video_stream()
            # ep_robot.close()
            print ('Exiting')
            exit(1)