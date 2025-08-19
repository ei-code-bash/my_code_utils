import sys,time
import numpy as np
import cv2
from convert import CornerModel,CornerPnPEstimator,TreeDanix
def guess_intrinsics(w,h,fov_deg=60):
    f=(w/(2*np.tan(0.5*np.deg2rad(fov_deg))))
    K = np.array([[f, 0, w / 2],
                  [0, f, h / 2],
                  [0, 0, 1]], dtype=np.float64)
    return K
def main():
    if len(sys.argv)<2:
        print("用法：python realtime_pnp.py <model_dir> [camera_id]")
        return
    model_dir=sys.argv[1]
    cam_id=int(sys.argv[2]) if len(sys.argv)>3 else 0
    model=CornerModel.load(model_dir)
    cap=cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ok, frame = cap.read()
    if not ok:
        print("打开摄像头失败");
        return
    #=============#
    #=============#
    h, w = frame.shape[:2]
    K = guess_intrinsics(w, h, fov_deg=60)
    dist = np.zeros((5, 1), np.float64)
    #=============#
    #=============#
    estimator=CornerPnPEstimator(K, dist,orb_nfeatures=1500,orb_fastThresold=12)
    use_subpix=False
    show_kps=False
    cv2.setUseOptimized(True)
    t0=time.time();cnt=0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img=frame
        res,vis=estimator.estimate_pose(img,model,ratio=0.75,subpix_refine=use_subpix,ransac_err_px=3.0,confidence=0.999
                                        ,iters=500)
        cnt+=1
        if cnt%10==0:
            dt=time.time()-t0
            fps=cnt/max(dt,1e-6)
        text=f"{'OK' if ok else 'FAILED'}"
        if res.sucess:
            text+=f"| inliers={len(res.inliers)} | err={res.reproj_error_px:.2f}px"
        text+=f"|subpix={'ON' if use_subpix else 'OFF' }"
        cv2.putText(vis,text,(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(50,220,50),2,cv2.LINE_AA)
        cv2.imshow('frame',vis)
        key=cv2.waitKey(1) & 0xFF
        if key ==27 or key == ord('q'):
            break
        elif key == ord('s'):
            use_subpix=not use_subpix
        elif key == ord('k'):
            show_kps=not show_kps
        elif key==ord('r'):
            nf = 1000 if estimator.orb.getMaxFeatures() > 1000 else 2000
            estimator.orb.setMaxFeatures(nf)
            print("set nfeatures =", nf)
    cap.release()
    cv2.destroyAllWindows()
if __name__=="__main__":
    main()
