#NEXTE 我永远喜欢雪之下雪乃
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict,List ,Tuple,Optional

import cv2
import numpy as np
import json
import os
def to_gray(img:np.ndarray)->np.ndarray:
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#角点位置亚像素级优化
def refine_subpinx(gray:np.ndarray,pts:np.ndarray,win=(5,5))->np.ndarray:
    if pts is None or len(pts) == 0:
        return pts
    pts=np.float32(pts).reshape(-1,1,2)
    criteria=(cv2.TermCriteria_EPS+cv2.TERM_CRITERIA_MAX_ITER,40,1e-3)
    cv2.cornerSubPix(gray,pts,win,(-1,-1),criteria)
    return pts.reshape(-1,2)

def TreeDanix(img:np.ndarray,K:np.ndarray,dist:np.ndarray,
              rvec: np.ndarray,tvec: np.ndarray,anis_len: float =0.05 )->np.ndarray:
    img=img.copy()
    anix_3d=np.float32(
        [[0,0,0],
         [anis_len,0,0],
         [0,anis_len,0],
         [0,0,anis_len]])
    proj, _=cv2.projectPoints(anix_3d,rvec,tvec,K,dist)
    p0 = tuple(int(v) for v in proj[0].ravel())
    px = tuple(int(v) for v in proj[1].ravel())
    py = tuple(int(v) for v in proj[2].ravel())
    pz = tuple(int(v) for v in proj[3].ravel())
    cv2.line(img, p0, px, (0, 0, 255), 3)  # X 红
    cv2.line(img, p0, py, (0, 255, 0), 3)  # Y 绿
    cv2.line(img, p0, pz, (255, 0, 0), 3)  # Z 蓝
    return img


@dataclass
class Corner3D:
    id: int
    xyz: np.ndarray
class CornerModel:
    def __init__(self,points3d: Dict[int,Tuple[float,float,float]],orb_nfeatures: int=2000, orb_fastThresold: int=12):
        self.points3d: Dict[int,np.ndarray]={i: np.asarray(p, np.float32).reshape(3) for i, p in points3d.items()}#将值转换为array数组
        self.desc_by_id: Dict[int, List[np.ndarray]] = {i: [] for i in self.points3d.keys()}
        self.orb = cv2.ORB_create(nfeatures=orb_nfeatures, fastThreshold=orb_fastThresold)
    def add_reference(self,image: np.ndarray,id_to_uv:Dict[int,tuple[float,float]],subpix:bool=True,kp_size: float=31.0)->int:
        gray = to_gray(image)
        if subpix:
            pts_in=np.float32([id_to_uv][k] for k in id_to_uv).reshape(-1,2)
            pts_refined=refine_subpinx(gray,pts_in)#亚像素微调
            uv_map={k:tuple(pts_refined[i]) for i,k in enumerate(id_to_uv)}
        else:
            uv_map=id_to_uv
        kps: List[cv2.KeyPoint]=[]
        id_order: List[int]=[]
        for cid,(u,v) in uv_map.items():
            if cid not in self.points3d:
                continue
            kps.append(cv2.KeyPoint(x=float(u),y=float(v),_size=kp_size))
            id_order.append(cid)
        kps,desc=self.orb.compute(gray,kps)
        ok=0
        if desc is not None and len(kps)==len(id_order):
            for i,cid in enumerate(id_order):
                self.desc_by_id[cid].append(desc[i:i+1])
                ok+=1
        return ok
    def build_descriptor_bank(self)->Tuple[np.ndarray, np.ndarray]:
        descs=[]
        ids=[]
        for cid,chunks in self.desc_by_id.items():
            if len(chunks)==0:
                continue
            cat =np.concatenate(chunks,axis=0)
            descs.append(cat)
            ids.append(np.full((cat.shape[0],),cid,dtype=np.int32))
        if len(descs)==0:
            return np.zeros((0,32),dtype=np.uint8),np.zeros((0,),dtype=np.int32)
        return np.concatenate(descs,axis=0),np.concatenate(ids,axis=0)
    def save(self,path:str)->None:
        os.makedirs(path,exist_ok=True)
        meta={"points3d": {str(i): self.points3d[i].tolist() for i in self.points3d}}
        with open(os.path.join(path,"meta.json"),"w",encoding="utf-8") as f:
            json.dump(meta,f,ensure_ascii=False,indent=2)
        bank_descs,bank_ids=self.build_descriptor_bank()
        np.savez_compressed(os.path.join(path,"descs.npz"),bank_descs=bank_descs,bank_ids=bank_ids)

    @staticmethod
    def load(path: str) -> "CornerModel":
        """ 从目录加载模型 """
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        points3d = {int(k): tuple(v) for k, v in meta["points3d"].items()}
        model = CornerModel(points3d)
        data = np.load(os.path.join(path, "descs.npz"))
        bank_descs = data["bank_descs"]
        bank_ids = data["bank_ids"]
        # 反填 desc_by_id（把库按 id 分回去）
        for cid in np.unique(bank_ids):
            model.desc_by_id[int(cid)] = [bank_descs[bank_ids == cid]]
        return model
@dataclass
class PnPResult:
    sucess: bool
    rvec: np.ndarray
    tvec: np.ndarray
    inliers: Optional[np.ndarray]#RANSAC的内点数
    reproj_error_px: Optional[float]#平均重投影误差
    used_pair: int#参与求解的点数

class CornerPnPEstimator:
    def __init__(self,K: np.ndarray,dist: np.ndarray,orb_nfeatures: int=2500, orb_fastThresold: int=12):
        self.K=np.asarray(K,np.float64)
        self.dist=np.asarray(dist,np.float64)
        self.orb = cv2.ORB_create(nfeatures=orb_nfeatures, fastThreshold=orb_fastThresold)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_query(self,image:np.ndarray):
        gray = to_gray(image)
        kps,desc=self.orb.detectAndCompute(gray,None)
        return kps,desc

    def match_to_model(self, query_desc: np.ndarray, model: CornerModel,
                       ratio: float = 0.75) -> Tuple[List[cv2.DMatch], np.ndarray, np.ndarray, List[int]]:
        # 1) 取出模型的描述子库与 id 库
        bank_descs, bank_ids = model.build_descriptor_bank()
        if bank_descs.shape[0] == 0 or query_desc is None or len(query_desc) == 0:
            return [], np.zeros((0, 3), np.float32), np.zeros((0, 2), np.float32), []

        # 2) KNN 匹配，k=2，做 Lowe ratio test
        knn = self.matcher.knnMatch(query_desc, bank_descs, k=2)
        prelim = []
        for m, n in knn:
            if m.distance < ratio * n.distance:
                prelim.append(m)

        # 3) 对每个模型角点 id 仅保留一条“最佳”（距离最小）匹配
        best_for_id: Dict[int, cv2.DMatch] = {}
        for m in prelim:
            cid = int(bank_ids[m.trainIdx])
            if cid not in best_for_id or m.distance < best_for_id[cid].distance:
                best_for_id[cid] = m
        good = list(best_for_id.values())
        if len(good) == 0:
            return [], np.zeros((0, 3), np.float32), np.zeros((0, 2), np.float32), []

        # 4) 根据匹配构造 3D-2D 对：
        #    obj_pts 用 3D 模型坐标；img_pts 先占位（后面由 keypoint 像素坐标填充）
        obj_pts = []
        img_pts = []
        qidx = []
        for m in good:
            cid = int(bank_ids[m.trainIdx])
            obj_pts.append(model.points3d[cid])
            img_pts.append([0, 0])  # 占位
            qidx.append(m.queryIdx)  # 查询图的 keypoint 索引
        return good, np.asarray(obj_pts, np.float32), np.asarray(img_pts, np.float32), qidx

    def solve_pnp(self,obj_pts_3d:np.ndarray,img_pts_2d:np.ndarray,
                  use_planar_ippe: bool =False,ransac_err_px: float=3.0,confidence:float=0.999,iters: int=1000)->PnPResult:
        if len(obj_pts_3d) < 4:
            return PnPResult(False,None,None,None,None,use_pairs=len(obj_pts_3d))
        flags=cv2.SOLVEPNP_ITERATIVE
        if use_planar_ippe and hasattr(cv2,"SOLVEPNP_ITERATIVE"):
            flags=cv2.SOLVEPNP_IPPE
        ok,rvec,tvec,inliers=cv2.solvePnPRansac(obj_pts_3d,img_pts_2d,self.K,self.dist,iterationsCount=iters,
                                                reprojectionError=ransac_err_px,confidence=confidence,flags=flags)
        if not ok or inliers is None or len(inliers)<4:
            return PnPResult(False,None,None,None,None,use_pairs=len(obj_pts_3d))
        proj, _ =cv2.projectPoints(obj_pts_3d[inliers.ravel()],rvec,tvec,self.K,self.dist)
        proj=proj.reshape(-1,2)
        err=float(np.linalg.norm(proj-img_pts_2d[inliers.ravel()],anis=1).mean())
        return PnPResult(True,rvec,tvec,inliers,err,used_pairs=len(obj_pts_3d))
    def estimate_pose(self,image: np.ndarray,model: CornerModel,ratio: float=0.75,subpix_refine: bool=True
                      ,ransac_err_px: float =3.0,confidence: float =0.999,iters: int =1000) -> Tuple[PnPResult,np.ndarray]:
        kps,qdesc=self.detect_query(image)
        good,obj3d,img2d,qidx=self.match_to_model(qdesc,model,ratio=ratio)
        if len(good) == 0:
            return PnPResult(False,None,None,None,None,0),image
        pts=np.float32(kps[i].pt for i in qidx)
        if subpix_refine:
            pts=refine_subpinx(to_gray(image),pts)
        img2d[:,:]=pts
        res=self.solve_pnp(obj3d,img2d,use_planar_ippe=False,ransac_err_px=ransac_err_px,confidence=confidence,iters=iters)
        vis=image.copy()
        for i,p in enumerate(pts):
            cv2.circle(vis,(int(p[0]),int(p[1])),4,(0,0,255),-1,lineType=cv2.LINE_AA)
        if res.sucess:
            vis=TreeDanix(vis,self.K,self.dist,res.rvec,res.tvec,anis_len=0.1)
            text=f"inliers={len(res.inliers)} err={res.reproj_error_px:.2f}px"
        else:
            text="Pnp failed"
        cv2.putText(vis,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(50,220,50),2,cv2.LINE_AA)
        return res,vis
if __name__=="__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "用法：\n  构建模型: python corners3dpnp.py build <model_dir> <ref.jpg> <ref_points.json>\n  求位姿: python corners3dpnp.py pose  <model_dir> <query.jpg>")
        sys.exit(0)

    mode = sys.argv[1].lower()

    if mode == "build":
        if len(sys.argv) < 5:
            print("参数不足：python corners3dpnp.py build <model_dir> <ref.jpg> <ref_points.json>")
            sys.exit(1)
        model_dir, ref_img_path, json_path = sys.argv[2], sys.argv[3], sys.argv[4]
        ref_img = cv2.imread(ref_img_path)
        if ref_img is None:
            print("读取参考图失败")
            sys.exit(1)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        points3d = {int(k): tuple(v) for k, v in data["points3d"].items()}
        id_to_uv = {int(k): tuple(v) for k, v in data["id_to_uv"].items()}

        model = CornerModel(points3d)
        ok = model.add_reference(ref_img, id_to_uv, subpix=True)
        print(f"参考图已加入：{ok} 个角点描述子")
        model.save(model_dir)
        print(f"模型已保存到：{model_dir}")

    elif mode == "pose":
        if len(sys.argv) < 4:
            print("参数不足：python corners3dpnp.py pose <model_dir> <query.jpg>")
            sys.exit(1)
        model_dir, qry_img_path = sys.argv[2], sys.argv[3]
        qry_img = cv2.imread(qry_img_path)
        if qry_img is None:
            print("读取查询图失败")
            sys.exit(1)
        model = CornerModel.load(model_dir)

        # TODO: 替换为你的相机内参（标定获得）
        h, w = qry_img.shape[:2]
        fx = fy = 1000.0
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((5, 1), np.float64)

        estimator = CornerPnPEstimator(K, dist)
        res, vis = estimator.estimate_pose(qry_img, model, ratio=0.75, subpix_refine=True)
        if res.success:
            print(f"PnP OK: inliers={len(res.inliers)}, reproj_err={res.reproj_error_px:.2f}px")
            print("rvec:", res.rvec.ravel())
            print("tvec:", res.tvec.ravel())
        else:
            print("PnP 失败（可能匹配不足或外观差异太大）")
        cv2.imwrite("pose_vis.jpg", vis)
        print("已保存可视化：pose_vis.jpg")
    else:
        print("未知模式，使用 build / pose")