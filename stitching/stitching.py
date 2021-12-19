import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
T=np.array(
        [[1,0,0],
         [0,1,50],
         [0,0,1]])
T2=np.float32(
        [[1,0,0],
         [0,1,50]])

def plot_img(rows, cols, index, img, title):
    ax = plt.subplot(rows,cols,index)
    if(len(img.shape) == 3):
        ax_img = plt.imshow(img[...,::-1]) 
    else:
        ax_img = plt.imshow(img, cmap='gray')
    plt.axis('on')
    if(title != None): plt.title(title) 
    return ax_img, ax
    
def update_good_correpondences(ratio_dist,matches):
    good_correspondences.clear()
    for m,n in matches:#매치 결과 상위 2개 m,n
        if m.distance/n.distance < ratio_dist:
            good_correspondences.append(m)#조건 만족시 추가

def stitch(img1,img2,kp1,kp2):

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_correspondences ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_correspondences ]).reshape(-1,1,2)
    H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
    TH=np.dot(T,H)
    
    
    stitch_plane_rows = img1.shape[0] +100
    stitch_plane_cols = img1.shape[1] + img2.shape[1]
    
    
    result1 = cv.warpPerspective(img2, TH, (stitch_plane_cols, stitch_plane_rows),flags=cv.INTER_LINEAR, borderMode=cv.BORDER_TRANSPARENT)
    result2 = cv.warpAffine(img1, T2, (stitch_plane_cols, stitch_plane_rows))
    #plot_img(4, 1, 1, result1, None)
    #plot_img(4, 1, 2, result2, None)
    
    h,w, = img2.shape[:2]
    pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
    dst = cv.perspectiveTransform(pts,TH)
    intdst_x=np.int32(dst[2,0,0])+1
    

    and_img = cv.bitwise_and(result1, result2)
    and_img_gray = cv.cvtColor(and_img, cv.COLOR_BGR2GRAY)
    th, mask = cv.threshold(and_img_gray, 1, 255, cv.THRESH_BINARY)
    #plot_img(4, 1, 3, mask, None)

    result1_gray=cv.cvtColor(result1, cv.COLOR_BGR2GRAY)
    result2_gray=cv.cvtColor(result2, cv.COLOR_BGR2GRAY)

    result3 = np.zeros((stitch_plane_rows, intdst_x,3), np.uint8)
    for y in range(stitch_plane_rows):
        for x in range(intdst_x):
            mask_v = mask[y, x]
            if(mask_v > 0):
                result3[y, x] = np.uint8(result1[y,x] * 0.5 + result2[y,x] * 0.5)
            else:
                if result1_gray[y,x]==0 and result2_gray[y,x]==0:
                    result3[y, x] = result2[y,x]
                elif result1_gray[y,x]!=0 and result2_gray[y,x]==0:
                    result3[y, x] = result1[y,x]
                elif result1_gray[y,x]==0 and result2_gray[y,x]!=0:
                    result3[y, x] = result2[y,x]
                else:
                    result3[y, x] = result2[y,x]

    #plot_img(4, 1, 4, result3, None)
    #plt.show()
    
    return result3

#main
img1 = cv.imread("1.jpg");img2 = cv.imread("2.jpg");img3 = cv.imread("3.jpg");img4 = cv.imread("4.jpg");img5 = cv.imread("5.jpg")

img1=cv.resize(img1,dsize=(500,500),interpolation=cv.INTER_AREA)
img2=cv.resize(img2,dsize=(500,500),interpolation=cv.INTER_AREA)
img3=cv.resize(img3,dsize=(500,500),interpolation=cv.INTER_AREA)
img4=cv.resize(img4,dsize=(500,500),interpolation=cv.INTER_AREA)
img5=cv.resize(img5,dsize=(500,500),interpolation=cv.INTER_AREA)

sift = cv.SIFT_create()

#step1(3+4)
kp1, des1 = sift.detectAndCompute(img3, None)
kp2, des2 = sift.detectAndCompute(img4, None)

flann = cv.FlannBasedMatcher({"algorithm":1, "trees":5}, {"checks":50})
matches = flann.knnMatch(des1 , des2, k=2)

good_correspondences = []#좋은 매칭결과 리스트
update_good_correpondences(0.3,matches)
result1=stitch(img3,img4,kp1,kp2)
plot_img(4, 1, 1, result1, None)

#step2(2+ (3,4))
img2=cv.flip(img2,1)#좌우반전
result1=cv.flip(result1,1)#좌우반전
kp1, des1 = sift.detectAndCompute(img2, None)
kp2, des2 = sift.detectAndCompute(result1, None)


flann = cv.FlannBasedMatcher({"algorithm":1, "trees":5}, {"checks":50})
matches = flann.knnMatch(des2 , des1, k=2)

good_correspondences = []#좋은 매칭결과 리스트
update_good_correpondences(0.5,matches)

result2=stitch(result1,img2,kp2,kp1)
result2=cv.flip(result2,1)
plot_img(4, 1, 2, result2, None)

img2=cv.flip(img2,1)#좌우반전
result1=cv.flip(result1,1)#좌우반전



#step3((2,3,4)+5)
kp1, des1 = sift.detectAndCompute(result2, None)
kp2, des2 = sift.detectAndCompute(img5, None)

flann = cv.FlannBasedMatcher({"algorithm":1, "trees":5}, {"checks":50})
matches = flann.knnMatch(des1 , des2, k=2)

good_correspondences = []#좋은 매칭결과 리스트
update_good_correpondences(0.5,matches)
result3=stitch(result2,img5,kp1,kp2)
plot_img(4, 1, 3, result3, None)


#step4(1+(2,3,4,5))
img1=cv.flip(img1,1)#좌우반전
result3=cv.flip(result3,1)#좌우반전
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(result3, None)


flann = cv.FlannBasedMatcher({"algorithm":1, "trees":5}, {"checks":50})
matches = flann.knnMatch(des2 , des1, k=2)

good_correspondences = []#좋은 매칭결과 리스트
update_good_correpondences(0.5,matches)

result4=stitch(result3,img1,kp2,kp1)
result4=cv.flip(result4,1)

plot_img(1, 1, 1, result4, None)

img1=cv.flip(img1,1)#좌우반전
result3=cv.flip(result3,1)#좌우반전
plt.show()