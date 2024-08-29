import cv2

img1 = cv2.imread("C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap\\S1\\1\\heatmap_1_1_3.png")
img2 = cv2.imread("C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg4\\S1\\1\\heatmap_1_1_3.png")
img3 = cv2.imread("C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg9\\S1\\1\\heatmap_1_1_3.png")
img4 = cv2.imread("C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg16\\S1\\1\\heatmap_1_1_3.png")
img5 = cv2.imread("C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg25\\S1\\1\\heatmap_1_1_3.png")
img6 = cv2.imread("C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg36\\S1\\1\\heatmap_1_1_3.png")

list_img = []

for i in range(6):
    list_img.append(globals()[f"img{i+1}"])

img_h = cv2.hconcat(list_img)

cv2.imwrite("C:\\Users\\hibik\\Desktop\\kenkyu\\a.png", img_h)