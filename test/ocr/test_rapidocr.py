from rapidocr_onnxruntime import RapidOCR
import time
start_time = time.time()
engine = RapidOCR(det_use_cuda=False,cls_use_cuda=False,rec_use_cuda=False)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"rapidocr初始化耗时: {elapsed_time:.2f}秒")
start_time = time.time()
img_path = '../z_using_files/pics/page_1.png'
result, _ = engine(img_path)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"rapidocr耗时: {elapsed_time:.2f}秒")
for buck in result:
    print(buck[1])
# print(result)
