import argparse
import time
import numpy as np
import onnx
import onnxruntime as ort
from tqdm import tqdm
from data_utils import load_image, preprocess, postprocess

parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode', type=str, default='int8', choices=['int8', 'fp32']
)
parser.add_argument(
    '--rep', type=int, default=100
)

if __name__ == '__main__':
    opt = parser.parse_args()
    session = ort.InferenceSession(f'RLFN_{opt.mode}.onnx', providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    orig_img = load_image('ms3_01.png')
    test_input = preprocess(orig_img, (300, 300))
    
    start = time.time()
    for i in tqdm(range(opt.rep)):
        raw_result = session.run([], {input_name:test_input})[0]
    time_elpased = time.time() - start
    print(f"# [{opt.mode}] total elpased time = {time_elpased:.2f}s")

    time_elpased /= opt.rep
    print(f"# [{opt.mode}] average elapsed time = {time_elpased:.5f}s")    

    pp_result = postprocess(raw_result)
    
    import cv2
    cv2.imwrite(f'ort_result_{opt.mode}.png', pp_result)
    