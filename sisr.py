import numpy as np
import onnx
import onnxruntime as ort

from data_utils import load_image, preprocess, postprocess

if __name__ == '__main__':
    session = ort.InferenceSession('RLFN_fp32.onnx', providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    orig_img = load_image('ms3_01.png')
    test_input = preprocess(orig_img, (300, 300))
    
    raw_result = session.run([], {input_name:test_input})[0]
    
    pp_result = postprocess(raw_result)
    
    import cv2
    cv2.imwrite('ort_result_fp32.png', pp_result)