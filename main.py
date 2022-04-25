import cv2
import numpy as np
import onnxruntime

# model_path = '02.model/PAN(timm-mobilenetv3_small_100)_452_1.02M_0.8028/best_model.onnx'
# model_path = '02.model/DeepLabV3Plus(timm-mobilenetv3_small_100)_452_2.16M_0.8385/best_model_simplifier.onnx'
model_path = '02.model/UnetPlusPlus(timm-mobilenetv3_small_100)_452_3.71M_0.8715/best_model_simplifier.onnx'

sess = onnxruntime.InferenceSession(model_path)

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

cap = cv2.VideoCapture('XX.image/01.mp4')

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    h, w, _ = img.shape

    x = cv2.resize(img, dsize=(512, 512))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = (x / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1).astype('float32')
    x = x.reshape(-1, 3, 512, 512)

    out = sess.run([output_name], {input_name: x})
    out = np.array(out).squeeze()

    masks = np.where(out > 0.5, 1, 0)
    masks = masks.transpose(1, 2, 0).astype(np.float32) # [skin, clothes, hair]
    masks = cv2.resize(masks, (w, h))

    hair = masks[:, :, 2:3]

    result = img.copy()
    result = 0.5 * result * hair + result * (1 - hair)

    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[:, :, 2] = 30 # red
    colored_hair = color * hair

    result = result + colored_hair
    result = result.astype(np.uint8)

    cv2.imshow('mask', masks)
    cv2.imshow('result', result)
    if cv2.waitKey(1) == ord('q'):
        break
