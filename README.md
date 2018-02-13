# Description
This project provide a **single** tensorflow model implemented the mtcnn face detector.
 It is very handy for face detection in python. The model is converted and
 modified from the original author's caffe model.
 
 For more detail about mtcnn, see the
  [original project](https://github.com/kpzhang93/MTCNN_face_detection_alignment).

# Requirement
- tensorflow >= 1.5.0 (older version may work as well, but it is not tested)
- opencv python binding (for reading image and show the result)

# Run
```bash
python mtcnn.py test_image.jpg
```

# Result
![result.jpg](./result.jpg)

# Input and Ouput
## Input: 
 BGR image.
## Output:
- box: bouding box, 2D float tensor with format [[y1, x1, y2, x2], ...]
- prob: confidence, 1D float tensor with format [x, ...]
- landmarks: face landmarks, 2D float tensor with format[[y1, y2, y3, y4, y5, x1, x2, x3, x4, x5], ...]

# Note
- Because the model is designed to work with opencv, so the input image format is BGR instead of RGB.

# TODO
- [ ] Upload the model convert script (The code is too dirty right now).
