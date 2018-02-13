# Description
This project provide a single tensorflow model implement the mtcnn face detector. It is very handy for face detection in python. The tensorflow model is converted and modified from the original author's caffe model.For more detail about mtcnn, see the [original project](https://github.com/kpzhang93/MTCNN_face_detection_alignment)

# Requirement
- tensorflow >= 1.5.0 (older version may work as well, but it is not tested in the moment)
- opencv python binding (for reading image and show the result)

# Run
```bash
python mtcnn.py test_image.jpg
```

# Result
![result.jpg](./result.jpg)

# Note
- Because the model is designed for working with opencv, so the input image format is BGR instead of RGB.