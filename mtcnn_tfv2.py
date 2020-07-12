import argparse

import tensorflow as tf
import cv2


def mtcnn_fun(img, min_size, factor, thresholds):
    with open('./mtcnn.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef.FromString(f.read())

    with tf.device('/cpu:0'):
        prob, landmarks, box = tf.compat.v1.import_graph_def(graph_def,
            input_map={
                'input:0': img,
                'min_size:0': min_size,
                'thresholds:0': thresholds,
                'factor:0': factor
            },
            return_elements=[
                'prob:0',
                'landmarks:0',
                'box:0']
            , name='')
    print(box, prob, landmarks)
    return box, prob, landmarks

# wrap graph function as a callable function
mtcnn_fun = tf.compat.v1.wrap_function(mtcnn_fun, [
    tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[3], dtype=tf.float32)
])

def main(args):
    img = cv2.imread(args.image)

    bbox, scores, landmarks = mtcnn_fun(img, 40, 0.7, [0.6, 0.7, 0.8])
    bbox, scores, landmarks = bbox.numpy(), scores.numpy(), landmarks.numpy()

    print('total box:', len(bbox))
    for box, pts in zip(bbox, landmarks):
        box = box.astype('int32')
        img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 3)

        pts = pts.astype('int32')
        for i in range(5):
            img = cv2.circle(img, (pts[i+5], pts[i]), 1, (0, 255, 0), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tensorflow mtcnn')
    parser.add_argument('image', help='image path')
    args = parser.parse_args()
    main(args)