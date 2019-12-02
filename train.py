from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle

def drawBox(path, boxList):
    data = pyplot.imread(path)
    pyplot.imshow(data)
    ax = pyplot.gca()
    for box in boxList:
        y1,x1,y2,x2 = box
        w,h = x2-x1, y2-y1
        rect = Rectangle((x1,y1), w,h, fill=False, color='blue')
        ax.add_patch(rect)
    pyplot.show()


class TestConfig(Config):
    NAME = 'test'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80


rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)

imgPath = 'dog_2.jpg'
img = load_img(imgPath)
img = img_to_array(img)

results = rcnn.detect([img])
drawBox(imgPath, results[0]['rois'])