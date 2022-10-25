from lib import *
from make_datapath import make_datapath_list


class Anno_xml(object):
    def __init__(self, classes):
        self.classes = classes
    
    def __call__(self, xml_path, width, height):
        # include image annotation
        ret = []
        # read file xml
        xml = ET.parse(xml_path).getroot()
        
        for obj in xml.iter('object'): # duyệt trong file xml
            difficult = int(obj.find("difficult").text) # tìm obj có chứa difficult sau đó lấy phần text của nó 
            #không dùng ảnh có difficult=1 vì nó khó huấn luyện
            if difficult == 1:
                continue
            # information for bounding box    
            bndbox = []
            name = obj.find("name").text.lower().strip() #lower để tất cả các chữ cái viết thường, và strip để bỏ kí tự space hoặc xuống dòng
            bbox = obj.find("bndbox")
            pts = ["xmin", "ymin", "xmax", "ymax"] # 4 thông tin của bndbox trong file xml
            for pt in pts:
                pixel = int(bbox.find(pt).text) - 1 #do các giá trị pixel trong VOC data lớn hơn  quy chuẩn trong xử lý ảnh 1 đvi
                if pt == "xmin" or pt == "xmax":
                    pixel /= width # ratio of width
                else:
                    pixel /= height # ratio of height 
                bndbox.append(pixel)
            label_id = self.classes.index(name) # id name đưa vào classes của các tập trong VOC thì có được id label
            bndbox.append(label_id)
            ret += [bndbox]
        return np.array(ret) #[[xmin, ymin, xmax, ymax, label_id], ......]

# test class Anno_xml
if __name__ == "__main__":
    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"] # các classes trong VOC
    anno_xml = Anno_xml(classes)

    root_path = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    idx = 1
    img_file_path = val_img_list[idx] #lấy phần từ có idx=1 trong val_img_list

    img = cv2.imread(img_file_path) # [height, width, 3 channels:BGR]
    height, width, channels = img.shape # get size img
    # print("Size img {}, {}, {}".format(height, width, channels))
    # xml_path, width, height
    annotation_infor = anno_xml(val_annotation_list[idx], width, height)
    print(annotation_infor)
    # ket qua: [[ 0.09        0.03003003  0.998       0.996997   18.        ],[ 0.122       0.56756757  0.164       0.72672673 14.        ]]