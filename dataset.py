from lib import *
from make_datapath import make_datapath_list
from transform import DataTransform
from extract_inform_annotation import Anno_xml


class MyDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, anno_xml):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.anno_xml = anno_xml
    
    def __len__(self):
        return len(self.img_list) 
    
    def __getitem__(self, index): # lấy ra các phần tử
        img, gt, height, width = self.pull_item(index)

        return img, gt # gt là các thông tin xmax, xmin,ymax, ymin,
    
    def pull_item(self, index): # đọc ra các ảnh theo vị trí index của nó
        img_file_path = self.img_list[index]
        img = cv2.imread(img_file_path) #BGR
        height, width, channels = img.shape

        # get anno information
        anno_file_path = self.anno_list[index] # lấy ra file list trước
        ann_info = self.anno_xml(anno_file_path, width, height)

        # preprocessing
        img, boxes, labels = self.transform(img, self.phase, ann_info[:, :4], ann_info[:, 4]) # 1 ảnh có nhiều box và label nên để số nhiều boxes, labels cho dễ hiểu

        # BGR -> RGB, (height, width, channels) -> (channels, height, width)
        img = torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)

        # ground truth
        gt = gt = np.hstack((boxes, np.expand_dims(labels, axis=1))) # stack theo chiều ngang

        return img, gt, height, width

# phải customize hàm my_collate_fn vì với obj detection 
# thì 1 ảnh có nhiều nhãn khác với classification 1 ảnh chỉ có 1 nhãn 
# và chỉ cần gọi torch.utils.data.DataLoader là có thể chia các tập dataloader
def my_collate_fn(batch): 
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0]) #sample[0]=img
        targets.append(torch.FloatTensor(sample[1])) # sample[1]=annotation# chuyển targer về dạng tensor bằng cách torch.FloatTensor
    #[3, 300, 300] là định dạng ban đầu của imgs
    # (batch_size, 3, 300, 300)
    imgs = torch.stack(imgs, dim=0) # chuyển đổi imgs từ dạng list sang tensor,(batch_size, 3, 300, 300)

    return imgs, targets


if __name__ == "__main__":
    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    # prepare train, valid, annotation list ## bước này giống trong make_data_path
    root_path = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    # prepare data transform ## bước này cần hàm trong transform.py và extract_inform_annotation.py
    color_mean = (104, 117, 123)
    input_size = 300

    train_dataset = MyDataset(train_img_list, train_annotation_list, phase="train",
    transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))

    val_dataset = MyDataset(val_img_list, val_annotation_list, phase="val",
    transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))

    # print(len(train_dataset))
    # print(train_dataset.__getitem__(1))

    batch_size = 4
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

    dataloader_dict = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    batch_iter = iter(dataloader_dict["val"])
    images, targets = next(batch_iter) # get 1 sample
    print(images.size())  # ket qua:torch.Size([4, 3, 300, 300])
    print(len(targets)) # ket qua: 4 # tức là 4 nhóm annotation
    print(targets[0].size())  # ket qua: torch.Size([1, 5]) # tức là 1 object có 5 thành phần xmin, ymin, xmax, ymax, label