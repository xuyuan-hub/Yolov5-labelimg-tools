import cv2
import xml.etree.ElementTree as ET

def txt2xml(text,img,labels):
    '''
    此函数用于将yolov5结果中的txt文件转为支持labelimg的xml文件.返回xml树
    Args:
        text: 由yolov5产生的txt文件
        img: 对应txt文件的图片
        labels: 对应图片以及txt的标签列表
    Returns:
        输出的xmlfile文件路径及名称
    >>> import cv2
    >>> import xml.etree.ElementTree as ET
    >>> xmlname = txt2xml(text="11-2-7.txt",img="11-2-7.png",labels=['bacteria'])
    >>> xmlname
    '11-2-7.xml'
    >>> tree = ET.parse(xmlname)
    >>> root = tree.getroot()
    >>> root.tag
    'annotation'
    >>> size = root.find('size')
    >>> int(size.find('width').text)
    1280
    >>> int(size.find('height').text)
    960
    >>> root.find('filename').text
    '11-2-7.png'
    >>> sample = root.find('object')
    >>> sample.find("name").text
    'bacteria'
    '''

    xmlfile = text.split('.')[-2] + '.xml'
    img1 = cv2.imread(img,0)
    [height,width] = img1.shape
    tree = ET.Element("annotation")
    t_folder = ET.SubElement(tree,'folder')
    t_fname = ET.SubElement(tree,'filename')
    t_path = ET.SubElement(tree,'path')
    t_source = ET.SubElement(tree,'source')
    t_size = ET.SubElement(tree, 'size')
    t_seg = ET.SubElement(tree, 'segmented')
    t_folder.text = 'annotations'
    t_fname.text = img
    t_path.text = img
    t_db = ET.SubElement(t_source, 'database')
    t_db.text = 'Unknown'
    t_seg.text = str(0)
    t_width = ET.SubElement(t_size, 'width')
    t_height = ET.SubElement(t_size, 'height')
    t_depth = ET.SubElement(t_size, 'depth')
    t_width.text = str(width)
    t_height.text = str(height)
    t_depth.text = str(1)
    txtfile = open(text)
    txt_lines = txtfile.readlines()

    for line in txt_lines:
        bacteria = ET.SubElement(tree, 'object')
        t_name = ET.SubElement(bacteria, 'name')
        t_pose = ET.SubElement(bacteria, 'pose')
        t_truc = ET.SubElement(bacteria, 'truncated')
        t_diff = ET.SubElement(bacteria, 'difficult')
        t_bndx = ET.SubElement(bacteria, 'bndbox')
        index = line.split(' ')[0]
        t_name.text = labels[int(index)]
        t_pose.text = 'Unspecified'
        t_truc.text = str(0)
        t_diff.text = str(0)
        bndx_xm = ET.SubElement(t_bndx, 'xmin')
        bndx_ym = ET.SubElement(t_bndx, 'ymin')
        bndx_xM = ET.SubElement(t_bndx, 'xmax')
        bndx_yM = ET.SubElement(t_bndx, 'ymax')
        x = float(line.split(' ')[1]) * width
        y = float(line.split(' ')[2]) * height
        w = float(line.split(' ')[3]) * width
        h = float(line.split(' ')[4].strip()) * height
        xmax = int((2 * x + w) / 2)
        ymax = int((2 * y + h) / 2)
        xmin = int((2 * x - w) / 2)
        ymin = int((2 * y - h) / 2)
        bndx_xm.text, bndx_ym.text, bndx_xM.text, bndx_yM.text = str(xmin), str(ymin), str(xmax), str(ymax)

    trees = ET.ElementTree(tree)

    trees.write(xmlfile)
    return xmlfile

def xml2txt(xml,labels):
    '''
    此函数用于将labelimg产生的xml文件转换为用于训练的txt文件
    Args:
        xml: 由labelimg产生的xml文件

    Returns:
        txt文件路径
    >>> fname = xml2txt(xml="11-2-7.xml",labels=["bacteria"])
    >>> fname
    '11-2-7.txt'
    >>> f = open(fname,'r')
    >>> line = f.readlines()[0].split(' ')
    >>> line[0]
    '0'
    >>> line[1]
    '0.39492187500000003'
    >>> line[2]
    '0.4067708333333333'
    >>> line[3]
    '0.00390625'
    '''

    tfname = xml.split('.')[-2] + '.txt'
    tfile = open(tfname, 'w')
    f = open(xml)
    xmltree = f.read()
    root = ET.fromstring(xmltree)
    f.close()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in labels:
            print("%s not in classes list please check it!"%cls)
            continue
        cls_id = labels.index(cls)
        xmlbox = obj.find('bndbox')
        box = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        size = (width,height)
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        x = (box[0] + box[1]) / 2.0 * dw
        y = (box[2] + box[3]) / 2.0 * dh
        w = (box[1] - box[0]) * dw
        h = (box[3] - box[2]) * dh
        b_box = (x, y, w, h)
        tfile.write(str(cls_id) + " " + " ".join([str(a) for a in b_box]) + '\n')

    tfile.close()
    return tfname

if __name__ == '__main__':
    import doctest
    doctest.testmod(xml2txt)

