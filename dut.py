import xml.etree.ElementTree
import os

folder = '/ais/gobi4/rjliao/Projects/CSC2541/data/TUD/cvpr10_tud_stadtmitte'
xml_fname = os.path.join(folder, 'TUD-Stadtmitte.xml')

tree = xml.etree.ElementTree.parse(xml_fname).getroot()

obj_data = {}
frame_start = None
frame_end = 0
for frame in tree.findall('frame'):
    nframe = frame.attrib['number']
    if frame_start is None:
        frame_start = nframe
        frame_end = nframe
    else:
        frame_start = min(frame_start, nframe)
        frame_end = max(frame_end, nframe)
    for obj_list in frame.findall('object_list'):
        for obj in obj_list.findall('object'):
            idx = obj.attrib['id']
            for box in obj.findall('box'):
                h = float(box.attrib['h'])
                w = float(box.attrib['h'])
                x = float(box.attrib['xc'])
                y = float(box.attrib['yc'])
            if idx in obj_data:
                obj_data[idx].append([nframe, h, w, x, y])
            else:
                obj_data[idx] = []

        print nframe, idx, h, w, x, y
