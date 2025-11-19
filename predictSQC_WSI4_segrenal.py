
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import sys
import json
import time
import numpy as np

# ----- OpenSlide DLL -----
dll_path = r'C:\Users\M300305\Desktop\openslide-win64-20230414\bin'
os.add_dll_directory(dll_path)
path = dll_path + os.pathsep + os.environ['PATH']
os.environ['PATH'] = path

# ----- Libs -----
import openslide as osl
import uuid
from PIL import Image, ImageDraw
from xml.dom import minidom
from matplotlib.patches import Ellipse
import cv2
import tensorflow as tf
from tensorflow import keras

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.debugging.set_log_device_placement(True)

# ----- Config / Model load -----
with open("configPR_SQC_IY.json") as json_file:
    paramsObj = json.load(json_file)

exec('from ' + paramsObj['modelPy'] + ' import prepModel as getModel', globals(), globals())

model = getModel(
    trainFlag='thyroidAI',
    root_dir=paramsObj['root_dir'],
    task=paramsObj['task'],
    image_path=paramsObj['image_path'],
    val_path=paramsObj['image_path'],
    label_path=paramsObj['label_path'],
    batch_size=paramsObj['batch_size'],
    n_classes=paramsObj['n_classes'],
    n_channel=paramsObj['n_channel'],
    image_shape=paramsObj['image_shape'],
    label_shape=paramsObj['label_shape'],
    image_format=paramsObj['image_format'],
    label_format=paramsObj['label_format'],
    model=paramsObj['model'],
    optimizer=paramsObj['optimizer'],
    datagen='',
    afold='',
    project_folder=paramsObj['project_folder'],
    logs_folder=paramsObj['logs_folder'],
    weights_folder=paramsObj['weights_folder'],
    preds_folder=paramsObj['preds_folder'],
    weight_file_name=paramsObj['weight_file_name'],
    output_label=paramsObj['time_stamp'],
    time_stamp=paramsObj['time_stamp'],
    init_model=paramsObj['init_model'],
    n_epochs=paramsObj['n_epochs'],
    shuffle=paramsObj['shuffle'],
    loss=paramsObj['loss'],
    weight_loss=paramsObj['weight_loss'],
    class_weights=paramsObj['class_weights'],
    lr=paramsObj['lr'],
    lr_decay=paramsObj['lr_decay'],
)

# ------------------------------
# Helpers
# ------------------------------

def convertContoursToZP(patch, c, r, offc, offr, l, label, lset, params, approx_eps_ratio=0.01):
    """
    patch: binary mask (0/1) in the local bbox
    c, r:   top-left of bbox in smask coordinates (pixels of smask)
    offc,offr: global top-left (level-0 px) of the stitched smask region
    l:      how many level-0 pixels one smask pixel spans (blevel=0 -> l = DS_PRED = 4)
    label:  class id (1,2,3,5)
    params: per-class morphology parameters
    approx_eps_ratio: Ramer–Douglas–Peucker epsilon as % of contour perimeter
    """
    color_fill = ['rgba(255,0,0,0.3)', 'rgba(0,255,0,0.3)', 'rgba(0,0,255,0.3)', 'rgba(255,255,0,0.3)',
                  'rgba(0,128,0,0.3)', 'rgba(255,0,255,0.3)', 'rgba(192,255,25,0.3)', 'rgba(68,236,165,0.3)',
                  'rgba(128,0,0,0.3)', 'rgba(192,250,192,0.3)', 'rgba(0,0,128,0.3)', 'rgba(0,128,128,0.3)',
                  'rgba(248,127,81,0.3)', 'rgba(128,0,128,0.3)', 'rgba(128,128,0,0.3)'] * 100
    color_stroke = ['rgba(255,0,0,1)', 'rgba(0,255,0,1)', 'rgba(0,0,255,1)', 'rgba(255,255,0,1)',
                    'rgba(0,128,0,1)', 'rgba(255,0,255,1)', 'rgba(192,255,25,1)', 'rgba(68,236,165,1)',
                    'rgba(128,0,0,1)', 'rgba(192,250,192,1)', 'rgba(0,0,128,1)', 'rgba(0,128,128,1)',
                    'rgba(248,127,81,1)', 'rgba(128,0,128,1)', 'rgba(128,128,0,1)'] * 100

    # ensure 0/255 for OpenCV
    bin255 = (patch > 0).astype('uint8') * 255

    # Morphological tidy
    k = int(params['gaussian_kernel'])
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(bin255, (k, k), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(blurred, kernel, iterations=int(params['dilate_iteration']))
    eroded = cv2.erode(dilated, kernel, iterations=int(params['erode_iteration']))
    _, binary = cv2.threshold(eroded, 127, 255, cv2.THRESH_BINARY)

    # External contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    out = {}
    for cnt in contours:
        if len(cnt) <= 10:
            continue

        # Optional contour simplification (smoother polygons)
        eps = approx_eps_ratio * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, eps, True)

        pts = []
        for pt in cnt:
            xpt = offc + (c + pt[0][0]) * l
            ypt = offr + (r + pt[0][1]) * l
            pts.append(f"{xpt},{ypt}")
        if not pts:
            continue

        poly = " ".join(pts)
        id_ = str(uuid.uuid4())
        # map label (1,2,3,5) -> index (0,1,2,3,4...) for colors
        color_idx = int(label) - 1 if int(label) > 0 else 0

        out[id_] = {
            "type": "Annotation",
            "body": [{
                "type": "TextualBody",
                "value": str(lset[label]),
                "purpose": "commenting",
                "creator": {"id": "tester", "name": "rainer"},
                "created": str(time.ctime()),
                "modified": str(time.ctime())
            }],
            "target": {
                "selector": {
                    "type": "SvgSelector",
                    "value": f"<svg><polygon points=\"{poly}\"></polygon></svg>"
                }
            },
            "objStrokeColor": color_stroke[color_idx],
            "objFillColor": color_fill[color_idx],
            "@context": "http://www.w3.org/ns/anno.jsonld",
            "id": id_
        }
    return out


def getPoints(x):
    """Parse polygon/ellipse or fragment; return points at original JSON scale."""
    if x['target']['selector']['type'] == 'FragmentSelector':
        t = x['target']['selector']['value']
        t = t[t.find(':')+1:].split(',')
        t = list(map(float, t))
        mp = [t[0]+t[2]/2, t[1]+t[3]/2]
        m  = [t[0], t[1], t[0]+t[2], t[1]+t[3]]
        pm = [(t[0], t[1]), (t[0]+t[2], t[1]), (t[0]+t[2], t[1]+t[3]), (t[0], t[1]+t[3])]
    else:
        t = x['target']['selector']['value']
        doc = minidom.parseString(t)
        if doc.getElementsByTagName('polygon') != []:
            t = [path.getAttribute('points') for path in doc.getElementsByTagName('polygon')][0]
            doc.unlink()
            t = t.split(' ')
            pm = []
            mpx, mpy = [], []
            for p in t:
                xp = float(p.split(',')[0]); yp = float(p.split(',')[1])
                mpx.append(xp); mpy.append(yp); pm.append((xp, yp))
            mp = [np.mean(mpx), np.mean(mpy)]
            m  = [min(mpx), min(mpy), max(mpx), max(mpy)]
        elif doc.getElementsByTagName('ellipse') != []:
            cx = float([path.getAttribute('cx') for path in doc.getElementsByTagName('ellipse')][0])
            cy = float([path.getAttribute('cy') for path in doc.getElementsByTagName('ellipse')][0])
            rx = float([path.getAttribute('rx') for path in doc.getElementsByTagName('ellipse')][0])
            ry = float([path.getAttribute('ry') for path in doc.getElementsByTagName('ellipse')][0])
            mp = [cx, cy]
            m  = [cx-rx, cy-ry, cx+rx, cy+ry]
            e  = Ellipse((cx, cy), rx, ry, 0)
            pm = [(each[0], each[1]) for each in e.get_verts()]
        else:
            pm, m, mp = [], [], []
    return pm, m, mp


# ------------------------------
# Main
# ------------------------------
if __name__ == '__main__':
    from openslide.deepzoom import DeepZoomGenerator

    # --------- INPUTS ----------
    fname = 'FL_Eval__scan_1404_.svs'
    annofile = f"//mfad/mcfdept/ai-ebus/AI-EBUS Project/z/Kidney/External Data/Annotations/test_annotation@mayo.edu/{fname}.json"
    slide = osl.OpenSlide(f"//mfad/mcfdept/ai-ebus/AI-EBUS Project/z/Kidney/External Data/{fname}")

    # ROI level (annotation rasterization level)
    dslevel = 1                             # draw mask at this level
    factor  = slide.level_downsamples[dslevel]      # float scale vs level-0
    slevel  = int(np.log2(factor))                  # must be integer (2 for 4x)

    # Read ROI-sized image and make an empty mask
    img  = slide.read_region((0, 0), dslevel, slide.level_dimensions[dslevel])
    mask = Image.new('L', img.size, 0)

    with open(annofile, 'r') as rd:
        anno = json.loads(rd.read())

    # Rasterize JSON polygons to mask at dslevel
    for each in list(anno['anno'].values()):
        pm, _, _ = getPoints(each)
        # scale down to dslevel
        pm = [(p[0] / factor, p[1] / factor) for p in pm]
        ImageDraw.Draw(mask).polygon(pm, outline=255, fill=255)

    # --------- DeepZoom setup ----------
    ts   = 1024   # physical tile size we ask from DeepZoom
    sts  = 1024    # same here (kept for clarity)
    opts = {'tile_size': sts, 'overlap': 0, 'limit_bounds': True}

    dgz1   = DeepZoomGenerator(slide, **opts)
    blevel = 0
    level  = dgz1.level_count - blevel - 1          # highest-res DZ level
    cols, rows = dgz1.level_tiles[level]

    # --------- Build tile list intersecting ROI ----------
    # At blevel=0, mask grid step in mask space = sts / factor
    zs = (sts / factor) * (2 ** blevel)             # here: sts/factor
    ncols = int(np.ceil(mask.size[0] / zs))
    nrows = int(np.ceil(mask.size[1] / zs))

    ptiles = {}
    rlist  = []
    for i in range(ncols):
        temp = []
        for j in range(nrows):
            x0 = int(i * zs);  y0 = int(j * zs)
            x1 = int((i + 1) * zs);  y1t = int((j + 1) * zs)
            stile = np.array(mask.crop((x0, y0, x1, y1t)))
            if np.any(stile > 0):
                temp.append((i, j))
                rlist.append(j)
        if temp:
            ptiles[i] = temp

    if not rlist:
        print('[WARN] No tiles under ROI — check annotations & dslevel.')
        sys.exit(0)

    y1 = min(rlist)
    y2 = max(rlist)
    xcols = list(ptiles.keys())

    # --------- Model / stitching ----------
    model.initModel()

    # Your model.run3 returns a 1024x1024 class map per tile; we stitch at 1/4 (256)
    DS_PRED      = 2 ** 2                              # 4
    TILE_OUT     = int(sts / DS_PRED)                  # 256
    smask_height = (y2 - y1 + 1) * TILE_OUT
    smask_width  = (xcols[-1] - xcols[0] + 1) * TILE_OUT
    smask = np.zeros((smask_height, smask_width), dtype='uint8')

    m = xcols[0]
    k = y1

    for i in xcols:
        for (_, r) in ptiles[i]:
            # Bounds guard
            if not (0 <= i < cols and 0 <= r < rows):
                print(f'[WARN] skip OOB tile ({i},{r}) at level {level}')
                continue
            tile = dgz1.get_tile(level, (i, r))
            if tile.mode != 'RGB':
                tile = tile.convert('RGB')

            tmask = model.run3(np.array(tile))  # 1024x1024 labels
            tmask = np.array(Image.fromarray(tmask).resize((TILE_OUT, TILE_OUT), Image.NEAREST))

            rr0 = (r - k) * TILE_OUT;  rr1 = rr0 + TILE_OUT
            cc0 = (i - m) * TILE_OUT;  cc1 = cc0 + TILE_OUT
            smask[rr0:rr1, cc0:cc1] = tmask

    # --------- Enforce ROI (clip predictions to red mask) ----------
    # Global origin of stitched smask in level-0 px:
    offx_lvl0 = xcols[0] * sts
    offy_lvl0 = y1        * sts

    # Convert that origin to dslevel pixels:
    soffx = int(offx_lvl0 / (2 ** slevel))
    soffy = int(offy_lvl0 / (2 ** slevel))

    # Relationship between mask spaces: #ROI pixels per smask pixel
    ds_ratio = 2 ** (slevel - blevel)                 # here: 2**slevel

    # How big a crop from the ROI mask corresponds to smask?
    # (DS_PRED = 4 means: smask pixel spans 4x4 level-0 px)
    xfactor = 2                                       # because DS_PRED = 2**2
    h_need = int((2 ** xfactor) * smask.shape[0] / ds_ratio)
    w_need = int((2 ** xfactor) * smask.shape[1] / ds_ratio)

    roi_crop = np.array(mask)[soffy:soffy + h_need, soffx:soffx + w_need]
    roi_resz = Image.fromarray(roi_crop, mode='L').resize((smask.shape[1], smask.shape[0]), Image.NEAREST)
    pmask    = (np.array(roi_resz) > 0).astype('uint8')

    # Hard clip
    smask = smask * pmask

    # --------- To polygons ----------
    annotations = {'anno': {}}

    # Classes & postproc
    classlist = [1, 2, 3, 5]
    params = {
        1: {'gaussian_kernel': 5, 'dilate_iteration': 5, 'erode_iteration': 5},  # Glomerulus
        2: {'gaussian_kernel': 5, 'dilate_iteration': 5, 'erode_iteration': 5},  # Glom-sclerotic
        3: {'gaussian_kernel': 5, 'dilate_iteration': 5, 'erode_iteration': 5},  # Artery
        5: {'gaussian_kernel': 5, 'dilate_iteration': 5, 'erode_iteration': 5},  # IFTA
    }
    lset = {1: 'glomerulus', 2: 'glomerulus-sclerotic', 3: 'artery', 5: 'ifta'}

    # area limits in smask pixels (not level-0)
    classSize    = {1: 1000, 2: 1000, 3: 1000, 5: 1000}
    classMaxSize = {1: 20000000, 2: 20000, 3: 25000, 5: 1000000000}

    # how many level-0 pixels does one smask pixel span? (blevel=0)
    l_px = DS_PRED

    try:
        for cls in classlist:
            # isolate class
            nmask = (smask == cls).astype('uint8')

            # label connected components (skip background)
            nb, im_sep, stats, _ = cv2.connectedComponentsWithStats(nmask, connectivity=8)
            if nb <= 1:
                continue

            valid = np.where((np.arange(nb) != 0) &
                             (stats[:, 4] > classSize[cls]) &
                             (stats[:, 4] < classMaxSize[cls]))[0]
            for lab in valid:
                x = int(stats[lab, 0]);  y = int(stats[lab, 1])
                w = int(stats[lab, 2]);  h = int(stats[lab, 3])

                patch = np.zeros((h, w), dtype='uint8')
                patch[im_sep[y:y + h, x:x + w] == lab] = 1

                anno = convertContoursToZP(
                    patch=patch, c=x, r=y,
                    offc=offx_lvl0, offr=offy_lvl0,
                    l=l_px, label=cls, lset=lset, params=params[cls],
                    approx_eps_ratio=0.005  # tweak 0.005–0.02 for more/less smoothing
                )
                annotations['anno'].update(anno)
    except Exception as e:
        import traceback
        print('[ERROR] contour extraction failed:', e)
        traceback.print_exc()

    outname = fname + 'pred.json'
    with open(outname, "w") as f:
        json.dump(annotations, f)
    print(f"[INFO] Wrote: {os.path.abspath(outname)}")