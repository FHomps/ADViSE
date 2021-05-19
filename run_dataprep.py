# -- Dataset preparation utilities and commands --
# This file is intended to be run from Spyder, which is more convenient to use
# than Jupyter for those kinds of operations (easier visualization).
 
# Dataset preparation works by running the prepareDataset function twice:
    
# The first time, it will generate an empty mask image in png format.
# This image should be edited with any image edition software by covering
# unwanted spots (data from HiRise is not perfect) in pure red. Precision in
# the coverage is not important, only the middle pixel of each spot matters.

# The second time, the mask file will be loaded into a list of ignored spots.
# This list is compared with the list of any previous masks loaded from a
# pickled file; if the unpickled ignore list differs from the mask's (or does
# not exist), it will be updated and partitioned tensors for the satellite
# pictures and slope will be created.
# Additional .part.png partition files are also created, which provide a
# visual summary of the formatted datasets.

from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = 5000000000
import numpy as np
import struct
import torch
from os.path import join, exists
import os
import pickle

def printDict(d):
    for key, value in d.items():
        print(key + ": " + str(value))

# Loads PDS .IMG file header information into a dictionary
def getPDSHeader(filename):
    with open(filename) as f:
        def readline():
            s = []
            while len(s) == 0:
                s = f.readline().split()
            return s
        
        header = {}
        
        key = ''
        value = ''
        string_value = False
        while True:
            s = readline()
            if s[0] == 'END':
                break
            if s[0] != '/*':
                if not string_value and len(s) > 1 and s[1] == '=':
                    if key != '':
                        if len(value) == 1:
                            header[key] = value[0]
                        else:
                            header[key] = value
                    key = s[0]
                    if len(s) > 2 and s[2][0] == '"':
                        string_value = True
                        value = ' '.join(s[2:])[1:]
                        if s[-1][-1] == '"':
                            value = value[:-1]
                            string_value = False
                    else:
                        value = s[2:]                    
                else:
                    if string_value:
                        value += ' '.join(s)
                        if s[-1][-1] == '"':
                            value = value[:-1]
                            string_value = False
                    else:
                        value += s                         
                
        return header

def loadHeightmap(filename):
    header = getPDSHeader(filename)
    a = np.fromfile(filename, dtype='<f4').reshape(int(header["LINES"])+1, int(header["LINE_SAMPLES"]))[1:]
    return a, header

def getSlope(heightmap):
    G = np.gradient(heightmap)
    return np.sqrt(np.square(G[0].clip(-10, 10)) + np.square(G[1].clip(-10, 10))).clip(0, 1)

def dist(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
def angle(a, b): # Argument of the vector from point a to point b
    return np.angle(b[0] - a[0] + 1j * (b[1] - a[1]))

def rotate(a, r, c = (0, 0)): # Rotates point a by r radians with rotation center c
    v = a[0] - c[0], a[1] - c[1]
    cos = np.cos(r)
    sin = np.sin(r)
    return (cos * v[0] - sin * v[1] + c[0], sin * v[0] + cos * v[1] + c[1])
    

def absmod(x, m): # Modulus going into the negatives to minimize absolute value
    x = x % m
    if x > m / 2:
        return x - m
    return x

# Returns the rotation and cropping parameters to be applied to the image to straighten it
# Extra cropping is optionally applied to the top left of the image / dataset
def getCropFactors(heightmap, header, aggressive_crop = False, extra_crop = (0, 0)):
    lines = len(heightmap)
    cols = len(heightmap[0])
    missing_constant = struct.unpack('!f', bytes.fromhex(header["MISSING_CONSTANT"][3:-1]))

    top = None    
    left = None
    bot = None
    right = None
    for i in range(lines):
        for j in range(cols):
            if heightmap[i, j] != missing_constant:
                top = (j, lines - i)
                break
        if top != None:
            break
    for j in range(cols):
        for i in range(lines):
            if heightmap[i, j] != missing_constant:
                left = (j, lines - i)
                break
        if left != None:
            break
    for i in range(lines)[::-1]:
        for j in range(cols):
            if heightmap[i, j] != missing_constant:
                bot = (j, lines - i)
                break
        if bot != None:
            break
    for j in range(cols)[::-1]:
        for i in range(lines):
            if heightmap[i, j] != missing_constant:
                right = (j, lines - i)
                break
        if right != None:
            break
    
    # Since the image is not a perfect rectangle, to preserve data integrity, rotation should be done
    # to align the biggest side of the rectangle
    if dist(top, left) > dist(bot, left):
        rot = -absmod(angle(top, left), np.pi / 2)
    else:
        rot = -absmod(angle(bot, left), np.pi / 2)

    # Compute the corners of the image once rotated
    c = (float(cols) / 2, float(lines) / 2)
    if rot > 0:
        tl, tr, bl, br = rotate(top, rot, c), rotate(right, rot, c), rotate(left, rot, c), rotate(bot, rot, c)
    else:
        tl, tr, bl, br = rotate(left, rot, c), rotate(top, rot, c), rotate(bot, rot, c), rotate(right, rot, c)
    
    
    if extra_crop == None:
        extra_crop = (0, 0)
    
    if aggressive_crop == True:
        crop_box = (
            max(tl[0], bl[0]) + extra_crop[0],
            lines - min(tl[1], tr[1]) + extra_crop[1],
            min(tr[0], br[0]),
            lines - max(bl[1], br[1])
        )
    elif aggressive_crop == "top":
        crop_box = (
            max(tl[0], bl[0]) + extra_crop[0],
            lines - min(tl[1], tr[1]) + extra_crop[1],
            max(tr[0], br[0]),
            lines - min(bl[1], br[1])
        )
    else:
        crop_box = (
            min(tl[0], bl[0]) + extra_crop[0],
            lines - max(tl[1], tr[1]) + extra_crop[1],
            max(tr[0], br[0]),
            lines - min(bl[1], br[1])
        )
    
    return rot * 180.0 / np.pi, crop_box

# Partitions the image into a tensor, ignoring the specified squares
def toTensor(img, res, ignore=[]):
    x_range = range(0, img.size[0] - res, res)
    y_range = range(0, img.size[1] - res, res)
    N = len(x_range) * len(y_range)
    T = torch.empty((N - len(ignore), 1, res, res), dtype=torch.uint8)
    i_before = 0
    i_after = 0
    for x in x_range:
        for y in y_range:
            if i_before not in ignore:
                T[i_after] = torch.from_numpy(np.array(img.crop((x, y, x + res, y + res))))
                i_after += 1
            i_before += 1
    return T

def drawOutlinedText(drawer, x, y, text, font):
    shadow_coords = [(i, j) for i in range(x-1, x+2) for j in range(y-1, y+2) if not (i == x and j == y)]
    for ij in shadow_coords:
        drawer.text(ij, text, 0, font)
    drawer.text((x, y), text, 255, font)


# Saves a visual summary of the partition to the disk
def printPartition(img, res, filename, ignore=[], resize_factor=1, onlyGrid = False):
    if resize_factor == 1:
        imgc = img.copy()
    else:
        imgc = img.resize((round(img.size[0] * resize_factor), round(img.size[1] * resize_factor)))

    # /!\ If resize_factor does not properly divide res, the partitioning will drift off in accuracy, so be careful.
    res = round(res * resize_factor)
    x_range = range(0, imgc.size[0], res)
    y_range = range(0, imgc.size[1], res)
    drawer = ImageDraw.Draw(imgc)
    
    for x in x_range:
        drawer.line(((x, 0), (x, y_range[-1])), fill=0, width=5)
    for y in y_range:
        drawer.line(((0, y), (x_range[-1], y)), fill=0, width=5)
    
    if not onlyGrid:
        font = ImageFont.truetype("OpenSans.ttf", max(10, round(res / 70) * 10))
        i_before = 0
        i_after = 0
        
        for x in x_range[:-1]:
            for y in y_range[:-1]:
                if i_before in ignore:
                    drawer.line(((x+res, y), (x, y+res)), fill=0, width=6)
                    drawer.line(((x+res, y), (x, y+res)), fill=255, width=2)
                    drawOutlinedText(drawer, x+5, y, str(i_before), font)
                else:
                    drawOutlinedText(drawer, x+5, y, str(i_before) + " (" + str(i_after) + ')', font)
                    i_after += 1
                i_before += 1
    
    for x in x_range:
        drawer.line(((x, 0), (x, y_range[-1])), fill=255, width=3)
    for y in y_range:
        drawer.line(((0, y), (x_range[-1], y)), fill=255, width=3)
    
    imgc.save(filename)

# Makes it easier to manually type an ignore list by expanding ranges when needed
def expandIgnore(compressed_ignore):
    ignore = list()
    for item in compressed_ignore:
        try:
            ignore.extend(range(item[0], item[1]+1))
        except TypeError:
            ignore.append(item)
    return ignore

# Load an ignore list from a heavily resized mask image
def loadIgnoreFromMask(filename, res, resize_factor):
    mask = Image.open(filename)
    res = round(res * resize_factor)
    center_offset = round(res / 2)
    x_range = range(center_offset, mask.size[0] - center_offset, res)
    y_range = range(center_offset, mask.size[1] - center_offset, res)
    ignore_list = []
    i = 0
    for x in x_range:
        for y in y_range:
            if mask.getpixel((x, y)) == (255, 0, 0):
                ignore_list.append(i)
            i += 1
    return ignore_list

def prepareDataset(zone_name, grid_size=1024, pic_res_multiplier=4, variant=1, use_ignore_file=True, extra_crop=None):
    global G
    print("Zone: " + zone_name)
    
    if grid_size % pic_res_multiplier != 0:
        print("Warning: grid size is not a multiple of pic_res_multiplier")
    grid_size_hm = round(grid_size / pic_res_multiplier)
    
    zone_part_dir = join(parts_dir, zone_name + '_' + str(grid_size))
    os.makedirs(zone_part_dir, exist_ok=True)
    
    print("Loading heightmap...")
    H, header = loadHeightmap(join(hmdir, zone_name + ".IMG"))
    print("Computing slope...")
    G_np = getSlope(H)
    print("Computing image rotation correction factors...")
    rot, box = getCropFactors(H, header, aggressive_crop="top", extra_crop=extra_crop)
    G_np = (G_np * 255).astype(np.uint8)
    G = Image.fromarray(G_np)
    del G_np
    print("Correcting slope map rotation...")
    G = G.rotate(rot).crop(np.round(box))
    
    mask_file = join(zone_part_dir, "mask.png")
    
    if not exists(mask_file):
        print("Mask file doesn't exist, creating empty mask.")
        printPartition(G, grid_size_hm, join(zone_part_dir, "emptymask.png"), onlyGrid=True)
        return
    
    print("Loading mask file...")
    ignore = loadIgnoreFromMask(mask_file, grid_size_hm, 1)
    print("Mask loaded, ignores " + str(len(ignore)) + " spots.")
    
    if use_ignore_file:
        ignore_file = join(zone_part_dir, "ignore.pickle")
        if not exists(ignore_file):
            print("Ignore file not found, creating.")
            pickle.dump(ignore, open(ignore_file, 'wb'))
        else:
            print("Comparing existing ignore file to mask...")
            loaded_ignore = pickle.load(open(ignore_file, 'rb'))
            if loaded_ignore == ignore:
                cont_ans = input("Unchanged ignore list, still create tensors?\n")
                if (cont_ans.lower() not in ("y", "yes")):
                    return
            else:
                print("Ignore list changed, updating ignore file.")
                pickle.dump(ignore, open(ignore_file, 'wb'))

    print("Creating slope tensor...")    
    G_T = toTensor(G, grid_size_hm, ignore)
    torch.save(G_T, join(zone_part_dir, "slope.ts"))
    print("Printing slope partition...")
    printPartition(G, grid_size_hm, join(zone_part_dir, "slope.part.png"), ignore)
    del G, G_T    
    
    print("Loading satellite picture...")
    I = Image.open(join(picdir, zone_name + '_' + str(variant) + ".bmp"))
    print("Correcting image rotation...")
    I = I.rotate(rot)
    I = I.crop(np.array(box) * pic_res_multiplier)
    print("Creating satellite tensor...")
    I_T = toTensor(I, grid_size, ignore)
    torch.save(I_T, join(zone_part_dir, "sat_" + str(variant) + ".ts"))
    print("Printing satellite partition...")
    printPartition(I, grid_size, join(zone_part_dir, "sat_" + str(variant) + ".part.png"), ignore, 1 / pic_res_multiplier)
    del I, I_T



#%% Data prep

hmdir = "data/heightmaps"
picdir = "data/satellite_pictures"
parts_dir = "data/partitioned_datasets"

zones = (
    "Crommelin",
    "Firsoff", 
    "Hellespontus", 
    "Mawrth_Vallis",
    "Utopia_Planitia",
)

extraCrops = (
    (100, 0),
    None,
    None,
    None,
    None
)

grid_size = 2048

#%%
for z, xcrop in zip(zones, extraCrops):
    prepareDataset(z, grid_size=grid_size, variant=2, extra_crop = xcrop)
    print()

#%% Postprocessing
from postprocessing import getViabilityMap, saveProcessedSample

for z in zones:
    dsdir = z + '_' + str(grid_size)
    print("Loading slope tensor...")
    gt_T = torch.load(join(parts_dir, dsdir, "slope.ts"))
    saveProcessedSample(gt_T, join(parts_dir, dsdir, "slope_processed.sample.png"))
    
    print("Computing full VMap...")
    pgt_T = getViabilityMap(gt_T)
    print("Saving processed tensor...")
    torch.save(pgt_T, join(parts_dir, dsdir, "slope_processed.ts"))
    
#%% Unification

unified_ds_name = "Unified"
unified_ds_dir = join(parts_dir, unified_ds_name + '_' + str(grid_size))
os.makedirs(unified_ds_dir, exist_ok=True)

#%%

U_T = torch.ByteTensor([])
for z in zones:
    dsdir = join(parts_dir, z + '_' + str(grid_size))
    U_T = torch.cat((U_T, torch.load(join(dsdir, "sat_1.ts"))), 0)
np.save(join(unified_ds_dir, "sat_1.npy"), U_T.numpy())

#%%
U_T = torch.ByteTensor([])
for z in zones:
    dsdir = join(parts_dir, z + '_' + str(grid_size))
    U_T = torch.cat((U_T, torch.load(join(dsdir, "sat_2.ts"))), 0)
np.save(join(unified_ds_dir, "sat_2.npy"), U_T.numpy())

#%%
U_T = torch.ByteTensor([])
for z in zones:
    dsdir = join(parts_dir, z + '_' + str(grid_size))
    U_T = torch.cat((U_T, torch.load(join(dsdir, "slope.ts"))), 0)
np.save(join(unified_ds_dir, "slope.npy"), U_T.numpy())

#%%
U_T = torch.ByteTensor([])
for z in zones:
    dsdir = join(parts_dir, z + '_' + str(grid_size))
    U_T = torch.cat((U_T, torch.load(join(dsdir, "slope_processed.ts"))), 0)
np.save(join(unified_ds_dir, "slope_processed.npy"), U_T.numpy())