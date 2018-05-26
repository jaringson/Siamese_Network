###
# This code randomly places an image on a white background and then 
# randomly skews the image. The skewing uses ImageMagick and so saves
# a temporary file to disk and then loads it again.
#
###
from PIL import Image
from random import randint
import numpy as np
from scipy.misc import imread, imresize, imsave, imshow
import os

def place_background(image, from_center, rotate, scale_rand):
    
    target = Image.open(image)

    background = Image.open('white-background.png')

    b_w, b_h = background.size

    delta_x = randint(0,b_w-269)
    delta_y = randint(0,b_h-203)
    background = background.crop((delta_x, delta_y, delta_x + 269, delta_y + 203))

    b_w, b_h = background.size
    
    target = target.convert("RGBA")

    foregrounds = []

    target_r = target
    p_deg = 0
    if rotate:
        p_deg = randint(0,360)
        target_r = target.rotate(p_deg, expand=True)

    if scale_rand:
        p_size = randint(18,24)
    else:
        p_size = 18
    p_scale = 1.0*p_size/b_w
    target_r = target_r.resize((p_size + randint(100,150) ,p_size+ randint(100,150)), Image.ANTIALIAS)

    t_w,t_h = target_r.size
    foregrounds.append(target_r)

    for foreground in foregrounds:

        p_w = randint(int(from_center * (b_w//2 - t_w//2)),int((b_w-t_w) - from_center * ((b_w-t_w) - ((b_w//2) - t_w//2))))
	p_h = randint(int(from_center * (b_h//2 - t_h//2)),int((b_h-t_h) - from_center * ((b_h-t_h) - ((b_h//2) - t_h//2))))
        
        background.paste(foreground, (p_w,p_h), foreground)

    ### For debugging
    #background.convert('L').show()
    return background.convert('L')


def run(image, random_skew = False, random_place = False, from_center = 0, rotate = False, scale_rand = True, crop=-1):

    img = []

    if random_place:
    	img = place_background(image, from_center, rotate, scale_rand)
    else:
        img = Image.open(image).convert('L')

    ### Cropping only for Certain Pratt data.
    ### Please don't use for other data.
    if crop != -1:
        width = img.size[0]
        height = img.size[1]
        img = img.crop((0,0,width-crop,height)) 
    img = img.resize((50, 50), Image.ANTIALIAS)

    ### Necessary for ImageMagick to work
    img.save('./tf_logs/temp.jpg')
    points = 4
    all_points = []
    
    quad_c1 = randint(0,4)
    quad_c2 = quad_c1
    while quad_c2 == quad_c1:
        quad_c2 = randint(0,4)
    
    c1_2 = [0]
    if random_skew:
        c1_2 = [quad_c1, quad_c2]

    ### This part chooses two points in diffent quadurants of the image 
    ### and randomly chooses where the points will be skewed to.
    for c in c1_2:
        if c == 0:
            img = np.array(img).flatten()
            list = img.tolist()
            return list
        if c == 1:
            ### 1st Quad Skewing
            all_points.append(randint(3*img.size[0]/16,4*img.size[0]/16))
            all_points.append(randint(3*img.size[1]/16,4*img.size[1]/16))
            all_points.append(randint(2*img.size[0]/16,3*img.size[0]/16))
            all_points.append(randint(2*img.size[1]/16,3*img.size[1]/16))
        if c == 2:
            ### 2nd Quad Skewing
            all_points.append(randint(8*img.size[0]/16,9*img.size[0]/16))
            all_points.append(randint(7*img.size[1]/16,8*img.size[1]/16))
            all_points.append(randint(9*img.size[0]/16,10*img.size[0]/16))
            all_points.append(randint(6*img.size[1]/16,7*img.size[1]/16))
        if c == 3:
            ### 3rd Quad Skewing
            all_points.append(randint(7*img.size[0]/16,8*img.size[0]/16))
            all_points.append(randint(8*img.size[1]/16,9*img.size[1]/16))
            all_points.append(randint(6*img.size[0]/16,7*img.size[0]/16))
            all_points.append(randint(9*img.size[1]/16,10*img.size[1]/16))
        if c == 4:
            ### 4th Quad Skewing
            all_points.append(randint(4*img.size[0]/16,5*img.size[0]/16))
            all_points.append(randint(4*img.size[1]/16,5*img.size[1]/16))
            all_points.append(randint(5*img.size[0]/16,6*img.size[0]/16))
            all_points.append(randint(5*img.size[1]/16,6*img.size[1]/16))
    
    ### ImageMagick works through saving to disk
    os.system("convert ./tf_logs/temp.jpg \
	-colorspace  RGB\
	-distort Shepards \
        '%d,%d %d,%d  %d,%d %d,%d' \
	./tf_logs/temp2.png"%(tuple(all_points))) 
    img = Image.open('./tf_logs/temp2.png')
    #img.show()
    img = np.array(img)
    img = img.flatten()
    list = img.tolist()
    
    return list


if __name__ == '__main__':
    ### For debugging. Modify as needed. 
    run('test.jpg', from_center=1)
