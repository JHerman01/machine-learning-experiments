import cv2
import os
import re

'''
    Video generating techniques borrowed from:
    https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python/44948030
'''
def generate_video(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
    images.sort(key=natural_keys)
    
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _tail = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()



'''
    Natural sorting techniques borrowed from:
    https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
'''
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]