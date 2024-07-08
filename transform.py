from PIL import Image, ImageFilter, ImageEnhance
import random
import torchvision.transforms.functional as TF


class RandomFlipOrRotate(object):
    def __call__(self, sample):
        img1, img2 = \
            sample['img1'], sample['img2']

        rand = random.random()
        if rand < 1 / 6:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)

        elif rand < 2 / 6:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)

        elif rand < 3 / 6:
            img1 = img1.transpose(Image.ROTATE_90)
            img2 = img2.transpose(Image.ROTATE_90)

        elif rand < 4 / 6:
            img1 = img1.transpose(Image.ROTATE_180)
            img2 = img2.transpose(Image.ROTATE_180)

        elif rand < 5 / 6:
            img1 = img1.transpose(Image.ROTATE_270)
            img2 = img2.transpose(Image.ROTATE_270)


        return {'img1': img1, 'img2': img2}

class RandomFlipOrRotate_MINE(object):
    def __call__(self, sample):
        img1, img2 = \
            sample['img1'], sample['img2']

        rand = random.randint(0,4)

        if rand==0:
            #print('000')
            angles = [90, 180, 270]
            index = random.randint(0, 2)
            angle = angles[index]
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)
        elif rand==1:
            #print('1111')
            img1 = img1
            img2 = img2
        elif rand==2:
            #print('2222')
            enhancer = ImageEnhance.Sharpness(img1)
            img1 = enhancer.enhance(3)
            enhancer = ImageEnhance.Sharpness(img2)
            img2 = enhancer.enhance(3)            
        elif rand==3:
           # print('3333')
            enhancer = ImageEnhance.Contrast(img1)
            img1 = enhancer.enhance(0.6)
            enhancer = ImageEnhance.Contrast(img2)
            img2 = enhancer.enhance(0.6)
        else:
            #print('4444')
            img1 = img1.filter(ImageFilter.GaussianBlur(radius=1.5))
            img2 = img2.filter(ImageFilter.GaussianBlur(radius=1.5))

        return {'img1': img1, 'img2': img2}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': (img1, img2),
                'label': mask}

class RandomVerticalFlip(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': (img1, img2),
                'label': mask}

class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.75:
            rotate_degree = random.choice(self.degree)
            img1 = img1.transpose(rotate_degree)
            img2 = img2.transpose(rotate_degree)
            mask = mask.transpose(rotate_degree)

        return {'image': (img1, img2),
                'label': mask}


'''
class GBlur(object):
    def __call__(self, sample):
        img1, img2, change = \
            sample['img1'], sample['img2'], sample['change']

        rand_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=1.5))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=1.5))

        return {'img1': img1, 'img2': img2, 'change': change}

class Sharp(object):
    def __call__(self, sample):
        img1, img2, change = \
            sample['img1'], sample['img2'], sample['change']

        rand_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        enhancer = ImageEnhance.Sharpness(img1)
        img1 = enhancer.enhance(3)
        enhancer = ImageEnhance.Sharpness(img2)
        img2 = enhancer.enhance(3)

        return {'img1': img1, 'img2': img2, 'change': change}
    

class Contrast(object):
    def __call__(self, sample):
        img1, img2, change = \
            sample['img1'], sample['img2'], sample['change']

        rand_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        enhancer = ImageEnhance.Contrast(img1)
        img1 = enhancer.enhance(0.6)
        enhancer = ImageEnhance.Contrast(img2)
        img2 = enhancer.enhance(0.6)

        return {'img1': img1, 'img2': img2, 'change': change}
    
class Nothing(object):
    def __call__(self, sample):
        img1, img2, change = \
            sample['img1'], sample['img2'], sample['change']

        return {'img1': img1, 'img2': img2, 'change': change}
'''