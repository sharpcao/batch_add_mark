from PIL import Image
import os
import random
from torchvision.data.utils  import Dataset, Dataload
from torchvision import transforms 

to_tensor = transforms.ToTensor()

class myDataset(Dataset):
	def __init__(self, path_org = 'origin', path_wm = 'watermark',n = 100):
		super().__init__()
		self.n = n
		self.path_org = path_org
		self.path_wm = path_wm
		self.imglist= os.list.file(self.path_org)
		self.imglist_size = len(self.imglist)
		self.wmlist = os.list.file(self.path_wm)
		self.wmlist_size = len(self.wmlist)

	def __len__(self)
		return self.n

	def random_clip(img):
		w,h = img.size()
		x = random.randint(0,w -200)
		y = random.randint(0,h- 200)
		return img.crop((x,y,x+200,y+200))

	def get_random_pos(big_box,small_box):
		bw,bh = big_box
		sw,sh = small_box
		x = random.randint(0,bw-sw)
		y = random.randint(0,bh-sh)
		return(x,y)

	def __getitem__(self,idx):
		ix = random.randint(0,self.imglist_size)
		iwm = random.randint(0,self.wmlist_size)

		img = Image.open(self.imglist[ix]).convert("RGB")
		imgwm = Image.open(self.wmlist[iwm]).convert("RGBA")
		r,g,b,a = imgwm.split()

		x,y = get_random_pos(img.size(),(200,200))
		img = img.crop((x,y,x+200,y+200))

		target = Image.new("RGBA",(200,200),(0,0,0,0))
		target.paste(img,(0,0))

		x,y = get_random_pos((200,200),imgwm.size())
		target.paste(imgwm,(x,y),a)

		src_t = to_tensor(img)
		wm_t = to_tensor(target)
		return {"src_t":src_t, "wm_t":wm_t}

