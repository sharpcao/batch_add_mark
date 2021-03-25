import torch
from PIL import Image
from unet import UNet
from torchvision import transforms
import sys, getopt
to_img = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

def cleanpic(in_file, out_file,n = 1):
	md = torch.load('model_X323.pkl',map_location=torch.device('cpu'))
	md.eval()
	pic = Image.open(in_file).convert("RGB")
	w,h = pic.size
	pic = pic.resize((400, h * 400 // w ),Image.ANTIALIAS)
	tmp = to_tensor(pic).unsqueeze(0)
	with torch.no_grad():
		for i in range(n):
			print('+',end="")
			sys.stdout.flush()
			tmp = md(tmp)

	out_pic = to_img(tmp.squeeze(0))
	out_pic.save(out_file)

def main(argv):
	in_file = argv[0]
	out_file = 'out.png'
	repeat_num = 1
	try:
		opts,args = getopt.getopt(argv,"hi:o:n:")
	except getopt.GetoptError:
		print('cleanpic -i <inputfile> -o <outputfile> -n <repeat>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('cleanpic -i <inputfile> -o <outputfile> -n <repeat>')
			sys.exit()
		elif opt == '-i':
			in_file = arg 
		elif opt == '-o':
			out_file = arg
		elif opt == '-n':
			repeat_num = int(arg)
		else:
			print(arg)

	cleanpic(in_file,out_file,repeat_num)
	print('\ndone!')

if __name__ =='__main__':
	main(sys.argv[1:])


