#---------------------------------------------------------------------------
# See pp.README.txt for instructions.
#---------------------------------------------------------------------------

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn

import functools

#---------------------------------------------------------------------------

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer



##############################################################################
# Classes
##############################################################################


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(3, ngf, input_nc=3, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


#---------------------------------------------------------------------------

class Pix2PixModel:

	def __init__(self,projectname,resize=True):
		self.device = torch.device('cpu')

		norm_layer = get_norm_layer(norm_type='batch')
		self.netG = UnetGenerator(8, 64, norm_layer=norm_layer, use_dropout=False)

		self.projectname = projectname
		self.load_weights()

		if resize:
			self.prepare_image = self.prepare_image_resize
			self.postpare_image = self.postpare_image_resize
		else:
			self.prepare_image = self.prepare_image_crop
			self.postpare_image = self.postpare_image_crop

	def load_weights(self):
		loadpath = os.path.join('pre-trained',f'{self.projectname}.pth')

		if not os.path.exists(loadpath):
			loadpath = os.path.join('pre-trained',self.projectname,f'{self.projectname}.pth')

		if not os.path.exists(loadpath):
			loadpath = os.path.join('pre-trained',self.projectname,'latest_net_G.pth')

		if not os.path.exists(loadpath):
			loadpath = os.path.join('checkpoints',self.projectname,'latest_net_G.pth')

		self.netG.load_state_dict(torch.load(loadpath, map_location=self.device))

	def prepare_image_crop(self,imagedata):
		A = imagedata[:,:256]/255.0
		A = A.transpose(2, 0, 1)
		return torch.from_numpy(A[np.newaxis,:,:,:]).type(torch.FloatTensor)

	def postpare_image_crop(self,imagedata):
		B = np.transpose(imagedata.numpy(),(2,3,1,0))[:,:,:,0]
		return cv2.cvtColor(B*255,cv2.COLOR_RGB2BGR)

	def prepare_image_resize(self,imagedata):
		self.original = (imagedata.shape[1],imagedata.shape[0])
		A = cv2.resize(imagedata,(256,256))/255.0
		A = A.transpose(2, 0, 1)
		return torch.from_numpy(A[np.newaxis,:,:,:]).type(torch.FloatTensor)

	def postpare_image_resize(self,imagedata):
		B = np.transpose(imagedata.numpy(),(2,3,1,0))[:,:,:,0]
		return cv2.cvtColor(cv2.resize(B,self.original)*255,cv2.COLOR_RGB2BGR)

	def forward(self, input):
		real_A = self.prepare_image(input)
		with torch.no_grad():
			image = self.netG(real_A.to(self.device))  # G(A)
		return self.postpare_image(image)

	def project(self):
		return self.projectname

#---------------------------------------------------------------------------

if __name__ == '__main__':
	resize,project,inputpath = True,sys.argv[1],sys.argv[2]

	if sys.argv[1] == 'resize':
		resize,project,inputpath = True,sys.argv[2],sys.argv[3]

	if sys.argv[1] == 'crop':
		resize,project,inputpath = False,sys.argv[2],sys.argv[3]

	model = Pix2PixModel(project,resize)
	os.makedirs(os.path.join('results',project),exist_ok=True)

	if os.path.isdir(inputpath):
		for fname in os.listdir(inputpath):
			if is_image_file(fname):
				cv2.imwrite(os.path.join('results',project,fname),model.forward(cv2.imread(os.path.join(inputpath, fname))))

	elif is_image_file(inputpath):
		cv2.imwrite(os.path.join('results',project,os.path.basename(inputpath)),model.forward(cv2.imread(inputpath)))

#---------------------------------------------------------------------------
