import torch
import torch.nn as nn
import torch.nn.functional as F
from .MLP import MLP

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        in_ch = 3
        try:
            if opt.use_front_normal:
                in_ch += 3
            if opt.use_back_normal:
                in_ch += 3
        except:
            pass

        self.image_filter = HGFilter(opt.num_stack, opt.hg_depth, in_ch, opt.hg_dim, 
                                     opt.norm, opt.hg_down, False)

        self.mlp = MLP(
            filter_channels=opt.mlp_dim,
            merge_layer=opt.merge_layer,
            res_layers=opt.mlp_res_layers,
            norm=opt.mlp_norm,
            last_op=nn.Sigmoid())

        self.spatial_enc = DepthNormalizer(opt)

        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.phi = None

        self.intermediate_preds_list = []

        self.netF = None
        self.netB = None
        try:
            if opt.use_front_normal:
                self.netF = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")
            if opt.use_back_normal:
                self.netB = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")
        except:
            pass
        self.nmlF = None
        self.nmlB = None

    def forward(self, images):
        nmls = []
        with torch.no_grad():
            if self.netF is not None:
                self.nmlF = self.netF.forward(images).detach()
                nmls.append(self.nmlF)
            if self.netB is not None:
                self.nmlB = self.netB.forward(images).detach()
                nmls.append(self.nmlB)
        if len(nmls) != 0:
            nmls = torch.cat(nmls,1)
            if images.size()[2:] != nmls.size()[2:]:
                nmls = nn.Upsample(size=images.size()[2:], mode='bilinear', align_corners=True)(nmls)
            images = torch.cat([images,nmls],1)

        self.im_feat_list, self.normx = self.image_filter(images)

        self.im_feat_list = [self.im_feat_list[-1]]

        return self.normx, self.im_feat_list[-1]
class Discriminator(HGPIFuNetwNML):
    def __init__(self, opt):
        super(Discriminator, self).__init__(opt, projection_mode='orthogonal', criteria={'occ': nn.MSELoss()})
        self.name = 'hg_pifu'

    def forward(self, points, calibs, transforms=None):
        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]

        in_bb = (xyz >= -1) & (xyz <= 1)
        in_bb = in_bb[:, 0, :] & in_bb[:, 1, :] & in_bb[:, 2, :]
        in_bb = in_bb[:, None, :].detach().float()

        sp_feat = self.spatial_enc(xyz, calibs=calibs)

        intermediate_preds_list = []

        phi = None
        for i, im_feat in enumerate(self.im_feat_list):
            point_local_feat_list = [self.index(im_feat, xy), sp_feat]       
            point_local_feat = torch.cat(point_local_feat_list, 1)
            pred, phi = self.mlp(point_local_feat)
            pred = in_bb * pred

            intermediate_preds_list.append(pred)

        return intermediate_preds_list[-1], phi
import torch.optim as optim

# Initialize the Generator and Discriminator
generator = Generator(opt)
discriminator = Discriminator(opt)

# Define the GAN loss function (e.g., binary cross-entropy loss)
gan_loss_fn = nn.BCEWithLogitsLoss()

# Define the optimizers for the generator and discriminator
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (images, points, calibs, labels, gamma, points_nml, labels_nml) in enumerate(data_loader):
        # Train the discriminator
        real_labels = torch.ones(images.size(0), 1)
        fake_labels = torch.zeros(images.size(0), 1)

        generator_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()

        # Generate fake data using the generator
        noise = torch.randn(images.size(0), latent_dim)
        generated_norms, generated_images = generator(noise)

        # Calculate discriminator outputs for real and fake data
        real_output, _ = discriminator(images, points, calibs)
        fake_output, _ = discriminator(generated_images, points, calibs)

        # Calculate the discriminator loss
        real_loss = gan_loss_fn(real_output, real_labels)
        fake_loss = gan_loss_fn(fake_output, fake_labels)
        discriminator_loss = real_loss + fake_loss

        # Backpropagate and update the discriminator
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train the generator
        generator_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()

        # Generate fake data using the generator
        noise = torch.randn(images.size(0), latent_dim)
        generated_norms, generated_images = generator(noise)

        # Calculate discriminator output for generated data
        fake_output, _ = discriminator(generated_images, points, calibs)

        # Calculate the generator loss
        generator_loss = gan_loss_fn(fake_output, real_labels)

        # Backpropagate and update the generator
        generator_loss.backward()
        generator_optimizer.step()
