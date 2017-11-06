from .cycle_gan_model import CycleGANModel
from torchvision.transforms import ToPILImage, ToTensor
from torch.autograd import Variable
from PIL import Image
import torch
import util.util as util

class GlassesModel(CycleGANModel):
    def name(self):
        return 'GlassesModel'

    def initialize(self, opt):
        CycleGANModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_mask_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_mask_B = self.Tensor(nb, opt.output_nc, size, size)

        if self.isTrain:
            self.criterionMask = torch.nn.L1Loss()

    def set_input(self, input):
        CycleGANModel.set_input(self, input)

        AtoB = self.opt.which_direction == 'AtoB'
        input_mask_A = input['A_eyemask' if AtoB else 'B_eyemask']
        input_mask_B = input['B_eyemask' if AtoB else 'A_eyemask']
        self.input_mask_A.resize_(input_mask_A.size()).copy_(input_mask_A)
        self.input_mask_B.resize_(input_mask_B.size()).copy_(input_mask_B)

    def forward(self):
        CycleGANModel.forward(self)

        self.mask_A = self.input_mask_A
        self.mask_B = self.input_mask_B

        self.vmask_A = Variable(self.input_mask_A)
        self.vmask_B = Variable(self.input_mask_B)

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_MA = self.opt.lambda_MA
        lambda_MB = self.opt.lambda_MB
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        toPILImage = ToPILImage()
        toTensor = ToTensor()

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG_A.forward(self.real_A)
        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)
        # D_B(G_B(B))
        self.fake_A = self.netG_B.forward(self.real_B)
        pred_fake = self.netD_B.forward(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake, True)
        # Forward cycle loss
        self.rec_A = self.netG_B.forward(self.fake_B)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.rec_B = self.netG_A.forward(self.fake_A)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Mask loss A
        self.masked_real_A = torch.max(self.input_A, self.mask_A)
        self.masked_fake_B = torch.max(self.fake_B.detach(), self.vmask_A)
        self.loss_mask_A = self.criterionMask(self.masked_real_A, self.masked_fake_B) * lambda_MA

        # Mask loss B
        self.masked_real_B = torch.max(self.input_B, self.mask_B)
        self.masked_fake_A = torch.max(self.fake_A.detach(), self.vmask_B)
        self.loss_mask_B = self.criterionMask(self.masked_real_B, self.masked_fake_A) * lambda_MB

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_mask_A + self.loss_mask_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def get_current_errors(self):
        errors = CycleGANModel.get_current_errors(self)
        errors['M_A'] = self.loss_mask_A.data[0]
        errors['M_B'] = self.loss_mask_B.data[0]
        return errors

    def get_current_visuals(self):
        visuals = CycleGANModel.get_current_visuals(self)
        visuals['masked_real_A'] = util.tensor2im(self.masked_real_A)
        visuals['masked_fake_B'] = util.tensor2im(self.masked_fake_B.data)
        visuals['masked_real_B'] = util.tensor2im(self.masked_real_B)
        visuals['masked_fake_A'] = util.tensor2im(self.masked_fake_A.data)
        return visuals