import torch

from ip_drit.models import AbsorbanceImGenerator
from ip_drit.models import AbsorbanceImGeneratorWithConvTranspose
from ip_drit.models import AbsorbanceImGeneratorWithInterpolation
from ip_drit.models import AttributeEncoder
from ip_drit.models import ContentEncoder
from ip_drit.models import RealFakeDiscriminator
from ip_drit.models import ReLUUpsample2xWithConvTranspose2d
from ip_drit.models import ReLUUpsample2xWithInterpolation


class TestModel:
    def test_2x_interpolation_based_upsampler_shape_matches_expected(self):
        m = ReLUUpsample2xWithInterpolation(in_channels=4, out_channels=2)
        x = torch.randn(1, 4, 256, 256)
        y = m(x)
        assert y.shape == torch.Size([1, 2, 512, 512])

    def test_2x_convtranspose2d_based_upsampler_shape_matches_expected(self):
        m = ReLUUpsample2xWithConvTranspose2d(in_channels=4, out_channels=3)
        x = torch.randn(1, 4, 256, 256)
        y = m(x)
        assert y.shape == torch.Size([1, 3, 512, 512])

    def test_content_encoder_output_shape_matches_expected(self):
        m = ContentEncoder(in_channels=3, num_stain_vectors=32)
        x = torch.randn(1, 3, 512, 512)
        y = m(x)
        assert y.shape == torch.Size([1, 32, 128, 128])

    def test_attribute_encoder_output_shape_matches_expected(self):
        m = AttributeEncoder(in_channels=3, out_channels=16, num_stain_vectors=32, downsampling_factor=4)
        x = torch.randn(1, 3, 512, 512)
        y = m(x)
        assert y.shape == torch.Size([1, 16, 32])

    def test_pixel_shuffling_generator_output_shape_matches_expected(self):
        cont_enc = ContentEncoder(in_channels=3, num_stain_vectors=32)
        attr_enc = AttributeEncoder(in_channels=3, out_channels=48, num_stain_vectors=32, downsampling_factor=4)
        gen = AbsorbanceImGenerator(downsampling_factor=4)
        x = torch.randn(1, 3, 512, 512)
        z_c = cont_enc(x)
        z_a = attr_enc(x)
        y = gen(z_c, z_a)
        assert y.shape == x.shape

    def test_interpolation_based_generator_output_shape_matches_expected(self):
        cont_enc = ContentEncoder(in_channels=3, num_stain_vectors=32)
        attr_enc = AttributeEncoder(in_channels=3, out_channels=8, num_stain_vectors=32, downsampling_factor=4)
        gen = AbsorbanceImGeneratorWithInterpolation(in_channels=8, downsampling_factor=4)
        x = torch.randn(1, 3, 512, 512)
        z_c = cont_enc(x)
        z_a = attr_enc(x)
        y = gen(z_c, z_a)
        assert y.shape == x.shape

    def test_conv_transpose2d_based_generator_output_shape_matches_expected(self):
        cont_enc = ContentEncoder(in_channels=3, num_stain_vectors=32)
        attr_enc = AttributeEncoder(in_channels=3, out_channels=16, num_stain_vectors=32, downsampling_factor=4)
        gen = AbsorbanceImGeneratorWithConvTranspose(in_channels=16)
        x = torch.randn(1, 3, 512, 512)
        z_c = cont_enc(x)
        z_a = attr_enc(x)
        y = gen(z_c, z_a)
        assert y.shape == x.shape

    def test_discriminator_shape_matches_expected(self):
        x = torch.randn(1, 3, 512, 512)
        m = RealFakeDiscriminator(in_channels=3)
        y = m(x)
        assert y.shape == torch.Size([1, 1, 8, 8])
