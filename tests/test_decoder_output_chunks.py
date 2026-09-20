import unittest
import torch
from pixal3d.models.sc_vaes.sparse_unet_vae import chunked_decoder_output


class DecoderOutputChunks(unittest.TestCase):
    @torch.no_grad()
    def test_matches_norm_then_linear_with_tail_and_dtype_conversion(self):
        torch.manual_seed(17)
        for channels, outputs in [(64, 7), (64, 6), (128, 8)]:
            layer = torch.nn.Linear(channels, outputs)
            x = torch.randn(103, channels).half()
            expected = layer(torch.nn.functional.layer_norm(x.float(), (channels,)))
            actual = chunked_decoder_output(x, layer, torch.float32, chunk_rows=17)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
